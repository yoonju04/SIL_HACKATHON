"""Reference trajectory generator ("autonomous driving" part = planning only).

Input : a lane-centre route (x, y points), e.g. CARLA map waypoints.
Output: Trajectory(s, x, y, yaw, kappa, v, t) for the controllers to track.

The planner never sends steering or pedal commands. Design rules (agreed in
the architecture review, 2026-10-21):
  1. Anchored to the lane centreline. The trajectory is generated once per
     scenario from the map, never re-planned from the current ego pose, so a
     controller fault shows up as tracking error instead of being absorbed by
     a new plan.
  2. Fault-unaware and deterministic: same route + same config -> same
     trajectory (needed for "same scenario without the fault" reruns).
  3. Physically feasible speed profile: v^2 * |kappa| <= min(a_lat_max,
     friction_margin * mu * g), plus acceleration / deceleration limits.
     Scenarios that violate this would fail even without any fault.

Geometry: natural cubic spline through route knots (PythonRobotics
CubicSpline, MIT) -> C2 path, continuous curvature. Optional lane change:
quintic lateral offset d(s) (Werling et al., ICRA 2010; PythonRobotics
QuinticPolynomial, MIT).

Frame: right-handed (x forward, y left, yaw counter-clockwise). CARLA's
left-handed frame is converted in carla_route.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .cubic_spline import CubicSpline2D
from .quintic import QuinticPolynomial

G = 9.81


@dataclass
class PlannerConfig:
    ds: float = 0.5  # output resolution [m]
    knot_spacing_m: float = 3.0  # spline knots taken from the route every ~3 m
    v_max: float = 15.0  # target cruise speed [m/s]
    a_lat_max: float = 3.0  # lateral acceleration limit [m/s^2] (comfort ~2, test scenarios up to friction_margin*mu*g)
    mu: float = 0.9  # assumed road friction
    friction_margin: float = 0.5  # use at most this fraction of mu*g laterally
    a_accel: float = 1.5  # longitudinal acceleration limit [m/s^2]
    a_decel: float = 2.5  # longitudinal deceleration limit [m/s^2]
    v_start: float = 0.0  # speed at the first point (spawned cars start at rest)
    v_end: float | None = None  # None: no forced stop at the route end
    v_floor_time: float = 0.5  # floor used only to integrate time stamps


@dataclass(frozen=True)
class LaneChange:
    s_start: float  # arc length where the manoeuvre starts [m]
    length: float  # longitudinal length of the manoeuvre [m]
    offset: float  # lateral offset at the end, left positive [m] (e.g. 3.5)


@dataclass
class Trajectory:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray  # unwrapped [rad]
    kappa: np.ndarray  # [1/m], left turn positive
    v: np.ndarray  # reference speed [m/s]
    t: np.ndarray  # reference time stamps [s] from the first point
    lane_x: np.ndarray  # lane centreline the route came from (before offsets)
    lane_y: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s_query):
        """Linear interpolation of all channels at arc length(s) s_query."""
        sq = np.clip(np.asarray(s_query, float), self.s[0], self.s[-1])
        return {k: np.interp(sq, self.s, getattr(self, k))
                for k in ("x", "y", "yaw", "kappa", "v", "t")}

    def to_dict(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}
        return d

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @staticmethod
    def load(path: str) -> "Trajectory":
        with open(path) as fh:
            d = json.load(fh)
        meta = d.pop("meta", {})
        return Trajectory(**{k: np.asarray(v, float) for k, v in d.items()}, meta=meta)


def _resample_polyline(x, y, spacing):
    """Points every `spacing` metres along a polyline (keeps both ends)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    seg = np.hypot(np.diff(x), np.diff(y))
    keep = np.r_[True, seg > 1e-6]
    x, y = x[keep], y[keep]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if s[-1] < 2 * spacing:
        raise ValueError(f"route too short ({s[-1]:.1f} m)")
    n = max(2, int(round(s[-1] / spacing)) + 1)
    sk = np.linspace(0.0, s[-1], n)
    return np.interp(sk, s, x), np.interp(sk, s, y)


def _sample_spline(x, y, cfg: PlannerConfig):
    kx, ky = _resample_polyline(x, y, cfg.knot_spacing_m)
    sp = CubicSpline2D(kx, ky)
    s = np.arange(0.0, sp.length, cfg.ds)
    px, py = sp.position(s)
    return s, px, py, np.unwrap(sp.yaw(s)), sp.curvature(s)


class ReferencePlanner:
    def __init__(self, cfg: PlannerConfig | None = None):
        self.cfg = cfg or PlannerConfig()

    # ------------------------------------------------------------ geometry
    def _apply_lane_changes(self, s, x, y, yaw, changes):
        d = np.zeros_like(s)
        d_now = 0.0
        for lc in sorted(changes, key=lambda c: c.s_start):
            q = QuinticPolynomial(d_now, 0.0, 0.0, d_now + lc.offset, 0.0, 0.0, lc.length)
            inside = (s >= lc.s_start) & (s <= lc.s_start + lc.length)
            after = s > lc.s_start + lc.length
            d[inside] = q.point(s[inside] - lc.s_start)
            d[after] = d_now + lc.offset
            d_now += lc.offset
        return x - d * np.sin(yaw), y + d * np.cos(yaw), d

    # --------------------------------------------------------------- speed
    def speed_profile(self, s, kappa):
        c = self.cfg
        a_lat = min(c.a_lat_max, c.friction_margin * c.mu * G)
        v = np.minimum(c.v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-9)))
        ds = np.diff(s)
        v[0] = min(v[0], c.v_start)
        for i in range(1, len(v)):  # acceleration limit (forward pass)
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * c.a_accel * ds[i - 1]))
        if c.v_end is not None:
            v[-1] = min(v[-1], c.v_end)
        for i in range(len(v) - 2, -1, -1):  # deceleration limit (backward pass)
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * c.a_decel * ds[i]))
        return v, a_lat

    def _time_stamps(self, s, v):
        vm = np.maximum(0.5 * (v[1:] + v[:-1]), self.cfg.v_floor_time)
        return np.r_[0.0, np.cumsum(np.diff(s) / vm)]

    # ---------------------------------------------------------------- plan
    def plan(self, route_x, route_y, lane_changes: tuple[LaneChange, ...] = ()) -> Trajectory:
        c = self.cfg
        s, x, y, yaw, kappa = _sample_spline(route_x, route_y, c)
        lane_x, lane_y = x.copy(), y.copy()
        d_max = 0.0
        if lane_changes:
            ox, oy, d = self._apply_lane_changes(s, x, y, yaw, lane_changes)
            d_max = float(np.max(np.abs(d)))
            s, x, y, yaw, kappa = _sample_spline(ox, oy, c)
        v, a_lat = self.speed_profile(s, kappa)
        t = self._time_stamps(s, v)
        lat_acc = v**2 * np.abs(kappa)
        meta = {
            "planner": "ReferencePlanner (cubic spline centreline + feasible speed profile)",
            "config": asdict(c),
            "lane_changes": [asdict(lc) for lc in lane_changes],
            "max_lateral_offset_m": d_max,
            "a_lat_limit": a_lat,
            "max_lat_acc": float(lat_acc.max()),
            "feasible": bool(lat_acc.max() <= c.friction_margin * c.mu * G + 1e-6),
            "max_abs_kappa": float(np.abs(kappa).max()),
        }
        return Trajectory(s=s, x=x, y=y, yaw=yaw, kappa=kappa, v=v, t=t,
                          lane_x=lane_x, lane_y=lane_y, meta=meta)
