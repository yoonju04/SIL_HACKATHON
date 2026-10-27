"""Glue for one driving episode: build a controller, step it, log, summarise.

This is the integration layer. It is the only code that imports all three
modules (planner, mpc_controller, lqr_controller); the modules themselves do
not import each other.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np

import lqr_controller as lqr_pkg
import mpc_controller as mpc_pkg

CONTROLLERS = ("mpc", "lqr")


def build_controller(kind: str, geometry: dict, dt: float, traj, mpc_cfg=None, lqr_cfg=None):
    """Create the lateral controller and give it the planner's trajectory."""
    if kind == "mpc":
        ctrl = mpc_pkg.MPCLateralController(mpc_pkg.VehicleParams(**geometry), dt, mpc_cfg)
        ctrl.set_reference(traj.x, traj.y, traj.yaw, traj.kappa, traj.s, traj.v, traj.t)
    elif kind == "lqr":
        ctrl = lqr_pkg.LQRLateralController(lqr_pkg.VehicleParams(**geometry), dt, lqr_cfg)
        ctrl.set_reference(traj.x, traj.y, traj.yaw, traj.kappa, traj.s)
    else:
        raise ValueError(f"controller must be one of {CONTROLLERS}")
    return ctrl


@dataclass
class LateralResult:
    steer: float | None
    s: float
    e_y: float
    e_psi: float
    kappa: float
    ok: bool
    status: str
    solve_ms: float | None
    qp_iterations: int | None


def step_controller(kind: str, ctrl, t, x, y, yaw, vx, vy, r) -> LateralResult:
    if kind == "mpc":
        o = ctrl.compute(x, y, yaw, vx, vy, r, t)
        return LateralResult(o.steer, o.s, float(o.state[0]), float(o.state[2]), o.kappa, o.ok,
                             o.status, 1e3 * o.solve_time_s, o.qp_iterations)
    o = ctrl.compute(x, y, yaw, vx, vy, r)
    return LateralResult(o.steer, o.s, float(o.state[0]), float(o.state[2]), o.kappa, True,
                         "ok", None, None)


class LaneMonitor:
    """Signed distance to the lane centreline (independent of the controllers)."""

    def __init__(self, lane_x, lane_y):
        self.x = np.asarray(lane_x, float)
        self.y = np.asarray(lane_y, float)
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)
        self.l2 = np.maximum(self.dx**2 + self.dy**2, 1e-12)

    def lateral_offset(self, px: float, py: float) -> float:
        a = np.clip(((px - self.x[:-1]) * self.dx + (py - self.y[:-1]) * self.dy) / self.l2, 0.0, 1.0)
        qx, qy = self.x[:-1] + a * self.dx, self.y[:-1] + a * self.dy
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        i = int(np.argmin(d2))
        cross = self.dx[i] * (py - self.y[i]) - self.dy[i] * (px - self.x[i])
        return math.copysign(math.sqrt(d2[i]), cross)


@dataclass
class StepRow:
    t: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float
    s: float
    v_ref: float
    e_y: float  # w.r.t. the planned trajectory (what the controller tracks)
    e_psi: float
    e_y_lane: float  # w.r.t. the lane centreline (independent check)
    kappa: float
    steer_rad: float
    ok: bool
    status: str
    solve_ms: float | None
    qp_iterations: int | None


class EpisodeLog:
    def __init__(self):
        self.rows: list[StepRow] = []

    def add(self, row: StepRow):
        self.rows.append(row)

    def summary(self, dt: float, route_length: float | None = None) -> dict:
        if not self.rows:
            return {"steps": 0}
        a = {k: np.array([getattr(r, k) for r in self.rows], dtype=float)
             for k in ("t", "e_y", "e_y_lane", "steer_rad", "vx", "s")}
        moving = a["vx"] > 1.0  # ignore the launch from standstill
        ey = a["e_y"][moving] if moving.any() else a["e_y"]
        rate = np.abs(np.diff(a["steer_rad"])) / dt if len(self.rows) > 1 else np.zeros(1)
        solve = np.array([r.solve_ms for r in self.rows if r.solve_ms is not None])
        iters = np.array([r.qp_iterations for r in self.rows if r.qp_iterations is not None])
        out = {
            "steps": len(self.rows),
            "duration_s": float(a["t"][-1]),
            "distance_m": float(a["s"][-1] - a["s"][0]),
            "route_completed": bool(route_length is not None and a["s"][-1] > route_length - 20.0),
            "max_abs_e_y_m": float(np.max(np.abs(ey))),
            "rms_e_y_m": float(np.sqrt(np.mean(ey**2))),
            "max_abs_e_y_lane_m": float(np.max(np.abs(a["e_y_lane"]))),
            "max_steer_deg": float(np.degrees(np.max(np.abs(a["steer_rad"])))),
            "max_steer_rate_dps": float(np.degrees(rate.max())),
            "mean_speed_mps": float(a["vx"].mean()),
            "not_ok_steps": int(sum(not r.ok for r in self.rows)),
        }
        if solve.size:
            out.update(solve_ms_mean=float(solve.mean()), solve_ms_p99=float(np.percentile(solve, 99)),
                       solve_ms_max=float(solve.max()))
        if iters.size:
            out.update(qp_iterations_mean=float(iters.mean()), qp_iterations_max=int(iters.max()))
        return out

    def save(self, out_dir: str, traj, meta: dict, summary: dict):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "steps.jsonl"), "w") as fh:
            for r in self.rows:
                fh.write(json.dumps(asdict(r)) + "\n")
        traj.save(os.path.join(out_dir, "trajectory.json"))
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump({"meta": meta, "summary": summary}, fh, indent=2, default=str)

    def plot(self, out_dir: str, traj, title: str = ""):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib not installed; skipping plot")
            return None
        t = [r.t for r in self.rows]
        fig, ax = plt.subplots(2, 2, figsize=(12, 7))
        ax[0, 0].plot(traj.lane_x, traj.lane_y, "k--", lw=0.8, label="lane centre")
        ax[0, 0].plot(traj.x, traj.y, "C0", lw=1, label="planned")
        ax[0, 0].plot([r.x for r in self.rows], [r.y for r in self.rows], "C3", lw=1, label="driven")
        ax[0, 0].axis("equal")
        ax[0, 0].legend()
        ax[0, 0].set_title("path (right-handed frame)")
        ax[0, 1].plot(t, [r.e_y for r in self.rows], label="e_y (trajectory)")
        ax[0, 1].plot(t, [r.e_y_lane for r in self.rows], label="e_y (lane centre)")
        ax[0, 1].set_ylabel("m")
        ax[0, 1].legend()
        ax[1, 0].plot(t, np.degrees([r.steer_rad for r in self.rows]))
        ax[1, 0].set_ylabel("steer [deg] (left +)")
        ax[1, 1].plot(t, [r.vx for r in self.rows], label="v")
        ax[1, 1].plot(t, [r.v_ref for r in self.rows], label="v_ref")
        ax[1, 1].set_ylabel("m/s")
        ax[1, 1].legend()
        for a_ in ax.ravel()[1:]:
            a_.set_xlabel("t [s]")
        fig.suptitle(title)
        fig.tight_layout()
        path = os.path.join(out_dir, "episode.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
