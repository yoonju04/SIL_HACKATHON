"""Linear time-varying MPC lateral (steering) controller, after Autoware.

References
- Autoware Universe, control/autoware_mpc_lateral_controller (Apache-2.0,
  commit 3b3a5ad): src/mpc.cpp (generateMPCMatrix, executeOptimization,
  addSteerWeightR/F), model_predictive_control_algorithm.md, default weights
  in param/lateral_controller_defaults.param.yaml.
- P. Falcone et al., "Predictive Active Steering Control for Autonomous
  Vehicle Systems", IEEE TCST 15(3), 2007 (MPC for autonomous steering).
- B. Stellato et al., OSQP, 2020 (QP algorithm, see qp.py).

Prediction (condensed, Autoware): X = Aex x0 + Bex U + Wex over N steps of
prediction_dt, model matrices evaluated at the reference speed and curvature
of each step (time-varying). Outputs y = [e_y, e_psi].

Cost (Autoware structure):
  sum_i  y_{i+1}' Q_i y_{i+1}                      Q_i = diag(w_lat, w_head + v_i^2 w_head_v2)
       + R_i (u_i - u_ref_i)^2                     R_i = w_steer + v_i^2 w_steer_v2
       + v_i^2 w_lat_jerk / DT^2 (u_{i+1} - u_i)^2 (lateral jerk)
       + w_steer_rate / DT^2 (u_{i+1} - u_i)^2 + w_steer_rate / T^2 (u_0 - u_prev)^2
  terminal step uses w_terminal_lat / w_terminal_head.
Constraints (Autoware):
  |u_i| <= max steer;  |u_0 - u_prev| <= rate * T;  |u_{i+1} - u_i| <= rate * DT
  where u_prev = the MPC's OWN previous raw command (m_raw_steer_cmd_prev),
  not the command that was finally applied to the vehicle.
Output: u_0 -> clip -> 2nd-order Butterworth low-pass (3 Hz) -> command.

Differences from Autoware (documented):
- Only the "dynamics" vehicle model is implemented (Autoware's default is the
  kinematic model with steering delay).
- Steering-rate limit is one constant instead of curvature/speed maps.
- The steer-rate weight of the first step is applied as (u_0 - u_prev)^2;
  Autoware's addSteerWeightF omits u_prev in that term.
- Steering-acceleration weight (default 1e-6) is not implemented.
- Initial state uses measured e_y_dot / e_psi_dot instead of finite
  differences + low-pass filter.
- QP: own ADMM (OSQP algorithm) by default; osqp library optional.
- rate_reference="applied" reproduces the earlier steering_sil behaviour
  (rate limit anchored on the applied command) for comparison experiments.

Self-contained: imports numpy/scipy and this package only.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .filters import Butterworth2d
from .model import VehicleParams, discrete_matrices, reference_input
from .qp import ADMMQP, QPSettings, solve_osqp
from .reference import ReferencePath


@dataclass
class MPCConfig:
    horizon: int = 50  # mpc_prediction_horizon
    prediction_dt: float = 0.1  # mpc_prediction_dt [s]
    # weights: Autoware lateral_controller_defaults.param.yaml
    w_lat_error: float = 1.0
    w_heading_error: float = 0.0
    w_heading_error_squared_vel: float = 0.3
    w_steering_input: float = 1.0
    w_steering_input_squared_vel: float = 0.25
    w_lat_jerk: float = 0.1
    w_steer_rate: float = 0.0
    w_terminal_lat_error: float = 1.0
    w_terminal_heading_error: float = 0.1
    zero_ff_steer_deg: float = 0.5  # |u_ref| below this is set to 0 (curvature noise)
    steer_rate_lim_dps: float = 60.0  # Autoware maps: 40-60 deg/s
    steering_lpf_cutoff_hz: float = 3.0
    input_delay_s: float = 0.0  # delay compensation (Autoware input_delay); 0 = off
    min_speed: float = 2.0  # model is singular at v = 0
    model_speed: str = "reference"  # "reference" (Autoware) or "measured"
    rate_reference: str = "own_raw"  # "own_raw" (Autoware) or "applied"
    qp_solver: str = "admm"  # "admm", "unconstrained_fast" (Autoware option), "osqp"
    qp: QPSettings = field(default_factory=QPSettings)


@dataclass
class MPCOutput:
    steer: float | None  # command after low-pass filter [rad]
    raw_steer: float | None  # first optimal input before the filter
    ok: bool
    status: str
    solve_time_s: float  # wall-clock (machine dependent)
    qp_iterations: int  # deterministic effort measure
    seq: int  # increments on every computation (for stale detection)
    stamp_s: float
    state: np.ndarray
    s: float
    kappa: float
    predicted_e_y: np.ndarray | None = None
    u_sequence: np.ndarray | None = None


def project_feasible(U, u_prev, lim, d0, d):
    """Sequentially clip U into the steering / steering-rate constraints."""
    out = np.empty_like(U)
    prev = u_prev
    for i, ui in enumerate(U):
        r = d0 if i == 0 else d
        out[i] = min(lim, max(-lim, min(prev + r, max(prev - r, ui))))
        prev = out[i]
    return out


class MPCLateralController:
    def __init__(self, params: VehicleParams, dt: float, cfg: MPCConfig | None = None):
        self.p = params
        self.dt = float(dt)  # control period
        self.cfg = cfg or MPCConfig()
        self.ref: ReferencePath | None = None
        self.qp = ADMMQP(self.cfg.qp)
        n = self.cfg.horizon
        Dm = np.eye(n)
        Dm[np.arange(1, n), np.arange(n - 1)] = -1.0
        self._A_con = np.vstack([np.eye(n), Dm])
        self.reset()

    # -------------------------------------------------------------- setup
    def set_reference(self, x, y, yaw, kappa, s=None, v=None, t=None):
        """Trajectory to track (arrays, right-handed frame). v, t: speed profile."""
        self.ref = ReferencePath(x, y, yaw, kappa, s, v, t)

    def reset(self, steer: float = 0.0):
        c = self.cfg
        self.raw_prev = float(steer)
        self.out_prev = float(steer)
        self.lpf = Butterworth2d(self.dt, c.steering_lpf_cutoff_hz, steer)
        self.qp.reset()
        self.seq = 0
        k = int(round(c.input_delay_s / self.dt))
        self.input_buffer = deque([float(steer)] * k, maxlen=k) if k > 0 else None

    # ------------------------------------------------------------ matrices
    def _build_qp(self, x0, hz, u_prev):
        c, p = self.cfg, self.p
        N, DT = c.horizon, c.prediction_dt
        v = np.maximum(hz["v"], c.min_speed)
        kap = hz["kappa"]
        Aex = np.zeros((4 * N, 4))
        Bex = np.zeros((4 * N, N))
        Wex = np.zeros(4 * N)
        Qd = np.zeros(2 * N)
        R1 = np.zeros(N)
        Uref = np.zeros(N)
        zero_ff = math.radians(c.zero_ff_steer_deg)
        for i in range(N):
            Ad, Bd, Wd = discrete_matrices(p, v[i], kap[i], DT)
            r = slice(4 * i, 4 * i + 4)
            if i == 0:
                Aex[r] = Ad
                Wex[r] = Wd
            else:
                rp = slice(4 * (i - 1), 4 * i)
                Aex[r] = Ad @ Aex[rp]
                Bex[r, :i] = Ad @ Bex[rp, :i]
                Wex[r] = Ad @ Wex[rp] + Wd
            Bex[r, i] = Bd
            terminal = i == N - 1
            w_lat = c.w_terminal_lat_error if terminal else c.w_lat_error
            w_head = c.w_terminal_heading_error if terminal else c.w_heading_error
            Qd[2 * i] = w_lat
            Qd[2 * i + 1] = w_head + v[i] ** 2 * c.w_heading_error_squared_vel
            R1[i] = c.w_steering_input + v[i] ** 2 * c.w_steering_input_squared_vel
            ur = reference_input(p, v[i], kap[i])
            Uref[i] = 0.0 if abs(ur) < zero_ff else ur
        # outputs y = [e_y, e_psi] -> rows 0 and 2 of each state block
        rows = np.ravel(np.column_stack([np.arange(N) * 4, np.arange(N) * 4 + 2]))
        CB = Bex[rows]
        cfree = (Aex @ x0 + Wex)[rows]
        QCB = Qd[:, None] * CB
        H = CB.T @ QCB + np.diag(R1)
        f = QCB.T @ cfree - R1 * Uref
        # lateral jerk and steering rate: (u_{i+1} - u_i)^2 terms
        wj = v[:-1] ** 2 * c.w_lat_jerk / DT**2 + c.w_steer_rate / DT**2
        idx = np.arange(N - 1)
        H[idx, idx] += wj
        H[idx + 1, idx + 1] += wj
        H[idx, idx + 1] -= wj
        H[idx + 1, idx] -= wj
        if c.w_steer_rate > 0:  # first step relative to the previous command
            w0 = c.w_steer_rate / self.dt**2
            H[0, 0] += w0
            f[0] -= w0 * u_prev
        H = 0.5 * (H + H.T)
        return H, f, Aex, Bex, Wex, rows

    # ---------------------------------------------------------------- step
    def compute(self, x: float, y: float, yaw: float, vx: float, vy: float = 0.0,
                yaw_rate: float = 0.0, t: float = 0.0, applied_prev: float | None = None) -> MPCOutput:
        """One control cycle. Pose / velocities in the right-handed frame.

        applied_prev: command applied to the vehicle in the previous cycle; only
        used when cfg.rate_reference == "applied".
        """
        if self.ref is None:
            raise RuntimeError("call set_reference() first")
        c, p = self.cfg, self.p
        t0 = time.perf_counter()
        self.seq += 1
        x0, s, kappa = self.ref.error_state(x, y, yaw, vx, vy, yaw_rate)
        hz = self.ref.horizon(s, c.horizon, c.prediction_dt, vx)
        if c.model_speed == "measured":
            hz["v"] = np.full(c.horizon, vx)
        if self.input_buffer is not None:  # delay compensation (Autoware)
            vd = max(vx, c.min_speed)
            Ad, Bd, Wd = discrete_matrices(p, vd, kappa, self.dt)
            for u_old in self.input_buffer:
                x0 = Ad @ x0 + Bd * u_old + Wd
        u_prev = self.raw_prev
        if c.rate_reference == "applied" and applied_prev is not None:
            u_prev = float(applied_prev)
        lim = p.max_steer_rad
        rate = math.radians(c.steer_rate_lim_dps)
        d0, d = rate * self.dt, rate * c.prediction_dt
        try:
            H, f, Aex, Bex, Wex, rows = self._build_qp(x0, hz, u_prev)
            if c.qp_solver == "unconstrained_fast":
                U = -np.linalg.solve(H, f)
                status, iters = "unconstrained", 0
            elif c.qp_solver == "osqp":  # pragma: no cover - optional dependency
                lo = np.r_[-lim * np.ones(c.horizon), u_prev - d0, -d * np.ones(c.horizon - 1)]
                hi = np.r_[lim * np.ones(c.horizon), u_prev + d0, d * np.ones(c.horizon - 1)]
                res = solve_osqp(H, f, self._A_con, lo, hi)
                U, status, iters = res.x, res.status, res.iterations
            else:
                lo = np.r_[-lim * np.ones(c.horizon), u_prev - d0, -d * np.ones(c.horizon - 1)]
                hi = np.r_[lim * np.ones(c.horizon), u_prev + d0, d * np.ones(c.horizon - 1)]
                res = self.qp.solve(H, f, self._A_con, lo, hi)
                U, status, iters = res.x, res.status, res.iterations
            if not np.all(np.isfinite(U)):
                raise FloatingPointError("non-finite QP solution")
            U = project_feasible(U, u_prev, lim, d0, d)
            ok = not status.startswith("error")
        except Exception as exc:  # numerical failure is reported, not raised
            elapsed = time.perf_counter() - t0
            return MPCOutput(steer=None, raw_steer=None, ok=False, status=f"error:{type(exc).__name__}",
                             solve_time_s=elapsed, qp_iterations=0, seq=self.seq, stamp_s=t,
                             state=x0, s=s, kappa=kappa)
        raw = float(U[0])
        self.raw_prev = raw  # Autoware: m_raw_steer_cmd_prev = Uex(0)
        steer = max(-lim, min(lim, self.lpf.filter(raw)))
        self.out_prev = steer
        if self.input_buffer is not None:
            self.input_buffer.append(steer)
        pred = (Aex @ x0 + Bex @ U + Wex)[rows][0::2]
        return MPCOutput(steer=steer, raw_steer=raw, ok=ok, status=status,
                         solve_time_s=time.perf_counter() - t0, qp_iterations=iters, seq=self.seq,
                         stamp_s=t, state=x0, s=s, kappa=kappa, predicted_e_y=pred, u_sequence=U)
