"""LQR lateral (steering) controller, after Apollo's lat_based_lqr_controller.

References
- Apollo, modules/control/controllers/lat_based_lqr_controller/lat_controller.cc
  (Apache-2.0, commit d53aa3d): model matrices, bilinear discretisation,
  iterative Riccati solution, feed-forward structure.
- R. Rajamani, Vehicle Dynamics and Control, 2nd ed., Springer 2012, ch. 2-3:
  lateral error dynamics and steady-state feed-forward.
- R. E. Kalman, "Contributions to the theory of optimal control", 1960.

State  x = [e_y, e_y_dot, e_psi, e_psi_dot]   (left / counter-clockwise +)
Input  delta = front road-wheel angle [rad]   (left +)
Model  x_dot = A(v) x + B delta  (+ curvature disturbance, handled by the
       feed-forward term)
       A[1,1] = -(cf+cr)/(m v)   A[1,2] = (cf+cr)/m   A[1,3] = (lr cr - lf cf)/(m v)
       A[3,1] = (lr cr - lf cf)/(Iz v)  A[3,2] = (lf cf - lr cr)/Iz
       A[3,3] = -(lf^2 cf + lr^2 cr)/(Iz v)
       B = [0, cf/m, 0, lf cf/Iz]
       cf, cr = axle cornering stiffness [N/rad] (2 x per-tyre value).
Discretisation (Apollo): Ad = (I - T/2 A)^-1 (I + T/2 A),  Bd = B T.
Gain: K from the discrete Riccati equation, iterated as in Apollo's
       common::math::SolveLQRProblem; cached per 0.25 m/s of speed.
Command delta = -K x + delta_ff, with the Rajamani feed-forward
       delta_ff = L k + Kv v^2 k - K[2] (lr k - lf m v^2 k / (cr L)),
       Kv = lr m / (cf L) - lf m / (cr L)   (understeer gradient)
       Apollo writes Kv and the last term with an extra 1/2 (per-tyre
       stiffness) while its A matrix uses axle stiffness; here the axle
       stiffness is used consistently, which gives zero steady-state lateral
       error on a constant-radius path for the linear model.

Not ported from Apollo: lead/lag compensation, MRAC steering adaptation,
look-ahead / look-back error shaping, reverse gear, mean filters.

Self-contained: imports numpy only and this package's reference.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .reference import ReferencePath


@dataclass(frozen=True)
class VehicleParams:
    mass: float = 1800.0  # [kg]
    iz: float = 3000.0  # yaw inertia [kg m^2]
    lf: float = 1.35  # CoG -> front axle [m]
    lr: float = 1.55  # CoG -> rear axle [m]
    cf: float = 1.2e5  # front axle cornering stiffness [N/rad]
    cr: float = 1.4e5  # rear axle cornering stiffness [N/rad]
    max_steer_rad: float = math.radians(30.0)  # road-wheel angle limit

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


@dataclass
class LQRConfig:
    # Apollo controller_conf.pb.txt: matrix_q = [0.05, 0.0, 1.0, 0.0], r = 1.
    # Checked offline at the 50 ms CARLA period (tests/test_lqr.py): max
    # |e_y| about 5 cm on the demo route, so Apollo's weights are kept.
    q: tuple = (0.05, 0.0, 1.0, 0.0)
    r: float = 1.0
    riccati_eps: float = 1e-9  # Apollo default 0.01 (absolute on P); tighter here
    riccati_max_iter: int = 2000  # Apollo default 150
    min_speed: float = 2.0  # the model is singular at v = 0; clamp below this
    speed_key_step: float = 0.25  # gain cache resolution [m/s]
    max_lat_acc: float | None = None  # Apollo FLAGS_set_steer_limit: |delta| <= atan(a L / v^2)
    preview_time_s: float = 0.0  # feed-forward curvature taken v*T metres ahead (0 = off, Apollo-like)


@dataclass
class LQROutput:
    steer: float  # road-wheel angle command [rad], clipped to the limit
    steer_feedback: float
    steer_feedforward: float
    state: np.ndarray  # [e_y, e_y_dot, e_psi, e_psi_dot]
    s: float
    kappa: float
    riccati_iterations: int
    gain: np.ndarray = field(repr=False, default=None)


def continuous_matrices(p: VehicleParams, v: float):
    v = float(v)
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 1] = -(p.cf + p.cr) / (p.mass * v)
    A[1, 2] = (p.cf + p.cr) / p.mass
    A[1, 3] = (p.lr * p.cr - p.lf * p.cf) / (p.mass * v)
    A[2, 3] = 1.0
    A[3, 1] = (p.lr * p.cr - p.lf * p.cf) / (p.iz * v)
    A[3, 2] = (p.lf * p.cf - p.lr * p.cr) / p.iz
    A[3, 3] = -(p.lf**2 * p.cf + p.lr**2 * p.cr) / (p.iz * v)
    B = np.array([[0.0], [p.cf / p.mass], [0.0], [p.lf * p.cf / p.iz]])
    return A, B


def discretize_apollo(A, B, ts):
    I = np.eye(A.shape[0])
    Ad = np.linalg.solve(I - 0.5 * ts * A, I + 0.5 * ts * A)
    return Ad, B * ts


def solve_lqr_iterative(A, B, Q, R, eps, max_iter):
    """Apollo common::math::SolveLQRProblem: iterate the Riccati recursion."""
    P = Q.copy()
    AT, BT = A.T, B.T
    it = 0
    for it in range(1, max_iter + 1):
        S = R + BT @ P @ B
        P_next = AT @ P @ A - AT @ P @ B @ np.linalg.solve(S, BT @ P @ A) + Q
        diff = np.max(np.abs(P_next - P))
        P = P_next
        if diff < eps:
            break
    K = np.linalg.solve(R + BT @ P @ B, BT @ P @ A)
    return K, P, it


class LQRLateralController:
    def __init__(self, params: VehicleParams, dt: float, cfg: LQRConfig | None = None):
        self.p = params
        self.dt = float(dt)
        self.cfg = cfg or LQRConfig()
        self.Q = np.diag(np.asarray(self.cfg.q, float))
        self.R = np.array([[float(self.cfg.r)]])
        self._gains: dict[float, tuple] = {}
        self.ref: ReferencePath | None = None

    # -------------------------------------------------------------- setup
    def set_reference(self, x, y, yaw, kappa, s=None):
        """Trajectory to track (arrays, right-handed frame)."""
        self.ref = ReferencePath(x, y, yaw, kappa, s)

    def gain(self, v: float):
        c = self.cfg
        key = round(max(v, c.min_speed) / c.speed_key_step) * c.speed_key_step
        if key not in self._gains:
            A, B = continuous_matrices(self.p, key)
            Ad, Bd = discretize_apollo(A, B, self.dt)
            K, P, it = solve_lqr_iterative(Ad, Bd, self.Q, self.R, c.riccati_eps, c.riccati_max_iter)
            self._gains[key] = (K.ravel(), P, it)
        return self._gains[key]

    def feedforward(self, v: float, kappa: float, K: np.ndarray) -> float:
        p = self.p
        L = p.wheelbase
        kv = p.lr * p.mass / (p.cf * L) - p.lf * p.mass / (p.cr * L)
        return (L * kappa + kv * v * v * kappa
                - K[2] * (p.lr * kappa - p.lf * p.mass * v * v * kappa / (p.cr * L)))

    # -------------------------------------------------------------- step
    def compute(self, x: float, y: float, yaw: float, vx: float, vy: float = 0.0,
                yaw_rate: float = 0.0) -> LQROutput:
        """One control cycle. Pose / velocities in the right-handed frame."""
        if self.ref is None:
            raise RuntimeError("call set_reference() first")
        state, s, kappa = self.ref.error_state(x, y, yaw, vx, vy, yaw_rate)
        v = max(vx, self.cfg.min_speed)
        K, _, it = self.gain(v)
        fb = -float(K @ state)
        kappa_ff = kappa if self.cfg.preview_time_s <= 0 else self.ref.kappa_at(s + v * self.cfg.preview_time_s)
        ff = self.feedforward(v, kappa_ff, K)
        limit = self.p.max_steer_rad
        if self.cfg.max_lat_acc is not None:
            limit = min(limit, math.atan(self.cfg.max_lat_acc * self.p.wheelbase / (v * v)))
        steer = max(-limit, min(limit, fb + ff))
        return LQROutput(steer=steer, steer_feedback=fb, steer_feedforward=ff, state=state,
                         s=s, kappa=kappa, riccati_iterations=it, gain=K)
