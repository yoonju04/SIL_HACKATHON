"""Dynamic bicycle error model used by the MPC (Autoware "dynamics" model).

Port of Autoware mpc_lateral_controller/src/vehicle_model/
vehicle_model_bicycle_dynamics.cpp (Apache-2.0, commit 3b3a5ad), which is the
error model of R. Rajamani, Vehicle Dynamics and Control, ch. 2:

  x = [e_y, e_y_dot, e_psi, e_psi_dot],  u = front road-wheel angle,
  x_dot = A(v) x + B u + w(v) * kappa * v

Discretisation (Autoware): bilinear for A,
  Ad = (I - T/2 A)^-1 (I + T/2 A),  Bd = (I - T/2 A)^-1 T B,
  Wd = (I - T/2 A)^-1 T kappa v w.
Reference (feed-forward) input: u_ref = L kappa + Kv v^2 kappa.
Autoware writes Kv = lr m / (2 cf L) - lf m / (2 cr L) while its A matrix
uses cf, cr as axle stiffness; here axle stiffness is used in both places
(Kv = lr m / (cf L) - lf m / (cr L)), consistent with Rajamani.

Self-contained: no import from the LQR package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


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


def continuous_matrices(p: VehicleParams, v: float):
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 1] = -(p.cf + p.cr) / (p.mass * v)
    A[1, 2] = (p.cf + p.cr) / p.mass
    A[1, 3] = (p.lr * p.cr - p.lf * p.cf) / (p.mass * v)
    A[2, 3] = 1.0
    A[3, 1] = (p.lr * p.cr - p.lf * p.cf) / (p.iz * v)
    A[3, 2] = (p.lf * p.cf - p.lr * p.cr) / p.iz
    A[3, 3] = -(p.lf**2 * p.cf + p.lr**2 * p.cr) / (p.iz * v)
    B = np.array([0.0, p.cf / p.mass, 0.0, p.lf * p.cf / p.iz])
    w = np.array([0.0, (p.lr * p.cr - p.lf * p.cf) / (p.mass * v) - v, 0.0,
                  -(p.lf**2 * p.cf + p.lr**2 * p.cr) / (p.iz * v)])
    return A, B, w


@lru_cache(maxsize=4096)
def _discrete_parts(p: VehicleParams, v: float, dt: float):
    A, B, w = continuous_matrices(p, v)
    I = np.eye(4)
    inv = np.linalg.inv(I - 0.5 * dt * A)
    return inv @ (I + 0.5 * dt * A), inv @ (dt * B), inv @ (dt * w)


def discrete_matrices(p: VehicleParams, v: float, kappa: float, dt: float):
    """Autoware DynamicsBicycleModel::calculateDiscreteMatrix (bilinear).

    The kappa-independent parts are cached per (v rounded to 0.01 m/s, dt);
    Wd is linear in kappa.
    """
    vq = round(float(v) * 100.0) / 100.0
    Ad, Bd, Wd1 = _discrete_parts(p, vq, float(dt))
    return Ad, Bd, Wd1 * (kappa * vq)


def reference_input(p: VehicleParams, v: float, kappa: float) -> float:
    """Steady-state steering for curvature kappa (Autoware calculateReferenceInput)."""
    L = p.wheelbase
    kv = p.lr * p.mass / (p.cf * L) - p.lf * p.mass / (p.cr * L)
    return L * kappa + kv * v * v * kappa
