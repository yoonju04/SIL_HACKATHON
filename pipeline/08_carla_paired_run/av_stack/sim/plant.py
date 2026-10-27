"""Offline vehicle: nonlinear single-track model + simple steer-by-wire actuator.

Deliberately different from the controllers' linear models (tyre force
saturation at mu*Fz, actuator lag / rate limit), so closed-loop tests see
model error. Right-handed frame (x forward, y left, yaw CCW).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlantParams:
    mass: float = 1800.0
    iz: float = 3000.0
    lf: float = 1.35
    lr: float = 1.55
    cf: float = 1.2e5  # axle cornering stiffness [N/rad]
    cr: float = 1.4e5
    max_steer_rad: float = math.radians(30.0)
    max_steer_rate_rad_s: float = math.radians(90.0)

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


@dataclass
class PlantState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    delta: float = 0.0  # actual road-wheel angle


class BicyclePlant:
    def __init__(self, p: PlantParams | None = None, mu: float = 0.9,
                 actuator_tau_s: float = 0.08, substeps: int = 10):
        self.p = p or PlantParams()
        self.mu, self.tau, self.substeps = mu, actuator_tau_s, substeps
        self.s = PlantState()

    def _tire(self, C, alpha, Fz):
        Fmax = self.mu * Fz
        return max(-Fmax, min(Fmax, -C * alpha))

    def step(self, delta_cmd: float, accel_cmd: float, dt: float) -> PlantState:
        """delta_cmd: road-wheel angle [rad]; accel_cmd: longitudinal accel [m/s^2]."""
        p, s = self.p, self.s
        h = dt / self.substeps
        g = 9.81
        Fzf = p.mass * g * p.lr / p.wheelbase
        Fzr = p.mass * g * p.lf / p.wheelbase
        for _ in range(self.substeps):
            dd = (delta_cmd - s.delta) / max(self.tau, 1e-3)
            dd = max(-p.max_steer_rate_rad_s, min(p.max_steer_rate_rad_s, dd))
            s.delta = max(-p.max_steer_rad, min(p.max_steer_rad, s.delta + dd * h))
            s.vx = max(0.0, s.vx + accel_cmd * h)
            if s.vx < 0.5:  # kinematic at very low speed (tyre model singular)
                s.vy = 0.0
                s.r = s.vx * math.tan(s.delta) / p.wheelbase
            else:
                af = math.atan2(s.vy + p.lf * s.r, s.vx) - s.delta
                ar = math.atan2(s.vy - p.lr * s.r, s.vx)
                Fyf = self._tire(p.cf, af, Fzf)
                Fyr = self._tire(p.cr, ar, Fzr)
                s.vy += ((Fyf * math.cos(s.delta) + Fyr) / p.mass - s.vx * s.r) * h
                s.r += ((p.lf * Fyf * math.cos(s.delta) - p.lr * Fyr) / p.iz) * h
            s.x += (s.vx * math.cos(s.yaw) - s.vy * math.sin(s.yaw)) * h
            s.y += (s.vx * math.sin(s.yaw) + s.vy * math.cos(s.yaw)) * h
            s.yaw += s.r * h
        return s
