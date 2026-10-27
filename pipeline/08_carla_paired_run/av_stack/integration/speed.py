"""Longitudinal speed tracking (kept simple on purpose: the study is lateral).

The planner's speed profile is looked up slightly ahead of the vehicle
(preview), otherwise a car at rest on a profile that starts at 0 m/s would
never start moving. PythonRobotics' LQR example uses the same kind of PID
speed loop.
"""
from __future__ import annotations

import numpy as np


def speed_reference(traj, s: float, vx: float, preview_s: float = 0.5, min_preview_m: float = 2.0) -> float:
    return float(np.interp(s + max(min_preview_m, preview_s * max(vx, 0.0)), traj.s, traj.v))


class SpeedPI:
    """PI speed loop -> (throttle, brake) in [0, 1] for CARLA."""

    def __init__(self, kp: float = 0.4, ki: float = 0.08, i_limit: float = 5.0):
        self.kp, self.ki, self.i_limit, self.i = kp, ki, i_limit, 0.0

    def __call__(self, v: float, v_ref: float, dt: float):
        e = v_ref - v
        self.i = max(-self.i_limit, min(self.i_limit, self.i + e * dt))
        u = self.kp * e + self.ki * self.i
        return (min(1.0, u), 0.0) if u >= 0 else (0.0, min(1.0, -u))


def accel_command(v: float, v_ref: float, kp: float = 1.0, a_min: float = -4.0, a_max: float = 2.0) -> float:
    """P speed loop -> longitudinal acceleration [m/s^2] for the offline plant."""
    return max(a_min, min(a_max, kp * (v_ref - v)))
