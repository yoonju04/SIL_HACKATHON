"""Reference matching and lateral error states for the LQR controller.

Follows Apollo LatController::ComputeLateralErrors (nearest point on the
trajectory by position, lateral error = cos(theta_ref)*dy - sin(theta_ref)*dx,
heading error = theta - theta_ref, heading error rate = yaw rate - kappa*v),
with one change: the nearest point is found on the *segment* between samples
(not only the nearest sample), which removes saw-tooth in e_y / s.

This file is deliberately self-contained (the MPC package has its own copy)
so the two controllers do not share code.
"""
from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ReferencePath:
    def __init__(self, x, y, yaw, kappa, s=None):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.yaw = np.unwrap(np.asarray(yaw, float))
        self.kappa = np.asarray(kappa, float)
        if s is None:
            s = np.r_[0.0, np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))]
        self.s = np.asarray(s, float)
        self._hint: int | None = None

    def project(self, px: float, py: float):
        """Return (s, e_y, yaw_ref, kappa_ref). e_y > 0: vehicle left of the path."""
        n = len(self.x)
        if self._hint is None:
            lo, hi = 0, n
        else:
            lo, hi = max(0, self._hint - 20), min(n, self._hint + 80)
        d2 = (self.x[lo:hi] - px) ** 2 + (self.y[lo:hi] - py) ** 2
        i = lo + int(np.argmin(d2))
        if self._hint is not None and d2.min() > 25.0:  # lost (>5 m): global search
            d2 = (self.x - px) ** 2 + (self.y - py) ** 2
            i = int(np.argmin(d2))
        self._hint = i
        j = min(i, n - 2)
        if i > 0:
            sx, sy = self.x[i] - self.x[i - 1], self.y[i] - self.y[i - 1]
            if (px - self.x[i]) * sx + (py - self.y[i]) * sy < 0:
                j = i - 1
        bx, by = self.x[j + 1] - self.x[j], self.y[j + 1] - self.y[j]
        seg = bx * bx + by * by
        a = min(1.0, max(0.0, ((px - self.x[j]) * bx + (py - self.y[j]) * by) / seg)) if seg > 0 else 0.0
        rx, ry = self.x[j] + a * bx, self.y[j] + a * by
        yaw_ref = (1 - a) * self.yaw[j] + a * self.yaw[j + 1]
        kappa = (1 - a) * self.kappa[j] + a * self.kappa[j + 1]
        s = self.s[j] + a * (self.s[j + 1] - self.s[j])
        dx, dy = px - rx, py - ry
        e_y = math.cos(yaw_ref) * dy - math.sin(yaw_ref) * dx
        return float(s), float(e_y), float(yaw_ref), float(kappa)

    def kappa_at(self, s: float) -> float:
        return float(np.interp(s, self.s, self.kappa))

    def error_state(self, px, py, yaw, vx, vy, yaw_rate):
        """[e_y, e_y_dot, e_psi, e_psi_dot] (Rajamani ch. 2 error coordinates)."""
        s, e_y, yaw_ref, kappa = self.project(px, py)
        e_psi = wrap(yaw - yaw_ref)
        # Apollo uses e_y_dot = v*sin(e_psi) (no side-slip). With the lateral
        # velocity available (CARLA / simulation) the exact rate is used:
        e_y_dot = vx * math.sin(e_psi) + vy * math.cos(e_psi)
        e_psi_dot = yaw_rate - kappa * vx
        return np.array([e_y, e_y_dot, e_psi, e_psi_dot]), s, kappa
