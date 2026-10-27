"""2nd-order Butterworth low-pass filter (bilinear transform).

Port of Autoware mpc_lateral_controller/src/lowpass_filter.cpp
Butterworth2dFilter (Apache-2.0, commit 3b3a5ad). Used on the steering
command (steering_lpf_cutoff_hz, default 3 Hz).
"""
from __future__ import annotations

import math


class Butterworth2d:
    def __init__(self, dt: float, f_cutoff_hz: float, initial_value: float = 0.0):
        fs = 1.0 / dt
        xi = 1.0 / math.tan(math.pi * f_cutoff_hz / fs)
        xi2 = xi * xi
        q = math.sqrt(2.0)
        self.b0 = 1.0 / (1.0 + q * xi + xi2)
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (xi2 - 1.0) * self.b0
        self.a2 = -(1.0 - q * xi + xi2) * self.b0
        self.reset(initial_value)

    def reset(self, value: float = 0.0):
        self.y1 = self.y2 = self.u1 = self.u2 = float(value)

    def filter(self, u0: float) -> float:
        y0 = (self.b2 * self.u2 + self.b1 * self.u1 + self.b0 * u0
              + self.a2 * self.y2 + self.a1 * self.y1)
        self.y2, self.y1 = self.y1, y0
        self.u2, self.u1 = self.u1, u0
        return y0
