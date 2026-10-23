"""Quintic polynomial with boundary position / velocity / acceleration.

Ported from PythonRobotics, PathPlanning/QuinticPolynomialsPlanner
(commit cdd0cc8), Copyright (c) 2016 - now Atsushi Sakai and contributors,
MIT License (see THIRD_PARTY_NOTICES.md).

Used here for a lateral offset d(s) along the lane (lane change), i.e. the
lateral part of Werling et al., "Optimal Trajectory Generation for Dynamic
Street Scenarios in a Frenet Frame", ICRA 2010 (quintic = jerk-optimal for
fixed boundary conditions). The independent variable is arc length s, not
time, so the shape does not depend on the speed profile.
"""
from __future__ import annotations

import numpy as np


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        self.a0, self.a1, self.a2 = xs, vxs, axs / 2.0
        A = np.array([[T**3, T**4, T**5],
                      [3 * T**2, 4 * T**3, 5 * T**4],
                      [6 * T, 12 * T**2, 20 * T**3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        self.a3, self.a4, self.a5 = np.linalg.solve(A, b)

    def point(self, t):
        t = np.asarray(t, float)
        return (self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3
                + self.a4 * t**4 + self.a5 * t**5)

    def first_derivative(self, t):
        t = np.asarray(t, float)
        return (self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2
                + 4 * self.a4 * t**3 + 5 * self.a5 * t**4)

    def second_derivative(self, t):
        t = np.asarray(t, float)
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3
