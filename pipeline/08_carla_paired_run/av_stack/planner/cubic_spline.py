"""Natural cubic spline in 1D and 2D (arc-length parameterised).

Ported from PythonRobotics, PathPlanning/CubicSpline/cubic_spline_planner.py
(commit cdd0cc8), Copyright (c) 2016 - now Atsushi Sakai and contributors,
MIT License (see THIRD_PARTY_NOTICES.md).

Changes from the original:
- evaluation is vectorised (numpy arrays in, numpy arrays out);
- the tridiagonal system is built the same way but solved once with numpy;
- values outside the knot range are clamped to the end segments instead of
  returning None.
The spline is C2 continuous, so curvature (which needs the 2nd derivative) is
continuous along the path. That matters for the controllers' feed-forward.
"""
from __future__ import annotations

import numpy as np


class CubicSpline1D:
    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        h = np.diff(x)
        if np.any(h <= 0):
            raise ValueError("x coordinates must be strictly increasing")
        self.x, self.nx = x, len(x)
        self.a = y.copy()
        A = self._calc_A(h)
        B = self._calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)
        self.d = (self.c[1:] - self.c[:-1]) / (3.0 * h)
        self.b = (self.a[1:] - self.a[:-1]) / h - h / 3.0 * (2.0 * self.c[:-1] + self.c[1:])

    def _calc_A(self, h):
        n = self.nx
        A = np.zeros((n, n))
        A[0, 0] = 1.0
        for i in range(n - 1):
            if i != n - 2:
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]
        A[0, 1] = 0.0
        A[n - 1, n - 2] = 0.0
        A[n - 1, n - 1] = 1.0
        return A

    def _calc_B(self, h, a):
        B = np.zeros(self.nx)
        B[1:-1] = 3.0 * (a[2:] - a[1:-1]) / h[1:] - 3.0 * (a[1:-1] - a[:-2]) / h[:-1]
        return B

    def _index(self, t):
        i = np.searchsorted(self.x, t, side="right") - 1
        return np.clip(i, 0, self.nx - 2)

    def position(self, t):
        t = np.asarray(t, float)
        i = self._index(t)
        dx = t - self.x[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3

    def first_derivative(self, t):
        t = np.asarray(t, float)
        i = self._index(t)
        dx = t - self.x[i]
        return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2

    def second_derivative(self, t):
        t = np.asarray(t, float)
        i = self._index(t)
        dx = t - self.x[i]
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dx

    def third_derivative(self, t):
        t = np.asarray(t, float)
        return 6.0 * self.d[self._index(t)]


class CubicSpline2D:
    """x(s), y(s) splines through 2D points; s = cumulative chord length."""

    def __init__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ds = np.hypot(np.diff(x), np.diff(y))
        keep = np.r_[True, ds > 1e-6]  # drop duplicate points
        x, y = x[keep], y[keep]
        self.s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def position(self, s):
        return self.sx.position(s), self.sy.position(s)

    def yaw(self, s):
        return np.arctan2(self.sy.first_derivative(s), self.sx.first_derivative(s))

    def curvature(self, s):
        dx, ddx = self.sx.first_derivative(s), self.sx.second_derivative(s)
        dy, ddy = self.sy.first_derivative(s), self.sy.second_derivative(s)
        return (ddy * dx - ddx * dy) / (dx**2 + dy**2) ** 1.5
