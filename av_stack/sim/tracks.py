"""Lane-centre routes for offline runs (same role as CARLA map waypoints)."""
from __future__ import annotations

import math

import numpy as np


def _build(segments, ds=1.0):
    pts = [(0.0, 0.0)]
    x = y = h = 0.0
    for kind, a, b in segments:
        if kind == "straight":
            for _ in range(int(a / ds)):
                x += ds * math.cos(h)
                y += ds * math.sin(h)
                pts.append((x, y))
        else:  # arc: a = radius, b = angle [rad], left positive
            n = max(1, int(abs(a * b) / ds))
            for _ in range(n):
                h += b / n
                x += ds * math.cos(h)
                y += ds * math.sin(h)
                pts.append((x, y))
    xs, ys = zip(*pts)
    return np.asarray(xs), np.asarray(ys)


def demo_route():
    """Straight 80 m -> left 90 deg R60 -> straight 50 m -> right 90 deg R45 -> straight 150 m."""
    return _build([("straight", 80, 0), ("arc", 60, math.pi / 2), ("straight", 50, 0),
                   ("arc", 45, -math.pi / 2), ("straight", 150, 0)])


def arc_route(radius=60.0, angle_deg=90.0, direction=1, straight=80.0, exit_len=150.0):
    ang = math.radians(angle_deg) * (1 if direction >= 0 else -1)
    return _build([("straight", straight, 0), ("arc", radius, ang), ("straight", exit_len, 0)])


def straight_route(length=400.0):
    return _build([("straight", length, 0)])
