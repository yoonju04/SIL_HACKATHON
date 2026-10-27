from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class LidarDecision:
    brake: bool
    speed_cap: float | None
    clearance_m: float | None


def lidar_guard(points: np.ndarray, speed: float) -> LidarDecision:
    """Request a stop from ego-frame LiDAR; no actor or scenario state is used."""
    cloud = np.asarray(points)
    valid = (np.isfinite(cloud[:, :3]).all(axis=1) & (cloud[:, 0] > 0)
             & (np.abs(cloud[:, 1]) < 1.2)
             & (cloud[:, 2] > .25) & (cloud[:, 2] < 2.0))
    distances = cloud[valid, 0]
    if len(distances) < 8:
        return LidarDecision(False, None, None)
    clearance = max(0., float(np.partition(distances, 7)[7])-2.5)
    deceleration, reaction, margin = 5., .25, 1.
    velocity = max(0., speed)
    stopping = margin + reaction*velocity + velocity*velocity/(2*deceleration)
    if clearance <= stopping:
        return LidarDecision(True, 0., clearance)
    usable = max(0., clearance-margin)
    cap = -deceleration*reaction + math.sqrt((deceleration*reaction)**2+2*deceleration*usable)
    return LidarDecision(False, cap, clearance)
