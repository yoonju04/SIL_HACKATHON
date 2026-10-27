from __future__ import annotations

import math

import numpy as np

from planner import PlannerConfig, ReferencePlanner, Trajectory
from planner.cubic_spline import CubicSpline2D


def checkpoints_to_reference(points: np.ndarray, pose: tuple[float, float, float],
                             target_speed: float, measured_speed: float,
                             backward: str = 'quadratic') -> Trajectory:
    """Model checkpoints (ego frame, first one ~2.5 m ahead) -> map-frame reference.

    The path is extended back past the vehicle. 'quadratic' continues the curvature of the
    first checkpoints (LQR has no preview and needs it at the matched point); 'linear' extends
    the first segment straight, which avoids fake e_y / e_psi at turn entry for the MPC
    (work/turn_entry_sweep.py: max path deviation 0.45 -> 0.38 m, LQR 0.34 -> 0.92 m).
    """
    if backward not in ('quadratic', 'linear'):
        raise ValueError('backward must be quadratic or linear')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Model must provide at least three finite XY checkpoints.')
    if not all(math.isfinite(value) for value in (*pose, target_speed, measured_speed)) or target_speed < 0:
        raise ValueError('Invalid model speed or sensor pose.')
    local = points.copy()
    keep = np.r_[True, np.linalg.norm(np.diff(local, axis=0), axis=1) > .05]
    local = local[keep]
    if len(local) < 3 or np.linalg.norm(np.diff(local, axis=0), axis=1).sum() < 1:
        raise ValueError('Degenerate predicted path.')
    direction = local[1] - local[0]
    direction /= np.linalg.norm(direction)
    backward_distance = max(float(local[0] @ direction), 0.)
    if backward == 'linear':
        extension = local[0] - np.outer([backward_distance+1., backward_distance], direction)
    else:
        fit_count = min(4, len(local))
        fit_station = np.r_[0., np.cumsum(np.linalg.norm(np.diff(local[:fit_count], axis=0), axis=1))]
        coefficients = np.polynomial.polynomial.polyfit(fit_station, local[:fit_count], 2)
        backward_station = np.array([-backward_distance-1., -backward_distance])
        extension = np.polynomial.polynomial.polyval(backward_station, coefficients).T
    local = np.vstack((extension, local))
    local = local[np.r_[True, np.linalg.norm(np.diff(local, axis=0), axis=1) > .05]]
    x, y, yaw = pose
    world_x = x + math.cos(yaw)*local[:, 0] - math.sin(yaw)*local[:, 1]
    world_y = -(y + math.sin(yaw)*local[:, 0] + math.cos(yaw)*local[:, 1])
    spline = CubicSpline2D(world_x, world_y)
    station = np.linspace(0, spline.length, math.ceil(spline.length/.25)+1)
    px, py = spline.position(station)
    yaw = np.unwrap(spline.yaw(station))
    curvature = spline.curvature(station)
    planner = ReferencePlanner(PlannerConfig(v_max=max(target_speed, .1),
                               v_start=target_speed, v_end=None))
    speed, _ = planner.speed_profile(station, curvature)
    speed = np.minimum(speed, target_speed)
    times = np.r_[0, np.cumsum(np.diff(station)/np.maximum(.5*(speed[1:]+speed[:-1]), .5))]
    metadata = dict(source='sensor model prediction',
                    checkpoint_semantics='path geometry with separately supplied target speed', capture_pose=pose)
    return Trajectory(station, px, py, yaw, curvature, speed, times, px.copy(), py.copy(), metadata)
