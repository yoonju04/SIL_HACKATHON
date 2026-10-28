"""Constant-velocity crossing prediction before the upstream lane-membership gate."""
from __future__ import annotations

from typing import Iterable

import carla
import numpy as np


def crossing_hazard(vehicle: carla.Vehicle, plan: Iterable[tuple]) -> carla.Actor | None:
    location = vehicle.get_location()
    speed = vehicle.get_velocity().length()
    lookahead = max(12.0, speed * speed / 5.0 + speed + vehicle.bounding_box.extent.x + 2.0)
    points = [(location.x, location.y)]
    points.extend((waypoint.transform.location.x, waypoint.transform.location.y)
                  for waypoint, _ in list(plan)[:25])
    if len(points) < 2:
        return None
    positions = np.asarray(points, dtype=float)
    segments = np.diff(positions, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    length2 = np.maximum(lengths * lengths, 1e-9)
    stations = np.r_[0.0, np.cumsum(lengths)]
    nearest_actor = None
    nearest_station = float("inf")
    for walker in vehicle.get_world().get_actors().filter("walker.pedestrian.*"):
        position = walker.get_location()
        if position.distance(location) > lookahead + 7.0:
            continue
        point = np.array((position.x, position.y))
        fractions = np.clip(np.sum((point - positions[:-1]) * segments, axis=1) / length2, 0.0, 1.0)
        projections = positions[:-1] + fractions[:, None] * segments
        distance2 = np.sum((projections - point)**2, axis=1)
        index = int(np.argmin(distance2))
        along = float(stations[index] + fractions[index] * lengths[index])
        if not 0.1 < along <= lookahead or lengths[index] < 1e-6:
            continue
        tangent = segments[index] / lengths[index]
        normal = np.array((-tangent[1], tangent[0]))
        lateral = float(np.dot(point - projections[index], normal))
        velocity = walker.get_velocity()
        lateral_speed = float(np.dot((velocity.x, velocity.y), normal))
        corridor = vehicle.bounding_box.extent.y + max(walker.bounding_box.extent.x, walker.bounding_box.extent.y) + 0.45
        inside = abs(lateral) <= corridor
        moving_toward = lateral * lateral_speed < -0.1
        crossing_time = max(0.0, abs(lateral) - corridor) / max(abs(lateral_speed), 0.1)
        arriving_soon = moving_toward and crossing_time <= along / max(speed, 0.5) + 1.0
        if (inside or arriving_soon) and along < nearest_station:
            nearest_actor, nearest_station = walker, along
    return nearest_actor
