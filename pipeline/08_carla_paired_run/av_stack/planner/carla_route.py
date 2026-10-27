"""Lane-centre route from a CARLA map (input to the planner).

Does not import `carla`: pass the map object and carla.LaneType.Driving in.
CARLA is left-handed (y right, yaw clockwise); the returned points are in the
planner's right-handed frame (y_rh = -y_carla).
"""
from __future__ import annotations

import numpy as np


def route_from_carla_map(world_map, start_location, length_m: float = 600.0,
                         step_m: float = 1.0, lane_type=None):
    """Follow the lane from the waypoint nearest to start_location.

    At junctions the first branch returned by Waypoint.next() is taken, so the
    route is deterministic for a given map and spawn point.
    Returns (x, y) numpy arrays in the right-handed frame.
    """
    kwargs = {"project_to_road": True}
    if lane_type is not None:
        kwargs["lane_type"] = lane_type
    wp = world_map.get_waypoint(start_location, **kwargs)
    xs, ys, dist = [], [], 0.0
    while dist <= length_m:
        loc = wp.transform.location
        xs.append(loc.x)
        ys.append(-loc.y)
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
        dist += step_m
    return np.asarray(xs), np.asarray(ys)
