from __future__ import annotations

import math
from typing import Callable

import carla
import numpy as np


class UrbanCrossings:
    def __init__(self, world: carla.World, plan: list, stations: np.ndarray) -> None:
        self.world = world
        self.crossings = []
        self.log: Callable = lambda *args: None
        self.reason: Callable[[], str] = lambda: ""
        self.hazard_id: Callable = lambda: None
        self.ego = None
        blueprint = world.get_blueprint_library().find("walker.pedestrian.0001")
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        for target in (200.0, 520.0):
            for index in np.argsort(np.abs(stations - target))[:80]:
                waypoint = plan[int(index)][0]
                if waypoint.is_junction or abs(float(stations[index]) - target) > 80.0:
                    continue
                center = waypoint.transform.location
                right = waypoint.transform.get_right_vector()
                candidate = center + carla.Location(x=5.0 * right.x, y=5.0 * right.y)
                sidewalk = world.get_map().get_waypoint(candidate, lane_type=carla.LaneType.Sidewalk)
                if sidewalk is None or sidewalk.transform.location.distance(center) > 9.0:
                    continue
                start = sidewalk.transform.location
                delta = carla.Vector3D(center.x - start.x, center.y - start.y, 0.0)
                distance = math.hypot(delta.x, delta.y)
                if distance < 2.0:
                    continue
                direction = carla.Vector3D(delta.x / distance, delta.y / distance, 0.0)
                spawn = carla.Location(start.x, start.y, start.z + 0.5)
                actor = world.try_spawn_actor(blueprint, carla.Transform(spawn))
                if actor is None:
                    continue
                actor.apply_control(carla.WalkerControl(speed=0.0))
                self.crossings.append(dict(actor=actor, s=float(stations[index]), direction=direction,
                                           distance=distance, phase="waiting", started=0.0,
                                           observed=False))
                break

    def tick(self, t: float, progress: float) -> None:
        for index, event in enumerate(self.crossings):
            if event["phase"] == "waiting" and 0.0 < event["s"] - progress < 24.0:
                event["phase"] = "crossing"
                event["started"] = t
                event["actor"].apply_control(carla.WalkerControl(direction=event["direction"], speed=1.25))
                self.log("pedestrian_crossing_started", t, crossing=index, station=event["s"])
            if event["phase"] == "crossing":
                elapsed = t - event["started"]
                if elapsed > (event["distance"] + 7.0) / 1.25:
                    event["actor"].apply_control(carla.WalkerControl(speed=0.0))
                    event["phase"] = "finished"
                    self.log("pedestrian_crossing_finished", t, crossing=index)
                if self.ego is not None:
                    gap = event["s"] - progress
                    if (self.reason() == "pedestrian" and self.hazard_id() == event["actor"].id and 0.0 < gap < 15.0
                            and self.ego.get_velocity().length() < 0.4 and not event["observed"]):
                        event["observed"] = True
                        self.log("ego_waited_for_crossing", t, crossing=index, gap_m=gap)

    def close(self) -> None:
        for event in self.crossings:
            if event["actor"].is_alive:
                event["actor"].destroy()
