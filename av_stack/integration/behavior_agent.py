"""Official CARLA behavior decisions with a non-actuating local-planner sink.

Private extension points are pinned to CARLA commit d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18.
The source patch is recorded in third_party/carla_0915/PATCHES.md. No upstream PID is stepped.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import carla

VENDOR = Path(__file__).resolve().parents[1] / "third_party" / "carla_0915"
sys.path.insert(0, str(VENDOR))

import agents  # noqa: E402
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402
from agents.navigation.local_planner import LocalPlanner, RoadOption  # noqa: E402
from integration.pedestrian_risk import crossing_hazard  # noqa: E402

if Path(agents.__file__).resolve().parent != VENDOR / "agents":
    raise ImportError("Another 'agents' package is loaded; run run_autonomy.py in a fresh process.")


class SpeedCapture:
    """Mutable decision sink: records km/h as m/s, never computes actuators."""

    def __init__(self) -> None:
        self.speed_mps = 0.0

    def run_step(self, target_speed: float, waypoint: carla.Waypoint) -> carla.VehicleControl:
        self.speed_mps = max(0.0, float(target_speed) / 3.6)
        return carla.VehicleControl()


class PlanningLocalPlanner(LocalPlanner):
    """Keep upstream waypoint progression while replacing its PID endpoint."""

    def _init_controller(self) -> None:
        self.capture = SpeedCapture()
        self._vehicle_controller = self.capture
        waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = waypoint, RoadOption.LANEFOLLOW
        self._waypoints_queue.append((waypoint, RoadOption.LANEFOLLOW))
        self.revision = 0
        self.route = []

    def set_global_plan(self, current_plan: Iterable[tuple[carla.Waypoint, RoadOption]],
                        stop_waypoint_creation: bool = True, clean_queue: bool = True) -> None:
        # Upstream API requires these arguments; queue purges are not route changes.
        super().set_global_plan(list(current_plan), stop_waypoint_creation, clean_queue)
        self.route = list(self._waypoints_queue)
        self.revision += 1


class PlanningAgent(BehaviorAgent):
    """Upstream behavior policy with observable reasons, without actuator ownership."""

    def __init__(self, vehicle: carla.Vehicle, dt: float, behavior: str) -> None:
        # BasicAgent constructs a PID object but never steps it during construction.
        options = {"dt": dt, "sampling_resolution": 2.0}
        super().__init__(vehicle, behavior=behavior, opt_dict=options)
        self._local_planner = PlanningLocalPlanner(vehicle, opt_dict=options, map_inst=self._map)
        self.reason = "cruise"
        self.hazard_actor_id: int | None = None

    def traffic_light_manager(self) -> bool:
        affected = super().traffic_light_manager()
        if affected:
            self.reason = "red_light"
            self.hazard_actor_id = self._last_traffic_light.id
        return affected

    def pedestrian_avoid_manager(self, waypoint: carla.Waypoint) -> tuple:
        walkers = self._world.get_actors().filter("*walker.pedestrian*")
        if not any(walker.get_location().distance(waypoint.transform.location) < 10.0 for walker in walkers):
            return False, None, -1.0
        result = super().pedestrian_avoid_manager(waypoint)
        if result[0]:
            self.reason = "pedestrian"
            self.hazard_actor_id = result[1].id
        return result

    def collision_and_car_avoid_manager(self, waypoint: carla.Waypoint) -> tuple:
        result = super().collision_and_car_avoid_manager(waypoint)
        if result[0]:
            self.reason = "vehicle"
            self.hazard_actor_id = result[1].id
        return result

    def run_step(self, debug: bool = False) -> carla.VehicleControl:
        self.reason = "cruise"
        self.hazard_actor_id = None
        pedestrian = crossing_hazard(self._vehicle, self._local_planner.get_plan())
        if pedestrian is not None:
            self.reason = "pedestrian"
            self.hazard_actor_id = pedestrian.id
            return self.emergency_stop()
        return super().run_step(debug=debug)
