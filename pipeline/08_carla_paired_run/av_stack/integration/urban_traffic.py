"""Episode-owned Traffic Manager vehicles and navigation-mesh pedestrians."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, TypedDict

import carla
if TYPE_CHECKING:
    from integration.behavior_agent import RoadOption


@dataclass(frozen=True)
class UrbanTrafficConfig:
    vehicle_count: int = 30
    walker_count: int = 60
    seed: int = 17
    tm_port: int = 8050


class TrafficMetadata(TypedDict):
    vehicles_requested: int
    vehicles_spawned: int
    walkers_requested: int
    walkers_spawned: int
    seed: int
    tm_port: int
    following_distance_m: float
    speed_reduction_percent: list[float]
    pedestrian_cross_factor: float
    pedestrian_speed_mps: list[float]


class UrbanTraffic:
    """Mutable owner of NPC actors; never owns the ego or world tick loop."""

    def __init__(self, client: carla.Client, config: UrbanTrafficConfig) -> None:
        self.client = client
        self.config = config
        self.world = client.get_world()
        self.random = random.Random(config.seed)
        self.vehicles: list[carla.Vehicle] = []
        self.walkers: list[carla.Walker] = []
        self.controllers: list[carla.WalkerAIController] = []
        self._vehicle_count = 0
        self._walker_count = 0
        self.tm = client.get_trafficmanager(config.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(config.seed)
        self.tm.set_global_distance_to_leading_vehicle(5.0)
        self.world.set_pedestrians_seed(config.seed)
        self.world.set_pedestrians_cross_factor(0.15)

    def spawn(self, ego: carla.Vehicle,
              route: list[tuple[carla.Waypoint, RoadOption]]) -> None:
        """Spawn moving traffic near the route with bounded placement attempts."""
        locations = [waypoint.transform.location for waypoint, _ in route]
        if not locations:
            raise ValueError("Urban traffic requires a nonempty route.")
        blueprints = self.world.get_blueprint_library()
        cars = sorted((bp for bp in blueprints.filter("vehicle.*")
                       if bp.id in ("vehicle.audi.a2", "vehicle.audi.tt", "vehicle.tesla.model3",
                                    "vehicle.lincoln.mkz_2020")),
                      key=lambda bp: bp.id)
        points = [point for point in self.world.get_map().get_spawn_points()
                  if point.location.distance(ego.get_location()) > 15.0
                  and min(point.location.distance(loc) for loc in locations) < 100.0]
        self.random.shuffle(points)
        for transform in points:
            if len(self.vehicles) >= self.config.vehicle_count or not cars:
                break
            blueprint = self.random.choice(cars)
            blueprint.set_attribute("role_name", "urban_traffic")
            if blueprint.has_attribute("color"):
                colors = blueprint.get_attribute("color").recommended_values
                if colors:
                    blueprint.set_attribute("color", self.random.choice(colors))
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is None:
                continue
            self.vehicles.append(vehicle)
            vehicle.set_autopilot(True, self.config.tm_port)
            self.tm.vehicle_percentage_speed_difference(vehicle, self.random.uniform(15.0, 25.0))
            self.tm.update_vehicle_lights(vehicle, True)

        pedestrians = sorted((bp for bp in blueprints.filter("walker.pedestrian.*")
                              if bp.id in ("walker.pedestrian.0001", "walker.pedestrian.0002",
                                           "walker.pedestrian.0003", "walker.pedestrian.0004")), key=lambda bp: bp.id)
        controller_bp = blueprints.find("controller.ai.walker")
        destinations: list[carla.Location] = []
        for _ in range(max(0, self.config.walker_count) * 40):
            if len(self.walkers) >= self.config.walker_count or not pedestrians:
                break
            location = self.world.get_random_location_from_navigation()
            if location is None or min(location.distance(loc) for loc in locations) >= 60.0:
                continue
            destination = self.world.get_random_location_from_navigation()
            if destination is None or min(destination.distance(loc) for loc in locations) >= 60.0:
                continue
            blueprint = self.random.choice(pedestrians)
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(blueprint, carla.Transform(location))
            if walker is None:
                continue
            self.walkers.append(walker)
            controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller is None:
                walker.destroy()
                self.walkers.pop()
                continue
            self.controllers.append(controller)
            destinations.append(destination)
        if self.controllers:
            # The caller owns synchronous ticking; do not wait for another client.
            self.world.tick()
        for controller, destination in zip(self.controllers, destinations):
            controller.start()
            controller.go_to_location(destination)
            controller.set_max_speed(self.random.uniform(1.0, 1.5))
        self._vehicle_count = len(self.vehicles)
        self._walker_count = len(self.walkers)
        print("Urban traffic: %d/%d vehicles, %d/%d moving pedestrians" % (
            self._vehicle_count, self.config.vehicle_count,
            self._walker_count, self.config.walker_count))

    def metadata(self) -> TrafficMetadata:
        """Return serializable spawn totals, retaining counts after cleanup."""
        return TrafficMetadata(
            vehicles_requested=self.config.vehicle_count, vehicles_spawned=self._vehicle_count,
            walkers_requested=self.config.walker_count, walkers_spawned=self._walker_count,
            seed=self.config.seed, tm_port=self.config.tm_port, following_distance_m=5.0,
            speed_reduction_percent=[15.0, 25.0], pedestrian_cross_factor=0.15,
            pedestrian_speed_mps=[1.0, 1.5])

    def close(self) -> None:
        """Stop owned walker AI before destruction, leaving unrelated actors alone."""
        try:
            for controller in self.controllers:
                if controller.is_alive:
                    controller.stop()
        finally:
            try:
                owned = [actor for actor in self.controllers + self.walkers + self.vehicles if actor.is_alive]
                if owned:
                    responses = self.client.apply_batch_sync(
                        [carla.command.DestroyActor(actor.id) for actor in owned], False)
                    errors = [response.error for response in responses
                              if response.error and response.error != "unable to destroy actor: not found"]
                    if errors:
                        raise RuntimeError("Urban traffic cleanup failed: " + "; ".join(errors))
                self.controllers.clear()
                self.walkers.clear()
                self.vehicles.clear()
            finally:
                self.tm.set_synchronous_mode(False)
                # CARLA exposes no getter; restore its documented default.
                self.world.set_pedestrians_cross_factor(0.0)
