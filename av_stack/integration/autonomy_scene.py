"""Owned demo actors and collision evidence for a single simulator episode."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import carla

from planner import Trajectory


@dataclass(frozen=True)
class Collision:
    frame: int
    actor_id: int
    actor_type: str
    impulse: float


class AutonomyScene:
    def __init__(self, vehicle: carla.Vehicle, evidence_directory: Path | None = None) -> None:
        self.world = vehicle.get_world()
        self.demo_actor = None
        self.collisions: list[Collision] = []
        self.evidence_directory = evidence_directory
        blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        impulse = event.normal_impulse
        self.collisions.append(Collision(event.frame, event.other_actor.id, event.other_actor.type_id,
                                         math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)))
        if self.evidence_directory is not None:
            with (self.evidence_directory / "collision_events_live.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(self.collisions[-1])) + "\n")

    def spawn_demo(self, trajectory: Trajectory, kind: str) -> None:
        if trajectory.length < 45.0:
            raise ValueError("Demo requires a route of at least 45 m.")
        point = trajectory.at(30.0)
        location = carla.Location(x=float(point["x"]), y=-float(point["y"]))
        ground = self.world.get_map().get_waypoint(location).transform.location
        location.z = ground.z + 0.6
        transform = carla.Transform(location, carla.Rotation(yaw=-math.degrees(float(point["yaw"]))))
        blueprint_name = {"pedestrian": "walker.pedestrian.0001", "vehicle": "vehicle.tesla.model3"}[kind]
        blueprint = self.world.get_blueprint_library().find(blueprint_name)
        self.demo_actor = self.world.try_spawn_actor(blueprint, transform)
        if self.demo_actor is None:
            raise RuntimeError("Demo spawn is occupied; use another spawn point or clear the scene.")
        if kind == "vehicle":
            self.demo_actor.apply_control(carla.VehicleControl(brake=1.0))
        else:
            self.demo_actor.apply_control(carla.WalkerControl(speed=0.0))

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "collisions.json").open("w", encoding="utf-8") as stream:
            json.dump([asdict(event) for event in self.collisions], stream, indent=2)

    def close(self) -> None:
        try:
            self.sensor.stop()
            self.sensor.destroy()
        finally:
            if self.demo_actor is not None:
                self.demo_actor.destroy()
