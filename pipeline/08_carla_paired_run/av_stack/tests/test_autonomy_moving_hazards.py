from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from test_autonomy_episode import Actor, BehaviorBridge, BehaviorConfig, EpisodeSettings, PlantVehicle, TickWorld, carla, run_episode


class MovingHazardWorld(TickWorld):
    def __init__(self, hazard_type: str) -> None:
        super().__init__()
        self.hazard_type = hazard_type
        self.minimum_clearance = float("inf")
        self.appearance_speed = 0.0

    def tick(self) -> int:
        frame = super().tick()
        if frame == 60:
            self.appearance_speed = self.vehicle.plant.s.vx
            self.hazard = Actor(self, self.hazard_type, self.vehicle.get_location().x + 8.0)
            if self.hazard_type.startswith("walker"):
                self.hazard.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(0.3, 0.3, 0.9))
        if self.hazard is not None and 60 <= frame < 160:
            clearance = self.hazard.x - self.vehicle.get_location().x - 2.0
            if self.hazard_type.startswith("walker"):
                clearance -= 0.3
            self.minimum_clearance = min(self.minimum_clearance, clearance)
        if frame == 160:
            self.hazard.state = carla.TrafficLightState.Green
            self.hazard.x = 400.0
        return frame


def exercise_moving_hazard(controller: str, hazard_type: str) -> None:
    world = MovingHazardWorld(hazard_type)
    vehicle = PlantVehicle(world)
    bridge = BehaviorBridge(vehicle, BehaviorConfig(speed_mps=5.0, route_length_m=40.0))
    with TemporaryDirectory(prefix="av-stack-moving-hazard-") as temporary:
        output = Path(temporary)
        settings = EpisodeSettings(controller, 30.0, 0.05, output)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_episode(vehicle, bridge, settings)
        records = [json.loads(line) for line in (output / "behavior.jsonl").read_text().splitlines()]
        states = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        stops = [record for record in records if 3.0 <= record["t"] < 8.0 and record["stop"]]
        assert world.appearance_speed > 4.0
        assert stops and all(row["throttle"] == 0.0 and row["brake"] == 1.0 for row in stops)
        assert any(4.0 <= row["t"] < 8.0 and row["vx"] < 0.3 for row in states)
        assert world.minimum_clearance > 0.0
        assert any(row["t"] >= 8.0 and row["throttle"] > 0.0 for row in records)
        assert summary["route_completed"]
        print(f"PASS {controller} {hazard_type}: appears at {world.appearance_speed:.2f}m/s, clearance {world.minimum_clearance:.2f}m")


if __name__ == "__main__":
    for controller in ("lqr", "mpc"):
        for hazard in ("traffic.traffic_light", "walker.pedestrian.0001"):
            exercise_moving_hazard(controller, hazard)
