from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_fixture import Actor, World, carla, make_ego


class MovingWalker(Actor):
    def __init__(self, world: World, location: carla.Location, velocity: carla.Vector3D) -> None:
        super().__init__(world, "walker.pedestrian.0004", location.x)
        self.location = location
        self.velocity = velocity
        self.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(0.3, 0.3, 0.9))

    def get_location(self) -> carla.Location:
        return self.location

    def get_velocity(self) -> carla.Vector3D:
        return self.velocity


def crossing_fixture():
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    ego = make_ego()
    ego.speed = 5.0
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=5.0, route_length_m=120.0))
    return ego, bridge


def test_crossing_walker_is_detected_before_entering_lane() -> None:
    from integration.pedestrian_risk import crossing_hazard
    ego, bridge = crossing_fixture()
    walker = MovingWalker(ego.world, carla.Location(ego.x + 14.0, 5.0), carla.Vector3D(0.0, -1.2, 0.0))
    threat = crossing_hazard(ego, bridge.agent.get_local_planner().get_plan())
    assert threat is not None and threat.id == walker.id


def test_parallel_sidewalk_walker_does_not_stop_ego() -> None:
    from integration.pedestrian_risk import crossing_hazard
    ego, bridge = crossing_fixture()
    MovingWalker(ego.world, carla.Location(ego.x + 14.0, 5.0), carla.Vector3D(1.2, 0.0, 0.0))
    assert crossing_hazard(ego, bridge.agent.get_local_planner().get_plan()) is None


def test_pedestrian_behind_ego_is_not_a_forward_threat() -> None:
    from integration.pedestrian_risk import crossing_hazard
    ego, bridge = crossing_fixture()
    MovingWalker(ego.world, carla.Location(ego.x - 3.0, 1.75), carla.Vector3D())
    assert crossing_hazard(ego, bridge.agent.get_local_planner().get_plan()) is None


def test_bridge_yields_to_predicted_crossing_without_using_upstream_pid() -> None:
    ego, bridge = crossing_fixture()
    walker = MovingWalker(ego.world, carla.Location(ego.x + 14.0, 5.0), carla.Vector3D(0.0, -1.2, 0.0))
    decision = bridge.step()
    assert decision.emergency_stop and decision.speed_mps == 0.0
    assert decision.reason == "pedestrian" and decision.hazard_actor_id == walker.id


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
