"""Run directly with the CARLA Python environment; no CARLA server required."""
import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_fixture import Actor, carla, make_ego, test_native_map_fixture


def test_bridge_available() -> None:
    # Given: the installed source tree.
    # When: looking up its behavior integration.
    spec = importlib.util.find_spec('integration.behavior_bridge')
    # Then: the adapter is importable.
    assert spec is not None


def test_clear_road_speed_conversion() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: cautious behavior on a 30 km/h road, with a 10 m/s cruise cap.
    ego = make_ego()
    ego.speed_limit = 30.0
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    # When: planning the clear-road behavior.
    decision = bridge.step()
    # Then: upstream's (30 - 6) km/h target is converted to m/s.
    assert math.isclose(decision.speed_mps, 24.0 / 3.6, abs_tol=1e-6)
    assert not decision.emergency_stop


def test_red_light_stops() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: a red signal trigger four metres ahead on the ego lane.
    ego = make_ego()
    Actor(ego.world, 'traffic.traffic_light', ego.x + 4.0)
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    # When: official traffic-light logic evaluates the signal.
    decision = bridge.step()
    # Then: motion stops before the signal.
    assert decision.emergency_stop
    assert decision.speed_mps == 0.0


def test_green_light_permits_motion() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: the same signal location with green state.
    ego = make_ego()
    light = Actor(ego.world, 'traffic.traffic_light', ego.x + 4.0)
    light.state = carla.TrafficLightState.Green
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    # When: official traffic-light logic evaluates the signal.
    decision = bridge.step()
    # Then: the requested cruise motion is permitted.
    assert not decision.emergency_stop
    assert decision.speed_mps > 0.0


def test_close_pedestrian_stops() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: a pedestrian six metres ahead, inside upstream's 10m walker search.
    ego = make_ego()
    pedestrian = Actor(ego.world, 'walker.pedestrian.0001', ego.x + 6.0)
    pedestrian.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(0.3, 0.3, 0.9))
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    # When: official pedestrian detection and braking distance logic runs.
    decision = bridge.step()
    # Then: the bridge requests an emergency stop.
    assert decision.emergency_stop
    assert decision.speed_mps == 0.0


def test_close_stopped_vehicle_stops() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: a stationary vehicle eight metres ahead.
    ego = make_ego()
    Actor(ego.world, 'vehicle.blocker', ego.x + 8.0)
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    # When: official vehicle obstacle logic runs.
    decision = bridge.step()
    # Then: the bridge requests an emergency stop.
    assert decision.emergency_stop
    assert decision.speed_mps == 0.0
    assert decision.reason == 'vehicle', decision.reason


def test_slow_lead_vehicle_reduces_speed() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: ego at 10 m/s and a slower vehicle within detection and TTC range.
    ego = make_ego()
    ego.speed = 10.0
    lead = Actor(ego.world, 'vehicle.lead', ego.x + 16.0)
    lead.speed = 5.0
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    # When: official car-following TTC policy selects a speed.
    decision = bridge.step()
    # Then: it slows without invoking the emergency-stop branch.
    assert not decision.emergency_stop
    assert 0.0 < decision.speed_mps < 5.0


def test_upstream_pid_never_runs() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: upstream PID is instrumented to fail if called.
    ego = make_ego()
    with patch('agents.navigation.controller.VehiclePIDController.run_step',
               side_effect=AssertionError('upstream PID executed')):
        bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
        # When: behavior plans a moving step.
        decision = bridge.step()
    # Then: the behavior still requests motion; fake apply_control also never ran.
    assert decision.speed_mps > 0.0


def test_queue_progress_preserves_reference() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: a built reference and the ego advanced past its first waypoint.
    ego = make_ego()
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    original = bridge.trajectory
    revision = bridge.revision
    queue = bridge.agent.get_local_planner().get_plan()
    old_length = len(queue)
    ego.x += 2.0
    # When: LocalPlanner purges passed waypoints.
    decision = bridge.step()
    # Then: queue progress does not rebuild the MPC/LQR reference.
    assert len(queue) < old_length
    assert bridge.trajectory is original
    assert bridge.revision == revision == decision.route_revision


def test_explicit_route_update_refreshes_reference() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    from agents.navigation.local_planner import RoadOption
    # Given: a bridge whose official agent is given a new shorter route.
    ego = make_ego()
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=10.0, route_length_m=120.0))
    original = bridge.trajectory
    revision = bridge.revision
    points = [ego.world.map.get_waypoint(carla.Location(x, 1.75, 0.0))
              for x in range(20, 82, 2)]
    bridge.agent.set_global_plan([(point, RoadOption.LANEFOLLOW) for point in points])
    # When: the bridge observes the new global plan.
    decision = bridge.step()
    # Then: reference and revision change together.
    assert bridge.trajectory is not original
    assert bridge.revision > revision
    assert decision.route_revision == bridge.revision


def test_configured_cruise_caps_upstream_speed() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: the official cautious policy permits 40 km/h but cruise is 4 m/s.
    ego = make_ego()
    bridge = BehaviorBridge(ego, BehaviorConfig(speed_mps=4.0, route_length_m=120.0))
    # When: clear-road speed is selected.
    decision = bridge.step()
    # Then: the configured cap wins after unit conversion.
    assert decision.speed_mps == 4.0


def test_destination_routes_with_official_global_planner() -> None:
    from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
    # Given: an explicit destination on the native map.
    ego = make_ego()
    config = BehaviorConfig(destination=carla.Location(100.0, 1.75, 0.0))
    # When: the bridge builds a route through official GlobalRoutePlanner.
    bridge = BehaviorBridge(ego, config)
    # Then: its route ends near the requested location.
    route = bridge.agent.get_local_planner().get_plan()
    assert abs(route[-1][0].transform.location.x - 100.0) < 3.0
    assert len(route) > 20


def test_angle_zero_is_in_forward_hazard_sector() -> None:
    from agents.tools.misc import is_within_distance
    # Given: an exact straight-ahead target within distance.
    origin = carla.Transform(carla.Location(), carla.Rotation())
    target = carla.Transform(carla.Location(4.0, 0.0, 0.0), carla.Rotation())
    # When: evaluating the forward hazard sector used by BehaviorAgent.
    detected = is_within_distance(target, origin, 5.0, [0, 90])
    # Then: collinear hazards are included.
    assert detected


def test_forward_sector_rejects_side_behind_and_far_targets() -> None:
    from agents.tools.misc import is_within_distance
    # Given: lateral, rearward, and out-of-range target positions.
    origin = carla.Transform(carla.Location(), carla.Rotation())
    targets = [carla.Location(0.0, 4.0, 0.0), carla.Location(-4.0, 0.0, 0.0),
               carla.Location(6.0, 0.0, 0.0)]
    # When: evaluating the same forward sector.
    detected = [is_within_distance(carla.Transform(point), origin, 5.0, [0, 90])
                for point in targets]
    # Then: the lower-bound fix does not broaden the side/rear/distance boundaries.
    assert detected == [False, False, False]


if __name__ == '__main__':
    checks = [value for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for check in checks:
        check()
        print('PASS ' + check.__name__)
    print('{} server-free checks passed'.format(len(checks)))
