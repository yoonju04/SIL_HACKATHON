from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys
import contextlib
import io
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.fault_injection import FaultConfig
from integration.supervised_lateral import SupervisedLateral
from planner import PlannerConfig, ReferencePlanner
from sim.plant import PlantParams
from sim.tracks import arc_route
from test_autonomy_episode import TickWorld, PlantVehicle
from integration.autonomy_episode import EpisodeSettings, run_episode
from integration.behavior_bridge import BehaviorBridge, BehaviorConfig


def test_curved_takeover_record_can_be_written_as_json() -> None:
    geometry = asdict(PlantParams())
    geometry.pop('max_steer_rate_rad_s')
    trajectory = ReferencePlanner(PlannerConfig(v_max=5, v_start=5)).plan(
        *arc_route(radius=35, straight=15, exit_len=20))
    point = trajectory.at(25)
    controller = SupervisedLateral(geometry, 0.05, trajectory, FaultConfig('solver_fail', 0, 2))
    controller.step(0, float(point['x']), float(point['y']), float(point['yaw']), 5, 0, 0)
    record = controller.record()
    encoded = json.dumps(dict(t=0, **record), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded['active_controller'] == 'lqr'
    assert decoded['safe_stop'] is False
    assert type(record['safe_stop']) is bool


def test_curved_episode_continues_after_transition_log_is_saved() -> None:
    world = TickWorld()
    vehicle = PlantVehicle(world)
    bridge = BehaviorBridge(vehicle, BehaviorConfig(speed_mps=5, route_length_m=100))
    route_x, route_y = arc_route(radius=35, straight=15, exit_len=20)
    bridge.trajectory = ReferencePlanner(PlannerConfig(v_max=5, v_start=5)).plan(
        route_x + vehicle.plant.s.x, route_y + vehicle.plant.s.y)
    with TemporaryDirectory(prefix='av-supervisor-json-') as directory:
        output = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_episode(vehicle, bridge, EpisodeSettings(
                'supervised', 10, 0.05, output, fault=FaultConfig('solver_fail', 5, 2)))
        transitions = [json.loads(row) for row in (output/'supervisor.jsonl').read_text().splitlines()]
        behavior = [json.loads(row) for row in (output/'behavior.jsonl').read_text().splitlines()]
        assert len(transitions) == 1 and transitions[0]['t'] == 5
        assert transitions[0]['safe_stop'] is False
        assert any(abs(row['steer_rad']) > 0.01 and row['steer_source'] == 'lqr' for row in behavior)
        assert behavior[-1]['t'] > 7 and behavior[-1]['steer_source'] == 'lqr'
        assert summary['supervisor']['switch_count'] == 1
        assert summary['supervisor']['safe_stop'] is False


if __name__ == '__main__':
    test_curved_takeover_record_can_be_written_as_json()
    print('PASS test_curved_takeover_record_can_be_written_as_json')
    test_curved_episode_continues_after_transition_log_is_saved()
    print('PASS test_curved_episode_continues_after_transition_log_is_saved')
