from __future__ import annotations

import contextlib
import io
import json
from dataclasses import asdict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.fault_injection import FaultConfig
from integration.unprotected_lateral import UnprotectedMPC
from integration.autonomy_episode import EpisodeSettings, run_episode
from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
from planner import ReferencePlanner, PlannerConfig
from sim.plant import BicyclePlant, PlantParams
from sim.tracks import arc_route
from test_autonomy_episode import PlantVehicle, TickWorld


def test_unprotected_failure_sends_neutral_and_recovers_after_fault_window() -> None:
    geometry = asdict(PlantParams())
    geometry.pop('max_steer_rate_rad_s')
    trajectory = ReferencePlanner().plan([0, 30, 60], [0, 0, 0])
    controller = UnprotectedMPC(geometry, 0.05, trajectory, FaultConfig('solver_fail', 1, 2))
    for index in range(80):
        t = index * 0.05
        result = controller.step(t, t*5, 0, 0, 5, 0, 0)
        failed = 1 <= t < 3
        assert controller.active == ('none' if failed else 'mpc')
        assert result.ok is not failed
        if failed:
            assert result.steer == 0
        record = json.loads(json.dumps(controller.record(), allow_nan=False))
        assert record['supervisor_enabled'] is False
        assert record['lqr_fallback_enabled'] is False


def test_unprotected_turn_departure_is_observable_in_real_plant() -> None:
    geometry = asdict(PlantParams())
    geometry.pop('max_steer_rate_rad_s')
    trajectory = ReferencePlanner(PlannerConfig(v_max=5, v_start=5)).plan(
        *arc_route(radius=35, straight=15, exit_len=20))
    controller = UnprotectedMPC(geometry, 0.05, trajectory, FaultConfig('solver_fail', 5, 20))
    plant = BicyclePlant()
    plant.s.x, plant.s.y, plant.s.yaw, plant.s.vx = trajectory.x[0], trajectory.y[0], trajectory.yaw[0], 5
    maximum_error = 0.0
    for index in range(240):
        state = plant.s
        result = controller.step(index*.05, state.x, state.y, state.yaw, state.vx, state.vy, state.r)
        plant.step(result.steer, 0, .05)
        maximum_error = max(maximum_error, abs(result.e_y))
    assert maximum_error > 2, maximum_error
    assert controller.active == 'none'
    print('PASS no-supervisor curved departure max_error=%.3fm' % maximum_error)


def test_unprotected_episode_records_no_fallback_and_keeps_speed_control() -> None:
    world = TickWorld()
    vehicle = PlantVehicle(world)
    bridge = BehaviorBridge(vehicle, BehaviorConfig(speed_mps=5, route_length_m=40))
    with TemporaryDirectory(prefix='av-no-supervisor-') as directory:
        output = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_episode(vehicle, bridge, EpisodeSettings(
                'mpc_unprotected', 25, .05, output, fault=FaultConfig('solver_fail', 3, 30)))
        rows = [json.loads(row) for row in (output/'behavior.jsonl').read_text().splitlines()]
        failed = [row for row in rows if row['t'] >= 3]
        assert failed and all(row['steer_source'] == 'none' and row['steer_rad'] == 0 for row in failed)
        assert any(row['throttle'] > 0 and not row['stop'] for row in failed)
        assert (output/'supervisor.jsonl').read_text() == ''
        assert summary['unprotected']['supervisor_enabled'] is False
        assert summary['unprotected']['lqr_fallback_enabled'] is False


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            test()
            print('PASS '+name)
