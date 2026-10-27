from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_autonomy_episode import TickWorld, PlantVehicle
from behavior_fixture import Actor, carla
from integration.autonomy_episode import EpisodeSettings, run_episode
from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
from integration.fault_injection import FaultConfig


def exercise_fault(mode: str, hazard: str = '') -> None:
    world = TickWorld()
    vehicle = PlantVehicle(world)
    if hazard:
        actor = Actor(world, hazard, 26.0)
        if hazard.startswith('walker'):
            actor.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(0.3, 0.3, 0.9))
        world.hazard = actor
    bridge = BehaviorBridge(vehicle, BehaviorConfig(speed_mps=5.0, route_length_m=40.0))
    with TemporaryDirectory(prefix='av-supervised-') as directory:
        output = Path(directory)
        settings = EpisodeSettings('supervised', 25.0, 0.05, output,
                                   fault=FaultConfig(mode, 1.0, 0.2))
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run_episode(vehicle, bridge, settings)
        records = [json.loads(line) for line in (output/'behavior.jsonl').read_text().splitlines()]
        assert summary['route_completed'], summary
        assert summary['max_abs_e_y_m'] < 0.1
        assert records[-1]['goal_distance_m'] <= 3 and vehicle.plant.s.vx < 0.3
        before = [r for r in records if r['t'] < 1]
        after = [r for r in records if r['t'] >= 1]
        assert all(r['steer_source'] == 'mpc' for r in before)
        expected = 'mpc' if mode == 'none' else 'lqr'
        assert all(r['steer_source'] == expected for r in after)
        assert summary['supervisor']['switch_count'] == (mode != 'none')
        assert not summary['supervisor']['safe_stop']
        if hazard:
            held = [r for r in records if 1 <= r['t'] < 2]
            assert held and all(r['brake'] == 1 and r['throttle'] == 0 for r in held)
        events = [json.loads(line) for line in (output/'supervisor.jsonl').read_text().splitlines()]
        assert len(events) == (mode != 'none')
        print('PASS closed-loop mode=%s hazard=%s complete=%.2fs' % (mode, hazard or 'clear', summary['duration_s']))


if __name__ == '__main__':
    for mode in ('none', 'solver_fail', 'nan', 'stale', 'exception', 'delay'):
        exercise_fault(mode)
    exercise_fault('solver_fail', 'traffic.traffic_light')
    exercise_fault('nan', 'walker.pedestrian.0001')
