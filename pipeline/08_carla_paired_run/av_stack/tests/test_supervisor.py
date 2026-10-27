from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.fault_injection import FaultConfig, MPCFaultInjector
from integration.supervised_lateral import MPCSupervisor, SupervisedLateral
from mpc_controller import MPCOutput
from planner import PlannerConfig, ReferencePlanner
from sim.plant import PlantParams
from sim.plant import BicyclePlant
from sim.tracks import arc_route


def output() -> MPCOutput:
    return MPCOutput(0.1, 0.1, True, 'solved', 0.001, 10, 1, 1.0,
                     np.zeros(4), 5.0, 0.0)


def test_healthy_mpc_is_not_rejected() -> None:
    supervisor = MPCSupervisor(0.5)
    assert supervisor.inspect(output(), 1.0, 2.0) is None


def test_invalid_outputs_are_rejected() -> None:
    cases = [(replace(output(), ok=False), 'solver_failure'),
             (replace(output(), steer=None), 'missing_steer'),
             (replace(output(), steer=math.nan), 'nonfinite_output'),
             (replace(output(), steer=0.7), 'steer_out_of_range'),
             (replace(output(), state=np.array([math.nan, 0, 0, 0])), 'nonfinite_output'),
             (replace(output(), stamp_s=0.0), 'stale_output')]
    for candidate, expected in cases:
        assert MPCSupervisor(0.5).inspect(candidate, 1.0, 2.0) == expected


def test_wall_deadline_and_repeated_sequence_are_detected() -> None:
    supervisor = MPCSupervisor(0.5)
    assert supervisor.inspect(output(), 1.0, 51.0) == 'deadline_miss'
    supervisor = MPCSupervisor(0.5)
    assert supervisor.inspect(output(), 1.0, 2.0) is None
    assert supervisor.inspect(replace(output(), stamp_s=1.05), 1.05, 2.0) == 'stale_sequence'


def test_fault_is_temporary_and_off_by_default() -> None:
    original = output()
    assert MPCFaultInjector(FaultConfig()).apply(original, 1.0) is original
    injector = MPCFaultInjector(FaultConfig('nan', 1.0, 0.2))
    assert injector.apply(original, 0.95) is original
    assert math.isnan(injector.apply(original, 1.0).steer)
    assert injector.apply(original, 1.2) is original


def test_intermittent_bias_alternates_on_and_off() -> None:
    injector = MPCFaultInjector(FaultConfig('intermittent_bias', 1.0, 10.0, 0.08, on_s=0.2, off_s=0.1))
    steers = [injector.apply(output(), 1.0 + k * 0.05).steer for k in range(8)]
    assert [round(steer - 0.1, 3) for steer in steers] == [0.08] * 4 + [0.0] * 2 + [0.08] * 2


def test_saturate_holds_exactly_the_limit_and_passes_the_range_check() -> None:
    for side in (1.0, -1.0):
        injector = MPCFaultInjector(FaultConfig('saturate', 1.0, 1.0, side), max_steer=0.5)
        faulty = injector.apply(output(), 1.0)
        assert faulty.steer == 0.5 * side
        assert MPCSupervisor(0.5).inspect(faulty, 1.0, 2.0) is None


def test_output_delay_sends_an_old_command_under_a_fresh_header() -> None:
    injector = MPCFaultInjector(FaultConfig('output_delay', 1.0, 1.0, 2))
    supervisor = MPCSupervisor(0.5)
    for k in range(6):
        t = 0.8 + k * 0.05
        fresh = replace(output(), steer=0.01 * k, seq=k + 1, stamp_s=t)
        sent = injector.apply(fresh, t)
        assert sent.seq == fresh.seq and sent.stamp_s == t
        assert sent.steer == (fresh.steer if t < 1.0 - 1e-9 else round(0.01 * (k - 2), 10))
        assert supervisor.inspect(sent, t, 2.0) is None


def test_real_mpc_to_lqr_switch_latches_after_temporary_failure() -> None:
    params = asdict(PlantParams())
    params.pop('max_steer_rate_rad_s')
    trajectory = ReferencePlanner(PlannerConfig(v_max=5.0, v_start=5.0)).plan(
        [0.0, 30.0, 60.0], [0.0, 0.0, 0.0])
    controller = SupervisedLateral(params, 0.05, trajectory, FaultConfig('solver_fail', 1.0, 0.2))
    for index in range(30):
        t = index * 0.05
        result = controller.step(t, t * 5, 0.0, 0.0, 5.0, 0.0, 0.0)
        assert result.ok and math.isfinite(result.steer)
        assert controller.active == ('mpc' if t < 1.0 else 'lqr')
    assert controller.switch_count == 1
    assert controller.switched_at == 1.0
    assert controller.reason == 'solver_failure'


def test_tracking_divergence_requires_persistence() -> None:
    supervisor = MPCSupervisor(0.5)
    candidate = replace(output(), state=np.array([1.1, 0, 0, 0]))
    assert supervisor.inspect(candidate, 1.0, 2.0) is None
    candidate = replace(candidate, seq=2, stamp_s=1.25)
    assert supervisor.inspect(candidate, 1.25, 2.0) == 'tracking_divergence'


def test_lqr_nonfinite_output_requires_safe_stop() -> None:
    params = asdict(PlantParams())
    params.pop('max_steer_rate_rad_s')
    trajectory = ReferencePlanner().plan([0, 30, 60], [0, 0, 0])
    controller = SupervisedLateral(params, 0.05, trajectory, FaultConfig('nan', 0, 1))
    original = controller.lqr.compute
    controller.lqr.compute = lambda *args: replace(original(*args), steer=math.nan)
    result = controller.step(0, 0, 0, 0, 5, 0, 0)
    assert controller.stop_required and controller.active == 'safe_stop'
    assert math.isfinite(result.steer) and not result.ok


def test_takeover_tracks_both_turn_directions_with_nonlinear_physics() -> None:
    for direction in (-1, 1):
        params = asdict(PlantParams())
        params.pop('max_steer_rate_rad_s')
        route = arc_route(radius=35, straight=15, exit_len=20, direction=direction)
        trajectory = ReferencePlanner(PlannerConfig(v_max=5, v_start=5)).plan(*route)
        controller = SupervisedLateral(params, 0.05, trajectory, FaultConfig('nan', 5, 0.2))
        plant = BicyclePlant()
        plant.s.x, plant.s.y, plant.s.yaw = trajectory.x[0], trajectory.y[0], trajectory.yaw[0]
        plant.s.vx = 5
        errors = []
        for index in range(500):
            state = plant.s
            result = controller.step(index*0.05, state.x, state.y, state.yaw, state.vx, state.vy, state.r)
            errors.append(abs(result.e_y))
            plant.step(result.steer, 0, 0.05)
            if result.s >= trajectory.length - 3:
                break
        assert result.s >= trajectory.length - 3
        assert max(errors) < 0.5, max(errors)
        assert controller.switched_at == 5.0 and controller.switch_count == 1
        assert abs(trajectory.at(25)['kappa']) > 0.02
        print('PASS curved takeover direction=%d max_error=%.3fm' % (direction, max(errors)))


if __name__ == '__main__':
    for name, check in list(globals().items()):
        if name.startswith('test_') and callable(check):
            check()
            print('PASS ' + name)
