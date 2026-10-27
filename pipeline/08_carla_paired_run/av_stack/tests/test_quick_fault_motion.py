from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.fault_injection import FaultConfig
from integration.supervised_lateral import SupervisedLateral
from integration.unprotected_lateral import UnprotectedMPC
from planner import PlannerConfig, ReferencePlanner
from sim.plant import BicyclePlant, PlantParams
from sim.tracks import arc_route


def test_fault_at_25_seconds_occurs_moving_on_a_curve() -> None:
    geometry = asdict(PlantParams())
    geometry.pop('max_steer_rate_rad_s')
    trajectory = ReferencePlanner(PlannerConfig(v_max=5, v_start=5)).plan(
        *arc_route(radius=35, straight=110, exit_len=70))
    for kind in (SupervisedLateral, UnprotectedMPC):
        control = kind(geometry, .05, trajectory, FaultConfig('solver_fail', 25, 60))
        plant = BicyclePlant()
        maximum_error = 0.0
        for index in range(800):
            state = plant.s
            result = control.step(index*.05, state.x, state.y, state.yaw, state.vx, state.vy, state.r)
            if index == 500:
                assert state.vx > 4
                assert abs(result.kappa) > .01
            maximum_error = max(maximum_error, abs(result.e_y))
            plant.step(result.steer, max(-4, min(2, 5-state.vx)), .05)
        if kind is SupervisedLateral:
            assert maximum_error < .5, maximum_error
            assert control.active == 'lqr'
        else:
            assert maximum_error > 2, maximum_error
            assert control.active == 'none'
        print('PASS %s fault=25s moving curve max_error=%.3fm' % (kind.__name__, maximum_error))


if __name__ == '__main__':
    test_fault_at_25_seconds_occurs_moving_on_a_curve()
