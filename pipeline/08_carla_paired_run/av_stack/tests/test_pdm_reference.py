"""PDM-Lite route -> av_stack reference, closed loop on the independent plant (no CARLA server).

PDM-Lite hands the steering function its dense route (CARLA frame, 10 points per metre,
starting at the ego's matched point). The test drives a straight -> R=12 m right turn with
MPC, LQR and the supervised switch (MPC fails right before the turn) and checks the path error.
"""
from dataclasses import asdict
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.pdm_agent import route_to_reference, PDMLiteOwnSteerAgent  # noqa: F401  (import chain check)
from integration.behavior_control import update_reference
from integration.episode import build_controller, step_controller
from integration.fault_injection import FaultConfig
from integration.supervised_lateral import SupervisedLateral
from sim import BicyclePlant, PlantParams

RADIUS, STRAIGHT = 12.0, 30.0


def carla_route(ds=0.1):
    """Right turn in the right-handed frame, returned in CARLA's left-handed frame (y flipped)."""
    s = np.arange(0, STRAIGHT + RADIUS * math.pi / 2 + 120, ds)
    a = np.clip((s - STRAIGHT) / RADIUS, 0, math.pi / 2)
    after = np.maximum(s - STRAIGHT - RADIUS * math.pi / 2, 0)
    x = np.where(s < STRAIGHT, s, STRAIGHT + RADIUS * np.sin(a))
    y = np.where(s < STRAIGHT, 0, -RADIUS * (1 - np.cos(a))) - after
    return np.column_stack((x, -y))


def drive(mode, speed=7.0, fault=FaultConfig()):
    params = PlantParams()
    geometry = asdict(params)
    geometry.pop('max_steer_rate_rad_s')
    plant = BicyclePlant(params)
    plant.s.vx = speed
    route = carla_route()
    controller, worst = None, 0.
    for frame in range(int((STRAIGHT + RADIUS * math.pi / 2 + 20) / speed / .05)):
        st = plant.s
        rh = np.column_stack((route[:, 0], -route[:, 1]))
        d = np.hypot(rh[:, 0] - st.x, rh[:, 1] - st.y)
        i = int(np.argmin(d))
        worst = max(worst, float(d[i]))
        behind = min(i, 30)
        reference = route_to_reference(route[i - behind:], st.vx, behind)
        if controller is None:
            controller = (SupervisedLateral(geometry, .05, reference, fault) if mode == 'supervised'
                          else build_controller(mode, geometry, .05, reference))
        if mode == 'supervised':
            controller.set_reference(reference, st.vx)
            result = controller.step(frame * .05, st.x, st.y, st.yaw, st.vx, st.vy, st.r)
        else:
            update_reference(controller, reference, st.vx)
            result = step_controller(mode, controller, frame * .05, st.x, st.y, st.yaw, st.vx, st.vy, st.r)
        plant.step(result.steer, 0., .05)
        plant.s.vx = speed
    return worst, controller


def test_reference_is_right_handed_and_starts_behind_the_ego() -> None:
    reference = route_to_reference(carla_route()[100:], 5.)  # route starts 10 m into the straight
    assert abs(reference.x[0] - 7.) < .1 and abs(reference.y[0]) < 1e-6  # 3 m stub behind it
    assert reference.kappa.min() < -1 / RADIUS * .8  # right turn -> negative curvature (left +)
    assert route_to_reference(carla_route()[:20], 5.) is None  # too short near the route end


def test_mpc_lqr_and_switch_follow_a_pdm_route_turn() -> None:
    results = {}
    for mode in ('mpc', 'lqr'):
        results[mode], _ = drive(mode)
    fault = FaultConfig('solver_fail', (STRAIGHT - 7.) / 7., 500.)  # 1 s before the turn
    results['supervised'], controller = drive('supervised', fault=fault)
    assert controller.active == 'lqr' and controller.switch_count == 1
    for mode, worst in results.items():
        print('PDM route R=%.0f m right turn @7 m/s  %-10s max path error %.3f m' % (RADIUS, mode, worst))
        assert worst < .3, (mode, worst)


if __name__ == '__main__':
    test_reference_is_right_handed_and_starts_behind_the_ego()
    test_mpc_lqr_and_switch_follow_a_pdm_route_turn()
    print('PASS PDM-Lite route reference, MPC/LQR tracking and takeover')
