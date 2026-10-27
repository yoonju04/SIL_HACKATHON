from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.behavior_control import BehaviorSpeedControl
from integration.tfpp_reference import checkpoints_to_reference
from mpc_controller.reference import ReferencePath


def test_speed_reduction_brakes_despite_accumulated_acceleration() -> None:
    control = BehaviorSpeedControl(.05)
    for _ in range(100):
        control.command(2., 5., False)
    pedals = control.command(4., 3.5, False)
    assert pedals.throttle == 0
    assert pedals.brake > 0


def test_predicted_offset_survives_reference_conversion() -> None:
    points = np.array([[2.5, 1.], [4., 1.], [6., 1.], [9., 1.]])
    trajectory = checkpoints_to_reference(points, (0., 0., 0.), 4., 3.)
    reference = ReferencePath(trajectory.x, trajectory.y, trajectory.yaw,
                              trajectory.kappa, trajectory.s, trajectory.v)
    _, lateral_error, _, _ = reference.project(0., 0.)
    assert abs(lateral_error-1.) < 1e-6
    assert np.allclose(trajectory.y, -1.)


def test_curved_prediction_preserves_curvature_at_vehicle() -> None:
    distance = np.linspace(2.5, 11.5, 10)
    radius = 12.
    points = np.column_stack((radius*np.sin(distance/radius), radius*(1-np.cos(distance/radius))))
    trajectory = checkpoints_to_reference(points, (0., 0., 0.), 4., 4.)
    reference = ReferencePath(trajectory.x, trajectory.y, trajectory.yaw,
                              trajectory.kappa, trajectory.s, trajectory.v)
    _, lateral_error, heading, curvature = reference.project(0., 0.)
    assert abs(lateral_error) < .1
    assert abs(heading) < .07
    assert abs(curvature+1/radius) < .03


def test_straight_model_speed_is_not_reset_to_spawn_acceleration_profile() -> None:
    points = np.column_stack((np.linspace(2.5, 11.5, 10), np.zeros(10)))
    reference = checkpoints_to_reference(points, (0., 0., 0.), 10., 2.)
    assert np.allclose(reference.v, 10.)


if __name__ == '__main__':
    for check in (test_speed_reduction_brakes_despite_accumulated_acceleration,
                  test_predicted_offset_survives_reference_conversion,
                  test_curved_prediction_preserves_curvature_at_vehicle,
                  test_straight_model_speed_is_not_reset_to_spawn_acceleration_profile):
        check()
        print('PASS', check.__name__)
