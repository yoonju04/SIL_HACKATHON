from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.tfpp_lidar_guard import lidar_guard


def test_front_vehicle_requests_braking_before_contact() -> None:
    points = np.tile([5., .1, .8], (16, 1))
    decision = lidar_guard(points, 4.)
    assert decision.brake
    assert decision.speed_cap == 0
    assert decision.clearance_m == 2.5


def test_ground_and_adjacent_lane_do_not_force_braking() -> None:
    ground = np.tile([3., 0., -.5], (20, 1))
    side = np.tile([3., 3., .8], (20, 1))
    decision = lidar_guard(np.vstack((ground, side)), 4.)
    assert not decision.brake
    assert decision.speed_cap is None


def test_distant_front_vehicle_limits_speed_without_minimum_speed_floor() -> None:
    points = np.tile([12., 0., .8], (16, 1))
    decision = lidar_guard(points, 2.)
    assert not decision.brake
    assert decision.speed_cap is not None
    assert 0 < decision.speed_cap < 10


if __name__ == '__main__':
    test_front_vehicle_requests_braking_before_contact()
    test_ground_and_adjacent_lane_do_not_force_braking()
    test_distant_front_vehicle_limits_speed_without_minimum_speed_floor()
    print('PASS LiDAR stopping distance, adjacent lane and ground rejection')
