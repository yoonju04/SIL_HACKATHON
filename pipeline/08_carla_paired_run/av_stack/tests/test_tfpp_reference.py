from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.tfpp_reference import checkpoints_to_reference


def test_sensor_plan_is_anchored_at_capture_pose_with_right_handed_coordinates() -> None:
    points = np.array([[2.5, 1], [4, 2], [6, 3], [9, 4]], dtype=float)
    reference = checkpoints_to_reference(points, (10, 20, math.pi/2), 4, 3)
    assert not np.allclose([reference.x[0], reference.y[0]], [10, -20])
    assert np.allclose([reference.x[-1], reference.y[-1]], [6, -29])
    assert np.isfinite(reference.kappa).all()
    assert reference.v.max() <= 4 + 1e-6
    assert reference.meta['source'] == 'sensor model prediction'


def test_zero_speed_is_preserved_without_fabricating_timestamps() -> None:
    points = np.array([[2.5, 0], [4, 0], [6, 0], [9, 0]], dtype=float)
    reference = checkpoints_to_reference(points, (0, 0, 0), 0, 0)
    assert np.all(reference.v == 0)
    assert np.allclose([reference.x[-1], reference.y[-1]], [9, 0])


if __name__ == '__main__':
    test_sensor_plan_is_anchored_at_capture_pose_with_right_handed_coordinates()
    test_zero_speed_is_preserved_without_fabricating_timestamps()
    print('PASS TFPP coordinate, capture-pose and speed contracts')
