import conftest  # noqa: F401  (sets sys.path)
import math

import numpy as np

from planner import LaneChange, PlannerConfig, ReferencePlanner, Trajectory
from sim.tracks import demo_route


def test_circle_curvature_and_speed_limit():
    R = 50.0
    th = np.linspace(0, math.pi, 400)
    traj = ReferencePlanner(PlannerConfig(v_max=20.0, a_lat_max=2.0, v_start=20.0)).plan(
        R * np.sin(th), R * (1 - np.cos(th)))
    k = traj.kappa[20:-20]
    assert abs(k.mean() - 1 / R) < 1e-4 and k.std() < 1e-3
    assert np.all(traj.v**2 * np.abs(traj.kappa) <= 2.0 + 1e-6)
    assert traj.meta["feasible"]


def test_speed_profile_limits_and_time():
    cfg = PlannerConfig(v_max=15.0, a_accel=1.5, a_decel=2.5, v_start=0.0)
    traj = ReferencePlanner(cfg).plan(*demo_route())
    dv2 = np.diff(traj.v**2) / np.diff(traj.s)
    assert dv2.max() <= 2 * cfg.a_accel + 1e-6 and dv2.min() >= -2 * cfg.a_decel - 1e-6
    assert traj.v[0] == 0.0 and np.all(np.diff(traj.t) > 0)
    assert traj.meta["max_lat_acc"] <= cfg.friction_margin * cfg.mu * 9.81 + 1e-6


def test_lane_change_reaches_offset_and_keeps_lane_reference():
    xs = np.arange(0.0, 300.0, 1.0)
    traj = ReferencePlanner(PlannerConfig()).plan(xs, 0 * xs, (LaneChange(100.0, 60.0, 3.5),))
    assert abs(np.interp(250.0, traj.x, traj.y) - 3.5) < 1e-3
    assert abs(np.interp(50.0, traj.x, traj.y)) < 1e-6
    assert np.allclose(traj.lane_y, 0.0)


def test_deterministic_and_roundtrip(tmp_path=None):
    a = ReferencePlanner().plan(*demo_route())
    b = ReferencePlanner().plan(*demo_route())
    assert np.array_equal(a.x, b.x) and np.array_equal(a.v, b.v)
    import os
    import tempfile
    d = tempfile.mkdtemp()
    a.save(os.path.join(d, "t.json"))
    c = Trajectory.load(os.path.join(d, "t.json"))
    assert np.allclose(a.kappa, c.kappa) and c.meta["feasible"] == a.meta["feasible"]


if __name__ == "__main__":
    for fn in (test_circle_curvature_and_speed_limit, test_speed_profile_limits_and_time,
               test_lane_change_reaches_offset_and_keeps_lane_reference, test_deterministic_and_roundtrip):
        fn()
        print("PASS", fn.__name__)
