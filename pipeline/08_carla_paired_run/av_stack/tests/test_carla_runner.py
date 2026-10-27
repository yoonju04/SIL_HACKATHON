"""Drive run_carla.py against the fake, left-handed CARLA module.

If a frame conversion had the wrong sign the car would leave the route within
seconds; the last test proves the check is sensitive to that.
"""
import conftest  # noqa: F401  (sets sys.path)
import sys
import tempfile

import fake_carla

sys.modules["carla"] = fake_carla
import integration.carla_io as carla_io  # noqa: E402
import run_carla  # noqa: E402


def _run(*extra):
    with tempfile.TemporaryDirectory() as d:
        return run_carla.main(["--duration", "45", "--out", d, *extra])


def test_mpc_tracks_route_in_fake_carla():
    s = _run("--controller", "mpc")
    assert s["route_completed"] and s["max_abs_e_y_m"] < 0.2 and s["not_ok_steps"] == 0, s


def test_lqr_tracks_route_in_fake_carla():
    s = _run("--controller", "lqr")
    assert s["route_completed"] and s["max_abs_e_y_m"] < 0.2, s


def test_lane_change_in_fake_carla():
    s = _run("--controller", "mpc", "--lane-change", "100:50:-3.5")
    assert s["max_abs_e_y_m"] < 0.2 and 3.0 < s["max_abs_e_y_lane_m"] < 3.8, s


def test_wrong_steer_sign_is_detected():
    orig = carla_io.to_carla_steer
    run_carla.to_carla_steer = lambda d, m: -orig(d, m)
    try:
        s = _run("--controller", "lqr", "--duration", "20")
    finally:
        run_carla.to_carla_steer = orig
    assert s["max_abs_e_y_m"] > 1.0, s


if __name__ == "__main__":
    for fn in (test_mpc_tracks_route_in_fake_carla, test_lqr_tracks_route_in_fake_carla,
               test_lane_change_in_fake_carla, test_wrong_steer_sign_is_detected):
        fn()
        print("PASS", fn.__name__)
