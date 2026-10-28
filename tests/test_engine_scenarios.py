import os
import random

import pytest

from engine import scenarios as SC
from engine.config import FIXTURE_SPACE, import_scenario_space_tool, load_yaml

ROUTES = ["bench2drive:26956", "town10_long"]
POOL = ["F-01", "F-02", "F-03", "F-04", "F-05", "F-06", "F-07"]


@pytest.fixture(scope="module")
def space():
    return load_yaml(FIXTURE_SPACE)


def test_space_filling_is_valid(space):
    cands = SC.space_filling(space, ROUTES, POOL, 20, random.Random(0))
    assert len(cands) == 20
    for c in cands:
        assert SC.validate_against_space(space, c, ROUTES) == []
        ok, tags, _ = SC.gate(space, c, ROUTES, None, 3.0)
        assert ok and "G-01_pending_baseline" in tags


def test_gate_rejects_missing_event(space):
    c = SC.space_filling(space, ROUTES, POOL, 1, random.Random(1))[0]
    c["onset_event"] = "curve_apex"
    ok, tags, _ = SC.gate(space, c, ROUTES, {"straight": 10.0, "curve_apex": None, "kappa_at": {}}, 3.0)
    assert not ok


def test_cli_mapping(space):
    c = SC.make_candidate(space, route_id="town10_long", speed=30, fault_id="F-04", params={"magnitude_rad": 0.05, "duration_s": 2.0},
                          onset_event="straight", offset=0.5, source="t")
    args = SC.fault_cli_local(space, c, 25.0)
    assert args[:4] == ["--mpc-fault", "bias", "--fault-start", "25.00"]
    assert "--fault-magnitude" in args and "--max-speed" in args


def test_detect_route_events():
    ticks = []
    for i in range(1200):
        t = i * 0.05
        k = 1 / 30 if 30 <= t < 36 else 0.0
        ticks.append({"t": t, "vx": 8.0, "yaw_rate": 8.0 * k})
    ev = SC.detect_route_events(ticks, 0.05)
    assert 29.5 <= ev["curve_entry"] <= 31 and 35 <= ev["curve_exit"] <= 37 and ev["straight"] is not None


def test_detect_route_events_ignores_stopped_and_launch():
    """BUG-012: a stopped car reads kappa 0 and must not be picked as 'straight'; launch is skipped."""
    ticks = []
    for i in range(1200):
        t = i * 0.05
        if t < 10:
            v = 0.0                      # waiting at a light: yaw_rate 0 -> kappa 0
        elif t < 13:
            v = 8.0 * (t - 10) / 3       # launch
        else:
            v = 8.0
        k = 1 / 30 if 40 <= t < 46 else 0.0
        ticks.append({"t": t, "vx": v, "yaw_rate": v * k})
    ev = SC.detect_route_events(ticks, 0.05)
    assert ev["straight"] is not None and ev["straight"] > 12 and ev["speed_at"]["straight"] >= 0.6 * 8.0 - 1e-6
    assert 39.5 <= ev["curve_entry"] <= 41 and ev["speed_at"]["curve_entry"] >= 4.0


def test_detect_route_events_stop_breaks_segment_and_low_speed_curve_dropped():
    ticks = []
    for i in range(1200):
        t = i * 0.05
        v = 1.5 if 20 <= t < 30 else 8.0  # crawling: kappa from yaw_rate/vx is noisy here
        k = 1 / 20 if 22 <= t < 26 else 0.0
        ticks.append({"t": t, "vx": v, "yaw_rate": v * k})
    ev = SC.detect_route_events(ticks, 0.05)
    assert ev["curve_entry"] is None and ev["n_curves"] == 0
    assert ev["straight"] is not None and not (20 <= ev["straight"] < 30)


@pytest.mark.skipif(not os.environ.get("SIM_EVAL_ROOT"), reason="SIM_EVAL_ROOT not set (sim-eval checkout)")
def test_parity_with_sim_eval_to_cli():
    from engine.config import resolve_scenario_space
    tool = import_scenario_space_tool()
    assert tool is not None
    sp = load_yaml(resolve_scenario_space(allow_fixture=False)[0])
    rng = random.Random(3)
    for c in SC.space_filling(sp, ROUTES, POOL, 30, rng):
        scen = {"fault_id": c["fault"]["fault_id"], "params": c["fault"]["params"], "environment": c["environment"]}
        assert tool.to_cli(sp, scen, 21.5) == SC.fault_cli_local(sp, c, 21.5)
