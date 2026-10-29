"""integration3 #1-#3: saturate magnitude/ramp, tire friction, control-fault handover axes (catalog -> CLI -> mock sim)."""
import random

import pytest

from engine import fault_args as FA
from engine import scenarios as SC
from engine.config import SPACE_REL, REPO_ROOT, load_yaml
from engine.fixtures import synthetic_sim as SIM

ROUTES = ["bench2drive:26956", "town10_long"]
POOL = ["F-01", "F-02", "F-03", "F-04", "F-05", "F-06", "F-07"]


@pytest.fixture(scope="module")
def space():
    return load_yaml(REPO_ROOT / SPACE_REL)


def _sat(space, **params):
    p = {"duration_s": 0.5, "side": "positive", **params}
    return SC.make_candidate(space, route_id="town10_long", speed=30, fault_id="F-06", params=p,
                             onset_event="curve_entry", offset=0.0, source="t")


def test_saturate_catalog_has_magnitude_and_ramp(space):
    f = SC.faults_by_id(space)["F-06"]["params"]
    assert f["magnitude"]["range"] == [0.2, 1.0] and f["magnitude"]["default"] == 1.0
    assert f["ramp_s"]["range"] == [0.0, 0.3] and f["ramp_s"]["default"] == 0.0


def test_saturate_cli_flags_and_back_compat(space):
    c = _sat(space, magnitude=0.45, ramp_s=0.12)
    scen = {"fault_id": "F-06", "params": c["fault"]["params"], "environment": c["environment"]}
    args = FA.to_cli(space, scen, 20.0)
    assert args[args.index("--saturate-level") + 1] == "0.4500"
    assert args[args.index("--fault-ramp") + 1] == "0.1200"
    assert args[args.index("--fault-magnitude") + 1] == "1.0000"       # side stays the sign
    assert args == SC.fault_cli_local(space, c, 20.0)                   # engine fallback is the same mapping
    old = {"fault_id": "F-06", "params": {"duration_s": 0.5, "side": "negative"}, "environment": {"max_speed_kmh": 30}}
    old_args = FA.to_cli(space, old, 20.0)                              # plans from before integration3
    assert "--saturate-level" not in old_args and "--fault-ramp" not in old_args


def test_to_cli_parity_engine_fallback_on_real_catalog(space):
    rng = random.Random(3)
    for c in SC.space_filling(space, ROUTES, POOL, 40, rng):
        scen = {"fault_id": c["fault"]["fault_id"], "params": c["fault"]["params"], "environment": c["environment"]}
        assert FA.to_cli(space, scen, 21.5) == SC.fault_cli_local(space, c, 21.5)


def test_fault_dict_for_sim_carries_level_and_ramp(space):
    d = SC.fault_dict_for_sim(space, _sat(space, magnitude=0.3, ramp_s=0.2), 20.0)
    assert d["level"] == pytest.approx(0.3) and d["ramp_s"] == pytest.approx(0.2)
    d0 = SC.fault_dict_for_sim(space, _sat(space), 20.0)
    assert d0["level"] == 1.0 and d0["ramp_s"] == 0.0


def _peak_offset(level, ramp=0.0, duration=0.4, route="town10_long", speed=30.0):
    fault = dict(mode="saturate", start=20.0, duration=duration, magnitude=1.0, level=level, ramp_s=ramp)
    rows = SIM.simulate(route, speed, fault, seed=17)["rows"]
    return max(abs(r["e_y"]) for r in rows if r["t"] >= 20.0)


def test_synthetic_sim_responds_to_saturate_level_and_ramp():
    weak, strong = _peak_offset(0.25), _peak_offset(1.0)
    assert weak < strong
    assert _peak_offset(1.0, ramp=0.3) < strong
