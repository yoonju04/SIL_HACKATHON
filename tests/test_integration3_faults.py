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


# ----------------------------------------------------------------------------- #2 tire friction
def test_friction_axis_is_wired_and_sampled(space):
    fax = SC.friction_axis(space)
    assert fax and fax["cli"] == "--tire-friction-scale" and fax["range"] == [0.3, 1.0]
    cands = SC.space_filling(space, ROUTES, POOL, 30, random.Random(0))
    fr = [c["environment"][SC.FRICTION] for c in cands]
    assert min(fr) < 0.5 and max(fr) > 0.8
    for c in cands:
        assert SC.validate_against_space(space, c, ROUTES) == []
    bad = dict(cands[0], environment=dict(cands[0]["environment"], tire_friction_scale=0.1))
    assert any("tire_friction_scale" in p for p in SC.validate_against_space(space, bad, ROUTES))
    x, names = SC.numeric_features(cands[0], space, ROUTES, POOL)
    assert "tire_friction_u" in names


def test_friction_cli_and_runner_command(space, tmp_path):
    from engine.carla_runner import CarlaRunner, baseline_key, key_hash
    from engine.config import load_stage_configs
    c = SC.make_candidate(space, route_id="bench2drive:26956", speed=30, fault_id="F-04",
                          params={"magnitude_rad": 0.05}, onset_event="curve_entry", offset=0.2, source="t", friction=0.55)
    scen = {"fault_id": "F-04", "params": c["fault"]["params"], "environment": c["environment"]}
    args = FA.to_cli(space, scen, 30.0)
    assert args[args.index("--tire-friction-scale") + 1] == "0.55"
    r = CarlaRunner(tmp_path, load_stage_configs()["paired_runs"]["carla"]["routes"], 17, space, 15.0,
                    av_stack=tmp_path, python_exe="python")
    base = r.command(c, tmp_path / "b", None)                    # the baseline drives on the same tyres
    assert base[base.index("--tire-friction-scale") + 1] == "0.55"
    f = r.command(c, tmp_path / "f", r.fault_args(c, 31.0))
    assert f.count("--tire-friction-scale") == 1
    route = {"town": "Town05"}
    assert key_hash(baseline_key(c, route, 17)) != key_hash(baseline_key(dict(c, environment={"max_speed_kmh": 30}), route, 17))
    nominal = SC.make_candidate(space, route_id="bench2drive:26956", speed=30, fault_id="F-04",
                                params={"magnitude_rad": 0.05}, onset_event="curve_entry", offset=0.2, source="t")
    assert "--tire-friction-scale" not in r.command(nominal, tmp_path / "n", None)


class _Wheel:
    def __init__(self, f):
        self.tire_friction = f


class _PC:
    def __init__(self, wheels):
        self.wheels = wheels


class _Vehicle:
    def __init__(self):
        self.pc = _PC([_Wheel(3.5) for _ in range(4)])

    def get_physics_control(self):
        return _PC([_Wheel(w.tire_friction) for w in self.pc.wheels])   # CARLA returns copies

    def apply_physics_control(self, pc):
        self.pc = _PC([_Wheel(w.tire_friction) for w in pc.wheels])


def test_carla_io_scale_tire_friction_all_wheels():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "pipeline/08_carla_paired_run/av_stack"))
    from integration.carla_io import scale_tire_friction
    v = _Vehicle()
    info = scale_tire_friction(v, 0.5)
    assert info["applied"] and info["before"] == [3.5] * 4 and info["after"] == [1.75] * 4
    assert [w.tire_friction for w in v.pc.wheels] == [1.75] * 4

    class Broken:
        def get_physics_control(self):
            raise RuntimeError("no physics")
    bad = scale_tire_friction(Broken(), 0.5)
    assert bad["applied"] is False and "RuntimeError" in bad["error"]


def test_p02_threshold_scales_with_friction(tmp_path):
    from engine import monitor as MON
    from engine.config import load_stage_configs
    th = MON.thresholds_from_config(load_stage_configs())
    assert th["lat_acc_threshold_mps2"] == pytest.approx(4.41)
    assert MON.scale_thresholds_for_friction(th, 1.0)["lat_acc_threshold_mps2"] == pytest.approx(4.41)
    assert MON.scale_thresholds_for_friction(th, 0.5)["lat_acc_threshold_mps2"] == pytest.approx(2.205)

    class R:
        config = {"tire_friction": {"scale": 0.4, "applied": False}}
    assert MON.run_friction_scale(R()) == 1.0                  # not applied -> real tyres
    R.config = {"tire_friction": {"scale": 0.4, "applied": True}}
    assert MON.run_friction_scale(R()) == 0.4
    R.config = {}
    assert MON.run_friction_scale(R(), {"environment": {"tire_friction_scale": 0.7}}) == 0.7


def test_monitor_records_friction_threshold_on_mock_run(space, tmp_path):
    from engine import monitor as MON
    fault = SC.fault_dict_for_sim(space, _sat(space, magnitude=0.6), 20.0)
    for fr in (1.0, 0.6):
        d = SIM.write_run_folder(tmp_path / f"f{fr}", route="town10_long", max_speed_kmh=30, fault=fault,
                                 record_front=False, tire_friction_scale=fr)
        b = SIM.write_run_folder(tmp_path / f"b{fr}", route="town10_long", max_speed_kmh=30, fault=None,
                                 record_front=False, tire_friction_scale=fr)
        rec = MON.evaluate(d, baseline_dir=b)
        assert rec["metrics"]["tire_friction_scale"] == fr
        assert rec["metrics"]["p02_lat_acc_threshold_mps2"] == pytest.approx(4.41 * fr)


def test_synthetic_sim_responds_to_friction():
    def peak(fr):
        rows = SIM.simulate("bench2drive:28048", 40.0, None, seed=17, tire_friction_scale=fr)["rows"]
        return max(abs(r["e_y"]) for r in rows)
    assert peak(0.3) > peak(1.0) + 0.2
