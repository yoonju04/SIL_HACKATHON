"""engine/rare_seeds.py + run.Loop quiet_seeds mode: rare failure-type seeds in the quiet sub-threshold fault region."""
import copy
import math
import random

import pytest

from engine import failure_types as FT
from engine import rare_seeds as RS
from engine import scenarios as SC
from engine.config import load_stage_configs, load_yaml, resolve_scenario_space


def _ctx():
    cfg = load_stage_configs()
    space = load_yaml(resolve_scenario_space(False)[0])
    routes = [r["route_id"] for r in cfg["paired_runs"]["carla"]["routes"] if r.get("enabled", True)]
    ft = FT.load_config()
    return cfg, space, routes, cfg["orchestrator"]["fault_pool"], ft, ft["acquisition"]["seeds"]


def test_bands_are_derived_from_the_supervisor_code_values():
    cfg, space, _, _, _, scfg = _ctx()
    mw = float(cfg["safety_spec"]["evaluation_window"]["min_length_s"])
    b = RS.derived_bands(space, scfg, supervisor="tight", eval_window_min_s=mw)
    tick = math.radians(90.0) * 0.05
    assert b["rate_tick_rad"] == pytest.approx(tick, abs=1e-5)
    lo, hi = b["bias_magnitude_rad"]
    assert lo == 0.06 and hi < tick
    assert b["drift_rate_rad_s"][1] == pytest.approx(tick / mw, abs=1e-4)          # 0.0157 rad/s
    hot = copy.deepcopy(scfg)
    hot["bias_sub_threshold"]["magnitude_rad"] = [0.06, 0.09]                       # above the tick -> clipped
    assert RS.derived_bands(space, hot)["bias_magnitude_rad"][1] < tick
    leg = RS.derived_bands(space, scfg, supervisor="legacy")
    assert leg["rate_tick_rad"] is None and any("legacy" in n for n in leg["notes"])


def test_generated_seeds_land_in_their_region():
    cfg, space, routes, pool, ft, scfg = _ctx()
    bands = RS.derived_bands(space, scfg)
    seeds = []
    for s in range(6):
        seeds += RS.generate(space, routes, pool, scfg, random.Random(s), bands=bands, rare_runs=[], round_index=1,
                             speed_step=5.0)
    by = {}
    for c in seeds:
        by.setdefault(c["rare_seed_source"], []).append(c)
        assert c["source"] == "rare_seed"
        assert not SC.validate_against_space(space, c, routes), c
        mode, p = c["fault"]["mode"], c["fault"]["params"]
        assert mode in RS.QUIET_MODES
        if mode in ("bias", "intermittent_bias"):
            assert bands["bias_magnitude_rad"][0] - 1e-9 <= p["magnitude_rad"] <= bands["bias_magnitude_rad"][1] + 1e-9
        else:
            assert bands["drift_rate_rad_s"][0] - 1e-9 <= p["rate_rad_s"] <= bands["drift_rate_rad_s"][1] + 1e-9
    assert set(by) == {"bias_sub_threshold", "slow_drift", "curve_entry", "low_friction"}   # no rare runs given
    for c in by["bias_sub_threshold"]:
        assert c["environment"]["max_speed_kmh"] >= 25.0 - 1e-9
    for c in by["curve_entry"]:
        assert c["onset_event"] == "curve_entry"
        q = round(c["environment"]["max_speed_kmh"] / 5) * 5.0
        assert "curve_entry" not in SC.onset_exclusions(space, c["route_id"], q)
        assert -0.5 - 1e-9 <= c["onset_offset_s"] <= 0.5 + 1e-9
    for c in by["low_friction"]:
        assert 0.3 - 1e-9 <= c["environment"][SC.FRICTION] <= 0.6 + 1e-9


def test_low_friction_skipped_without_friction_axis():
    _, space, routes, pool, _, scfg = _ctx()
    sp = copy.deepcopy(space)
    sp["environment"].pop(SC.FRICTION)
    seeds = RS.generate(sp, routes, pool, scfg, random.Random(0), bands=RS.derived_bands(sp, scfg), rare_runs=[],
                        round_index=1)
    assert seeds and "low_friction" not in {c["rare_seed_source"] for c in seeds}


def _rec(route, mode="bias", params=None, **kw):
    r = {"sim_status": "ok", "switch_time": 10.0, "supervisor_profile": "tight", "run_id": kw.pop("run_id", "R"),
         "recovery_outcome": "safe_recovery", "lane_recovery_robustness": 0.5,
         "parameters": {"route_id": route, "environment": {"max_speed_kmh": 30.0, "tire_friction_scale": 0.7},
                        "fault": {"fault_id": {"bias": "F-04", "drift": "F-05"}[mode], "mode": mode,
                                  "params": params or {"magnitude_rad": 0.07}},
                        "onset_event": "curve_entry", "onset_offset_s": 0.2},
         "metrics": {"lat_jerk_excess_mps3": 1.0, "lat_acc_excess_mps2": 0.1}, "recovery_labels": {}}
    r.update(kw)
    return r


def test_known_rare_runs_and_neighbours():
    _, space, routes, pool, ft, scfg = _ctx()
    rt = "town10_long"
    recs = [_rec(rt, run_id="dep", recovery_outcome="recovered_after_departure"),
            _rec(rt, run_id="sr4", sr_checks={"sr": {"SR4": {"pass": False}}}),          # footprint only
            _rec(rt, run_id="jerk", metrics={"lat_jerk_excess_mps3": 30.0, "lat_acc_excess_mps2": 0.1}),  # not rare
            _rec(rt, run_id="legacy", recovery_outcome="unsettled", supervisor_profile="legacy"),
            _rec(rt, run_id="noswitch", switch_time=None, recovery_outcome="no_switch")]
    rare = RS.known_rare_runs(recs, scfg, ft, supervisor="tight")
    assert {r["run_id"] for r in rare} == {"dep", "sr4"}
    off = copy.deepcopy(scfg)
    off["rare_neighbours"]["include_sr4_footprint"] = False
    assert {r["run_id"] for r in RS.known_rare_runs(recs, off, ft, supervisor="tight")} == {"dep"}
    only = dict(scfg, sources=["rare_neighbours"])
    seeds = RS.generate(space, routes, pool, only, random.Random(0), bands=RS.derived_bands(space, scfg),
                        rare_runs=rare, round_index=2)
    assert len(seeds) == 2
    for c in seeds:
        assert c["rare_seed_source"] == "rare_neighbours" and c["rare_seed_of"]["run_id"] in {"dep", "sr4"}
        assert (c["route_id"], c["fault"]["fault_id"], c["onset_event"]) == (rt, "F-04", "curve_entry")
        assert c["scenario_id"] == SC.cand_id(c)


def test_saved_history_has_rare_runs():
    _, space, routes, pool, ft, scfg = _ctx()
    from engine.config import REPO_ROOT
    recs = RS.load_records(scfg["rare_neighbours"]["history"], REPO_ROOT)
    if not recs:
        pytest.skip("result_8..12 run records not in this checkout")
    rare = RS.known_rare_runs(recs, scfg, ft, supervisor="tight")
    assert len(rare) >= 6                      # 3 departures + 3 unsettled at least (more with SR4 / lat_acc)
    assert all(r["_rare_types"] for r in rare)


def test_loop_quiet_mode_picks_seeds_in_the_region(tmp_path):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=8, batch_size=6,
                rounds=2, quiet=True, rare_type_picks=2)
    assert loop.ft_cfg["acquisition"]["mode"] == "quiet_seeds"
    loop.run()
    picks = [p for e in loop.stats["rare_type_picks"] for p in e["picked"]]
    assert picks and all(e["mode"] == "quiet_seeds" for e in loop.stats["rare_type_picks"])
    bands = loop.rare_seed_bands()
    for e in loop.stats["rare_type_picks"]:
        srcs = [p["seed_source"] for p in e["picked"]]
        assert len(srcs) == len(set(srcs)) <= 2                    # max_picks_per_source 1
    for p in picks:
        mode, par = p["fault"]["mode"], p["fault"]["params"]
        if p["seed_source"] in ("bias_sub_threshold", "slow_drift", "curve_entry", "low_friction"):
            if mode in ("bias", "intermittent_bias"):
                assert par["magnitude_rad"] < bands["rate_tick_rad"]
            else:
                assert mode == "drift" and par["rate_rad_s"] <= bands["drift_rate_rad_s"][1] + 1e-9
        if p["seed_source"] == "curve_entry":
            assert p["onset_event"] == "curve_entry"
    ran = {r["scenario_id"] for r in loop.records}
    assert any(p["scenario_id"] in ran for p in picks)            # picks survive 07 diversity and run


def test_loop_classifier_mode_still_selectable(tmp_path):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=8, batch_size=4,
                rounds=1, quiet=True, rare_type_picks=1, rare_type_mode="classifier")
    assert loop.rare_seed_candidates(1) == []                    # no seeds in classifier mode
    loop.run()
    assert all(e["mode"] == "classifier" for e in loop.stats.get("rare_type_picks", []))
