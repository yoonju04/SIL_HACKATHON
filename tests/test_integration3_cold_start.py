"""integration3 #8c: cold start rule (orchestrator.yaml cold_start.rule), space filling fault cycling, round-0 refill."""
import copy
import random
from collections import Counter

from engine import scenarios as SC
from engine.config import load_stage_configs, load_yaml, resolve_scenario_space
from run import Loop


def _space():
    path, _ = resolve_scenario_space(False)
    return load_yaml(path)


def _ctx():
    cfg = load_stage_configs()
    space = _space()
    routes = [r["route_id"] for r in cfg["paired_runs"]["carla"]["routes"] if r.get("enabled", True)]
    pool = cfg["orchestrator"]["fault_pool"]
    controls = {f["id"] for f in space["faults"] if f.get("group") == "control"}
    return cfg, space, routes, pool, controls


def test_space_filling_cycles_every_fault_type():
    _, space, routes, pool, controls = _ctx()
    for seed in range(5):
        c = SC.space_filling(space, routes, pool, len(pool), random.Random(seed))
        assert sorted(x["fault"]["fault_id"] for x in c) == sorted(pool)
        c4 = SC.space_filling(space, routes, pool, 4, random.Random(seed))
        assert len({x["fault"]["fault_id"] for x in c4}) == 4


def test_space_filling_respects_control_cap():
    _, space, routes, pool, controls = _ctx()
    risk = [f for f in pool if f not in controls]
    for seed in range(5):
        c = SC.space_filling(space, routes, pool, 12, random.Random(seed), max_controls=1)
        ids = [x["fault"]["fault_id"] for x in c]
        assert sum(i in controls for i in ids) == 1
        assert set(risk) <= set(ids)
        c0 = SC.space_filling(space, routes, pool, 6, random.Random(seed), max_controls=0)
        assert not any(x["fault"]["fault_id"] in controls for x in c0)


def test_space_filling_spreads_onsets_per_fault():
    _, space, routes, pool, _ = _ctx()
    c = SC.space_filling(space, routes, pool, 4 * len(pool), random.Random(3))
    by_fault = {}
    for x in c:
        by_fault.setdefault(x["fault"]["fault_id"], set()).add(x["onset_event"])
    faults = SC.faults_by_id(space)
    for fid, evs in by_fault.items():
        allowed = {e for r in routes for e in SC.allowed_onset_events(space, r, faults[fid])}
        assert len(evs) >= min(3, len(allowed)), (fid, evs)
    for x in c:   # every combo legal (route_onset_exclusions)
        assert not SC.validate_against_space(space, x, routes), x


def test_cold_start_rule_default_is_12():
    loop = Loop(mode="mock", out="/tmp/_cs_rule", ablation="X-5", video_on=False, seed=0, quiet=True)
    cfg, _, _, pool, controls = _ctx()
    n_ctrl = sum(f in controls for f in pool)
    eligible = len(pool) - n_ctrl + min(n_ctrl, loop.control_cap)
    rule = cfg["orchestrator"]["cold_start"]["rule"]
    assert cfg["orchestrator"]["cold_start"]["total_runs"] is None
    expect = min(max(loop.min_train_for_acq, rule["per_fault_type"] * eligible), rule["max_runs"])
    assert loop.cold == expect == 12
    assert loop.rounds == 5   # (72 - 12) / 12


def test_cold_start_overrides():
    cfg = load_stage_configs()
    cfg2 = copy.deepcopy(cfg)
    cfg2["orchestrator"]["cold_start"]["total_runs"] = 24
    assert Loop(mode="mock", out="/tmp/_cs_o", cfg=cfg2, ablation="X-5", seed=0, quiet=True, video_on=False).cold == 24
    cfg3 = copy.deepcopy(cfg)
    cfg3["orchestrator"]["cold_start"]["rule"] = {"per_fault_type": 1, "max_runs": 12}
    assert Loop(mode="mock", out="/tmp/_cs_o", cfg=cfg3, ablation="X-5", seed=0, quiet=True, video_on=False).cold == 8   # min_train floor
    assert Loop(mode="mock", out="/tmp/_cs_o", cold_start=5, ablation="X-5", seed=0, quiet=True, video_on=False).cold == 5


def test_round0_runs_full_cold_start(tmp_path):
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=12, batch_size=4,
                rounds=0, quiet=True)
    loop.run()
    r0 = [r for r in loop.records if r["round"] == 0]
    assert len(r0) == 12
    ids = Counter(loop.candidates[r["scenario_id"]]["fault"]["fault_id"] for r in r0)
    controls = loop.control_ids
    assert sum(v for k, v in ids.items() if k in controls) <= loop.control_cap
    assert len(ids) >= 5


def test_space_filling_speed_step_checks_quantized_speed():
    _, space, routes, pool, _ = _ctx()
    for seed in range(5):
        for x in SC.space_filling(space, routes, pool, 21, random.Random(seed), speed_step=5.0):
            q = dict(x, environment=dict(x["environment"], max_speed_kmh=round(x["environment"]["max_speed_kmh"] / 5) * 5.0))
            assert not SC.onset_exclusions(space, x["route_id"], q["environment"]["max_speed_kmh"]) or \
                x["onset_event"] in SC.allowed_onset_events(space, x["route_id"], SC.faults_by_id(space)[x["fault"]["fault_id"]],
                                                            q["environment"]["max_speed_kmh"])
