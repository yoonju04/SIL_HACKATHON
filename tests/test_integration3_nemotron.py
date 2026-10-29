"""integration3 #4: Nemotron A boundary proposals (GP summary every round, range-extreme penalty, per-round share)."""
import json
import random

import pytest

from engine import nemotron as NM
from engine import scenarios as SC
from engine.config import REPO_ROOT, SPACE_REL, load_yaml
from engine.llm import StubClient

ROUTES = ["bench2drive:26956", "town10_long"]
POOL = ["F-01", "F-04", "F-06"]


@pytest.fixture(scope="module")
def space():
    return load_yaml(REPO_ROOT / SPACE_REL)


def _sat(space, **kw):
    p = {"duration_s": 0.5, "side": "positive", "magnitude": 0.6, "ramp_s": 0.1}
    p.update(kw.pop("params", {}))
    return SC.make_candidate(space, route_id="town10_long", speed=kw.pop("speed", 25), fault_id="F-06", params=p,
                             onset_event="curve_entry", offset=kw.pop("offset", 0.0), source="nemotron_a", **kw)


def test_prompt_is_boundary_seeking_not_failure_seeking():
    s = NM.A_SYSTEM
    assert "NOT to find failures" in s and "ON the success/failure boundary" in s
    assert "surrogate_summary" in s and "extreme_justification" in s


def test_extreme_detection_and_edge_distance(space):
    mid = _sat(space)
    assert NM.extreme_fields(mid, space) == [] and NM.edge_distance(mid, space) > 0.1
    old_style = _sat(space, params={"duration_s": 1.0, "magnitude": 1.0})   # result_2/5: saturate at the max
    assert {"params.duration_s", "params.magnitude"} <= set(NM.extreme_fields(old_style, space))
    assert NM.edge_distance(old_style, space) == pytest.approx(0.0)


def test_unjustified_extremes_are_tagged(space):
    sch = NM.candidate_schema(space, ROUTES, POOL, ["S1"], ["F-06", "S1"], 3)
    item = {"fault_id": "F-06", "params": {"duration_s": 1.0, "side": "positive", "magnitude": 1.0, "ramp_s": 0.0},
            "route_id": "town10_long", "max_speed_kmh": 30, "tire_friction_scale": 1.0, "onset_event": "curve_entry",
            "onset_offset_s": 0.0, "objective_ids": ["S1"], "evidence_ids": ["F-06"], "expected_mechanism": "x",
            "expected_side": "fail", "extreme_justification": "", "missing_information": ""}
    assert "extreme_justification" in sch["properties"]["candidates"]["items"]["anyOf"][0]["required"]
    c = NM.a_to_candidates({"candidates": [item]}, space, round_index=1)[0]
    assert "range_extreme_unjustified" in c["tags"] and c["range_extremes"]
    ok = NM.a_to_candidates({"candidates": [dict(item, extreme_justification="bracket the known failure")]}, space, round_index=1)[0]
    assert "range_extreme_unjustified" not in ok.get("tags", [])


def test_surrogate_summary_picks_ambiguous_regions(space):
    a = dict(_sat(space), gp_mean={"recovery_success": 0.1}, gp_std={"recovery_success": 2.0})    # ambiguous
    b = dict(_sat(space, speed=35), gp_mean={"recovery_success": -15.0}, gp_std={"recovery_success": 0.5})   # clear fail
    recs = [{"sim_status": "ok", "parameters": {"fault": {"mode": "saturate"}}, "recovery_outcome": "unrecovered",
             "final_safe_recovery": False, "switch_time": 5.0}] * 3
    # integration3 #7: training std 4 (z) -> a's std is half of it: explored, ambiguous -> boundary
    s = NM.surrogate_summary(regions=[b, a], records=recs, candidates={}, space=space, metric="recovery_success",
                             switch_metric="switched_indicator", n_train=3, k=1, gp_info={"y_scale_z": 4.0})
    assert len(s["boundary_regions"]) == 1 and s["boundary_regions"][0]["max_speed_kmh"] == 25
    assert s["recent_outcomes_by_fault_mode"]["saturate"]["status"] == "failed_whenever_switched"
    assert s["gp"]["n_train"] == 3
    # with training std 1 the same std 2 is prior-level uncertainty: unexplored, not boundary evidence
    s = NM.surrogate_summary(regions=[b, a], records=recs, candidates={}, space=space, metric="recovery_success",
                             switch_metric="switched_indicator", n_train=3, k=1, gp_info={"y_scale_z": 1.0})
    assert [r["max_speed_kmh"] for r in s["unexplored_regions"]] == [25]
    assert [r["max_speed_kmh"] for r in s["boundary_regions"]] == [35]


def test_offline_stub_uses_boundary_regions(space):
    sch = NM.candidate_schema(space, ROUTES, POOL, ["S1"], ["F-06", "S1"], 4)
    region = {"fault_id": "F-06", "params": {"duration_s": 0.4, "side": "negative", "magnitude": 0.5, "ramp_s": 0.1},
              "route_id": "bench2drive:26956", "max_speed_kmh": 30, "tire_friction_scale": 0.7,
              "onset_event": "curve_apex", "onset_offset_s": 0.2}
    ctx = {"surrogate_summary": {"boundary_regions": [region]}}
    client = StubClient(NM.stub_handler(0), None)
    obj, st = NM.call_a(client, {"model": "stub/A"}, NM.a_messages(schema=sch, n=4, context=ctx), sch)
    assert st["schema_errors"] == 0 and st["items_valid"] == 4
    near = [c for c in obj["candidates"] if c["expected_side"] == "boundary" and c["fault_id"] == "F-06"
            and c["route_id"] == "bench2drive:26956" and c["onset_event"] == "curve_apex"]
    assert len(near) == 2 and all(abs(c["params"]["magnitude"] - 0.5) < 0.4 for c in near)


def test_nemotron_share_reserves_batch_slots(tmp_path, space):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=1, cold_start=4, batch_size=10,
                rounds=1, quiet=True)
    a = [dict(_sat(space, offset=0.1 * i), seed_scenario_id=f"a{i}") for i in range(5)]
    a[0]["tags"] = ["range_extreme_unjustified"]
    others = [dict(_sat(space, offset=-0.1 * i), source="space_filling") for i in range(1, 6)]
    picked = loop.nemotron_share(a + others, 10, lambda c: None)
    assert len(picked) == 3 and all(c["selected_by"] == "nemotron_a_share" for c in picked)
    assert a[0] not in picked                                   # unjustified extreme goes last
    loop.cfg["orchestrator"]["acquisition"]["batch_composition"]["nemotron_a"] = 0.0
    assert loop.nemotron_share(a + others, 10, lambda c: None) == []


def test_mock_loop_gives_a_summary_every_round_and_runs_a_candidates(tmp_path):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=6, batch_size=6,
                rounds=2, quiet=True)
    m = loop.run()
    for rnd in (1, 2):
        summ = json.loads((tmp_path / "o" / f"round_{rnd:02d}" / "03_gp_summary.json").read_text())
        assert "boundary_regions" in summ and "recent_outcomes_by_fault_mode" in summ
    calls = [c for c in loop.stats["a_calls"] if not c.get("refine")]
    assert len(calls) == 3                                      # cold start + every round
    after = [r for r in loop.records if r["round"] > 0]
    sel = [r["selector"] for r in after]
    assert sel.count("nemotron_a_share") >= 2                   # ceil(0.3 * 6) per round, some may be gated/capped
    assert m["nemotron_a"]["share_after_cold_start"] > 0
