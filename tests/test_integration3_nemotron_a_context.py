"""integration3 #7: what Nemotron A gets to judge the recovery boundary (units, acquisition-consistent ranking,
never_switched != failed, route geometry / V-03, outcome pairs, English view, schema placement, refine context)."""
import json

import pytest

from engine import feedback as FB
from engine import nemotron as NM
from engine import scenarios as SC
from engine.config import REPO_ROOT, SPACE_REL, load_yaml
from engine.llm import LLMError, StubClient
from engine.surrogate import boundary_score, boundary_score_parts

ROUTES = ["bench2drive:26956", "town10_long"]
POOL = ["F-01", "F-04", "F-06"]
M, SW = "recovery_success", "switched_indicator"


@pytest.fixture(scope="module")
def space():
    return load_yaml(REPO_ROOT / SPACE_REL)


def _bias(space, mag, speed=25, sid=None):
    c = SC.make_candidate(space, route_id="town10_long", speed=speed, fault_id="F-04", params={"magnitude_rad": mag},
                          onset_event="curve_entry", offset=0.0, source="nemotron_a")
    if sid:
        c["scenario_id"] = sid
    return c


def _rec(sid, mode="bias", switch=5.0, fs=None, outcome="safe_recovery", lrr=0.3, p01=0.2, **kw):
    return {"scenario_id": sid, "run_id": f"R-{sid}", "sim_status": "ok", "parameters": {"fault": {"mode": mode}},
            "switch_time": switch, "planned_fault_onset": 4.0, "final_safe_recovery": fs, "recovery_outcome": outcome,
            "lane_recovery_robustness": lrr, "P01": p01, "P02": 1.0, **kw}


def test_region_values_are_in_target_units_and_rank_matches_acquisition(space):
    native = lambda m, s: (m * 0.1, s * 0.1)   # noqa: E731  (orient scale 0.1, sign +1)
    a = dict(_bias(space, 0.05), gp_mean={M: 1.0, SW: 0.9, SW + "__latent": 2.0}, gp_std={M: 1.5, SW + "__latent": 0.5})
    b = dict(_bias(space, 0.09, speed=35), gp_mean={M: -6.0, SW: 0.2, SW + "__latent": -0.5},
             gp_std={M: 1.5, SW + "__latent": 0.4})
    kw = dict(beta=1.96, level_z=0.0, y_scale=4.0, switch_penalty=1.0)
    s = NM.surrogate_summary(regions=[a, b], records=[], candidates={}, space=space, metric=M, switch_metric=SW,
                             n_train=10, k=2, native=native, gp_info={"y_scale_z": 4.0, "units": "dimensionless"},
                             scorer=lambda c: boundary_score_parts(c, M, SW, **kw))
    got = {r["max_speed_kmh"]: r for r in s["boundary_regions"]}
    assert got[25]["predicted_lane_recovery_robustness"] == pytest.approx(0.1)        # z 1.0 -> lrr 0.1, not "1.0"
    assert got[25]["predicted_lane_recovery_robustness_std"] == pytest.approx(0.15)
    for c, r in ((a, got[25]), (b, got[35])):
        assert r["acquisition_score"] == pytest.approx(round(boundary_score(c, M, SW, **kw), 3))
    order = sorted((a, b), key=lambda c: boundary_score(c, M, SW, **kw), reverse=True)
    assert [r["max_speed_kmh"] for r in s["boundary_regions"]] == [c["environment"]["max_speed_kmh"] for c in order]
    assert got[35]["p_switch"] == 0.2
    # the 1 - P(switch) penalty is inside the recovery term, as in the orchestrator
    parts = boundary_score_parts(b, M, SW, **kw)
    assert parts["recovery_term"] == pytest.approx((1.96 * 1.5 - 6.0) / 4.0 - 0.8)


def test_never_switched_is_not_failed():
    recs = [_rec("a", switch=None, fs=False, outcome="no_switch"), _rec("b", switch=None, fs=False, outcome="no_switch"),
            _rec("c", mode="saturate", fs=False, outcome="unrecovered"),
            _rec("d", mode="drift", fs=False, outcome="safe_recovery", **{}),
            _rec("e", mode="drift", fs=True)]
    per = NM.outcomes_by_fault_mode(recs)
    assert per["bias"]["status"] == "never_switched" and per["bias"]["failed"] == 0 and per["bias"]["never_switched"] == 2
    assert per["saturate"]["status"] == "failed_whenever_switched"
    assert per["drift"]["status"] == "mixed"
    assert per["drift"]["failed_reasons"] == {"lat_acc_or_jerk_excess": 1}     # safe_recovery but not final
    assert FB.outcome_label(recs[0]) == "never_switched" and FB.outcome_label(None) == "not_run"
    assert "never_switched" in NM.A_SYSTEM and "stronger" in NM.A_SYSTEM


def test_outcome_examples_bracket_the_boundary(space):
    cs = {sid: _bias(space, m, sid=sid) for sid, m in (("r1", 0.02), ("r2", 0.05), ("f1", 0.06), ("f2", 0.1), ("n", 0.01))}
    recs = [_rec("r1", fs=True), _rec("r2", fs=True), _rec("f1", fs=False, outcome="unrecovered", lrr=-0.2, p01=-0.1),
            _rec("f2", fs=False, outcome="unrecovered"), _rec("n", switch=None, fs=False, outcome="no_switch")]
    ex = FB.outcome_examples(recs, cs, space, ROUTES, POOL)
    e = next(x for x in ex if x["fault_mode"] == "bias")
    assert e["recovered"]["params"]["magnitude_rad"] == 0.05 and e["failed"]["params"]["magnitude_rad"] == 0.06
    assert e["failed"]["outcome"]["lane_margin_min_m"] == -0.1 and e["failed"]["outcome"]["switched"] is True
    assert e["never_switched_runs"] == 1
    t = FB.tried_view(cs["f1"], recs[2])
    assert t["outcome"] == "failed:unrecovered" and t["switched"] is True and t["lane_margin_min_m"] == -0.1


def test_english_view_prefers_en_and_drops_untranslated():
    dropped = []
    v = NM.english_view({"note": "한국어", "note_en": "english", "x": {"check": "검사"}, "y": "ok"}, dropped)
    assert v == {"note": "english", "x": {}, "y": "ok"} and len(dropped) == 1


def test_prompt_is_english_with_real_config(space):
    from engine.config import load_stage_configs
    import re
    cfg = load_stage_configs()
    ctx = NM.a_context(profile=cfg["safety_profile"], prior=cfg["prior_knowledge"], space=space,
                       objective_cfg={"natural_switch_bound": "한", "natural_switch_bound_en": "en"}, routes=[],
                       few_shot=[], mechanisms=[], tried=[], include_unreviewed=True)
    txt = json.dumps(ctx, ensure_ascii=False)
    assert not re.search(r"[\uac00-\ud7a3]", txt)
    f4 = next(f for f in ctx["scenario_space"]["faults"] if f["id"] == "F-04")
    assert f4["description"].startswith("Adds a constant angle")
    assert ctx.get("untranslated_fields_omitted", 0) <= 15      # remaining Korean-only leaves are few


class _Rec(StubClient):
    def __init__(self, fail_schema=False):
        super().__init__(NM.stub_handler(0), None)
        self.seen, self.fail_schema = [], fail_schema

    def chat(self, model, messages, response_format=None, **kw):
        self.seen.append((messages, response_format))
        if self.fail_schema and (response_format or {}).get("type") == "json_schema":
            raise LLMError("HTTP 400 json_schema unsupported")
        return super().chat(model, messages, response_format=response_format, **kw)


def test_schema_only_in_response_format_except_fallback(space):
    sch = NM.candidate_schema(space, ROUTES, POOL, ["S1"], ["F-04", "S1"], 2)
    msgs = NM.a_messages(schema=sch, n=2, context={})
    assert "json_schema" not in json.loads(msgs[-1]["content"])
    c = _Rec()
    obj, st = NM.call_a(c, {"model": "stub/A"}, msgs, sch)
    assert st["response_format"] == "json_schema" and c.seen[0][1]["type"] == "json_schema"
    c = _Rec(fail_schema=True)
    obj, st = NM.call_a(c, {"model": "stub/A"}, msgs, sch)
    assert st["response_format"] == "json_object_fallback" and st["items_valid"] == 2
    assert "json_schema" in json.loads(c.seen[-1][0][-1]["content"])


def test_refine_gets_the_same_compact_context(space):
    sch = NM.candidate_schema(space, ROUTES, POOL, ["S1"], ["F-04", "S1"], 1)
    ctx = {"objective": {"judgment_criteria": {"recovered": "x"}}, "route_geometry": {"routes": {}},
           "outcome_examples": [{"fault_mode": "bias"}], "already_tried": [{"outcome": "recovered"}],
           "scenario_space": {}, "supervisor": {}, "routes": [],
           "surrogate_summary": {"gp": {"units": "u"}, "recent_outcomes_by_fault_mode": {"bias": {}}, "boundary_regions": [1]}}
    u = json.loads(NM.refine_messages(sch, [{"fault_id": "F-04"}], ctx)[-1]["content"])
    for k in ("objective", "route_geometry", "outcome_examples", "already_tried"):
        assert u[k] == ctx[k]
    assert u["surrogate"]["gp"]["units"] == "u" and "boundary_regions" not in u["surrogate"] and "json_schema" not in u


def test_stub_respects_v03_latest_offset(space):
    sch = NM.candidate_schema(space, ["town10_long"], ["F-04"], ["S1"], ["F-04", "S1"], 6)
    geom = {"routes": {"town10_long": {"events": {e: [[25, 10.0, 0.02, 7.0, 1.0, -0.8 if e == "curve_entry" else None]]
                                                   for e in ("straight", "curve_entry", "curve_apex", "curve_exit")}}}}
    client = StubClient(NM.stub_handler(3), None)
    obj, st = NM.call_a(client, {"model": "stub/A"}, NM.a_messages(schema=sch, n=6, context={"route_geometry": geom}), sch)
    assert st["items_valid"] == 6
    for c in obj["candidates"]:
        assert c["onset_event"] == "curve_entry" and c["onset_offset_s"] <= -0.85 + 1e-9


def test_loop_route_geometry_and_context(tmp_path):
    from run import Loop
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=6, batch_size=6,
                rounds=1, quiet=True)
    loop.run()
    g = json.loads((tmp_path / "o" / "round_01" / "03_route_geometry.json").read_text())
    assert g["columns"][-1] == "latest_onset_offset_s_for_V03" and "V03" in g["rules"]
    mw = float(loop.cfg["safety_spec"]["evaluation_window"]["min_length_s"])
    for rid, r in g["routes"].items():
        for ev, rows in r["events"].items():
            for row in rows:
                if row[1] is not None:
                    assert row[5] == pytest.approx(r["route_end_s_by_kmh"][f"{row[0]:g}"] - mw - row[1], abs=0.02)
                    assert row[4] == pytest.approx(row[3] ** 2 * abs(row[2]), rel=0.05, abs=0.02)
    jc = loop.judgment_criteria()
    assert "4.41" in jc["lat_acc_limit"] and "0.5 * 0.9" in jc["lat_acc_limit"] and "3 s" in jc["recovered"]
    z, sd = loop.native(-2.0, 1.0)
    assert (z, sd) == pytest.approx((-0.2, 0.1))
    summ = json.loads((tmp_path / "o" / "round_01" / "03_gp_summary.json").read_text())
    assert "dimensionless" in (summ["gp"] or {}).get("units", "dimensionless")
