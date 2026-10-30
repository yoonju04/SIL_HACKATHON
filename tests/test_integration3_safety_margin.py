"""integration3 #9: GP target = signed safety margin over every final_safe_recovery criterion."""
import json

import pytest

from engine import judgment as J
from engine.config import load_stage_configs
from engine.fixtures.synthetic_sim import write_run_folder
from engine.monitor import evaluate, thresholds_from_config

TH = thresholds_from_config(load_stage_configs())
LIM_J = TH["lat_jerk_threshold_mps3"]
LIM_A = TH["lat_acc_threshold_mps2"]
NEED = TH["settle_hold_s"]


def sm(lrr=0.5, collision=False, acc=0.0, jerk=0.0, hold=2.0, need=NEED, acc_lim=LIM_A):
    return J.safety_margin(lrr=lrr, collision=collision, acc_excess=acc, acc_limit=acc_lim, jerk_excess=jerk,
                           jerk_limit=LIM_J, settle_hold_s=hold, settle_need_s=need,
                           clip=(TH["robustness_clip_min"], TH["robustness_clip_max"]))


def labels(acc, jerk, acc_lim=LIM_A):
    # judgment labels use a strict '>' (lat_*_excess.primary)
    return {"lat_acc_excess": {"primary": acc > acc_lim}, "lat_jerk_excess": {"primary": jerk > LIM_J}}


def test_thresholds_are_the_judgment_config():
    assert LIM_J == 15.0 and NEED == 1.0         # safety_spec.yaml P-04 (integration4: 5 -> 15, data-based) / recovery_judgment


@pytest.mark.parametrize("acc,jerk,hold,lrr,outcome", [
    (0.0, 0.0, 2.0, 0.5, "safe_recovery"),
    (0.0, LIM_J, 2.0, 0.5, "safe_recovery"),            # exactly at the jerk limit: not an excess, margin 0 -> safe
    (0.0, LIM_J + 1e-6, 2.0, 0.5, "safe_recovery"),     # just over
    (LIM_A, 0.0, 2.0, 0.5, "safe_recovery"),            # exactly at the acc limit
    (LIM_A + 1e-6, 0.0, 2.0, 0.5, "safe_recovery"),
    (0.0, 0.0, NEED, 0.5, "safe_recovery"),             # held exactly settle_hold_s
    (0.0, 0.0, NEED - 0.05, 0.5, "unsettled"),
    (0.0, 0.0, 2.0, -0.3, "recovered_after_departure"),
    (0.0, 0.0, 0.0, -1.5, "unrecovered"),
    (0.0, 0.0, 2.0, 0.0, "safe_recovery"),              # touching the lane edge, no departure
])
def test_sign_equals_final_safe(acc, jerk, hold, lrr, outcome):
    m = sm(lrr=lrr, acc=acc, jerk=jerk, hold=hold)
    fs = J.final_safe_recovery(outcome, labels(acc, jerk))
    assert (m["value"] >= 0) == fs
    assert m["failing"] == [k for k, v in m["terms"].items() if v < 0]
    assert m["binding"] == min(m["terms"], key=m["terms"].get)


def test_terms_and_units():
    m = sm(lrr=0.6, acc=1.0, jerk=LIM_J + 1.0, hold=1.5)
    assert m["terms"]["lat_jerk"] == pytest.approx(-1.0 / LIM_J)
    assert m["terms"]["lat_acc"] == pytest.approx((LIM_A - 1.0) / LIM_A)
    assert m["terms"]["settle"] == pytest.approx((1.5 - NEED) / NEED)
    assert m["binding"] == "lat_jerk" and m["failing"] == ["lat_jerk"]
    assert m["observed"]["lat_jerk"] == {"value": LIM_J + 1.0, "unit": "m/s^3 over baseline", "limit": LIM_J}


def test_collision_binding_and_clip():
    m = sm(lrr=TH["collision_penalty"], collision=True, hold=0.0)
    assert m["binding"] == "collision" and "collision" in m["failing"] and m["value"] >= TH["robustness_clip_min"]
    assert sm(lrr=50.0)["value"] <= TH["robustness_clip_max"]


def test_undefined_inputs_give_none():
    assert sm(lrr=None) is None and sm(acc=None) is None and sm(jerk=None) is None
    assert sm(hold=None)["terms"].keys() == {"lane", "lat_acc", "lat_jerk"}


def _rec(**kw):
    r = {"sim_status": "ok", "switch_time": 20.0, "recovery_outcome": "safe_recovery", "final_safe_recovery": True,
         "lane_recovery_robustness": 0.5,
         "metrics": {"lat_acc_excess_mps2": 0.2, "lat_jerk_excess_mps3": 1.0, "p02_lat_acc_threshold_mps2": LIM_A}}
    for k, v in kw.items():
        if k == "metrics":
            r["metrics"].update(v)
        else:
            r[k] = v
    return r


def test_rescore_reasons():
    ok, why = J.rescore_safety_margin(_rec(), TH)
    assert why is None and ok["value"] > 0 and ok["rescored_from_record"] and "settle" not in ok["terms"]
    assert J.rescore_safety_margin(_rec(sim_status="error"), TH) == (None, "not sim_status ok")
    # integration4: a never-switched run is re-scored from its SR4 record (lane-only margin)
    assert J.rescore_safety_margin(_rec(switch_time=None), TH) == (None, "no switch: SR4 margin not logged")
    ns, why = J.rescore_safety_margin(_rec(switch_time=None, trigger_time=19.0, sr_checks={"sr": {"SR4": {
        "margin_m": -0.1, "nominal_margin_m": 0.8, "collision": False}}}), TH)
    assert why is None and ns["kind"] == "no_switch_lane" and ns["value"] == pytest.approx(-0.125) and ns["binding"] == "lane"
    assert J.rescore_safety_margin(_rec(recovery_outcome="unsettled", final_safe_recovery=False), TH)[1].startswith("unsettled")
    assert J.rescore_safety_margin(_rec(metrics={"lat_jerk_excess_mps3": None}), TH)[1].startswith("missing")
    assert J.rescore_safety_margin(_rec(final_safe_recovery=None), TH)[1] == "final_safe_recovery is None"
    # the logged label is re-derived with the current thresholds (P-04 5 -> 15 turns logged jerk failures safe)
    relab, why = J.rescore_safety_margin(_rec(final_safe_recovery=False, metrics={"lat_jerk_excess_mps3": 7.0}), TH)
    assert why is None and relab["value"] > 0
    # a contradictory record (safe_recovery with a negative lane term) is still excluded
    assert J.rescore_safety_margin(_rec(final_safe_recovery=False, lane_recovery_robustness=-0.3), TH)[1].startswith("rescored margin sign")
    bad, why = J.rescore_safety_margin(_rec(final_safe_recovery=False, metrics={"lat_jerk_excess_mps3": LIM_J + 2.0}), TH)
    assert why is None and bad["binding"] == "lat_jerk" and bad["value"] < 0
    # friction-scaled limit when p02 threshold was not logged
    r = _rec(metrics={"p02_lat_acc_threshold_mps2": None, "tire_friction_scale": 0.5, "lat_acc_excess_mps2": 0.3})
    assert J.rescore_safety_margin(r, TH)[0]["observed"]["lat_acc"]["limit"] == pytest.approx(LIM_A * 0.5)
    # an already-logged margin is recomputed (thresholds may have changed since it was written)
    d = sm(jerk=LIM_J + 3.0)
    again, _ = J.rescore_safety_margin(_rec(metrics={"safety_margin": d["value"], "safety_margin_detail": d}), TH)
    assert again["value"] == pytest.approx(ok["value"])


def test_monitor_writes_margin_matching_final_safe(tmp_path):
    base = write_run_folder(tmp_path / "b", route="town10_long", max_speed_kmh=30, fault=None, stop_after_fault_s=10,
                            record_front=False)
    seen = 0
    for i, mag in enumerate((0.03, 0.08, 0.2)):
        d = write_run_folder(tmp_path / f"f{i}", route="town10_long", max_speed_kmh=30, stop_after_fault_s=10,
                             fault=dict(mode="bias", start=20.0, duration=500.0, magnitude=mag), record_front=False)
        r = evaluate(d, baseline_dir=base)
        m = r["metrics"]
        if r.get("switch_time") is None:   # integration4: lane-only margin when the fault was effective
            assert m["safety_margin"] is None or m["safety_margin_detail"]["kind"] == "no_switch_lane"
            continue
        if r.get("final_safe_recovery") is None:
            assert m["safety_margin"] is None
            continue
        seen += 1
        assert (m["safety_margin"] >= 0) == r["final_safe_recovery"]
        assert m["safety_margin_binding"] == m["safety_margin_detail"]["binding"]
        assert m["settle_hold_max_s"] is not None
        assert r["gp_targets"]["recovery_success"] == pytest.approx(m["safety_margin"] / 0.1)
        json.dumps(r)
    assert seen >= 1


def test_select_gp_target_and_cli():
    import run
    cfg = load_stage_configs()["vulnerability_patterns"]["metric_rules"]
    name, mr = run.select_gp_target(cfg, None)
    assert name == "safety_margin" and mr["recovery_success"]["gp"]["target"] == "safety_margin"
    name, mr = run.select_gp_target(cfg, "lrr")
    assert mr["recovery_success"]["gp"]["target"] == "lane_recovery_robustness"
    assert cfg["recovery_success"]["gp"]["target"] == "safety_margin"            # config not mutated
    with pytest.raises(SystemExit):
        run.select_gp_target(cfg, "nope")


def test_history_rescoring(tmp_path):
    from run import Loop
    par = {"route_id": "town10_long", "environment": {"max_speed_kmh": 30.0}, "onset_event": "straight",
           "onset_offset_s": 0.0, "fault": {"fault_id": "F-04", "mode": "bias", "params": {"magnitude_rad": 0.05}}}
    rows = [dict(_rec(), parameters=par, supervisor_profile="tight"),
            dict(_rec(recovery_outcome="unsettled", final_safe_recovery=False), parameters=par, supervisor_profile="tight"),
            dict(_rec(switch_time=None, recovery_outcome="no_switch", final_safe_recovery=False), parameters=par,
                 supervisor_profile="tight")]
    hist = tmp_path / "h.jsonl"
    hist.write_text("\n".join(json.dumps(r) for r in rows))
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=4, batch_size=2,
                rounds=0, quiet=True)
    tg, why = loop.history_targets(rows[0])
    assert why is None and tg["recovery_success"] > 0
    assert loop.history_targets(rows[1])[1].startswith("safety_margin: unsettled")
    assert "recovery_success" not in loop.history_targets(rows[2])[0]
    loop.load_gp_history([str(hist)])
    g = loop.stats["gp_history"]
    assert g["gp_target"] == "safety_margin" and g["loaded"] == 2
    assert any(k.startswith("safety_margin: unsettled") for k in g["skipped"])


def test_nemotron_a_views_carry_margin_in_real_units():
    from engine import feedback as FB
    from engine import nemotron as NM
    d = sm(jerk=LIM_J + 2.0)
    r = _rec(final_safe_recovery=False, metrics={"safety_margin": d["value"], "safety_margin_binding": d["binding"],
                                                 "safety_margin_detail": d, "lat_jerk_excess_mps3": LIM_J + 2.0})
    v = FB.outcome_view(r)
    assert v["safety_margin"]["binding"] == "lat_jerk" and v["failed_criteria"] == ["lat_jerk"]
    assert v["safety_margin"]["terms"]["lat_jerk"] == {"margin": round(d["terms"]["lat_jerk"], 3), "observed": LIM_J + 2.0,
                                                       "limit": LIM_J, "unit": "m/s^3 over baseline"}
    assert "safety_margin_detail" not in v
    c = {"scenario_id": "s", "fault": {"fault_id": "F-04", "mode": "bias", "params": {}}, "route_id": "r",
         "environment": {"max_speed_kmh": 30.0}, "onset_event": "straight", "onset_offset_s": 0.0,
         "gp_mean": {"recovery_success": 0.0}, "gp_std": {"recovery_success": 1.0}}
    it = NM.region_item(c, {}, metric="recovery_success", native=lambda m, s: (m, s),
                        y_scale=0.1, level_z=0.0, target="safety_margin")
    assert it["predicted_safety_margin"] == 0.0 and "predicted_lane_recovery_robustness" not in it


def test_no_switch_margin_and_footprint_departure():
    # integration4: never-switched runs get a lane-only target; departures use the vehicle footprint (as SR4)
    m = J.no_switch_safety_margin(margins=[0.4, -0.05], lane_widths=[3.5, 3.5], bbox_half_width_m=0.95, collision=False,
                                  penalty=TH["collision_penalty"], clip=(TH["robustness_clip_min"], TH["robustness_clip_max"]))
    assert m["kind"] == "no_switch_lane" and m["value"] == pytest.approx(-0.05 / 0.8) and m["failing"] == ["lane"]
    c = J.no_switch_safety_margin(margins=[0.4], lane_widths=[3.5], bbox_half_width_m=0.95, collision=True,
                                  penalty=TH["collision_penalty"], clip=(TH["robustness_clip_min"], TH["robustness_clip_max"]))
    assert c["binding"] == "collision" and c["value"] <= TH["collision_penalty"]
    from engine.monitor import footprint_margin_m
    row = {"lane_width_m": 3.5, "lane_offset_m": 0.7, "bbox_half_width_m": 0.92, "bbox_half_length_m": 2.45,
           "lane_heading_err_rad": 0.08}
    assert J.lane_margin_m(3.5, 0.7, 0.92) > 0 > footprint_margin_m(row)    # centre-line says in, a corner is out
