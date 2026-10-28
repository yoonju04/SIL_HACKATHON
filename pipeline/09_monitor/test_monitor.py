"""Synthetic-trace tests for monitor.py v2 and feedback.py:  python pipeline/09_monitor/test_monitor.py"""
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feedback import (failure_table, gp_training_rows, hypothesis_tracking,  # noqa: E402
                      write_outputs)
from monitor import (RUN_RECORD_FIELDS, MonitorConfig, Trace, footprint_lane_margin,  # noqa: E402
                     lateral_acceleration, load_spec, measure_ftti, noise_floor, view)
import monitor  # noqa: E402


def evaluate_pair(*a, **k):
    """Tests read the flat view (runner record + monitor details)."""
    return view(monitor.evaluate_pair(*a, **k))

DT = 0.05
CFG = MonitorConfig(dt=DT, lane_width_default_m=3.5, vehicle_half_width_m=0.95,
                    vehicle_half_length_m=2.4, max_steer_rad=math.radians(70))
ONSET = 10.0


def trace(offset_fn=lambda x: 0.0, T=30.0, fault_at=None, switch_at=None, switch_back_at=None,
          revision_at=None, vx=15.0, r=0.0, collide_at=None, steer_fn=lambda x: 0.0, effective=True):
    t = np.round(np.arange(0.0, T, DT), 6)
    off = np.array([offset_fn(x) for x in t])
    ac = []
    for x in t:
        on_lqr = switch_at is not None and x >= switch_at and (switch_back_at is None or x < switch_back_at)
        ac.append("lqr" if on_lqr else "mpc")
    fa = np.zeros(len(t), bool) if fault_at is None else (t >= fault_at)
    fe = fa.copy() if effective else np.zeros(len(t), bool)
    rev = None if revision_at is None else (t >= revision_at).astype(int)
    return Trace(t=t, vx=np.full(len(t), vx), vy=np.zeros(len(t)), yaw_rate=np.full(len(t), r),
                 lane_offset_m=off, lane_heading_err=np.zeros(len(t)), active_controller=ac,
                 steer_cmd_rad=np.array([steer_fn(x) for x in t]), fault_active=fa, fault_effective=fe,
                 route_revision=rev, collisions_t=[] if collide_at is None else [collide_at])


def drift(rate, cap, start=ONSET):
    return lambda x: 0.0 if x < start else min(cap, rate * (x - start))


def dangerous_no_reaction():
    return trace(drift(1.0, 2.0), fault_at=ONSET)          # margin < 0 after ~0.8 s


def harmless_no_reaction():
    return trace(drift(0.2, 0.4), fault_at=ONSET)          # stays inside the lane


# -------------------------------------------------------------- geometry
def test_centered_margin_equals_lane_half_width_minus_half_width():
    assert abs(footprint_lane_margin([0.0], [0.0], [3.5], 0.95, 2.4)[0] - 0.8) < 1e-12


def test_heading_error_reduces_margin_through_the_corners():
    m1 = footprint_lane_margin([0.0], [math.radians(5)], [3.5], 0.95, 2.4)[0]
    expected = 1.75 - (2.4 * math.sin(math.radians(5)) + 0.95 * math.cos(math.radians(5)))
    assert abs(m1 - expected) < 1e-12


def test_lateral_acceleration_from_yaw_rate():
    assert np.allclose(lateral_acceleration(trace(vx=15.0, r=0.2), CFG), 3.0)


def test_load_spec_reads_numbers_from_yaml():
    cfg = load_spec(os.path.join(HERE, "safety_spec.yaml"), lane_width_default_m=3.5)
    assert cfg.sr2_limit_dps == 60.0 and cfg.sr1_ftti_fraction == 0.5 and cfg.window_s == 10.0
    assert cfg.sr5_enforce is False and cfg.gp_clip == (-2.0, 1.0) and cfg.lane_width_default_m == 3.5


# ------------------------------------------------------------------ FTTI
def test_ftti_finite_when_no_reaction_leaves_lane():
    f = measure_ftti(dangerous_no_reaction(), ONSET, CFG)
    assert f["valid"] and abs(f["ftti_s"] - 0.8) < DT + 1e-9, f


def test_ftti_infinite_when_no_reaction_stays_in_lane():
    assert math.isinf(measure_ftti(harmless_no_reaction(), ONSET, CFG)["ftti_s"])


# -------------------------------------------------------------- outcomes
def test_pass_quick_smooth_switch():
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2)
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "pass", rec
    sr1 = rec["sr"]["SR1"]
    assert sr1["applies"] and abs(sr1["latency_s"] - 0.2) < 1e-9 and abs(sr1["limit_s"] - 0.4) < DT
    assert abs(rec["margin_delta_m"] + 0.2) < 1e-9
    g = rec["sr_scores"]
    assert g["overall"] == min(g["SR1"], g["SR2"], g["SR4"]) == rec["sr_overall"]
    assert abs(g["SR4"] - (0.8 - 0.2) / 0.8) < 1e-9 and g["SR2"] == 1.0


def test_F1_no_switch_and_dangerous():
    fault = trace(drift(1.0, 2.0), fault_at=ONSET)
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "F1_no_or_late_switch" and rec["unsafe"], rec
    assert rec["sr"]["SR1"]["robustness"] == -1.0 and "SR1" in rec["failed_requirements"]


def test_F1_late_switch():
    fault = trace(drift(0.3, 0.3), fault_at=ONSET, switch_at=11.0)  # 1.0 s > 0.5 * FTTI(0.8)
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "F1_no_or_late_switch" and not rec["unsafe"], rec
    assert rec["sr"]["SR1"]["robustness"] < 0


def test_undetected_but_harmless_is_pass_with_note():
    fault = trace(drift(0.2, 0.4), fault_at=ONSET)  # never switches, stays in lane
    rec = evaluate_pair(fault, trace(), ONSET, CFG, harmless_no_reaction())
    assert rec["outcome_category"] == "pass" and rec["note"] == "undetected_harmless", rec
    assert rec["sr"]["SR1"]["applies"] is False


def test_fixed_limit_used_without_no_reaction_run():
    fault = trace(drift(0.2, 0.4), fault_at=ONSET)
    rec = evaluate_pair(fault, trace(), ONSET, CFG)
    assert rec["outcome_category"] == "F1_no_or_late_switch", rec
    assert rec["sr"]["SR1"]["limit_source"] == "fixed_fallback"


def test_F2_steer_jump_at_switch():
    steer = lambda x: 0.0 if x < 10.2 else math.radians(10)  # 10 deg in one tick = 200 deg/s
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2, steer_fn=steer)
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "F2_switch_transient", rec
    assert rec["sr"]["SR2"]["max_rate_dps"] > 60 and rec["sr_overall"] < 0


def test_F3_leaves_lane_after_timely_switch():
    fault = trace(drift(1.0, 1.2), fault_at=ONSET, switch_at=10.2)
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "F3_not_recovered" and rec["unsafe"], rec
    assert rec["P01"] < 0 and rec["sr"]["SR4"]["first_violation_t"] > ONSET


def test_collision_sets_gp_target_floor():
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2, collide_at=12.0)
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "F3_not_recovered" and rec["sr_overall"] == -2.0, rec


def test_route_end_a_few_ticks_apart_is_aligned_not_invalid():
    """BUG-016: both logs end inside the 10 s window (route completed) 0.1 s apart -> cut both at the earlier end."""
    fault = trace(drift(0.3, 0.2), T=17.0, fault_at=ONSET, switch_at=10.2)
    base = trace(T=17.1)
    rec = evaluate_pair(fault, base, ONSET, CFG, dangerous_no_reaction())
    assert rec["sim_status"] == "valid" and rec["outcome_category"] == "pass", rec
    assert abs(rec["evaluation_window"][1] - 16.95) < 1e-9
    assert any("V-08 aligned" in n for n in rec.get("notes") or [])
    short = evaluate_pair(trace(drift(0.3, 0.2), T=14.8, fault_at=ONSET, switch_at=10.2), trace(T=17.1), ONSET, CFG)
    assert short["sim_status"] == "invalid" and "V-03" in short["invalid_reason"]   # minimum window still enforced


def test_switch_before_fault_is_invalid():
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=5.0)
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "invalid" and "V-10" in rec["invalid_reason"], rec


def test_baseline_switch_is_only_recorded():
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2)
    rec = evaluate_pair(fault, trace(switch_at=5.0), ONSET, CFG, dangerous_no_reaction())
    assert rec["outcome_category"] == "pass" and rec["baseline_switch_count"] == 1, rec


def test_gp_targets_measure_how_much_each_sr_broke():
    steer = lambda x: 0.0 if x < 10.6 else math.radians(10)  # 200 deg/s at switch
    fault = trace(drift(1.0, 1.2), fault_at=ONSET, switch_at=10.6, steer_fn=steer)  # late + jump + off lane
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    g, m = rec["sr_scores"], rec["sr_measures"]
    assert g["SR1"] < 0 and abs(m["SR1_latency_s"] - 0.6) < 1e-9          # late by 0.2 s on a 0.4 s limit
    assert g["SR2"] == -2.0 and m["SR2_max_rate_dps"] > 190                # (60-200)/60 = -2.33, clipped
    assert g["SR4"] < 0 and m["SR4_margin_m"] < 0
    assert g["overall"] == min(g["SR1"], g["SR2"], g["SR4"])
    assert set(rec["failed_requirements"]) >= {"SR1", "SR2", "SR4"}


def test_not_applicable_sr_is_null_not_zero():
    rec = evaluate_pair(trace(drift(0.2, 0.4), fault_at=ONSET), trace(), ONSET, CFG, harmless_no_reaction())
    g = rec["sr_scores"]
    assert g["SR1"] is None and g["SR2"] is None and g["SR5"] is None and g["SR4"] is not None


def test_no_trigger():
    fault = trace(fault_at=ONSET, effective=False)
    rec = evaluate_pair(fault, trace(), ONSET, CFG)
    assert rec["outcome_category"] == "no_trigger" and rec["sr"]["SR1"]["applies"] is False, rec


def test_sr5_recovery_is_advisory():
    fault = trace(drift(0.5, 0.5), fault_at=ONSET, switch_at=10.2)  # never back inside 0.3 m band
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["sr"]["SR5"]["pass"] is False and rec["outcome_category"] == "pass", rec


def test_run_that_ends_in_a_crash_is_graded_not_invalid():
    fault = trace(drift(1.0, 2.0), T=12.0, fault_at=ONSET, collide_at=11.9)   # window only 2 s
    rec = evaluate_pair(fault, trace(), ONSET, CFG, dangerous_no_reaction())
    assert rec["sim_status"] == "valid" and rec["unsafe"], rec.get("invalid_reason")
    assert rec["sr_scores"]["SR4"] == -2.0 and rec["sr_measures"]["SR4_collision_t"] == 11.9


def test_collision_without_time_is_invalid():
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2)
    fault.collisions_untimed = 1
    rec = evaluate_pair(fault, trace(), ONSET, CFG)
    assert rec["outcome_category"] == "invalid" and "V-11" in rec["invalid_reason"]


# -------------------------------------------------------------- validity
def test_route_revision_in_window_is_invalid():
    fault = trace(fault_at=ONSET, switch_at=10.2, revision_at=12.0)
    rec = evaluate_pair(fault, trace(), ONSET, CFG)
    assert rec["outcome_category"] == "invalid" and "V-04" in rec["invalid_reason"], rec


def test_unsafe_baseline_rejects_pair():
    rec = evaluate_pair(trace(fault_at=ONSET, switch_at=10.2), trace(lambda x: 1.5), ONSET, CFG)
    assert rec["outcome_category"] == "invalid" and "G-04" in rec["invalid_reason"], rec


def test_missing_geometry_is_invalid():
    cfg = MonitorConfig(dt=DT)
    rec = evaluate_pair(trace(fault_at=ONSET, switch_at=10.2), trace(), ONSET, cfg)
    assert rec["outcome_category"] == "invalid" and "V-05" in rec["invalid_reason"]


def test_fault_in_baseline_is_invalid():
    rec = evaluate_pair(trace(fault_at=ONSET, switch_at=10.2), trace(fault_at=3.0), ONSET, CFG)
    assert "V-07" in rec["invalid_reason"]


def test_noise_floor():
    nf = noise_floor([0.80, 0.79, 0.81, 0.80, 0.80])
    assert nf["n"] == 5 and nf["delta_min_m"] == 0.1


# ============================================================== feedback
def _rec(md, arm, hid="H-01", **kw):
    r = {"sim_status": "valid", "outcome_category": "pass", "margin_delta_m": md,
         "hypothesis_arms": {hid: arm}, "sr_overall": 1 + md / 0.8}
    r.update(kw)
    return r


def test_stage2_supported_at_first_checkpoint():
    rng = np.random.default_rng(1)
    recs = [_rec(-0.5 + 0.02 * rng.standard_normal(), "condition") for _ in range(10)]
    recs += [_rec(0.0 + 0.02 * rng.standard_normal(), "control") for _ in range(10)]
    h = hypothesis_tracking(recs, delta_min_m=0.1)["hypotheses"]["H-01"]
    assert h["status"] == "supported" and h["status_history"][0]["checkpoint"] == 10, h


def test_stage2_refuted_and_testing():
    recs = [_rec(0.0, "condition") for _ in range(10)] + [_rec(0.0, "control") for _ in range(10)]
    assert hypothesis_tracking(recs, delta_min_m=0.1)["hypotheses"]["H-01"]["status"] == "refuted"
    few = recs[:5] + recs[10:15]
    h = hypothesis_tracking(few, delta_min_m=0.1)["hypotheses"]["H-01"]
    assert h["status"] == "testing" and h["next_checkpoint"] == 10


def test_stage2_excludes_invalid_runs():
    recs = [_rec(-0.5, "condition") for _ in range(10)] + [_rec(0.0, "control") for _ in range(10)]
    recs += [{"sim_status": "invalid", "outcome_category": "invalid", "hypothesis_arms": {"H-01": "control"}}]
    h = hypothesis_tracking(recs, delta_min_m=0.1)["hypotheses"]["H-01"]
    assert h["excluded_runs"] == 1 and h["runs_control"] == 10
    assert h["descriptive"]["condition"]["n"] == 10
    assert h["descriptive"]["condition"]["mean"] == -0.5
    assert h["descriptive"]["control"]["mean"] == 0.0
    assert h["descriptive"]["mean_difference"] == -0.5


def test_hypothesis_partial_samples_show_descriptive_stats_without_decision():
    recs = [_rec(-0.2, "condition"), _rec(-0.4, "condition"), _rec(0.0, "control")]
    recs.append(_rec(float('nan'), "control"))
    h = hypothesis_tracking(recs)["hypotheses"]["H-01"]
    assert h['status'] == 'testing' and h['ci'] is None and h['d_estimate'] is None
    assert h['excluded_runs'] == 1 and h['descriptive']['control']['n'] == 1
    assert abs(h['descriptive']['mean_difference'] + 0.3) < 1e-10


def test_stage2_chattering_refute_needs_repeated_switch():
    hyp = {"H-03": {"metric": "switch_count", "direction": "positive",
                    "require_in_condition": {"metric": "switch_count", "min": 3}}}
    recs = [_rec(0, "condition", "H-03", switch_count=1) for _ in range(10)]
    recs += [_rec(0, "control", "H-03", switch_count=1) for _ in range(10)]
    h = hypothesis_tracking(recs, hyp)["hypotheses"]["H-03"]
    assert h["status_history"][0]["status"] == "inconclusive" and h["status_history"][0]["guard"]


def _fail(sid, gp, ftype, onset_event, oc="F1_no_or_late_switch", **kw):
    r = {"sim_status": "valid", "outcome_category": oc, "scenario_id": sid, "run_id": f"{sid}-{gp}",
         "sr_overall": gp, "failed_requirements": ["SR1"],
         "sr_scores": {"SR1": gp, "SR2": None, "SR4": 0.5, "SR5": None, "SR6": 0.9, "overall": gp},
         "sr": {"SR1": {"applies": True, "pass": False, "robustness": gp, "latency_s": None, "limit_s": 0.4}},
         "environment": {"route": "Town04_highway", "speed_mps": 20},
         "fault_plan": {"fault_type": ftype, "magnitude": 0.1, "duration_s": 2.0, "onset_event": onset_event},
         "switch_time": None, "detection_latency_s": None, "switch_count": 0}
    r.update(kw)
    return r


def test_stage3_verified_ranked_diverse_top_k():
    recs = [_fail("S1", -1.0, "bias", "curve_entry"), _fail("S1", -0.9, "bias", "curve_entry"),
            _fail("S2", -0.5, "bias", "curve_entry"), _fail("S2", -0.6, "bias", "curve_entry"),  # same combo
            _fail("S3", -0.3, "drift", "straight"), _fail("S3", -0.2, "drift", "straight"),
            _fail("S4", -1.5, "nan", "curve_apex"),                                          # only once
            _fail("S5", -2.0, "bias", "curve_entry", stress_test=True),
            _fail("S5", -2.0, "bias", "curve_entry", stress_test=True)]
    ft = failure_table(recs, k=5)
    ids = [r["scenario_id"] for r in ft["verified"]]
    assert ids == ["S1", "S3"], ids
    assert ft["candidates"][0]["scenario_id"] == "S4"
    row = ft["verified"][0]
    assert row["reproduction_count"] == 2 and row["how_much"]["SR1"]["switched"] is False
    assert row["environment"]["route"] == "Town04_highway" and "전환 없음" in row["summary"]


# metric_rules as in 02 vulnerability_patterns.yaml (new_pipeline), subset
RULES = {
    "recovery_success": {"support_direction": "negative",
                         "gp": {"target": "recovery_robustness", "form": "absolute_post_switch", "scale": 0.1}},
    "pre_switch_max_lateral_error_m": {"support_direction": "positive",
                                       "gp": {"target": "pre_switch_max_lateral_error_m",
                                              "form": "paired_delta_if_fault_else_absolute", "scale": 0.1}},
    "switch_count": {"support_direction": "positive",
                     "gp": {"target": "switch_count", "form": "paired_delta_if_fault_else_absolute", "scale": 1.0}},
}
INFO = {"run_id": "r1", "scenario_id": "S1", "hypothesis_ids": ["H-01"],
        "environment": {"route": "Town04"}, "arm": "condition",
        "fault_plan": {"fault_type": "bias", "onset_event": "curve_entry", "magnitude": 0.08}}


def test_runner_record_has_exactly_the_runner_fields():
    raw = monitor.evaluate_pair(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG,
                                dangerous_no_reaction(), INFO, RULES)
    assert tuple(raw) == RUN_RECORD_FIELDS
    json.dumps(raw)
    assert raw["outcome_category"] == "switched_safe" and raw["P03"] is True
    assert raw["parameters"]["fault_plan"]["fault_type"] == "bias" and raw["parameters"]["arm"] == "condition"
    assert raw["metrics"]["monitor"]["failure_type"] == "pass"
    bad = monitor.evaluate_pair(trace(fault_at=ONSET, switch_at=10.2, revision_at=12.0), trace(), ONSET, CFG, None, INFO)
    assert tuple(bad) == RUN_RECORD_FIELDS and bad["outcome_category"] == "simulation_invalid"


def test_runner_outcome_labels():
    lab = lambda f, nr=dangerous_no_reaction(): monitor.evaluate_pair(f, trace(), ONSET, CFG, nr)["outcome_category"]
    assert lab(trace(drift(1.0, 2.0), fault_at=ONSET)) == "triggered_no_switch"
    assert lab(trace(drift(1.0, 1.2), fault_at=ONSET, switch_at=10.2)) == "switched_unsafe"
    assert lab(trace(fault_at=ONSET, effective=False)) == "no_trigger"


def test_team_metrics_follow_02_definitions():
    steer = lambda x: 0.0 if x < 10.2 else math.radians(3)
    raw = monitor.evaluate_pair(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2, steer_fn=steer), trace(),
                                ONSET, CFG, dangerous_no_reaction(), INFO, RULES)
    m, b = raw["metrics"], raw["baseline_metrics"]
    lat = 1.75 - 0.95                                              # lane_half_width - vehicle_half_width
    lane_term, rate_term = (lat - 0.2) / lat, (90 - 60) / 90      # 3 deg in one 0.05 s tick = 60 deg/s
    assert abs(m["recovery_robustness"] - min(lane_term, rate_term)) < 1e-9 and m["recovery_success"] is True
    assert abs(m["pre_switch_max_lateral_error_m"] - 0.06) < 1e-9   # |e_y| at 10.2 s on a 0.3 m/s drift
    assert abs(m["command_jump_rad"] - math.radians(3)) < 1e-12
    assert m["switch_count"] == 1 and b["switch_count"] == 0 and b["recovery_robustness"] is None
    assert m["deadline_miss_rate"] is None and m["mpc_constraint_violation_rate"] is None   # not logged -> null
    z = raw["gp_targets"]
    assert abs(z["recovery_success"] - m["recovery_robustness"] / 0.1) < 1e-9
    assert z["switch_count"] == -1.0                                 # -(1 - 0) / 1
    assert abs(z["pre_switch_max_lateral_error_m"] + (0.06 - b["pre_switch_max_lateral_error_m"]) / 0.1) < 1e-9


def test_no_switch_means_recovery_false_and_no_robustness_target():
    raw = monitor.evaluate_pair(trace(drift(1.0, 2.0), fault_at=ONSET), trace(), ONSET, CFG,
                                dangerous_no_reaction(), INFO, RULES)
    assert raw["metrics"]["recovery_success"] is False and raw["metrics"]["recovery_robustness"] is None
    assert "recovery_success" not in raw["gp_targets"]               # undefined, never 0


def test_stage6_rows_use_runner_gp_targets():
    good = monitor.evaluate_pair(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG,
                                 dangerous_no_reaction(), {**INFO, "parameters": {"seed": 3}}, RULES)
    bad = monitor.evaluate_pair(trace(fault_at=ONSET, revision_at=12.0), trace(), ONSET, CFG, None, INFO, RULES)
    rows = gp_training_rows([good, bad])
    assert len(rows) == 1 and rows[0]["gp_targets"] == good["gp_targets"]
    assert rows[0]["x"]["fault.fault_type"] == "bias" and rows[0]["x"]["param.seed"] == 3
    assert "monitor" not in rows[0]["metrics"] and rows[0]["sr_scores"]["SR4"] is not None


def test_stage2_reads_team_metric_from_metrics():
    def run(arm, rob):
        r = monitor.evaluate_pair(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG,
                                  dangerous_no_reaction(), {**INFO, "arm": arm})
        r["metrics"]["recovery_success"] = rob
        return r
    recs = [run("condition", False) for _ in range(10)] + [run("control", True) for _ in range(10)]
    hyp = {"H-01": {"metric": "recovery_success", "direction": "negative", "delta_min": 0.05}}
    h = hypothesis_tracking(recs, hyp)["hypotheses"]["H-01"]
    assert h["status"] == "supported" and h["d_estimate"] == -1.0, h


def test_end_to_end_pair_to_files():
    raw = monitor.evaluate_pair(trace(drift(1.0, 2.0), fault_at=ONSET), trace(), ONSET, CFG,
                                dangerous_no_reaction(), INFO, RULES)
    with tempfile.TemporaryDirectory() as d:
        paths = write_outputs([raw, {**raw, "run_id": "r2"}], d)
        ft = json.loads(open(paths["stage3"]).read())
        assert ft["verified"][0]["scenario_id"] == "S1" and ft["verified"][0]["fault"]["fault_type"] == "bias"
        assert open(paths["stage6"]).read().count("\n") == 2
        first = json.loads(open(paths["run_records"]).readline())
        assert tuple(first) == RUN_RECORD_FIELDS


def test_dashboard_one_page_latest_left_all_right():
    from dashboard import refresh
    info = {"run_id": "r1", "scenario_id": "S1", "environment": {"route": "Town04"},
            "fault_plan": {"fault_type": "bias", "onset_event": "curve_entry", "magnitude": 0.08}}
    rec = monitor.evaluate_pair(trace(drift(1.0, 2.0), fault_at=ONSET), trace(), ONSET, CFG,
                                dangerous_no_reaction(), info)
    harmless = monitor.evaluate_pair(trace(drift(0.2, 0.4), fault_at=ONSET), trace(), ONSET, CFG,
                                     harmless_no_reaction(), {**info, "run_id": "r2"})
    bad = monitor.evaluate_pair(trace(fault_at=ONSET, revision_at=12.0), trace(), ONSET, CFG, None,
                                {**info, "run_id": "r3"})
    with tempfile.TemporaryDirectory() as d:
        recs = os.path.join(d, "rr.jsonl")
        with open(recs, "w", encoding="utf-8") as fh:
            for r in (rec, harmless, bad):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        page = refresh(recs, os.path.join(d, "screen")).read_text(encoding="utf-8")
        assert 'http-equiv="refresh"' in page and 'data-latest="r3"' in page
        assert page.count('class="run"') == 3 and page.count("<tr data-id=") == 3
        assert "F1 전환 실패" in page and "통과 (관측 창 내 전환 불필요)" in page and "V-04" in page
        assert not os.path.exists(os.path.join(d, "screen", "runs"))
        empty = refresh(os.path.join(d, "none.jsonl"), os.path.join(d, "s2")).read_text(encoding="utf-8")
        assert "아직 끝난 시나리오가 없습니다" in empty


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
