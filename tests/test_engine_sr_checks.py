"""engine/sr_checks.py: the teammate's SR1/SR2/SR4/SR6 grading (ported from junwoo/sim-eval a5242b2
pipeline/09_monitor/test_monitor.py) + its wiring into engine.monitor.evaluate and engine.dashboard.

Changes against the original tests (overlaps now follow OUR implementation):
  * SR5 is derived from recovery v2 (recovery_outcome), so the old "SR5 advisory band" test checks the derived view.
  * F3 is decided by final_safe_recovery when the recovery result is passed in (SR4-only fallback otherwise).
  * The window follows our safety_spec evaluation_window (onset .. simulation end), not onset + 10 s.
  * P-02 aux (0.5*mu*g) / team metrics / runner_skeleton schema / stage 2-3-6 feedback tests are dropped
    (replaced by engine.judgment + engine.feedback, tested in test_engine_judgment / test_engine_monitor).
"""
import json
import math

import numpy as np
import pytest

from engine import sr_checks as SR
from engine.config import load_yaml, PIPELINE_DIR
from engine.sr_checks import SRConfig, Trace, footprint_lane_margin, lateral_acceleration, measure_ftti, noise_floor

DT = 0.05
CFG = SRConfig(dt=DT, lane_width_default_m=3.5, vehicle_half_width_m=0.95, vehicle_half_length_m=2.4,
               max_steer_rad=math.radians(70))
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


def grade(*a, **k):
    return SR.grade(*a, **k)


def invalid_text(g):
    return "; ".join(g["invalid_reasons"])


# -------------------------------------------------------------- geometry
def test_centered_margin_equals_lane_half_width_minus_half_width():
    assert abs(footprint_lane_margin([0.0], [0.0], [3.5], 0.95, 2.4)[0] - 0.8) < 1e-12


def test_heading_error_reduces_margin_through_the_corners():
    m1 = footprint_lane_margin([0.0], [math.radians(5)], [3.5], 0.95, 2.4)[0]
    expected = 1.75 - (2.4 * math.sin(math.radians(5)) + 0.95 * math.cos(math.radians(5)))
    assert abs(m1 - expected) < 1e-12


def test_lateral_acceleration_from_yaw_rate():
    assert np.allclose(lateral_acceleration(trace(vx=15.0, r=0.2)), 3.0)


def test_config_reads_our_safety_spec():
    spec = load_yaml(PIPELINE_DIR / "09_monitor" / "safety_spec.yaml")
    cfg = SR.config_from_spec(spec, lane_width_default_m=3.5)
    assert cfg.sr2_limit_dps == 60.0 and cfg.sr1_ftti_fraction == 0.5 and cfg.sr6_max_rate_dps == 90.0
    assert cfg.window_s == 10.0 and cfg.min_window_s == 5.0     # teammate: onset + 10 s, min 5 s
    assert cfg.sr1_fixed_limit_s == 0.5
    assert cfg.gp_clip == (-2.0, 1.0) and cfg.lane_width_default_m == 3.5


# ------------------------------------------------------------------ FTTI
def test_ftti_finite_when_no_reaction_leaves_lane():
    f = measure_ftti(dangerous_no_reaction(), ONSET, CFG)
    assert f["valid"] and abs(f["ftti_s"] - 0.8) < DT + 1e-9, f


def test_ftti_infinite_when_no_reaction_stays_in_lane():
    assert math.isinf(measure_ftti(harmless_no_reaction(), ONSET, CFG)["ftti_s"])


# -------------------------------------------------------------- outcomes
def test_pass_quick_smooth_switch():
    g = grade(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG, dangerous_no_reaction())
    assert g["sr_outcome"] == "pass", g
    sr1 = g["sr"]["SR1"]
    assert sr1["applies"] and abs(sr1["latency_s"] - 0.2) < 1e-9 and abs(sr1["limit_s"] - 0.4) < DT
    assert abs(g["sr_measures"]["SR4_margin_delta_m"] + 0.2) < 1e-9
    s = g["sr_scores"]
    assert s["overall"] == min(s["SR1"], s["SR2"], s["SR4"])
    assert abs(s["SR4"] - (0.8 - 0.2) / 0.8) < 1e-9 and s["SR2"] == 1.0
    assert g["detection_latency_s"] == pytest.approx(0.2) and g["ftti_s"] == pytest.approx(0.8, abs=DT)


def test_F1_no_switch_and_dangerous():
    g = grade(trace(drift(1.0, 2.0), fault_at=ONSET), trace(), ONSET, CFG, dangerous_no_reaction())
    assert g["sr_outcome"] == "F1_no_or_late_switch" and not g["sr"]["SR4"]["pass"], g
    assert g["sr"]["SR1"]["robustness"] == -1.0 and "SR1" in g["failed_requirements"]
    assert g["sr_labels"]["late_or_no_switch"] is True


def test_F1_late_switch():
    g = grade(trace(drift(0.3, 0.3), fault_at=ONSET, switch_at=11.0), trace(), ONSET, CFG, dangerous_no_reaction())
    assert g["sr_outcome"] == "F1_no_or_late_switch" and g["sr"]["SR4"]["pass"], g
    assert g["sr"]["SR1"]["robustness"] < 0


def test_undetected_but_harmless_is_pass_with_note():
    g = grade(trace(drift(0.2, 0.4), fault_at=ONSET), trace(), ONSET, CFG, harmless_no_reaction())
    assert g["sr_outcome"] == "pass" and g["note"] == "undetected_harmless", g
    assert g["sr"]["SR1"]["applies"] is False and g["ftti_s"] is None


def test_fixed_limit_used_without_no_reaction_run():
    g = grade(trace(drift(0.2, 0.4), fault_at=ONSET), trace(), ONSET, CFG)
    assert g["sr_outcome"] == "F1_no_or_late_switch", g
    assert g["sr"]["SR1"]["limit_source"] == "fixed_fallback"


def test_F2_steer_jump_at_switch():
    steer = lambda x: 0.0 if x < 10.2 else math.radians(10)  # 10 deg in one tick = 200 deg/s
    g = grade(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2, steer_fn=steer), trace(), ONSET, CFG,
              dangerous_no_reaction())
    assert g["sr_outcome"] == "F2_switch_transient", g
    assert g["sr"]["SR2"]["max_rate_dps"] > 60 and g["sr_scores"]["overall"] < 0
    assert g["sr_labels"]["switch_transient"] is True


def test_F3_leaves_lane_after_timely_switch_sr4_fallback():
    g = grade(trace(drift(1.0, 1.2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG, dangerous_no_reaction())
    assert g["sr_outcome"] == "F3_not_recovered" and not g["sr"]["SR4"]["pass"], g
    assert g["sr_measures"]["SR4_margin_m"] < 0 and g["sr"]["SR4"]["first_violation_t"] > ONSET


def test_F3_follows_our_recovery_v2():
    """Inside the lane by SR4, but recovery v2 says unsettled / not final-safe -> F3 (ours wins over SR5 band)."""
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2)
    g = grade(fault, trace(), ONSET, CFG, dangerous_no_reaction(),
              recovery={"recovery_outcome": "unsettled", "final_safe_recovery": False})
    assert g["sr_outcome"] == "F3_not_recovered" and g["sr"]["SR4"]["pass"]
    assert g["sr"]["SR5"] == {"applies": True, "derived_from": "engine.judgment recovery v2", "pass": False,
                              "recovery_outcome": "unsettled", "final_safe_recovery": False}
    ok = grade(fault, trace(), ONSET, CFG, dangerous_no_reaction(),
               recovery={"recovery_outcome": "safe_recovery", "final_safe_recovery": True})
    assert ok["sr_outcome"] == "pass" and ok["sr"]["SR5"]["pass"] is True


def test_collision_sets_score_floor():
    g = grade(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2, collide_at=12.0), trace(), ONSET, CFG,
              dangerous_no_reaction())
    assert g["sr_outcome"] == "F3_not_recovered" and g["sr_scores"]["overall"] == -2.0, g


def test_switch_before_fault_is_invalid():
    g = grade(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=5.0), trace(), ONSET, CFG, dangerous_no_reaction())
    assert g["sr_outcome"] == "invalid" and "V-10" in invalid_text(g), g


def test_baseline_switch_is_only_recorded():
    g = grade(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2), trace(switch_at=5.0), ONSET, CFG,
              dangerous_no_reaction())
    assert g["sr_outcome"] == "pass" and g["baseline_switch_count"] == 1, g


def test_scores_measure_how_much_each_sr_broke():
    steer = lambda x: 0.0 if x < 10.6 else math.radians(10)
    g = grade(trace(drift(1.0, 1.2), fault_at=ONSET, switch_at=10.6, steer_fn=steer), trace(), ONSET, CFG,
              dangerous_no_reaction())
    s, m = g["sr_scores"], g["sr_measures"]
    assert s["SR1"] < 0 and abs(m["SR1_latency_s"] - 0.6) < 1e-9
    assert s["SR2"] == -2.0 and m["SR2_max_rate_dps"] > 190
    assert s["SR4"] < 0 and m["SR4_margin_m"] < 0
    assert s["overall"] == min(s["SR1"], s["SR2"], s["SR4"])
    assert set(g["failed_requirements"]) >= {"SR1", "SR2", "SR4"}


def test_not_applicable_sr_is_null_not_zero():
    g = grade(trace(drift(0.2, 0.4), fault_at=ONSET), trace(), ONSET, CFG, harmless_no_reaction())
    s = g["sr_scores"]
    assert s["SR1"] is None and s["SR2"] is None and s["SR4"] is not None
    assert g["sr"]["SR5"]["applies"] is False and g["sr_labels"]["switch_transient"] is None


def test_no_trigger():
    g = grade(trace(fault_at=ONSET, effective=False), trace(), ONSET, CFG)
    assert g["sr_outcome"] == "no_trigger" and g["sr"]["SR1"]["applies"] is False, g


def test_sr6_actuator_limit():
    steer = lambda x: 0.0 if x < 10.2 else min(math.radians(80), math.radians(5) * (x - 10.2) / DT)
    g = grade(trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2, steer_fn=steer), trace(), ONSET, CFG,
              dangerous_no_reaction())
    assert g["sr"]["SR6"]["pass"] is False and g["sr_labels"]["actuator_limit_exceeded"] is True
    assert "SR6" in g["failed_requirements"]


def test_run_that_ends_in_a_crash_is_graded_not_invalid():
    g = grade(trace(drift(1.0, 2.0), T=12.0, fault_at=ONSET, collide_at=11.9), trace(), ONSET, CFG,
              dangerous_no_reaction())
    assert g["valid"], invalid_text(g)
    assert g["sr_scores"]["SR4"] == -2.0 and g["sr_measures"]["SR4_collision_t"] == 11.9


def test_collision_without_time_is_invalid():
    fault = trace(drift(0.3, 0.2), fault_at=ONSET, switch_at=10.2)
    fault.collisions_untimed = 1
    g = grade(fault, trace(), ONSET, CFG)
    assert g["sr_outcome"] == "invalid" and "V-11" in invalid_text(g)


# -------------------------------------------------------------- validity
def test_route_revision_in_window_is_invalid():
    g = grade(trace(fault_at=ONSET, switch_at=10.2, revision_at=12.0), trace(), ONSET, CFG)
    assert g["sr_outcome"] == "invalid" and "V-04" in invalid_text(g), g


def test_unsafe_baseline_rejects_pair():
    g = grade(trace(fault_at=ONSET, switch_at=10.2), trace(lambda x: 1.5), ONSET, CFG)
    assert g["sr_outcome"] == "invalid" and "G-04" in invalid_text(g), g


def test_missing_geometry_is_invalid():
    g = grade(trace(fault_at=ONSET, switch_at=10.2), trace(), ONSET, SRConfig(dt=DT))
    assert g["sr_outcome"] == "invalid" and "V-05" in invalid_text(g)


def test_fault_in_baseline_is_invalid():
    g = grade(trace(fault_at=ONSET, switch_at=10.2), trace(fault_at=3.0), ONSET, CFG)
    assert "V-07" in invalid_text(g)


def test_window_is_onset_plus_10s_for_fault_and_baseline():
    """Teammate's original: both runs use [onset, onset + 10 s]; a longer baseline log is fine."""
    g = grade(trace(drift(0.3, 0.2), T=25.0, fault_at=ONSET, switch_at=10.2), trace(T=60.0), ONSET, CFG,
              dangerous_no_reaction())
    assert g["valid"] and g["evaluation_window"][1] == pytest.approx(ONSET + 10.0), invalid_text(g)


def test_fault_run_shorter_than_baseline_window_is_v08():
    """Fault log ends 7 s after onset (>= 5 s min) but the baseline has the full 10 s -> V-08 (original)."""
    g = grade(trace(drift(0.3, 0.2), T=17.0, fault_at=ONSET, switch_at=10.2), trace(T=60.0), ONSET, CFG,
              dangerous_no_reaction())
    assert g["sr_outcome"] == "invalid" and "V-08" in invalid_text(g), invalid_text(g)


def test_no_baseline_is_not_graded():
    g = grade(trace(fault_at=ONSET, switch_at=10.2), None, ONSET, CFG)
    assert g["valid"] is False and "baseline" in invalid_text(g)


def test_noise_floor():
    nf = noise_floor([0.80, 0.79, 0.81, 0.80, 0.80])
    assert nf["n"] == 5 and nf["delta_min_m"] == 0.1


# ============================================================ our run folders
@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    from engine.fixtures.synthetic_sim import write_run_folder
    d = tmp_path_factory.mktemp("sr")
    fault = dict(mode="saturate", start=20.0, duration=500.0, magnitude=1.0)
    base = write_run_folder(d / "base", route="bench2drive:26956", max_speed_kmh=30, fault=None, record_front=False)
    run = write_run_folder(d / "run", route="bench2drive:26956", max_speed_kmh=30, fault=fault,
                           stop_after_fault_s=15, record_front=False)
    nore = write_run_folder(d / "nore", route="bench2drive:26956", max_speed_kmh=30, fault=fault,
                            stop_after_fault_s=15, record_front=False, controller="mpc_unprotected")
    return base, run, nore


def test_trace_adapter_reads_our_run_folder(folders):
    from engine.monitor import load_av_stack_run
    tr = SR.trace_from_run(load_av_stack_run(folders[1]))
    assert tr.steer_cmd_rad is not None and tr.max_steer_rad and tr.half_width_m and tr.lane_width_m is not None
    assert tr.fault_active is not None and tr.fault_active.any() and "lqr" in tr.active_controller
    nr = SR.trace_from_run(load_av_stack_run(folders[2]))
    assert set(nr.active_controller) == {"mpc"}       # no supervisor


def test_evaluate_adds_sr_fields_to_our_run_record(folders):
    from engine import judgment as J
    from engine.monitor import evaluate
    base, run, nore = folders
    rec = evaluate(run, baseline_dir=base, no_reaction_dir=nore)
    assert set(J.RUN_RECORD_FIELDS) <= set(rec)
    for k in ("recovery_outcome", "recovery_labels", "final_safe_recovery", "lane_recovery_robustness",
              "lat_acc_excess_mps2", "gp_targets"):
        assert k in rec
    assert rec["sr_checks"]["valid"], rec["sr_checks"]["invalid_reasons"]
    assert rec["sr_outcome"] in SR.SR_OUTCOMES
    assert rec["sr_checks"]["sr"]["SR1"]["limit_source"] == "ftti_measured" and rec["ftti_s"] is not None
    assert rec["sr_checks"]["sr"]["SR5"]["recovery_outcome"] == rec["recovery_outcome"]
    if rec["final_safe_recovery"] is False and rec["sr_outcome"] not in ("F1_no_or_late_switch", "F2_switch_transient"):
        assert rec["sr_outcome"] == "F3_not_recovered"
    # SR checks never change our labels
    from engine.monitor import compute_run_record, load_av_stack_run, thresholds_from_config
    from engine.config import load_stage_configs
    cfg = load_stage_configs()
    plain = compute_run_record(load_av_stack_run(run), load_av_stack_run(base), th=thresholds_from_config(cfg),
                               metric_rules=cfg["vulnerability_patterns"]["metric_rules"])
    for k in ("recovery_outcome", "final_safe_recovery", "outcome_category", "gp_targets"):
        assert rec[k] == plain[k]
    json.dumps(rec, default=str)


def test_monitor_cli_writes_jsonl(folders, tmp_path, capsys):
    from engine.monitor import main
    base, run, nore = folders
    out = tmp_path / "rr.jsonl"
    assert main([str(run), "--baseline", str(base), "--no-reaction", str(nore), "--out", str(out)]) == 0
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["sr_outcome"] and "_timeline" in rec


def test_dashboard_one_page_latest_left_all_right(folders, tmp_path):
    from engine.dashboard import refresh
    from engine.monitor import evaluate
    base, run, nore = folders
    a = evaluate(run, baseline_dir=base, no_reaction_dir=nore, run_id="r1")
    b = evaluate(run, baseline_dir=base, run_id="r2")
    c = evaluate(tmp_path / "missing", baseline_dir=base, run_id="r3")
    recs = tmp_path / "rr.jsonl"
    recs.write_text("".join(json.dumps(r, default=str, ensure_ascii=False) + "\n" for r in (a, b, c)), encoding="utf-8")
    page = refresh(recs, tmp_path / "screen").read_text(encoding="utf-8")
    assert 'http-equiv="refresh"' in page and 'data-latest="r3"' in page
    assert page.count('class="run"') == 3 and page.count("<tr data-id=") == 3
    assert "무효" in page and "recovery v2" in page
    empty = refresh(tmp_path / "none.jsonl", tmp_path / "s2").read_text(encoding="utf-8")
    assert "아직 끝난 시나리오가 없습니다" in empty
