"""engine/sr_checks.py: adapter from the engine to the SR monitor v3.1 (pipeline/09_monitor/monitor.py).

The grading rules themselves are tested next to the monitor (pipeline/09_monitor/test_monitor.py, test_spec_v3.py).
Here: the adapter's field mapping, the trace adapter on our run folders, and the wiring into engine.monitor.evaluate
and engine.dashboard.
"""
import json
import math

import numpy as np
import pytest

from engine import sr_checks as SR
from engine.config import load_yaml, PIPELINE_DIR

DT = 0.05
CFG = SR.MonitorConfig(dt=DT, lane_width_default_m=3.5, vehicle_half_width_m=0.95, vehicle_half_length_m=2.4,
                       max_steer_rad=math.radians(70))
ONSET = 10.0


def trace(offset_fn=lambda x: 0.0, T=30.0, fault_at=None, switch_at=None, steer_fn=lambda x: 0.0, collide_at=None):
    t = np.round(np.arange(0.0, T, DT), 6)
    ac = ["lqr" if switch_at is not None and x >= switch_at else "mpc" for x in t]
    fa = np.zeros(len(t), bool) if fault_at is None else (t >= fault_at)
    return SR.Trace(t=t, vx=np.full(len(t), 15.0), vy=np.zeros(len(t)), yaw_rate=np.zeros(len(t)),
                    lane_offset_m=np.array([offset_fn(x) for x in t]), lane_heading_err=np.zeros(len(t)),
                    active_controller=ac, steer_cmd_rad=np.array([steer_fn(x) for x in t]), fault_active=fa,
                    fault_effective=fa.copy(), collisions_t=[] if collide_at is None else [collide_at])


def drift(rate, cap, start=ONSET):
    return lambda x: 0.0 if x < start else min(cap, rate * (x - start))


def test_config_reads_the_v31_sections_of_our_safety_spec():
    spec = load_yaml(PIPELINE_DIR / "09_monitor" / "safety_spec.yaml")
    cfg = SR.config_from_spec(spec, dt=DT)
    assert cfg.sr2_limit_dps == spec["requirements"]["SR2"]["limit_dps"]
    assert cfg.window_s == spec["evaluation_window"]["length_s"] and cfg.p02_paired
    assert "sr_checks" not in spec          # old Monitor v2 numbers are gone


def test_pass_maps_to_sr_fields():
    g = SR.grade(trace(drift(.3, .2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG)
    assert g["valid"] and g["sr_outcome"] == "pass" and g["monitor"] == "sr_monitor_v3.1"
    assert g["sr_labels"] == {"late_or_no_switch": False, "switch_transient": False, "footprint_departure": False,
                              "actuator_limit_exceeded": False}
    assert g["aux"]["P02_lat_acc"]["evaluation"] == "paired_baseline" and g["diagnostics"]["role"] == "report_only"


def test_no_switch_on_a_dangerous_fault_is_F1():
    g = SR.grade(trace(drift(1.0, 2.0), fault_at=ONSET), trace(), ONSET, CFG)
    assert g["sr_outcome"] == "F1_no_or_late_switch" and g["sr_labels"]["late_or_no_switch"]
    assert g["sr_labels"]["footprint_departure"] and "SR4" in g["failed_requirements"]


def test_recovery_v2_is_recorded_next_to_sr5_without_changing_it():
    g = SR.grade(trace(drift(.3, .2), fault_at=ONSET, switch_at=10.2), trace(), ONSET, CFG,
                 recovery={"recovery_outcome": "unrecovered", "final_safe_recovery": False})
    assert g["sr_outcome"] == "pass"
    assert g["sr"]["SR5"]["recovery_v2"]["recovery_outcome"] == "unrecovered"


def test_invalid_and_missing_baseline_are_not_graded():
    g = SR.grade(trace(fault_at=ONSET, switch_at=5.0), trace(), ONSET, CFG)   # switched before the fault
    assert not g["valid"] and g["sr_outcome"] == "invalid" and any("V-10" in r for r in g["invalid_reasons"])
    g = SR.grade(trace(fault_at=ONSET), None, ONSET, CFG)
    assert not g["valid"] and "no baseline" in g["invalid_reasons"][0]


def test_noise_floor():
    assert SR.noise_floor([0.8])["delta_min_m"] is None
    r = SR.noise_floor([0.80, 0.82, 0.78, 0.81, 0.79])
    assert r["delta_min_m"] == max(2 * r["noise_floor_m"], 0.1)


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
    sr5 = rec["sr_checks"]["sr"].get("SR5") or {}
    if sr5.get("applies"):
        assert sr5["recovery_v2"]["recovery_outcome"] == rec["recovery_outcome"]
    assert rec["sr_checks"]["monitor"] == "sr_monitor_v3.1"
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
    assert page.count('class="run"') == 3 and page.count("<tr class=") == 3
    assert ("Invalid" in page or "Not graded" in page) and "Driving Safety" in page
    empty = refresh(tmp_path / "none.jsonl", tmp_path / "s2").read_text(encoding="utf-8")
    assert "No finished scenarios yet" in empty
