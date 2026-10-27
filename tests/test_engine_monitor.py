"""09 monitor on SYNTHETIC run folders (engine/fixtures/synthetic_sim.py)."""
import json

import pytest

from engine import judgment as J
from engine.config import load_stage_configs
from engine.fixtures.synthetic_sim import write_run_folder
from engine.monitor import compute_run_record, load_av_stack_run, monitor_timeline


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    base = write_run_folder(d / "base", route="town10_long", max_speed_kmh=30, fault=None, record_front=False)
    small = write_run_folder(d / "solver", route="town10_long", max_speed_kmh=20,
                             fault=dict(mode="solver_fail", start=12.0, duration=5.0, magnitude=0.05),
                             stop_after_fault_s=15, record_front=False)
    big = write_run_folder(d / "sat", route="bench2drive:26956", max_speed_kmh=30,
                           fault=dict(mode="saturate", start=20.0, duration=500.0, magnitude=1.0),
                           stop_after_fault_s=15, record_front=False)
    return base, small, big


def test_loader_joins_files_and_marks_synthetic(runs):
    run = load_av_stack_run(runs[1])
    assert run.synthetic and not run.missing_files
    t = run.ticks[10]
    for k in ("vx", "e_y_lane", "active_controller", "lane_offset_m", "ctrl_steer"):
        assert k in t
    assert run.dt == pytest.approx(0.05)


def test_record_has_every_run_record_field(runs):
    base, small, _ = runs
    rules = load_stage_configs()["vulnerability_patterns"]["metric_rules"]
    rec = compute_run_record(load_av_stack_run(small), load_av_stack_run(base), run_id="t", metric_rules=rules)
    missing = [f for f in J.RUN_RECORD_FIELDS if f not in rec]
    assert not missing
    assert rec["switch_time"] is not None and rec["trigger_source"] == "solver_failure"
    assert rec["recovery_outcome"] in ("safe_recovery", "unsettled", "recovered_after_departure")
    assert "recovery_success" in rec["gp_targets"]


def test_baseline_has_no_switch(runs):
    rec = compute_run_record(load_av_stack_run(runs[0]))
    assert rec["switch_time"] is None and rec["recovery_outcome"] == "no_switch"


def test_departure_or_collision_is_failure(runs):
    base, _, big = runs
    rec = compute_run_record(load_av_stack_run(big), load_av_stack_run(base))
    assert rec["final_safe_recovery"] is False
    assert rec["recovery_outcome"] in ("unrecovered", "recovered_after_departure")
    tl = monitor_timeline(load_av_stack_run(big), rec)
    assert tl["recovery_outcome_decided_by_monitor"] == rec["recovery_outcome"]
    assert tl["lane_departure_intervals_s"]


def test_missing_files_is_invalid(tmp_path):
    (tmp_path / "steps.jsonl").write_text(json.dumps({"t": 0.0}) + "\n")
    rec = compute_run_record(load_av_stack_run(tmp_path))
    assert rec["recovery_outcome"] == "simulation_invalid" and rec["sim_status"] == "invalid"
