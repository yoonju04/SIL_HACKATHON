"""integration3 #8: tighter av_stack supervisor (0.6 m / 0.35 rad / 0.15 s + 90 deg/s MPC command rate check).

av_stack side: pipeline/08_carla_paired_run/av_stack/tests/test_supervisor.py (cherry-picked d26144d).
Here: the mock simulator follows the same profiles, the runner passes --supervisor, the monitor and --gp-history use
the run's profile, and the rate check never fires on the recorded fault-free CARLA baselines (result_1..6)."""
import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from engine import scenarios as SC
from engine.config import REPO_ROOT, load_stage_configs, load_yaml, resolve_scenario_space
from engine.fixtures import synthetic_sim as SIM
from engine.fixtures.synthetic_sim import write_run_folder
from engine.monitor import compute_run_record, load_av_stack_run

AV_STACK = REPO_ROOT / "pipeline" / "08_carla_paired_run" / "av_stack"


def test_mock_profiles_match_av_stack():
    code = ("import json,sys; sys.path.insert(0, '.'); from integration.supervised_lateral import SUPERVISOR_PROFILES as P;"
            "print(json.dumps({k: [v.max_lateral_error_m, v.max_heading_error_rad, v.divergence_hold_s, v.max_steer_rate_dps]"
            " for k, v in P.items()}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=AV_STACK, capture_output=True, text=True, check=True).stdout
    av = json.loads(out)
    mock = {k: [v["e_y_m"], v["e_psi_rad"], v["hold_s"], v["max_steer_rate_dps"]] for k, v in SIM.SUPERVISOR_PROFILES.items()}
    assert av == mock
    space = load_yaml(resolve_scenario_space(False)[0])
    td = space["supervisor"]["tracking_divergence"]
    assert [td["e_y_m"], td["e_psi_rad"], td["hold_s"]] == av["tight"][:3]
    assert space["supervisor"]["steer_rate_exceeded"]["max_dps"] == av["tight"][3]
    assert load_stage_configs()["paired_runs"]["runner"]["supervisor"] == "tight"


def _first_switch(rows):
    return next(((r["switched_at"], r["reason"]) for r in rows if r["active"] == "lqr"), (None, None))


def test_mock_saturate_step_trips_rate_check_at_onset_legacy_does_not():
    fault = dict(mode="saturate", start=20.0, duration=0.6, magnitude=1.0, level=1.0, ramp_s=0.0)
    t, why = _first_switch(SIM.simulate("town10_long", 30.0, fault, supervisor="tight")["rows"])
    assert why == "steer_rate_exceeded" and t == pytest.approx(20.0, abs=0.051)
    t_l, why_l = _first_switch(SIM.simulate("town10_long", 30.0, fault, supervisor="legacy")["rows"])
    assert why_l in (None, "tracking_divergence") and (t_l is None or t_l > t + 0.2)


def test_mock_slow_saturate_ramp_and_drift_are_not_rate_events():
    # ramp 0.3 s to 0.2 x max steer: about 0.14 rad / 0.3 s = 27 deg/s (+ MPC) < 90 deg/s; drift <= 0.05 rad/s
    # (duration long: the END of a saturate is a one-tick step back to the MPC command and does trip the check)
    ramp = dict(mode="saturate", start=20.0, duration=500.0, magnitude=1.0, level=0.2, ramp_s=0.3)
    drift = dict(mode="drift", start=20.0, duration=500.0, magnitude=0.05)
    for f in (ramp, drift):
        _, why = _first_switch(SIM.simulate("town10_long", 30.0, f, supervisor="tight")["rows"])
        assert why != "steer_rate_exceeded", f


def test_mock_tight_bias_switches_no_later_than_legacy():
    f = dict(mode="bias", start=20.0, duration=500.0, magnitude=0.05)
    t_t, _ = _first_switch(SIM.simulate("bench2drive:28035", 30.0, f, supervisor="tight")["rows"])
    t_l, _ = _first_switch(SIM.simulate("bench2drive:28035", 30.0, f, supervisor="legacy")["rows"])
    assert t_t is not None and (t_l is None or t_t <= t_l)


def test_mock_fault_free_runs_no_rate_false_positive():
    # false-positive check on the synthetic routes (curve entries included), all speeds / friction corners.
    # Default tyres: never a switch. Low friction: the SYNTHETIC car understeers off the path (legacy switches there
    # too); tight may switch earlier, but only by tracking_divergence after a real > 0.6 m / 0.35 rad error.
    for route in SIM.ROUTES:
        for kmh in (10, 25, 40):
            for fr in (0.3, 0.5, 1.0):
                rows = SIM.simulate(route, kmh, None, tire_friction_scale=fr, supervisor="tight")["rows"]
                t, why = _first_switch(rows)
                if fr == 1.0:
                    assert t is None, (route, kmh, why)
                elif t is not None:
                    assert why == "tracking_divergence", (route, kmh, fr, why)
                    pre = [r for r in rows if r["t"] <= t]
                    assert max(max(abs(r["e_y"]) - 0.6, abs(r["e_psi"]) - 0.35) for r in pre) > 0


def _replay_tight(controls, max_steer):
    """av_stack MPCSupervisor tight rules on logged MPC rows: (reason, t) of the first fire or None."""
    bad = prev = None
    for r in controls:
        if r.get("active_controller") != "mpc":
            break
        if prev is not None and r["t"] > prev["t"]:
            if abs(r["steer"] - prev["steer"]) * max_steer / (r["t"] - prev["t"]) > math.radians(90.0) + 1e-9:
                return "steer_rate_exceeded", r["t"]
        if abs(r.get("tracking_error") or 0) > 0.6 or abs(r.get("heading_error") or 0) > 0.35:
            bad = r["t"] if bad is None else bad
            if r["t"] - bad >= 0.15 - 1e-9:
                return "tracking_divergence", r["t"]
        else:
            bad = None
        prev = r
    return None


def test_tight_rules_never_fire_on_recorded_fault_free_carla_baselines():
    dirs = sorted(glob.glob(str(REPO_ROOT / "result" / "result_*" / "runs" / "baselines" / "*" / "agent_controls.jsonl")))
    if not dirs:
        pytest.skip("no recorded baselines in result/")
    seen = set()
    for f in dirs:
        d = Path(f).parent
        if d.name in seen:
            continue
        seen.add(d.name)
        cfg = json.loads((d / "agent_config.json").read_text())
        if cfg.get("mpc_fault") not in (None, "none"):
            continue
        rows = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        assert _replay_tight(rows, cfg["geometry"]["max_steer_rad"]) is None, d
    assert len(seen) >= 40   # result_1..6: 44 distinct baselines


def test_runner_passes_supervisor_and_mock_records_it(tmp_path):
    from engine.carla_runner import CarlaRunner, MockRunner
    space = load_yaml(resolve_scenario_space(False)[0])
    routes = load_stage_configs()["paired_runs"]["carla"]["routes"]
    c = SC.make_candidate(space, route_id="town10_long", speed=30, fault_id="F-04", params={"magnitude_rad": 0.05},
                          onset_event="straight", offset=0.0, source="t")
    cmd = CarlaRunner(tmp_path, routes, 17, space, 15.0, "tight", av_stack=tmp_path, python_exe="python").command(c, tmp_path / "x", None)
    assert cmd[cmd.index("--supervisor") + 1] == "tight"
    assert "--supervisor" not in CarlaRunner(tmp_path, routes, 17, space, 15.0, av_stack=tmp_path, python_exe="python").command(c, tmp_path / "y", None)
    d, _ = MockRunner(tmp_path / "m", routes, 17, space, 15.0, "legacy").ensure_baseline(c)
    assert json.loads((d / "agent_config.json").read_text())["supervisor"] == "legacy"


def test_monitor_normalizes_switch_error_by_the_runs_profile(tmp_path):
    f = dict(mode="bias", start=20.0, duration=500.0, magnitude=0.08)
    recs = {}
    for prof in ("tight", "legacy"):
        d = write_run_folder(tmp_path / prof, route="town10_long", max_speed_kmh=30, fault=f, stop_after_fault_s=10,
                             record_front=False, supervisor=prof)
        recs[prof] = compute_run_record(load_av_stack_run(d))
        assert recs[prof]["supervisor_profile"] == prof
    for prof, (ey, ep) in {"tight": (0.6, 0.35), "legacy": (1.0, 0.6)}.items():
        ss = recs[prof]["switch_state"]
        if ss and ss.get("normalized_switch_error") is not None:
            assert ss["normalized_switch_error"] == pytest.approx(max(abs(ss["e_y_at_switch"]) / ey, abs(ss["e_psi_at_switch"]) / ep))


def test_gp_history_skips_runs_from_another_supervisor(tmp_path):
    from run import Loop
    base = {"sim_status": "ok", "gp_targets": {"lane_recovery_robustness": 0.3}, "switch_time": 21.0,
            "parameters": {"route_id": "town10_long", "environment": {"max_speed_kmh": 30.0}, "onset_event": "straight",
                           "onset_offset_s": 0.0, "fault": {"fault_id": "F-04", "mode": "bias", "params": {"magnitude_rad": 0.05}}}}
    hist = tmp_path / "h.jsonl"
    hist.write_text("\n".join(json.dumps(dict(base, **extra)) for extra in ({}, {"supervisor_profile": "legacy"},
                                                                              {"supervisor_profile": "tight"})))
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=4, batch_size=2,
                rounds=0, quiet=True)
    assert loop.supervisor == "tight"
    loop.load_gp_history([str(hist)])
    assert loop.stats["gp_history"]["loaded"] == 1
    assert loop.stats["gp_history"]["skipped"] == {"supervisor legacy != tight": 2}
