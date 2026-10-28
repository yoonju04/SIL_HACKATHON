"""09 Monitor SR checks: adapter from the engine to the SR monitor v3.1 in pipeline/09_monitor/monitor.py.

The grading itself (SR1..SR6, validity V-01..V-11 / G-04, P-02 paired with the baseline, R79-style lateral
signal, diagnostics) lives in pipeline/09_monitor/monitor.py and reads pipeline/09_monitor/safety_spec.yaml
(sections requirements, evaluation_window, gp_target, diagnostics, ... and safety_specification P-02).
This module only
  * turns engine.monitor.RunData into the monitor's Trace (trace_from_run),
  * builds its MonitorConfig from the engine's spec dict (config_from_spec),
  * maps its result onto the run_record sr_* fields that engine.monitor.attach_sr_checks spreads (grade).
SR results are reference fields: they never change recovery_outcome, outcome_category or GP targets, which
engine.judgment (recovery v2) decides. The recovery v2 result is recorded next to SR5 for comparison.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import PIPELINE_DIR

SR_OUTCOMES = ("invalid", "no_trigger", "F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered", "pass")
FAIL_OUTCOMES = ("F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered")
MONITOR_DIR = PIPELINE_DIR / "09_monitor"


def _load_monitor():
    """pipeline/09_monitor/monitor.py as a module (its folder must be importable for `performance`)."""
    if str(MONITOR_DIR) not in sys.path:
        sys.path.insert(0, str(MONITOR_DIR))
    spec = importlib.util.spec_from_file_location("sr_monitor_v3", MONITOR_DIR / "monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_monitor()
Trace = M.Trace
MonitorConfig = M.MonitorConfig
footprint_lane_margin = M.footprint_lane_margin
measure_ftti = M.measure_ftti


def config_from_spec(spec: dict | None, **overrides) -> MonitorConfig:
    """MonitorConfig from the engine's safety_spec dict; defaults when the SR sections are absent (tests)."""
    if spec and spec.get("requirements") and spec.get("evaluation_window"):
        return M.config_from_spec(spec, None, **overrides)
    cfg = MonitorConfig()
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def _num_or_nan(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def trace_from_run(run) -> Trace:
    """Adapter: engine.monitor.RunData (our joined steps/behavior/monitor_probe/agent_controls ticks) -> Trace.

    steer_cmd_rad: behavior/steps `steer_rad` if logged, else agent_controls `steer` (CARLA
    normalised, left-handed) * agent_config geometry.max_steer_rad with the sign flipped to right-handed."""
    ticks = run.ticks
    t = np.array([r["t"] for r in ticks], float)

    def col(name):
        vals = [r.get(name) for r in ticks]
        return None if all(v is None for v in vals) else vals

    def arr(name, fallback=None):
        vals = col(name) or (col(fallback) if fallback else None)
        return None if vals is None else np.array([_num_or_nan(v) for v in vals], float)

    max_steer = ((run.summary.get("meta") or {}).get("vehicle_params") or {}).get("max_steer_rad") \
        or (run.config.get("geometry") or {}).get("max_steer_rad")
    steer = col("steer_rad")
    if steer is not None and all(v is not None for v in steer):
        steer_cmd = np.array(steer, float)
    elif col("ctrl_steer") is not None and max_steer:
        cs = col("ctrl_steer")
        steer_cmd = None if any(v is None for v in cs) else -np.array(cs, float) * float(max_steer)
    else:
        steer_cmd = None
    first_probe = next((r for r in ticks if r.get("bbox_half_width_m") is not None), {})
    collisions, info, untimed = [], [], 0
    for e in run.collisions or []:
        tc = e.get("t", e.get("time_s"))
        if tc is None:
            untimed += 1
            continue
        collisions.append(float(tc))
        info.append({"t": float(tc), "other_actor": e.get("other_actor", e.get("other")), "impulse": e.get("impulse")})
    fa, fe, rev = col("fault_injected"), col("fault_effective"), col("route_revision")
    ac = col("active_controller") or col("steer_source")
    return Trace(
        t=t, vx=arr("vx"), vy=arr("vy"), yaw_rate=arr("yaw_rate"),
        lane_offset_m=arr("lane_offset_m", "e_y_lane"), lane_heading_err=arr("lane_heading_err_rad", "e_psi"),
        lane_width_m=arr("lane_width_m"), ay_mps2=arr("ay_mps2"), steer_cmd_rad=steer_cmd,
        active_controller=ac, fault_active=None if fa is None else np.array([bool(v) for v in fa]),
        fault_effective=None if fe is None else np.array([bool(v) for v in fe]),
        route_revision=None if rev is None else np.array([v or 0 for v in rev]),
        collisions_t=collisions, collisions_info=info, collisions_untimed=untimed,
        half_width_m=first_probe.get("bbox_half_width_m"), half_length_m=first_probe.get("bbox_half_length_m"),
        max_steer_rad=max_steer,
        safe_stop=bool((run.summary.get("supervisor") or {}).get("safe_stop", False)))


def grade(fault: Trace, baseline: Trace | None, onset: float, cfg: MonitorConfig | None = None,
          no_reaction: Trace | None = None, recovery: dict | None = None) -> dict:
    """SR monitor v3.1 verdict for one (fault, baseline[, no-reaction]) set, in the engine's sr_* field shape.

    Returns {valid, invalid_reasons, sr_outcome, sr, sr_scores, sr_measures, failed_requirements, sr_labels,
             detection_latency_s, ftti_s, note, aux, diagnostics, ...}."""
    cfg = cfg or MonitorConfig()
    out: dict[str, Any] = {"valid": False, "invalid_reasons": [], "sr_outcome": "invalid", "sr": {}, "sr_scores": None,
                           "sr_measures": None, "failed_requirements": [], "sr_labels": None,
                           "detection_latency_s": None, "ftti_s": None, "note": None, "monitor": "sr_monitor_v3.1"}
    if baseline is None:
        out["invalid_reasons"] = ["no baseline run (SR4 needs the paired no-fault run)"]
        return out
    d = M.grade_pair(fault, baseline, onset, cfg, no_reaction)
    out.update(actual_fault_onset=d.get("actual_fault_onset"), trigger_time=d.get("trigger_time"))
    if d.get("sim_status") != "valid":
        out["invalid_reasons"] = (d.get("invalid_reason") or "invalid").split("; ")
        return out
    sr = d["sr"]
    sr1 = sr.get("SR1") or {}
    if recovery is not None and sr.get("SR5"):
        sr["SR5"]["recovery_v2"] = {"recovery_outcome": recovery.get("recovery_outcome"),
                                    "final_safe_recovery": recovery.get("final_safe_recovery")}
    applies = lambda k: bool((sr.get(k) or {}).get("applies"))
    labels = {  # extra boolean labels next to recovery_labels (None = not applicable)
        "late_or_no_switch": (not sr1["pass"]) if applies("SR1") else None,
        "switch_transient": (not sr["SR2"]["pass"]) if applies("SR2") else None,
        "footprint_departure": d["P01"] < 0,
        "actuator_limit_exceeded": (not sr["SR6"]["pass"]) if applies("SR6") else None,
    }
    ftti = sr1.get("ftti_s")
    out.update(valid=True, invalid_reasons=[], sr_outcome=d["failure_type"], sr=sr, sr_scores=d["sr_scores"],
               sr_measures=d["sr_measures"], failed_requirements=d["failed_requirements"], sr_labels=labels,
               detection_latency_s=d.get("detection_latency_s"),
               ftti_s=ftti if ftti is None or not math.isinf(ftti) else None, note=d.get("note"),
               unsafe=d.get("unsafe"), assessment_complete=d.get("assessment_complete"),
               unassessed_requirements=d.get("unassessed_requirements"), aux=d.get("aux"),
               diagnostics=d.get("diagnostics"), switch_time=d.get("switch_time"), switch_count=d.get("switch_count"),
               baseline_switch_count=d.get("baseline_switch_count"), evaluation_window=d.get("evaluation_window"),
               notes=d.get("notes") or [], spec_provenance=d.get("spec_provenance"))
    return out


def noise_floor(baseline_margins: list[float]) -> dict:
    """Spread of SR4 margins over repeated baselines -> minimum meaningful margin delta (calibration helper)."""
    a = np.asarray([v for v in baseline_margins if v is not None], float)
    if a.size < 2:
        return {"n": int(a.size), "noise_floor_m": None, "delta_min_m": None}
    nf = float(a.std(ddof=1))
    return {"n": int(a.size), "noise_floor_m": nf, "delta_min_m": max(2 * nf, 0.1)}
