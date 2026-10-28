"""09 Monitor v3: grades MPC->LQR switching safety on SR1..SR6 (safety_spec.yaml).

What is verified
  (1) a fault happened but the switch to LQR did not happen or came too late  -> SR1 (F1)
  (2) the switch happened but the switch moment or the driving afterwards
      was unsafe                                     -> SR2 (F2), SR4/collision (F3)

Rules
  * Read-only and post-hoc: reads run folders after the episode, sends nothing.
  * Grading uses simulator truth (pose, lane geometry, collisions) and the
    injector log; the supervisor under test never sees either.
  * Unknown inputs make a run invalid instead of being guessed.

Inputs per graded pair
  fault run, baseline run (same scenario, no fault), optional no-reaction run
  (same fault, no supervisor; PR #1 `--controller mpc_unprotected`) used to
  measure the FTTI that sets the SR1 limit.
Outputs: one run_record per pair (feedback.py turns records into the files
for stages 2, 3 and 6).
"""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

G = 9.81
OUTCOMES = ("invalid", "no_trigger", "F1_no_or_late_switch",
            "F2_switch_transient", "F3_not_recovered", "pass")


# ----------------------------------------------------------------- config
@dataclass
class MonitorConfig:
    dt: float = 0.05
    window_s: float = 10.0
    min_window_s: float = 5.0
    tick_tolerance: float = 0.25
    onset_tolerance_s: float = 0.051
    # SR1
    sr1_ftti_fraction: float = 0.5
    sr1_fixed_limit_s: float = 0.5
    # SR2
    sr2_limit_dps: float = 60.0
    sr2_window_s: float = 0.5
    # SR4 geometry fallbacks (used only when the probe did not log them)
    lane_width_default_m: float | None = None
    vehicle_half_width_m: float | None = None
    vehicle_half_length_m: float | None = None
    # SR5
    sr5_band_m: float = 0.3
    sr5_within_s: float = 3.0
    sr5_hold_s: float = 1.0
    sr5_enforce: bool = False
    # SR6
    max_steer_rad: float | None = None
    sr6_max_rate_dps: float = 90.0
    # Diagnostic literature references, not safety label thresholds.
    diagnostic_low_speed_mps: float = 0.5
    diagnostic_post_window_s: float = 5.0
    comfort_reference_limits: dict = field(default_factory=lambda: {
        "max_abs_lateral_accel_mps2": 4.89, "max_abs_yaw_rate_rad_s": 0.95,
        "max_abs_yaw_accel_rad_s2": 1.93})
    sr6_enforce: bool = True
    spec_provenance: dict = field(default_factory=lambda: {"status": "inline_unversioned"})
    # auxiliary
    mu: float = 0.9
    ay_smoothing_ticks: int = 5
    # UN R79 Annex 8 2.4 style signal processing (sampled at our 20 Hz, so no compliance claim)
    ay_filter_cutoff_hz: float = 0.5
    ay_filter_order: int = 4
    jerk_average_s: float = 0.5
    # P-02 is judged against the paired baseline: limit = max(threshold, baseline peak) + allowance
    p02_paired: bool = True
    p02_allowance_mps2: float = 0.3
    # GP target
    gp_clip: tuple = (-2.0, 1.0)
    gp_collision_value: float = -2.0
    # team metrics (02 vulnerability_patterns.yaml outcome_definitions)
    t_hold_s: float = 5.0                    # recovery: no admissible constraint violated for T_hold after the switch
    a_y_max_mps2: float | None = None        # admissible lateral accel; None -> 0.5 * mu * g (same as P-02)
    steer_rate_max_dps: float | None = None  # admissible steering rate; None -> sr6_max_rate_dps
    deadline_ms: float | None = None         # MPC configured_time_limit_ms (safety_profile); None -> metric null
    post_switch_sat_window_s: float = 1.0
    saturation_tol: float = 0.995            # |u| >= tol * limit counts as saturated


def load_spec(path: str | Path, **overrides) -> MonitorConfig:
    """Build MonitorConfig from safety_spec.yaml (single source of the numbers)."""
    import yaml  # local import: the grading functions themselves need only numpy

    with open(path, encoding="utf-8") as fh:
        s = yaml.safe_load(fh)
    return config_from_spec(s, hashlib.sha256(Path(path).read_bytes()).hexdigest(), **overrides)


def config_from_spec(s: dict, sha256: str | None = None, **overrides) -> MonitorConfig:
    """MonitorConfig from an already-loaded safety_spec dict (engine/sr_checks.py passes the engine's copy)."""
    r = s["requirements"]
    w = s["evaluation_window"]
    g = s.get("gp_target", {})
    cfg = MonitorConfig(
        window_s=float(w["length_s"]), min_window_s=float(w["min_length_s"]),
        sr1_ftti_fraction=float(r["SR1"]["limit"]["ftti_fraction"]),
        sr1_fixed_limit_s=float(r["SR1"]["limit"]["fixed_fallback_s"]),
        sr2_limit_dps=float(r["SR2"]["limit_dps"]), sr2_window_s=float(r["SR2"]["window_after_switch_s"]),
        sr5_band_m=float(r["SR5"]["band_m"]), sr5_within_s=float(r["SR5"]["within_s"]),
        sr5_hold_s=float(r["SR5"]["hold_s"]), sr5_enforce=bool(r["SR5"]["enforce"]),
        sr6_max_rate_dps=float(r["SR6"]["max_rate_dps"]),
        gp_clip=tuple(g.get("clip", (-2.0, 1.0))), gp_collision_value=float(g.get("collision_value", -2.0)),
    )
    tm = s.get("team_metrics") or {}
    if tm.get("t_hold_s") is not None:
        cfg.t_hold_s = float(tm["t_hold_s"])
    if tm.get("post_switch_saturation_window_s") is not None:
        cfg.post_switch_sat_window_s = float(tm["post_switch_saturation_window_s"])
    p02 = next((x for x in (s.get("safety_specification") or {}).get("properties", []) if x.get("id") == "P-02"), {})
    if p02.get("lateral_acc_threshold_mps2") is not None:
        cfg.a_y_max_mps2 = float(p02["lateral_acc_threshold_mps2"])
    if p02.get("evaluation") is not None:
        cfg.p02_paired = p02["evaluation"] == "paired_baseline"
    if p02.get("paired_allowance_mps2") is not None:
        cfg.p02_allowance_mps2 = float(p02["paired_allowance_mps2"])
    sp = (s.get("diagnostics") or {}).get("lateral_signal") or {}
    cfg.ay_filter_cutoff_hz = float(sp.get("filter_cutoff_hz", cfg.ay_filter_cutoff_hz))
    cfg.ay_filter_order = int(sp.get("filter_order", cfg.ay_filter_order))
    cfg.jerk_average_s = float(sp.get("jerk_moving_average_s", cfg.jerk_average_s))
    diagnostics = s.get("diagnostics") or {}
    cfg.diagnostic_low_speed_mps = float(diagnostics.get("low_speed_exclusion_mps", 0.5))
    cfg.diagnostic_post_window_s = float(diagnostics.get("post_switch_window_s", 5.0))
    if diagnostics.get("comfort_reference_limits") is not None:
        cfg.comfort_reference_limits = dict(diagnostics["comfort_reference_limits"])
    cfg.sr6_enforce = bool(r["SR6"].get("enforce", True))
    cfg.spec_provenance = {"schema_version": s.get("meta", {}).get("schema_version"),
                           "status": s.get("meta", {}).get("status"),
                           "sha256": sha256}
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    for key in ('dt','window_s','min_window_s','sr1_ftti_fraction','sr1_fixed_limit_s','sr2_limit_dps',
                'sr2_window_s','sr5_band_m','sr5_within_s','sr5_hold_s','sr6_max_rate_dps',
                'diagnostic_post_window_s','diagnostic_low_speed_mps','mu','ay_filter_cutoff_hz','jerk_average_s'):
        if not math.isfinite(getattr(cfg,key)) or getattr(cfg,key) <= 0:
            raise ValueError(f"invalid positive spec parameter: {key}")
    if cfg.sr1_ftti_fraction > 1 or cfg.min_window_s > cfg.window_s:
        raise ValueError('invalid FTTI fraction or evaluation window')
    if cfg.a_y_max_mps2 is not None and (not math.isfinite(cfg.a_y_max_mps2) or cfg.a_y_max_mps2 <= 0):
        raise ValueError('invalid lateral acceleration limit')
    return cfg


# ------------------------------------------------------------------ trace
@dataclass
class Trace:
    t: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    yaw_rate: np.ndarray
    lane_offset_m: np.ndarray
    lane_heading_err: np.ndarray | None = None
    lane_width_m: np.ndarray | None = None
    ay_mps2: np.ndarray | None = None
    steer_cmd_rad: np.ndarray | None = None  # what was actually sent to the vehicle
    active_controller: list | None = None
    fault_active: np.ndarray | None = None
    fault_effective: np.ndarray | None = None
    route_revision: np.ndarray | None = None
    solve_ms: np.ndarray | None = None        # MPC solve time per tick (steps.jsonl), for deadline_miss_rate
    mpc_violation: np.ndarray | None = None   # MPC constraint violated per tick, if the MPC logs it
    collisions_t: list = field(default_factory=list)
    collisions_info: list = field(default_factory=list)   # [{t, other_actor, impulse}] when logged
    collisions_untimed: int = 0                            # events logged without a time -> run invalid
    half_width_m: float | None = None
    half_length_m: float | None = None
    max_steer_rad: float | None = None
    safe_stop: bool = False
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------- helpers
def footprint_lane_margin(offset, heading_err, lane_width, half_w, half_l):
    """Signed margin of the vehicle rectangle to the intended lane's boundaries [m]."""
    offset = np.asarray(offset, float)
    dpsi = np.zeros_like(offset) if heading_err is None else np.asarray(heading_err, float)
    s, c = np.sin(dpsi), np.cos(dpsi)
    corners = np.stack([offset + xc * s + yc * c for xc in (half_l, -half_l) for yc in (half_w, -half_w)])
    half_lane = 0.5 * np.asarray(lane_width, float)
    return np.minimum(half_lane - corners.max(axis=0), half_lane + corners.min(axis=0))


def lateral_acceleration(tr: Trace, cfg: MonitorConfig):
    if tr.ay_mps2 is not None:
        return np.asarray(tr.ay_mps2, float)
    ay = np.gradient(np.asarray(tr.vy, float), tr.t) + np.asarray(tr.vx, float) * np.asarray(tr.yaw_rate, float)
    k = max(1, cfg.ay_smoothing_ticks)
    if k > 1:
        ay = np.convolve(np.pad(ay, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return ay


def filtered_lateral_acceleration(tr: Trace, cfg: MonitorConfig):
    """Lateral acceleration low-passed as in UN R79 Annex 8 2.4 (4th-order Butterworth, 0.5 Hz), zero phase.

    Returns (ay, method). Falls back to the unfiltered signal when the trace is too short for the filter
    or scipy is missing, and says so in `method`.
    """
    ay = lateral_acceleration(tr, cfg)
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return ay, "unfiltered (scipy unavailable)"
    b, a = butter(cfg.ay_filter_order, cfg.ay_filter_cutoff_hz, fs=1.0 / cfg.dt)
    if len(ay) <= 3 * max(len(a), len(b)):
        return ay, "unfiltered (trace shorter than filter padding)"
    return filtfilt(b, a, ay), "butterworth order %d, %.2f Hz, zero phase" % (cfg.ay_filter_order,
                                                                          cfg.ay_filter_cutoff_hz)


def _first(mask, t):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if idx.size else None


def _switch_times(tr: Trace) -> list[float]:
    if tr.active_controller is None:
        return []
    ac = list(tr.active_controller)
    return [float(tr.t[i]) for i in range(1, len(ac)) if ac[i] != ac[i - 1]]


def _clip(x, lo, hi):
    return float(min(hi, max(lo, x)))


# ---------------------------------------------------------- run checks
@dataclass
class RunCheck:
    valid: bool
    reasons: list
    window: tuple
    win: np.ndarray | None = None
    margin: np.ndarray | None = None  # SR4 margin per tick
    nominal_margin: float | None = None


def check_run(tr: Trace, onset: float, cfg: MonitorConfig, kind: str) -> RunCheck:
    """Validity (V-01..V-09) and the SR4 margin series. kind: fault | baseline | no_reaction."""
    t = np.asarray(tr.t, float)
    reasons: list[str] = []
    chans = [tr.vx, tr.vy, tr.yaw_rate, tr.lane_offset_m]
    if len(t) < 2 or any(len(c) != len(t) for c in chans):
        return RunCheck(False, [f"{kind}: V-01 empty or misaligned log"], (onset, onset))
    for channel in (tr.lane_heading_err, tr.lane_width_m, tr.ay_mps2, tr.steer_cmd_rad, tr.fault_active,
                    tr.fault_effective, tr.route_revision):
        if channel is not None and len(channel) != len(t):
            return RunCheck(False, [f"{kind}: V-01 optional channel misaligned"], (onset, onset))
    numeric_optional = [x for x in (tr.lane_heading_err, tr.lane_width_m, tr.ay_mps2, tr.steer_cmd_rad) if x is not None]
    if not all(np.all(np.isfinite(np.asarray(c, float))) for c in [t, *chans, *numeric_optional]):
        reasons.append(f"{kind}: V-01 non-finite values")
    if np.any(np.abs(np.diff(t) - cfg.dt) > cfg.tick_tolerance * cfg.dt):
        reasons.append(f"{kind}: V-02 irregular ticks")
    end = min(onset + cfg.window_s, float(t[-1]))
    ended_by_crash = any(onset - 1e-9 <= c <= end + cfg.dt for c in tr.collisions_t)
    if tr.collisions_untimed:
        reasons.append(f"{kind}: V-11 collision logged without time (cannot place it in the window)")
    if end - onset < cfg.min_window_s and not tr.safe_stop and not ended_by_crash and kind != "no_reaction":
        reasons.append(f"{kind}: V-03 window shorter than minimum")
    win = (t >= onset - 1e-9) & (t <= end + 1e-9)
    if not win.any():
        return RunCheck(False, reasons + [f"{kind}: V-03 no samples in window"], (onset, end))
    if tr.route_revision is not None:
        rev = np.asarray(tr.route_revision)[win]
        if np.any(rev != rev[0]):
            reasons.append(f"{kind}: V-04 reference route changed in window")
    if kind == "baseline" and tr.fault_active is not None and np.any(np.asarray(tr.fault_active, bool)):
        reasons.append("baseline: V-07 fault active in baseline")
    if kind in ("fault", "no_reaction") and tr.fault_active is None:
        reasons.append(f"{kind}: V-09 no injector log")
    if any(value is not None and (not math.isfinite(value) or value <= 0)
           for value in (tr.half_width_m, tr.half_length_m, tr.max_steer_rad)):
        reasons.append(f"{kind}: V-05 non-positive vehicle parameter")
    half_w = tr.half_width_m or cfg.vehicle_half_width_m
    half_l = tr.half_length_m or cfg.vehicle_half_length_m
    lane_w = tr.lane_width_m if tr.lane_width_m is not None else (
        None if cfg.lane_width_default_m is None else np.full(len(t), cfg.lane_width_default_m))
    margin = nominal = None
    if half_w is None or half_l is None or lane_w is None:
        reasons.append(f"{kind}: V-05 lane width or vehicle footprint unknown")
    elif half_w <= 0 or half_l <= 0 or np.any(np.asarray(lane_w, float) <= 2*half_w):
        reasons.append(f"{kind}: V-05 invalid lane width or vehicle footprint")
    else:
        margin = footprint_lane_margin(tr.lane_offset_m, tr.lane_heading_err, lane_w, half_w, half_l)
        nominal = float(0.5 * np.median(np.asarray(lane_w, float)[win]) - half_w)
    return RunCheck(not reasons, reasons, (onset, end), win, margin, nominal)


def measure_ftti(no_reaction: Trace, onset: float, cfg: MonitorConfig) -> dict:
    """FTTI from a no-reaction run: time from the actual fault start to the first SR4/collision violation."""
    chk = check_run(no_reaction, onset, cfg, "no_reaction")
    if not chk.valid:
        return {"ftti_s": None, "valid": False, "reasons": chk.reasons}
    t = np.asarray(no_reaction.t, float)
    start = _first(np.asarray(no_reaction.fault_active, bool), t)
    start = onset if start is None else start
    viol = _first(chk.win & (chk.margin < 0), t)
    hits = [c for c in no_reaction.collisions_t if chk.window[0] <= c <= chk.window[1]]
    times = [x for x in (viol, min(hits) if hits else None) if x is not None]
    ftti = (min(times) - start) if times else math.inf
    return {"ftti_s": ftti, "valid": True, "reasons": []}


# ------------------------------------------------------------- SR checks
def grade_pair(fault: Trace, baseline: Trace, onset: float, cfg: MonitorConfig | None = None,
                  no_reaction: Trace | None = None, run_info: dict | None = None) -> dict:
    """SR1..SR6 verdict for one pair (fault vs baseline [+ no-reaction]). Kept inside run_record.metrics["monitor"]."""
    cfg = cfg or MonitorConfig()
    info = dict(run_info or {})
    fc = check_run(fault, onset, cfg, "fault")
    bc = check_run(baseline, onset, cfg, "baseline")
    reasons = fc.reasons + bc.reasons
    t = np.asarray(fault.t, float)

    # injector truth
    actual_onset = trigger = None
    if fault.fault_active is not None:
        actual_onset = _first(np.asarray(fault.fault_active, bool), t)
        if actual_onset is not None and abs(actual_onset - onset) > cfg.onset_tolerance_s:
            reasons.append("fault: V-06 actual onset differs from planned onset")
        eff = fault.fault_effective if fault.fault_effective is not None else fault.fault_active
        trigger = _first(np.asarray(eff, bool), t)
    if fc.valid and bc.valid and abs((fc.window[1] - fc.window[0]) - (bc.window[1] - bc.window[0])) > cfg.dt \
            and not fault.safe_stop and not fault.collisions_t:
        reasons.append("V-08 fault and baseline windows differ")

    if bc.valid and (np.min(bc.margin[bc.win]) < 0 or
                     any(bc.window[0] <= c <= bc.window[1] for c in baseline.collisions_t)):
        reasons.append("G-04 baseline (no fault) already breaks SR4: pair rejected")

    if fc.valid and fault.active_controller is not None:
        t0 = actual_onset if actual_onset is not None else onset
        if any(s < t0 - 1e-9 for s in _switch_times(fault)):
            reasons.append("fault: V-10 already switched before the fault started (switching cannot be graded)")

    rec: dict[str, Any] = {k: info.get(k) for k in (
        "run_id", "scenario_id", "seed_scenario_id", "round", "selector", "profile_id", "source_commit",
        "feature_schema_version", "hypothesis_ids", "hypothesis_arms", "parameters", "reason_items",
        "gp_mean", "gp_std", "stress_test", "environment", "fault_plan")}
    rec.update(planned_fault_onset=onset, actual_fault_onset=actual_onset, trigger_time=trigger)
    if reasons:
        rec.update(sim_status="invalid", invalid_reason="; ".join(reasons), failure_type="invalid",
                   unsafe=False, sr={}, sr_overall=None, sr_scores=None, failed_requirements=[])
        return rec

    # ---- switches (SUT behaviour)
    f_sw = _switch_times(fault)
    b_sw = _switch_times(baseline)  # recorded only (no requirement on fault-free switching)
    post_sw = [s for s in f_sw if s >= (actual_onset if actual_onset is not None else onset) - 1e-9
               and s <= fc.window[1] + 1e-9]
    switch_time = post_sw[0] if post_sw else None
    sr: dict[str, dict] = {}

    # SR4 lane keeping (+ collision)
    w = fc.win
    p01_f = float(np.min(fc.margin[w]))
    p01_b = float(np.min(bc.margin[bc.win]))
    m_nom = fc.nominal_margin
    hits = [c for c in fault.collisions_t if fc.window[0] <= c <= fc.window[1]]
    sr4_fail = p01_f < 0 or bool(hits)
    sr["SR4"] = {"applies": True, "pass": not sr4_fail, "robustness": p01_f / m_nom,
                 "margin_m": p01_f, "baseline_margin_m": p01_b, "nominal_margin_m": m_nom,
                 "first_violation_t": _first(w & (fc.margin < 0), t), "collision": bool(hits)}
    margin_delta = p01_f - p01_b

    # SR1 switch latency
    ftti = None
    limit_src = "fixed_fallback"
    if no_reaction is not None:
        fr = measure_ftti(no_reaction, onset, cfg)
        if fr["valid"]:
            ftti, limit_src = fr["ftti_s"], "ftti_measured"
        else:
            rec.setdefault("notes", []).append("no_reaction run invalid: " + "; ".join(fr["reasons"]))
    triggered = trigger is not None
    if not triggered:
        sr1 = {"applies": False, "reason": "fault never effective"}
    elif ftti is not None and math.isinf(ftti):
        sr1 = {"applies": False, "reason": "undetected fault is harmless (no-reaction run keeps SR4)",
               "ftti_s": None}
    else:
        limit = cfg.sr1_ftti_fraction * ftti if ftti is not None else cfg.sr1_fixed_limit_s
        # Never expand a hazard budget to one simulation tick; a sub-tick budget is still binding.
        latency = (switch_time - actual_onset) if switch_time is not None else math.inf
        rob = -1.0 if limit <= 0 or math.isinf(latency) else _clip((limit - latency) / limit, -1.0, 1.0)
        sr1 = {"applies": True, "pass": limit > 0 and latency <= limit, "robustness": rob, "latency_s": _num(latency),
               "limit_s": limit, "limit_source": limit_src, "ftti_s": _num(ftti)}
    sr["SR1"] = sr1

    # SR2 switch smoothness (on the command actually sent)
    if switch_time is None or fault.steer_cmd_rad is None:
        sr["SR2"] = {"applies": False, "reason": "no switch" if switch_time is None else "steer_cmd not logged"}
    else:
        m = ((t >= switch_time - cfg.dt - 1e-9) & (t <= switch_time + cfg.sr2_window_s + 1e-9)
             & (t <= fc.window[1] + 1e-9))
        st = np.asarray(fault.steer_cmd_rad, float)
        rates = np.abs(np.diff(st)) / np.diff(t)
        rate_mask = m[1:] & m[:-1]
        max_rate = float(np.degrees(rates[rate_mask].max())) if rate_mask.any() else 0.0
        sr["SR2"] = {"applies": True, "pass": max_rate <= cfg.sr2_limit_dps,
                     "robustness": (cfg.sr2_limit_dps - max_rate) / cfg.sr2_limit_dps,
                     "max_rate_dps": max_rate, "limit_dps": cfg.sr2_limit_dps,
                     "coverage_complete": switch_time + cfg.sr2_window_s <= fc.window[1] + 1e-9}

    # SR3 is explicitly unassessed until a vehicle/ODD-specific stability envelope is supplied.
    sr["SR3"] = {"applies": False, "pass": None, "role": "report_only",
                 "reason": "validated beta-yaw stability/recoverability envelope unavailable"}

    # SR5 recovery (advisory unless enforced)
    if switch_time is None:
        sr["SR5"] = {"applies": False, "reason": "no switch"}
    else:
        off = np.abs(np.asarray(fault.lane_offset_m, float))
        t_rec = None
        for i in np.flatnonzero((t >= switch_time) & (t <= fc.window[1])):
            j = int(np.searchsorted(t, t[i] + cfg.sr5_hold_s - 1e-9))
            if j < len(t) and t[j] <= fc.window[1] + 1e-9 and np.all(off[i:j+1] <= cfg.sr5_band_m):
                t_rec = float(t[i] - switch_time)
                break
        rob = -1.0 if t_rec is None else (cfg.sr5_within_s - t_rec) / cfg.sr5_within_s
        sr["SR5"] = {"applies": True, "enforced": cfg.sr5_enforce,
                     "pass": t_rec is not None and t_rec <= cfg.sr5_within_s,
                     "robustness": rob, "recover_time_s": t_rec,
                     "band_m": cfg.sr5_band_m, "within_s": cfg.sr5_within_s, "hold_s": cfg.sr5_hold_s}

    # SR6 actuator limits
    max_steer = fault.max_steer_rad or cfg.max_steer_rad
    if fault.steer_cmd_rad is None or max_steer is None:
        sr["SR6"] = {"applies": False, "reason": "steer_cmd or max steer unknown"}
    else:
        st = np.asarray(fault.steer_cmd_rad, float)[w]
        tw = t[w]
        peak = float(np.max(np.abs(st)))
        whole_rate = np.abs(np.diff(np.asarray(fault.steer_cmd_rad, float))) / np.diff(t)
        rate = float(np.degrees(np.max(whole_rate[w[1:]]))) if np.any(w[1:]) else 0.0
        rob = min(1 - peak / max_steer, 1 - rate / cfg.sr6_max_rate_dps)
        sr["SR6"] = {"applies": True, "pass": rob >= -1e-9, "robustness": rob,
                     "max_steer_deg": math.degrees(peak), "max_rate_dps": rate,
                     "limit_rate_dps": cfg.sr6_max_rate_dps, "limit_steer_deg": math.degrees(max_steer),
                     "enforced": cfg.sr6_enforce}

    # auxiliary P-02 on the R79-filtered signal, judged against the paired baseline
    ay, ay_method = filtered_lateral_acceleration(fault, cfg)
    ay_base, _ = filtered_lateral_acceleration(baseline, cfg)
    thr = cfg.a_y_max_mps2 if cfg.a_y_max_mps2 is not None else 0.5 * cfg.mu * G
    peak_f = float(np.max(np.abs(ay[w])))
    peak_b = float(np.max(np.abs(ay_base[bc.win])))
    limit = max(thr, peak_b) + cfg.p02_allowance_mps2 if cfg.p02_paired else thr
    aux = {"P02_lat_acc": {"threshold_mps2": thr, "max_abs_mps2": peak_f, "baseline_max_abs_mps2": peak_b,
                           "evaluation": "paired_baseline" if cfg.p02_paired else "absolute",
                           "effective_limit_mps2": limit, "robustness": limit - peak_f,
                           "absolute_robustness": thr - peak_f, "signal": ay_method}}

    # ---- outcome (priority order from the spec)
    failed = [k for k, v in sr.items() if v.get("applies") and not v.get("pass")
              and not (k == "SR5" and not cfg.sr5_enforce)
              and not (k == "SR6" and not cfg.sr6_enforce)]
    unsafe = not sr["SR4"]["pass"] or bool(cfg.sr6_enforce and sr["SR6"].get("applies") and not sr["SR6"]["pass"])
    note = None
    if not triggered:
        outcome = "no_trigger"
    elif sr["SR1"].get("applies") and not sr["SR1"]["pass"]:
        outcome = "F1_no_or_late_switch"
    elif sr["SR2"].get("applies") and not sr["SR2"]["pass"]:
        outcome = "F2_switch_transient"
    elif not sr["SR4"]["pass"] or (cfg.sr6_enforce and sr["SR6"].get("applies") and not sr["SR6"]["pass"]) or (cfg.sr5_enforce and sr["SR5"].get("applies") and not sr["SR5"]["pass"]):
        outcome = "F3_not_recovered"
    else:
        outcome = "pass"
        if switch_time is None and triggered:
            note = "undetected_harmless"

    # ---- GP targets (stage 6): how much EACH requirement was kept or broken.
    # Every value is on the same scale: 1 = fully kept, 0 = exactly at the limit, < 0 = broken
    # (-1 = broken by the full limit, clipped at -2). None = requirement did not apply in this run.
    lo, hi = cfg.gp_clip
    gpt: dict[str, Any] = {}
    gpt["SR1"] = sr["SR1"]["robustness"] if sr["SR1"].get("applies") else None
    gpt["SR2"] = _clip(sr["SR2"]["robustness"], lo, hi) if sr["SR2"].get("applies") else None
    gpt["SR4"] = cfg.gp_collision_value if sr["SR4"]["collision"] else _clip(p01_f / m_nom, lo, hi)
    gpt["SR5"] = _clip(sr["SR5"]["robustness"], lo, hi) if sr["SR5"].get("applies") else None
    gpt["SR6"] = _clip(sr["SR6"]["robustness"], lo, hi) if sr["SR6"].get("applies") else None
    enforced = [gpt[k] for k in ("SR1", "SR2", "SR4") + (("SR6",) if cfg.sr6_enforce else ()) + (("SR5",) if cfg.sr5_enforce else ())
                if gpt[k] is not None]
    gpt["overall"] = min(enforced)
    gp = gpt["overall"]
    # raw amounts behind each score (units, not normalised) so people and the GP side can read them
    gp_measures = {
        "SR1_latency_s": sr["SR1"].get("latency_s"), "SR1_limit_s": sr["SR1"].get("limit_s"),
        "SR1_no_switch": bool(sr["SR1"].get("applies")) and switch_time is None,
        "SR2_max_rate_dps": sr["SR2"].get("max_rate_dps"),
        "SR4_margin_m": p01_f, "SR4_margin_delta_m": margin_delta, "SR4_collision": sr["SR4"]["collision"],
        "SR4_collision_t": min(hits) if hits else None,
        "SR4_collision_other": next((c.get("other_actor") for c in fault.collisions_info
                                     if hits and c.get("t") == min(hits)), None),
        "SR5_recover_time_s": sr["SR5"].get("recover_time_s"),
        "SR6_max_steer_deg": sr["SR6"].get("max_steer_deg"), "SR6_max_rate_dps": sr["SR6"].get("max_rate_dps"),
    }

    pre = w & (t <= (switch_time if switch_time is not None else fc.window[1]))
    rec.update(
        sim_status="valid", invalid_reason=None, failure_type=outcome, note=note, unsafe=unsafe,
        failed_requirements=failed, sr=sr, aux=aux,
        switch_time=switch_time, switch_count=len(f_sw), detection_latency_s=sr["SR1"].get("latency_s"),
        P01=p01_f, baseline_P01=p01_b, margin_delta_m=margin_delta, sr_overall=gp,
        sr_scores=gpt, sr_measures=gp_measures, baseline_switch_count=len(b_sw),
        pre_switch_max_lateral_error_m=float(np.max(np.abs(np.asarray(fault.lane_offset_m, float)[pre]))),
        max_steering_rate_dps=sr.get("SR6", {}).get("max_rate_dps"),
        evaluation_window=list(fc.window),
    )
    from performance import switch_diagnostics
    rec["diagnostics"] = switch_diagnostics(fault, cfg, fc, actual_onset if actual_onset is not None else onset,
                                            switch_time, ay)
    rec["spec_provenance"] = cfg.spec_provenance
    missing = [k for k in ("SR2", "SR6") if (k == "SR6" or switch_time is not None)
               and not sr[k].get("applies") and (k != "SR6" or cfg.sr6_enforce)]
    if switch_time is not None and sr['SR2'].get('coverage_complete') is False:
        missing.append('SR2_transition_coverage')
    rec["assessment_complete"] = not missing
    rec["unassessed_requirements"] = missing
    if sr1.get("applies") is False and ftti is not None and math.isinf(ftti):
        sr1["ftti_right_censored"] = True
        sr1["reason"] = "no hazard observed within no-reaction evaluation horizon; not globally harmless"
    return rec


# ============================================================ runner contract
# scripts/runner_skeleton.py (new_pipeline) + 09 error_table.yaml results_schema.run_record:
# exactly these fields. Our SR1..SR6 verdict travels inside metrics["monitor"].
RUN_RECORD_FIELDS = (
    "run_id", "scenario_id", "seed_scenario_id", "round", "selector",
    "profile_id", "source_commit", "feature_schema_version", "hypothesis_ids",
    "parameters", "planned_fault_onset", "actual_fault_onset", "trigger_time",
    "switch_time", "switch_count", "outcome_category", "P01", "P02", "P03",
    "margin_delta_m", "switch_state_label", "metrics", "baseline_metrics", "gp_targets",
    "gp_mean", "gp_std", "sim_status", "invalid_reason",
)
RUNNER_OUTCOMES = ("no_trigger", "triggered_no_switch", "switched_safe", "switched_unsafe", "simulation_invalid")
METRIC_NAMES = ("recovery_success", "recovery_robustness", "pre_switch_max_lateral_error_m", "switch_count",
                "lateral_error_rmse_m", "deadline_miss_rate", "mpc_constraint_violation_rate",
                "post_switch_saturation_ratio", "command_jump_rad", "max_steering_rate_dps", "margin_delta_m")


def _geometry(tr: Trace, cfg: MonitorConfig):
    half_w = tr.half_width_m or cfg.vehicle_half_width_m
    lane_w = tr.lane_width_m if tr.lane_width_m is not None else np.full(len(tr.t), cfg.lane_width_default_m)
    return np.asarray(lane_w, float), float(half_w)


def run_metrics(tr: Trace, chk: RunCheck, cfg: MonitorConfig, onset: float, switch_time: float | None,
                triggered: bool, is_fault: bool, a_lim: float | None = None) -> dict:
    """Team metrics (02 outcome_definitions) for one run. Not computable -> None (never 0).

    a_lim: lateral-acceleration limit for recovery; pass P-02's paired effective limit for a fault run.
    """
    t = np.asarray(tr.t, float)
    w = chk.win
    e_y = np.asarray(tr.lane_offset_m, float)
    lane_w, half_w = _geometry(tr, cfg)
    lat_lim = 0.5 * lane_w - half_w                          # lane_half_width - vehicle_half_width
    a_lim = a_lim or cfg.a_y_max_mps2 or 0.5 * cfg.mu * G
    r_lim = math.radians(cfg.steer_rate_max_dps or cfg.sr6_max_rate_dps)
    d_lim = tr.max_steer_rad or cfg.max_steer_rad
    ay = np.abs(filtered_lateral_acceleration(tr, cfg)[0])
    st = None if tr.steer_cmd_rad is None else np.asarray(tr.steer_cmd_rad, float)
    rate = None
    if st is not None:
        rate = np.abs(np.diff(st, prepend=st[0])) / cfg.dt
        rate[0] = rate[1] if len(rate) > 1 else 0.0

    m: dict[str, Any] = {k: None for k in METRIC_NAMES}
    m["switch_count"] = len([s for s in _switch_times(tr) if chk.window[0] - 1e-9 <= s <= chk.window[1] + 1e-9])
    m["lateral_error_rmse_m"] = float(np.sqrt(np.mean(e_y[w] ** 2)))
    if rate is not None:
        m["max_steering_rate_dps"] = float(np.degrees(np.max(rate[w])))
    if tr.solve_ms is not None and cfg.deadline_ms is not None:
        sm = np.asarray(tr.solve_ms, float)[w]
        sm = sm[np.isfinite(sm)]
        m["deadline_miss_rate"] = float(np.mean(sm > cfg.deadline_ms)) if sm.size else None
    if tr.mpc_violation is not None:
        m["mpc_constraint_violation_rate"] = float(np.mean(np.asarray(tr.mpc_violation, bool)[w]))

    if is_fault and triggered:
        start = onset
        end = switch_time if switch_time is not None else chk.window[1]
        pre = w & (t >= start - 1e-9) & (t <= end + 1e-9)
        m["pre_switch_max_lateral_error_m"] = float(np.max(np.abs(e_y[pre]))) if pre.any() else None
    elif not is_fault:
        m["pre_switch_max_lateral_error_m"] = float(np.max(np.abs(e_y[w])))  # baseline: whole window

    if is_fault and triggered and switch_time is None:
        m["recovery_success"] = False   # 02: no switch at all counts as recovery failure (decision only)
    if switch_time is not None:
        post = (t >= switch_time - 1e-9) & (t <= switch_time + cfg.t_hold_s + 1e-9)
        ratios = [(lat_lim[post] - np.abs(e_y[post])) / lat_lim[post], (a_lim - ay[post]) / a_lim]
        if st is not None and d_lim:
            ratios.append((d_lim - np.abs(st[post])) / d_lim)
        if rate is not None:
            ratios.append((r_lim - rate[post]) / r_lim)
        rob = float(min(np.min(x) for x in ratios))
        m["recovery_robustness"] = rob
        m["recovery_success"] = rob >= 0
        k = int(np.searchsorted(t, switch_time - 1e-9))
        if st is not None and k > 0:
            m["command_jump_rad"] = float(abs(st[k] - st[k - 1]))
            win1 = (t >= switch_time - 1e-9) & (t <= switch_time + cfg.post_switch_sat_window_s + 1e-9)
            sat = np.zeros(int(win1.sum()), bool)
            if d_lim:
                sat |= np.abs(st[win1]) >= cfg.saturation_tol * d_lim
            sat |= rate[win1] >= cfg.saturation_tol * r_lim
            m["post_switch_saturation_ratio"] = float(sat.mean()) if sat.size else None
    return m


def load_metric_rules(path: str | Path) -> dict:
    """metric_rules from 02 vulnerability_patterns.yaml (source of GP target form, scale, direction)."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("metric_rules") or {}


def runner_gp_targets(metric_rules: dict, metrics: dict, baseline_metrics: dict | None, has_fault: bool) -> dict:
    """Same formula as runner_skeleton.gp_targets: y by gp.form, then z = sign * y / scale."""
    out = {}
    for name, rule in (metric_rules or {}).items():
        gp = rule["gp"]
        v = metrics.get(gp["target"])
        if v is None:
            continue
        if gp["form"] == "absolute_post_switch" or not has_fault:
            y = float(v)
        elif gp["form"] == "paired_delta_if_fault_else_absolute":
            b = (baseline_metrics or {}).get(gp["target"])
            if b is None:
                continue
            y = float(v) - float(b)
        else:
            raise ValueError(f"unknown GP target form: {gp['form']}")
        sign = 1.0 if rule["support_direction"] == "negative" else -1.0
        out[name] = sign * y / gp["scale"]
    return out


def _runner_outcome(detail: dict) -> str:
    ft = detail.get("failure_type")
    if ft == "invalid":
        return "simulation_invalid"
    if ft == "no_trigger":
        return "no_trigger"
    if detail.get("switch_time") is None:
        return "triggered_no_switch"
    sr4 = (detail.get("sr") or {}).get("SR4") or {}
    p02 = ((detail.get("aux") or {}).get("P02_lat_acc") or {}).get("robustness")
    ok = sr4.get("pass") and (p02 is None or p02 >= 0)
    return "switched_safe" if ok else "switched_unsafe"


def evaluate_pair(fault: Trace, baseline: Trace, onset: float, cfg: MonitorConfig | None = None,
                  no_reaction: Trace | None = None, run_info: dict | None = None,
                  metric_rules: dict | None = None) -> dict:
    """Grade one pair and return the run_record in the runner's schema (RUN_RECORD_FIELDS, nothing else).

    metrics          team metrics of the fault run + "monitor": our SR1..SR6 verdict and details
    baseline_metrics team metrics of the baseline run
    gp_targets       per-metric z (only when metric_rules is given; the runner can also compute it)
    """
    cfg = cfg or MonitorConfig()
    info = dict(run_info or {})
    d = grade_pair(fault, baseline, onset, cfg, no_reaction, info)
    params = dict(info.get("parameters") or {})
    for k in ("environment", "fault_plan", "arm"):
        if info.get(k) is not None:
            params.setdefault(k, info[k])
    rec = {k: info.get(k) for k in RUN_RECORD_FIELDS}
    rec.update(parameters=params or None, planned_fault_onset=onset, actual_fault_onset=d.get("actual_fault_onset"),
               trigger_time=d.get("trigger_time"), outcome_category=_runner_outcome(d),
               sim_status=d["sim_status"], invalid_reason=d.get("invalid_reason"), switch_state_label=None)
    if d["sim_status"] != "valid":
        rec.update(metrics={"monitor": d}, baseline_metrics=None, gp_targets={})
        return rec
    fc = check_run(fault, onset, cfg, "fault")
    bc = check_run(baseline, onset, cfg, "baseline")
    act = d.get("actual_fault_onset")
    m = run_metrics(fault, fc, cfg, onset if act is None else act, d["switch_time"], d.get("trigger_time") is not None,
                    True, d["aux"]["P02_lat_acc"]["effective_limit_mps2"])
    bm = run_metrics(baseline, bc, cfg, onset, None, False, False)
    m["margin_delta_m"] = d["margin_delta_m"]
    from performance import switch_diagnostics
    baseline_phases = switch_diagnostics(baseline, cfg, bc, onset, d['switch_time'],
                                         filtered_lateral_acceleration(baseline, cfg)[0])['phases']
    d['diagnostics']['baseline_phases_at_fault_anchor'] = baseline_phases
    delta_fields = ('lateral_rmse_m','lateral_mae_m','max_abs_lateral_error_m',
                    'max_abs_lateral_accel_mps2','max_abs_lateral_jerk_mps3')
    d['diagnostics']['paired_deltas'] = {
        phase: {key: values[key]-baseline_phases[phase][key] for key in delta_fields
                if values.get(key) is not None and baseline_phases.get(phase,{}).get(key) is not None}
        for phase, values in d['diagnostics']['phases'].items()}
    # state at the switch, for the recoverability oracle (02 recoverability_oracle_spec; artifacts still null)
    if d["switch_time"] is not None:
        t = np.asarray(fault.t, float)
        k = int(np.searchsorted(t, d["switch_time"] - 1e-9))
        e_y = np.asarray(fault.lane_offset_m, float)
        e_psi = np.zeros_like(e_y) if fault.lane_heading_err is None else np.asarray(fault.lane_heading_err, float)
        d["switch_state"] = {"e_y": float(e_y[k]), "e_y_dot": float(np.gradient(e_y, t)[k]),
                             "e_psi": float(e_psi[k]), "e_psi_dot": float(np.gradient(e_psi, t)[k])}
    m["monitor"] = d
    collision = bool(d["sr"]["SR4"]["collision"])
    rec.update(switch_time=d["switch_time"], switch_count=d["switch_count"], P01=d["P01"],
               P02=d["aux"]["P02_lat_acc"]["robustness"], P03=not collision,
               margin_delta_m=d["margin_delta_m"], metrics=m, baseline_metrics=bm,
               gp_targets=runner_gp_targets(metric_rules, m, bm, has_fault=True) if metric_rules else {})
    return rec


def view(rec: dict) -> dict:
    """Flat view for feedback.py / dashboard.py: runner fields + our monitor details side by side.
    outcome_category becomes our F-label; the runner's label stays in runner_outcome."""
    d = (rec.get("metrics") or {}).get("monitor") if isinstance(rec.get("metrics"), dict) else None
    if not d or "runner_outcome" in rec:   # old-style record, or already a view
        return rec
    out = dict(rec)
    out.update(d)
    out["runner_outcome"] = rec.get("outcome_category")
    out["outcome_category"] = d.get("failure_type")
    for k in ("metrics", "baseline_metrics", "gp_targets", "parameters"):
        out[k] = rec.get(k)        # runner values win over same-named monitor details
    params = rec.get("parameters") or {}
    for k in ("environment", "fault_plan"):
        if out.get(k) is None:
            out[k] = params.get(k)
    if out.get("hypothesis_arms") is None and params.get("arm") and rec.get("hypothesis_ids"):
        out["hypothesis_arms"] = {h: params["arm"] for h in rec["hypothesis_ids"]}
    return out




def _num(x):
    if x is None:
        return None
    return None if (isinstance(x, float) and math.isinf(x)) else float(x)


def noise_floor(baseline_margins: list[float]) -> dict:
    a = np.asarray([v for v in baseline_margins if v is not None], float)
    if a.size < 2:
        return {"n": int(a.size), "noise_floor_m": None, "delta_min_m": None}
    nf = float(a.std(ddof=1))
    return {"n": int(a.size), "noise_floor_m": nf, "delta_min_m": max(2 * nf, 0.1)}


# ------------------------------------------------------------ av_stack I/O
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_av_stack_run(run_dir: str | Path) -> Trace:
    """Trace from an av_stack run folder (steps / behavior / monitor_probe / collisions / summary)."""
    d = Path(run_dir)
    steps = _read_jsonl(d / "steps.jsonl")
    beh = {round(r["t"], 4): r for r in _read_jsonl(d / "behavior.jsonl")}
    probe = {round(r["t"], 4): r for r in _read_jsonl(d / "monitor_probe.jsonl")}
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
    config = json.loads((d / "agent_config.json").read_text(encoding="utf-8")) if (d / "agent_config.json").exists() else {}
    controls = {round(r["t"], 4): r for r in _read_jsonl(d / "agent_controls.jsonl")}
    t = np.array([r["t"] for r in steps], float)
    keys = [round(x, 4) for x in t]

    def col(src, name):
        vals = [src.get(k, {}).get(name) for k in keys]
        return None if all(v is None for v in vals) else vals

    if probe:
        off = np.array(col(probe, "lane_offset_m"), float)
        head, width, ay = col(probe, "lane_heading_err_rad"), col(probe, "lane_width_m"), col(probe, "ay_mps2")
        first = next(iter(probe.values()))
        half_w, half_l = first.get("bbox_half_width_m"), first.get("bbox_half_length_m")
    else:
        off = np.array([r["e_y_lane"] for r in steps], float)
        head, width, ay, half_w, half_l = [r["e_psi"] for r in steps], None, None, None, None
    collisions, info, untimed = [], [], 0
    if (d / "collisions.json").exists():
        for e in json.loads((d / "collisions.json").read_text()):
            tc = e.get("t", e.get("time_s"))
            if tc is None:
                untimed += 1
                continue
            collisions.append(float(tc))
            info.append({"t": float(tc), "other_actor": e.get("other_actor", e.get("other")),
                         "impulse": e.get("impulse")})
    summ = summary.get("summary", summary) if isinstance(summary, dict) else {}
    meta = summary.get("meta", {}) if isinstance(summary, dict) else {}
    max_steer = ((meta.get("vehicle_params") or {}).get("max_steer_rad")
                 or (config.get("geometry") or {}).get("max_steer_rad"))
    # Only actual, normalized CARLA commands may be converted to road-wheel radians.
    # Never treat a normalized value as radians or substitute missing commands with zero.
    steer = col(beh, "steer_rad") or [r.get("steer_rad") for r in steps]
    normalized = col(controls, "steer")
    if any(v is None for v in steer) and normalized is not None and max_steer is not None:
        steer = [None if v is None else float(v) * max_steer for v in normalized]
    fa = col(beh, "fault_injected")
    fe = col(probe, "fault_effective")
    rev = col(beh, "route_revision")
    solve = [r.get("solve_ms") for r in steps]
    viol_key = next((k for k in ("mpc_constraint_violation", "constraint_violated") if steps and k in steps[0]), None)
    return Trace(
        t=t, vx=np.array([r["vx"] for r in steps], float), vy=np.array([r["vy"] for r in steps], float),
        yaw_rate=np.array([r["yaw_rate"] for r in steps], float), lane_offset_m=off,
        lane_heading_err=None if head is None else np.array(head, float),
        lane_width_m=None if width is None else np.array(width, float),
        ay_mps2=None if ay is None else np.array(ay, float),
        steer_cmd_rad=None if steer is None or any(v is None for v in steer) else np.array(steer, float),
        active_controller=col(beh, "active_controller") or col(beh, "steer_source"),
        fault_active=None if fa is None else np.array([bool(v) for v in fa]),
        fault_effective=None if fe is None else np.array([bool(v) for v in fe]),
        route_revision=None if rev is None else np.array([v or 0 for v in rev]),
        solve_ms=None if all(v is None for v in solve) else np.array([np.nan if v is None else v for v in solve], float),
        mpc_violation=None if viol_key is None else np.array([bool(r.get(viol_key)) for r in steps]),
        collisions_t=collisions, collisions_info=info, collisions_untimed=untimed, half_width_m=half_w, half_length_m=half_l,
        max_steer_rad=max_steer,
        safe_stop=bool((summ.get("supervisor") or {}).get("safe_stop", False)), meta=meta)


# -------------------------------------------------------------------- CLI
def main(argv=None):
    import argparse
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="09 Monitor v3: grade a fault/baseline pair on SR1..SR6")
    ap.add_argument("--fault-run", required=True)
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--no-reaction-run", default=None, help="same fault without supervisor (FTTI for SR1)")
    ap.add_argument("--onset", type=float, required=True, help="planned fault onset [s]")
    ap.add_argument("--spec", default=str(here / "safety_spec.yaml"))
    ap.add_argument("--lane-width", type=float, default=None)
    ap.add_argument("--half-width", type=float, default=None)
    ap.add_argument("--half-length", type=float, default=None)
    ap.add_argument("--run-info", default=None, help="json file: scenario_id, parameters, hypothesis_arms ...")
    ap.add_argument("--out", default=None, help="append the run_record to this jsonl")
    ap.add_argument("--patterns", default=None,
                    help="02_hypotheses/vulnerability_patterns.yaml: metric_rules for gp_targets")
    ap.add_argument("--deadline-ms", type=float, default=None, help="MPC configured_time_limit_ms (deadline_miss_rate)")
    a = ap.parse_args(argv)
    cfg = load_spec(a.spec, lane_width_default_m=a.lane_width, vehicle_half_width_m=a.half_width,
                    vehicle_half_length_m=a.half_length, deadline_ms=a.deadline_ms)
    info = json.loads(Path(a.run_info).read_text()) if a.run_info else None
    rec = evaluate_pair(load_av_stack_run(a.fault_run), load_av_stack_run(a.baseline_run), a.onset, cfg,
                        load_av_stack_run(a.no_reaction_run) if a.no_reaction_run else None, info,
                        load_metric_rules(a.patterns) if a.patterns else None)
    rec["metrics"]["monitor"]["run_dirs"] = {"fault": str(Path(a.fault_run).resolve()), "baseline": str(Path(a.baseline_run).resolve()),
                       "no_reaction": str(Path(a.no_reaction_run).resolve()) if a.no_reaction_run else None}
    if a.out:  # write first so a closed stdout cannot lose the record
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    print(json.dumps(rec, indent=2, default=str, ensure_ascii=False))
    return rec


if __name__ == "__main__":
    main()
