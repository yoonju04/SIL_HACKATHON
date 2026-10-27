"""09 Monitor SR checks (ported from junwoo/sim-eval a5242b2 pipeline/09_monitor/monitor.py "Monitor v2").

What this module still decides (the teammate's judgment parts that our recovery v2 does not cover):
  SR1  switch timing vs FTTI: latency fault-effective -> LQR switch <= ftti_fraction * FTTI, where FTTI is measured
       on a no-reaction run (same fault, `--controller mpc_unprotected`); without that run a fixed limit is used.
  SR2  switch transient: max steering-command rate in [t_sw - dt, t_sw + window] <= limit_dps.
  SR4  lane keeping with the vehicle FOOTPRINT (rectangle corners incl. heading error) + collision.
  SR6  actuator limits: |steer| <= max_steer and steering rate <= max_rate_dps over the evaluation window.
  V-01..V-11 / G-04 validity of the (fault, baseline[, no-reaction]) set.

What it does NOT decide any more (our implementation wins where the two overlapped):
  SR5 recovery and the F3 outcome  -> engine.judgment.classify_recovery / final_safe_recovery (recovery v2).
                                      SR5 is only a derived view of recovery_outcome.
  P-02 lateral acceleration -> recovery_labels.lat_acc_excess (threshold now the teammate's 0.5*mu*g = 4.41 m/s^2).
  team metrics / runner_skeleton 28-field run_record / feedback / explain -> engine.monitor / engine.feedback / engine.explainer.

All numbers come from pipeline/09_monitor/safety_spec.yaml section sr_checks = the teammate's original values
(window onset + 10 s, min 5 s; SR1 0.5 x FTTI / 0.5 s; SR2 60 deg/s in 0.5 s; SR6 90 deg/s). PROVISIONAL.
Input: our engine's av_stack run folder, already loaded by engine.monitor.load_av_stack_run (RunData).
Output: a plain dict that engine.monitor.evaluate() spreads into run_record fields (sr_outcome, sr_checks, ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SR_OUTCOMES = ("invalid", "no_trigger", "F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered", "pass")
FAIL_OUTCOMES = ("F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered")


# ----------------------------------------------------------------- config
@dataclass
class SRConfig:
    dt: float = 0.05
    window_s: float = 10.0              # teammate's evaluation window: planned onset + 10 s
    min_window_s: float = 5.0           # shorter -> V-03 invalid (safe stop / crash excepted)
    tick_tolerance: float = 0.25
    onset_tolerance_s: float = 0.051
    sr1_ftti_fraction: float = 0.5
    sr1_fixed_limit_s: float = 0.5
    sr2_limit_dps: float = 60.0
    sr2_window_s: float = 0.5
    sr6_max_rate_dps: float = 90.0
    lane_width_default_m: float | None = None
    vehicle_half_width_m: float | None = None
    vehicle_half_length_m: float | None = None
    max_steer_rad: float | None = None
    gp_clip: tuple = (-2.0, 1.0)
    gp_collision_value: float = -2.0


def config_from_spec(spec: dict | None, **overrides) -> SRConfig:
    """SRConfig from our safety_spec.yaml dict, section sr_checks (the teammate's original numbers)."""
    sc = (spec or {}).get("sr_checks") or {}
    cfg = SRConfig()
    ew = sc.get("evaluation_window") or {}
    if ew.get("length_s") is not None:
        cfg.window_s = float(ew["length_s"])
    if ew.get("min_length_s") is not None:
        cfg.min_window_s = float(ew["min_length_s"])
    sr1, sr2, sr6 = sc.get("SR1") or {}, sc.get("SR2") or {}, sc.get("SR6") or {}
    val, gp = sc.get("validity") or {}, sc.get("gp_score") or {}
    for attr, src, key in (("sr1_ftti_fraction", sr1, "ftti_fraction"), ("sr1_fixed_limit_s", sr1, "fixed_fallback_s"),
                           ("sr2_limit_dps", sr2, "limit_dps"), ("sr2_window_s", sr2, "window_after_switch_s"),
                           ("sr6_max_rate_dps", sr6, "max_rate_dps"), ("tick_tolerance", val, "tick_tolerance"),
                           ("onset_tolerance_s", val, "onset_tolerance_s"), ("gp_collision_value", gp, "collision_value")):
        if src.get(key) is not None:
            setattr(cfg, attr, float(src[key]))
    if gp.get("clip") is not None:
        cfg.gp_clip = tuple(float(x) for x in gp["clip"])
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
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
    steer_cmd_rad: np.ndarray | None = None   # command actually sent to the vehicle [rad, right-handed]
    active_controller: list | None = None
    fault_active: np.ndarray | None = None
    fault_effective: np.ndarray | None = None
    route_revision: np.ndarray | None = None
    collisions_t: list = field(default_factory=list)
    collisions_info: list = field(default_factory=list)
    collisions_untimed: int = 0
    half_width_m: float | None = None
    half_length_m: float | None = None
    max_steer_rad: float | None = None
    safe_stop: bool = False


def _num_or_nan(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def trace_from_run(run) -> Trace:
    """Adapter: engine.monitor.RunData (our joined steps/behavior/monitor_probe/agent_controls ticks) -> Trace.

    steer_cmd_rad: behavior/steps `steer_rad` if logged (autonomy_episode), else agent_controls `steer` (CARLA
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


# --------------------------------------------------------------- helpers
def footprint_lane_margin(offset, heading_err, lane_width, half_w, half_l):
    """Signed margin of the vehicle rectangle to the intended lane's boundaries [m]."""
    offset = np.asarray(offset, float)
    dpsi = np.zeros_like(offset) if heading_err is None else np.asarray(heading_err, float)
    s, c = np.sin(dpsi), np.cos(dpsi)
    corners = np.stack([offset + xc * s + yc * c for xc in (half_l, -half_l) for yc in (half_w, -half_w)])
    half_lane = 0.5 * np.asarray(lane_width, float)
    return np.minimum(half_lane - corners.max(axis=0), half_lane + corners.min(axis=0))


def lateral_acceleration(tr: Trace, smoothing_ticks: int = 5):
    """Logged ay if present, else d(vy)/dt + vx*yaw_rate (moving average). Informational only (labels: our P-02)."""
    if tr.ay_mps2 is not None:
        return np.asarray(tr.ay_mps2, float)
    ay = np.gradient(np.asarray(tr.vy, float), tr.t) + np.asarray(tr.vx, float) * np.asarray(tr.yaw_rate, float)
    k = max(1, smoothing_ticks)
    if k > 1:
        ay = np.convolve(np.pad(ay, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return ay


def _first(mask, t):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if idx.size else None


def switch_times(tr: Trace) -> list[float]:
    if tr.active_controller is None:
        return []
    ac = list(tr.active_controller)
    return [float(tr.t[i]) for i in range(1, len(ac)) if ac[i] != ac[i - 1]]


def _clip(x, lo, hi):
    return float(min(hi, max(lo, x)))


def _num(x):
    if x is None:
        return None
    return None if (isinstance(x, float) and math.isinf(x)) else float(x)


# ---------------------------------------------------------- run checks
@dataclass
class RunCheck:
    valid: bool
    reasons: list
    window: tuple
    win: np.ndarray | None = None
    margin: np.ndarray | None = None
    nominal_margin: float | None = None


def check_run(tr: Trace, onset: float, cfg: SRConfig, kind: str) -> RunCheck:
    """Validity (V-01..V-11) and the SR4 footprint margin series. kind: fault | baseline | no_reaction.
    Window = [onset, min(onset + window_s, end of log)] for every run (teammate's original)."""
    t = np.asarray(tr.t, float)
    reasons: list[str] = []
    chans = [tr.vx, tr.vy, tr.yaw_rate, tr.lane_offset_m]
    if len(t) < 2 or any(c is None or len(c) != len(t) for c in chans):
        return RunCheck(False, [f"{kind}: V-01 empty or misaligned log"], (onset, onset))
    if not all(np.all(np.isfinite(np.asarray(c, float))) for c in chans):
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
    half_w = tr.half_width_m or cfg.vehicle_half_width_m
    half_l = tr.half_length_m or cfg.vehicle_half_length_m
    lane_w = tr.lane_width_m if tr.lane_width_m is not None else (
        None if cfg.lane_width_default_m is None else np.full(len(t), cfg.lane_width_default_m))
    margin = nominal = None
    if half_w is None or half_l is None or lane_w is None:
        reasons.append(f"{kind}: V-05 lane width or vehicle footprint unknown")
    else:
        margin = footprint_lane_margin(tr.lane_offset_m, tr.lane_heading_err, lane_w, half_w, half_l)
        nominal = float(0.5 * np.median(np.asarray(lane_w, float)[win]) - half_w)
    return RunCheck(not reasons, reasons, (onset, end), win, margin, nominal)


def measure_ftti(no_reaction: Trace, onset: float, cfg: SRConfig) -> dict:
    """FTTI from a no-reaction run: actual fault start -> first SR4 (footprint) or collision violation."""
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


# ------------------------------------------------------------- SR grading
def grade(fault: Trace, baseline: Trace | None, onset: float, cfg: SRConfig | None = None,
          no_reaction: Trace | None = None, recovery: dict | None = None) -> dict:
    """SR1/SR2/SR4/SR6 verdict for one (fault, baseline[, no-reaction]) set.

    recovery: our recovery v2 result for the same run {recovery_outcome, final_safe_recovery} (engine.judgment).
      It replaces the teammate's SR5 and decides F3. Without it (unit tests), F3 falls back to SR4 only.
    Returns {valid, invalid_reasons, sr_outcome, sr: {SR1, SR2, SR4, SR5(derived), SR6}, sr_scores, sr_measures,
             failed_requirements, sr_labels, detection_latency_s, ftti_s, note, ...}."""
    cfg = cfg or SRConfig()
    out: dict[str, Any] = {"valid": False, "invalid_reasons": [], "sr_outcome": "invalid", "sr": {}, "sr_scores": None,
                           "sr_measures": None, "failed_requirements": [], "sr_labels": None,
                           "detection_latency_s": None, "ftti_s": None, "note": None}
    if baseline is None:
        out["invalid_reasons"] = ["no baseline run (SR4 needs the paired no-fault run)"]
        return out
    fc = check_run(fault, onset, cfg, "fault")
    bc = check_run(baseline, onset, cfg, "baseline")
    reasons = fc.reasons + bc.reasons
    t = np.asarray(fault.t, float)
    actual_onset = trigger = None
    if fault.fault_active is not None and len(t):
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
        if any(s < t0 - 1e-9 for s in switch_times(fault)):
            reasons.append("fault: V-10 already switched before the fault started (switching cannot be graded)")
    out.update(actual_fault_onset=actual_onset, trigger_time=trigger)
    if reasons:
        out["invalid_reasons"] = reasons
        return out

    f_sw = switch_times(fault)
    post_sw = [s for s in f_sw if s >= (actual_onset if actual_onset is not None else onset) - 1e-9
               and s <= fc.window[1] + 1e-9]
    switch_time = post_sw[0] if post_sw else None
    sr: dict[str, dict] = {}

    # SR4 footprint lane keeping + collision
    w = fc.win
    p01_f = float(np.min(fc.margin[w]))
    p01_b = float(np.min(bc.margin[bc.win]))
    m_nom = fc.nominal_margin
    hits = [c for c in fault.collisions_t if fc.window[0] <= c <= fc.window[1]]
    sr4_fail = p01_f < 0 or bool(hits)
    sr["SR4"] = {"applies": True, "pass": not sr4_fail, "robustness": p01_f / m_nom,
                 "margin_m": p01_f, "baseline_margin_m": p01_b, "nominal_margin_m": m_nom,
                 "first_violation_t": _first(w & (fc.margin < 0), t), "collision": bool(hits)}

    # SR1 switch latency vs FTTI
    ftti, limit_src = None, "fixed_fallback"
    notes = []
    if no_reaction is not None:
        fr = measure_ftti(no_reaction, onset, cfg)
        if fr["valid"]:
            ftti, limit_src = fr["ftti_s"], "ftti_measured"
        else:
            notes.append("no_reaction run invalid: " + "; ".join(fr["reasons"]))
    triggered = trigger is not None
    if not triggered:
        sr1 = {"applies": False, "reason": "fault never effective"}
    elif ftti is not None and math.isinf(ftti):
        sr1 = {"applies": False, "reason": "undetected fault is harmless (no-reaction run keeps SR4)", "ftti_s": None}
    else:
        limit = max(cfg.sr1_ftti_fraction * ftti if ftti is not None else cfg.sr1_fixed_limit_s, cfg.dt)
        latency = (switch_time - actual_onset) if switch_time is not None else math.inf
        rob = -1.0 if math.isinf(latency) else _clip((limit - latency) / limit, -1.0, 1.0)
        sr1 = {"applies": True, "pass": latency <= limit, "robustness": rob, "latency_s": _num(latency),
               "limit_s": limit, "limit_source": limit_src, "ftti_s": _num(ftti)}
    sr["SR1"] = sr1

    # SR2 switch transient (command actually sent)
    if switch_time is None or fault.steer_cmd_rad is None:
        sr["SR2"] = {"applies": False, "reason": "no switch" if switch_time is None else "steer_cmd not logged"}
    else:
        m = (t >= switch_time - cfg.dt - 1e-9) & (t <= switch_time + cfg.sr2_window_s + 1e-9)
        st = np.asarray(fault.steer_cmd_rad, float)
        rates = np.abs(np.diff(st)) / np.diff(t)
        rate_mask = m[1:] & m[:-1]
        max_rate = float(np.degrees(rates[rate_mask].max())) if rate_mask.any() else 0.0
        sr["SR2"] = {"applies": True, "pass": max_rate <= cfg.sr2_limit_dps,
                     "robustness": (cfg.sr2_limit_dps - max_rate) / cfg.sr2_limit_dps,
                     "max_rate_dps": max_rate, "limit_dps": cfg.sr2_limit_dps}

    # SR5: derived from OUR recovery v2 (no own band/within/hold numbers)
    rec_out = (recovery or {}).get("recovery_outcome")
    if switch_time is None:
        sr["SR5"] = {"applies": False, "reason": "no switch"}
    elif recovery is None:
        sr["SR5"] = {"applies": False, "reason": "recovery v2 result not given"}
    else:
        sr["SR5"] = {"applies": True, "derived_from": "engine.judgment recovery v2",
                     "pass": rec_out == "safe_recovery", "recovery_outcome": rec_out,
                     "final_safe_recovery": (recovery or {}).get("final_safe_recovery")}

    # SR6 actuator limits
    max_steer = fault.max_steer_rad or cfg.max_steer_rad
    if fault.steer_cmd_rad is None or max_steer is None:
        sr["SR6"] = {"applies": False, "reason": "steer_cmd or max steer unknown"}
    else:
        st = np.asarray(fault.steer_cmd_rad, float)[w]
        tw = t[w]
        peak = float(np.max(np.abs(st)))
        rate = float(np.degrees(np.max(np.abs(np.diff(st)) / np.diff(tw)))) if len(st) > 1 else 0.0
        rob = min(1 - peak / max_steer, 1 - rate / cfg.sr6_max_rate_dps)
        sr["SR6"] = {"applies": True, "pass": rob >= -1e-9, "robustness": rob,
                     "max_steer_deg": math.degrees(peak), "max_rate_dps": rate,
                     "limit_steer_deg": math.degrees(max_steer), "limit_rate_dps": cfg.sr6_max_rate_dps}

    # outcome (priority order). F3 = our recovery v2 says not safe (fallback: SR4 only).
    final = (recovery or {}).get("final_safe_recovery")
    if recovery is not None and switch_time is not None:
        not_recovered = final is False or (final is None and sr4_fail)
    else:
        not_recovered = sr4_fail
    note = None
    if not triggered:
        outcome = "no_trigger"
    elif sr1.get("applies") and not sr1["pass"]:
        outcome = "F1_no_or_late_switch"
    elif sr["SR2"].get("applies") and not sr["SR2"]["pass"]:
        outcome = "F2_switch_transient"
    elif not_recovered:
        outcome = "F3_not_recovered"
    else:
        outcome = "pass"
        if switch_time is None:
            note = "undetected_harmless"
    failed = [k for k in ("SR1", "SR2", "SR4", "SR6") if sr[k].get("applies") and not sr[k].get("pass")]
    if sr["SR5"].get("applies") and not sr["SR5"]["pass"]:
        failed.append("SR5")

    lo, hi = cfg.gp_clip
    scores: dict[str, Any] = {
        "SR1": sr1["robustness"] if sr1.get("applies") else None,
        "SR2": _clip(sr["SR2"]["robustness"], lo, hi) if sr["SR2"].get("applies") else None,
        "SR4": cfg.gp_collision_value if sr["SR4"]["collision"] else _clip(p01_f / m_nom, lo, hi),
        "SR6": _clip(sr["SR6"]["robustness"], lo, hi) if sr["SR6"].get("applies") else None,
    }
    scores["overall"] = min(v for k, v in scores.items() if k in ("SR1", "SR2", "SR4") and v is not None)
    measures = {
        "SR1_latency_s": sr1.get("latency_s"), "SR1_limit_s": sr1.get("limit_s"),
        "SR1_no_switch": bool(sr1.get("applies")) and switch_time is None,
        "SR2_max_rate_dps": sr["SR2"].get("max_rate_dps"),
        "SR4_margin_m": p01_f, "SR4_margin_delta_m": p01_f - p01_b, "SR4_collision": sr["SR4"]["collision"],
        "SR4_collision_t": min(hits) if hits else None,
        "SR4_collision_other": next((c.get("other_actor") for c in fault.collisions_info
                                     if hits and c.get("t") == min(hits)), None),
        "SR6_max_steer_deg": sr["SR6"].get("max_steer_deg"), "SR6_max_rate_dps": sr["SR6"].get("max_rate_dps"),
    }
    labels = {  # extra boolean labels next to recovery_labels (None = not applicable)
        "late_or_no_switch": (not sr1["pass"]) if sr1.get("applies") else None,
        "switch_transient": (not sr["SR2"]["pass"]) if sr["SR2"].get("applies") else None,
        "footprint_departure": p01_f < 0,
        "actuator_limit_exceeded": (not sr["SR6"]["pass"]) if sr["SR6"].get("applies") else None,
    }
    out.update(valid=True, invalid_reasons=[], sr_outcome=outcome, sr=sr, sr_scores=scores, sr_measures=measures,
               failed_requirements=failed, sr_labels=labels, detection_latency_s=sr1.get("latency_s"),
               ftti_s=sr1.get("ftti_s") if ftti is None or not math.isinf(ftti) else None, note=note,
               switch_time=switch_time, switch_count=len(f_sw), baseline_switch_count=len(switch_times(baseline)),
               evaluation_window=list(fc.window), notes=notes)
    return out


def noise_floor(baseline_margins: list[float]) -> dict:
    """Spread of SR4 margins over repeated baselines -> minimum meaningful margin delta (calibration helper)."""
    a = np.asarray([v for v in baseline_margins if v is not None], float)
    if a.size < 2:
        return {"n": int(a.size), "noise_floor_m": None, "delta_min_m": None}
    nf = float(a.std(ddof=1))
    return {"n": int(a.size), "noise_floor_m": nf, "delta_min_m": max(2 * nf, 0.1)}
