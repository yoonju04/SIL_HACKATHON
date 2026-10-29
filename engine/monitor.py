"""09 Monitor: sim-eval av_stack run folder -> run_record (09_monitor/error_table.yaml).

Input files (written by sim-eval integration/pdm_agent.py + integration/monitor_logs.py, joined on t):
  steps.jsonl          t, vx, vy, yaw_rate, e_y_lane, e_psi              (monitor_logs.py:121-122)
  behavior.jsonl       t, active_controller, supervisor_reason, fault_injected (monitor_logs.py:123-124)
  monitor_probe.jsonl  t, lane_offset_m, lane_heading_err_rad, lane_width_m, ay_mps2, bbox_* (monitor_logs.py:80-91)
  collisions.json      [{t, other}]                                        (monitor_logs.py:115-116, 136)
  summary.json         {supervisor: {safe_stop}, meta: {fault_mode, fault_start, ...}} (monitor_logs.py:137-138)
  agent_controls.jsonl t, steer (CARLA normalised), tracking_error, heading_error, switched_at_s,
                       switch_count, primary_elapsed_ms, ...               (pdm_agent.py:404-418)
  agent_config.json    run options + geometry.max_steer_rad                (run_benchmark.py:102-107, tfpp_benchmark.py:111-113)
  front_frames.jsonl / front.mp4 / view.mp4 (only with --record-front)     (benchmark_recording.py, chase_view.py)

The judgment rules are the pure functions in engine/judgment.py. Thresholds are PROVISIONAL.
The monitor is the ONLY component that sets safety labels; 10_video_explainer never writes here.

Entry point for the loop and the CLI: evaluate(run_dir, spec, baseline_dir=..., no_reaction_dir=...) -> run_record.
It runs our recovery v2 (compute_run_record) and then the SR monitor v3.1 (pipeline/09_monitor/monitor.py through
engine/sr_checks.py) and adds its SR1..SR6 result as sr_* fields.

CLI (repository root; replaces both pipeline/09_monitor/monitor.py versions):
  python -m engine.monitor RUN_DIR --baseline BASELINE_DIR [--no-reaction NO_REACTION_DIR] [--spec safety_spec.yaml] [--out rr.jsonl]
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import judgment as J

REQUIRED_FILES = ("steps.jsonl", "behavior.jsonl", "monitor_probe.jsonl")


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a run killed mid-write can leave a partial last line
    return out


def _json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _key(t: Any) -> float:
    return round(float(t), 4)


@dataclass
class RunData:
    run_dir: Path
    ticks: list[dict]                       # joined rows sorted by t
    collisions: list[dict]
    summary: dict
    config: dict
    manifest: dict
    front_frames: list[dict]
    missing_files: list[str] = field(default_factory=list)

    @property
    def dt(self) -> float:
        return float(self.config.get("dt") or 0.05)

    @property
    def synthetic(self) -> bool:
        return bool(self.config.get("synthetic_fixture") or (self.summary.get("meta") or {}).get("synthetic_fixture"))

    def video(self, prefer: str = "front") -> Path | None:
        order = ("front.mp4", "view.mp4") if prefer == "front" else ("view.mp4", "front.mp4")
        for name in order:
            if (self.run_dir / name).is_file():
                return self.run_dir / name
        return None


def load_av_stack_run(run_dir: str | Path) -> RunData:
    """Load one av_stack run folder; rows of the monitor files are joined on t (rounded to 1e-4 s)."""
    run_dir = Path(run_dir)
    missing = [n for n in REQUIRED_FILES if not (run_dir / n).is_file()]
    tables = {
        "steps": _jsonl(run_dir / "steps.jsonl"),
        "behavior": _jsonl(run_dir / "behavior.jsonl"),
        "probe": _jsonl(run_dir / "monitor_probe.jsonl"),
        "controls": _jsonl(run_dir / "agent_controls.jsonl"),
    }
    joined: dict[float, dict] = {}
    for name in ("steps", "behavior", "probe"):
        for row in tables[name]:
            joined.setdefault(_key(row["t"]), {"t": _key(row["t"])}).update(
                {k: v for k, v in row.items() if k != "t"})
    for row in tables["controls"]:   # agent clock is unrounded there; only fill ticks the monitor files have
        k = _key(row["t"])
        if k in joined:
            joined[k].update({("ctrl_" + key): v for key, v in row.items() if key != "t"})
    ticks = [joined[k] for k in sorted(joined)]
    return RunData(run_dir=run_dir, ticks=ticks, collisions=_json(run_dir / "collisions.json", []),
                   summary=_json(run_dir / "summary.json", {}), config=_json(run_dir / "agent_config.json", {}),
                   manifest=_json(run_dir / "manifest.json", {}), front_frames=_jsonl(run_dir / "front_frames.jsonl"),
                   missing_files=missing)


def _fault_config(run: RunData) -> dict[str, Any]:
    cfg, meta = run.config, run.summary.get("meta") or {}
    mode = cfg.get("mpc_fault", meta.get("fault_mode", "none")) or "none"
    return dict(mode=mode, start=cfg.get("fault_start", meta.get("fault_start")),
                duration=cfg.get("fault_duration"), magnitude=cfg.get("fault_magnitude", meta.get("fault_magnitude")),
                on_s=cfg.get("fault_on_s"), off_s=cfg.get("fault_off_s"),
                level=cfg.get("fault_level", 1.0), ramp_s=cfg.get("fault_ramp_s", 0.0))


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def thresholds_from_config(stage_cfg: dict | None) -> dict[str, float]:
    """PROVISIONAL thresholds. 02 outcome_definitions.recovery_judgment first, then 09 safety_spec.yaml overrides
    (P-02, P-04, after_switch_window_s, safety_specification.recovery_judgment) — safety_spec is the single source."""
    th = dict(J.PROVISIONAL_RECOVERY)
    od = ((stage_cfg or {}).get("vulnerability_patterns") or {}).get("outcome_definitions") or {}
    rj = (od.get("recovery_judgment") or {}).get("thresholds") or {}
    for k in ("e_settle_m", "psi_settle_rad", "settle_hold_s", "window_s"):
        if isinstance(rj.get(k), dict) and _finite(rj[k].get("value")):
            th[k] = float(rj[k]["value"])
    ss = ((stage_cfg or {}).get("safety_spec") or {}).get("safety_specification") or {}
    props = {p.get("id"): p for p in ss.get("properties") or [] if isinstance(p, dict)}
    if _finite((props.get("P-02") or {}).get("lateral_acc_threshold_mps2")):
        th["lat_acc_threshold_mps2"] = float(props["P-02"]["lateral_acc_threshold_mps2"])
    if _finite((props.get("P-04") or {}).get("lateral_jerk_threshold_mps3")):
        th["lat_jerk_threshold_mps3"] = float(props["P-04"]["lateral_jerk_threshold_mps3"])
    if _finite(ss.get("after_switch_window_s")):
        th["window_s"] = float(ss["after_switch_window_s"])
    rjs = ss.get("recovery_judgment") or {}          # safety_spec wins over 02 (single source of thresholds)
    for k in ("e_settle_m", "psi_settle_rad", "settle_hold_s", "window_s", "filter_s", "collision_penalty"):
        if _finite(rjs.get(k)):
            th[k] = float(rjs[k])
    clip = rjs.get("robustness_clip")
    if isinstance(clip, (list, tuple)) and len(clip) == 2 and all(_finite(x) for x in clip):
        th["robustness_clip_min"], th["robustness_clip_max"] = float(clip[0]), float(clip[1])
    # BUG-022: P-01 uses the SR monitor's evaluation window (planned onset + length_s), not onset .. log end
    ew = ((stage_cfg or {}).get("safety_spec") or {}).get("evaluation_window") or {}
    th["eval_window_s"] = float(ew["length_s"]) if _finite(ew.get("length_s")) else 10.0
    return th


def run_friction_scale(run: "RunData | None", candidate: dict | None = None) -> float:
    """Tyre friction scale the run actually drove with (integration3). run_benchmark.py records what it applied in
    agent_config.json tire_friction {scale, applied}; a scale that could not be applied counts as 1.0 (CARLA tyres).
    Falls back to agent_config tire_friction_scale, then the candidate's environment, then 1.0."""
    cfg = (run.config if run is not None else None) or {}
    tf = cfg.get("tire_friction")
    if isinstance(tf, dict) and _finite(tf.get("scale")):
        return float(tf["scale"]) if tf.get("applied", True) else 1.0
    if _finite(cfg.get("tire_friction_scale")):
        return float(cfg["tire_friction_scale"])
    v = ((candidate or {}).get("environment") or {}).get("tire_friction_scale")
    return float(v) if _finite(v) else 1.0


def scale_thresholds_for_friction(th: dict[str, float], scale: float) -> dict[str, float]:
    """P-02 lateral-acc limit is 0.5 * mu * g with mu = 0.9 (safety_spec P-02 lateral_acc_formula); with a tyre
    friction scale mu becomes 0.9 * scale, so the limit scales linearly (scale 1.0 keeps 4.41 m/s^2)."""
    out = dict(th)
    out["tire_friction_scale"] = float(scale)
    out["lat_acc_threshold_nominal_mps2"] = th["lat_acc_threshold_mps2"]
    out["lat_acc_threshold_mps2"] = th["lat_acc_threshold_mps2"] * float(scale)
    return out


def footprint_margin_m(r: dict) -> float:
    """Signed margin of the vehicle rectangle (4 corners incl. heading error) to the intended lane, as SR4.

    Falls back to the centre-line margin (lane_width/2 - |offset| - half_width) when heading or length is missing."""
    w, off, hw = r["lane_width_m"], r["lane_offset_m"], r["bbox_half_width_m"]
    psi, hl = r.get("lane_heading_err_rad"), r.get("bbox_half_length_m")
    if not (_finite(psi) and _finite(hl)):
        return J.lane_margin_m(w, off, hw)
    s, c = math.sin(psi), math.cos(psi)
    corners = [off + xc * s + yc * c for xc in (hl, -hl) for yc in (hw, -hw)]
    return min(w / 2 - max(corners), w / 2 + min(corners))


def _rate_limit_dps(run: RunData) -> tuple[float, str]:
    # pdm_agent.py:86 TAKEOVER_RATE_DPS = 90 for the PDM path; SupervisedLateral default 60 (supervised_lateral.py:113)
    if (run.config.get("planner") or run.manifest.get("planner")) == "pdm":
        return 90.0, "config:pdm_agent.py:86 (not in run meta, logging L-2)"
    return 60.0, "config:supervised_lateral.py:113 (not in run meta, logging L-2)"


def _rmse(xs: list) -> float | None:
    xs = [float(x) for x in xs if _finite(x)]
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) if _finite(r.get(key)) else float("nan") for r in rows]


def compute_events(run: RunData, th: dict[str, float] | None = None) -> dict[str, Any]:
    """Event timeline used by run_record and by the 10 video explainer verification."""
    th = th or J.PROVISIONAL_RECOVERY
    ticks = run.ticks
    fault = _fault_config(run)
    ev: dict[str, Any] = {"fault": fault}
    actual = next((r["t"] for r in ticks if r.get("fault_injected")), None)
    ev["planned_fault_onset"] = fault["start"] if fault["mode"] != "none" else None
    ev["actual_fault_onset"] = actual
    # trigger / switch: behavior.jsonl active_controller leaves 'mpc' on the switch tick (same-tick switch,
    # supervised_lateral.py:144-148); agent_controls switched_at_s is the supervisor's own stamp.
    sw_row = next((r for r in ticks if r.get("active_controller") not in (None, "mpc")), None)
    detect_row = next((r for r in ticks if r.get("supervisor_reason")), None)
    # trigger = the fault first changed the MPC output (runner contract: no_trigger = fault had no effect).
    # av_stack logs fault_effective per tick (BUG-002); older logs without it fall back to the injection onset.
    if any(r.get("fault_effective") is not None for r in ticks):
        trig_t = next((r["t"] for r in ticks if r.get("fault_effective")), None)
    else:
        trig_t = actual
    switched_at = None
    for r in ticks:
        if _finite(r.get("ctrl_switched_at_s")):
            switched_at = float(r["ctrl_switched_at_s"])
            break
    ev["switch_time"] = _key(switched_at) if switched_at is not None else (sw_row["t"] if sw_row else None)
    ev["trigger_time"] = trig_t
    ev["detection_time"] = detect_row["t"] if detect_row else None   # supervisor first flagged the MPC
    ev["trigger_source"] = detect_row.get("supervisor_reason") if detect_row else None
    counts = [int(r["ctrl_switch_count"]) for r in ticks if _finite(r.get("ctrl_switch_count"))]
    ev["switch_count"] = max(counts) if counts else (1 if sw_row else 0)
    stop_row = next((r for r in ticks if r.get("active_controller") == "safe_stop"), None)
    ev["safe_stop_time"] = stop_row["t"] if stop_row else None
    # lane departures over the whole run
    margins = [J.lane_margin_m(r["lane_width_m"], r["lane_offset_m"], r["bbox_half_width_m"])
               if all(_finite(r.get(k)) for k in ("lane_width_m", "lane_offset_m", "bbox_half_width_m")) else float("nan")
               for r in ticks]
    intervals, start = [], None
    for r, m in zip(ticks, margins):
        if m < 0 and start is None:
            start = r["t"]
        elif not (m < 0) and start is not None:
            intervals.append([start, r["t"]]); start = None
    if start is not None:
        intervals.append([start, ticks[-1]["t"]])
    ev["departure_intervals"] = intervals
    # how far outside the lane (m) at each departed tick: lets the video check ask only for VISIBLE departures (BUG-020)
    ev["departure_depth"] = [[r["t"], round(-m, 3)] for r, m in zip(ticks, margins) if m < 0]
    t_ref = ev["switch_time"] if ev["switch_time"] is not None else (ev["planned_fault_onset"] or 0.0)
    after = [iv for iv in intervals if iv[1] >= t_ref - 1e-9]
    # if the car is already outside the lane at t_ref the "departure after switch" is t_ref itself
    ev["first_departure_after_switch"] = max(after[0][0], t_ref) if after else None
    ev["first_departure_time"] = intervals[0][0] if intervals else None
    ev["collision_times"] = [c.get("t") for c in run.collisions]
    if ticks:
        i_max = max(range(len(ticks)), key=lambda i: abs(ticks[i].get("lane_offset_m") or 0.0))
        ev["max_abs_lane_offset_m"] = abs(ticks[i_max].get("lane_offset_m") or 0.0)
        ev["max_abs_lane_offset_t"] = ticks[i_max]["t"]
    return ev


def _window(rows: list[dict], t0: float, t1: float) -> list[dict]:
    return [r for r in rows if t0 - 1e-9 <= r["t"] <= t1 + 1e-9]


# integration3 #8: av_stack supervisor profiles (integration/supervised_lateral.py SUPERVISOR_PROFILES):
# tracking_divergence (|e_y| m, |e_psi| rad). Runs written before `--supervisor` existed have no key = legacy.
SUPERVISOR_DIVERGENCE = {"tight": (0.6, 0.35), "legacy": (1.0, 0.6)}


def supervisor_profile(run: RunData) -> str:
    return str((run.config or {}).get("supervisor") or "legacy")


def supervisor_thresholds(run: RunData) -> tuple[float, float]:
    """(e_y_m, e_psi_rad) of the tracking_divergence rule the run was recorded with (normalized_switch_error)."""
    return SUPERVISOR_DIVERGENCE.get(supervisor_profile(run), SUPERVISOR_DIVERGENCE["legacy"])


def compute_run_record(run: RunData, baseline: RunData | None = None, *, candidate: dict | None = None,
                       metric_rules: dict | None = None, th: dict[str, float] | None = None,
                       run_id: str | None = None, round_index: int | None = None,
                       selector: str | None = None) -> dict[str, Any]:
    """One run_record dict with every field of RUN_RECORD_FIELDS (+ 'events' kept separately by callers)."""
    th = th or J.PROVISIONAL_RECOVERY
    cand = candidate or {}
    dt = run.dt
    ticks = run.ticks
    ev = compute_events(run, th)
    invalid_reason = None
    if run.missing_files:
        invalid_reason = "missing_files:" + ",".join(run.missing_files)
    elif not ticks:
        invalid_reason = "empty_logs"
    switched = ev["switch_time"] is not None
    triggered = ev["trigger_time"] is not None
    t_sw = ev["switch_time"]
    window_s = th["window_s"]
    vx_all = _series(ticks, "vx")
    s_all = J.travelled_distance([0.0 if math.isnan(v) else v for v in vx_all], dt)
    for r, s in zip(ticks, s_all):
        r["_s"] = s
    a_all = J.moving_average([0.0 if math.isnan(a) else a for a in _series(ticks, "ay_mps2")], th["filter_s"], dt)
    j_all = J.lateral_jerk([0.0 if math.isnan(a) else a for a in _series(ticks, "ay_mps2")], dt, th["filter_s"])
    for r, a, j in zip(ticks, a_all, j_all):
        r["_ay_f"], r["_jerk"] = a, j
    end_t = ticks[-1]["t"] if ticks else 0.0
    eval_start = ev["planned_fault_onset"] if ev["planned_fault_onset"] is not None else (ticks[0]["t"] if ticks else 0.0)
    if switched:
        win = _window(ticks, t_sw, t_sw + window_s)
        window_complete = bool(win) and end_t >= t_sw + window_s - dt / 2
    else:
        win = _window(ticks, eval_start, end_t)
        window_complete = bool(win)
    # A collision inside the available post-switch data is decisive (unrecovered) even if the run ended
    # before t_sw + window (CARLA/synthetic runs can stop after a crash). Without a collision, an
    # incomplete window stays simulation_invalid (never counted as safe).
    collision_after_switch = switched and any(_finite(c.get("t")) and c["t"] >= t_sw - 1e-9 for c in run.collisions)
    if switched and not window_complete and collision_after_switch and win:
        window_complete = True
    if invalid_reason is None and switched and not window_complete:
        invalid_reason = "route_ended_before_window_end"
    lane_ok = all(_finite(r.get("lane_width_m")) for r in win)
    if invalid_reason is None and not lane_ok:
        invalid_reason = "lane_width_nan"

    def margins_of(rows):
        return [J.lane_margin_m(r["lane_width_m"], r["lane_offset_m"], r["bbox_half_width_m"]) for r in rows]

    margins = margins_of(win) if lane_ok else []
    e_y = [float(r["lane_offset_m"]) for r in win] if lane_ok else []
    e_psi = [float(r["lane_heading_err_rad"]) for r in win] if lane_ok else []
    col_in = [c for c in run.collisions if _finite(c.get("t")) and win and win[0]["t"] - 1e-9 <= c["t"] <= win[-1]["t"] + 1e-9]
    outcome = J.classify_recovery(switched=switched, window_complete=window_complete and invalid_reason is None,
                                  margins=margins, e_y=e_y, e_psi=e_psi, collision=bool(col_in), dt=dt, th=th)
    if invalid_reason is not None:
        outcome = "simulation_invalid"
    # labels: baseline-relative (primary) and total (secondary)
    acc_total = max((abs(r["_ay_f"]) for r in win), default=None)
    jerk_total = max((abs(r["_jerk"]) for r in win), default=None)
    acc_excess = jerk_excess = None
    base_win = []
    if baseline is not None and baseline.ticks:
        b_vx = [0.0 if math.isnan(v) else v for v in _series(baseline.ticks, "vx")]
        b_s = J.travelled_distance(b_vx, baseline.dt)
        b_ay = [0.0 if math.isnan(a) else a for a in _series(baseline.ticks, "ay_mps2")]
        b_a = J.moving_average(b_ay, th["filter_s"], baseline.dt)
        b_j = J.lateral_jerk(b_ay, baseline.dt, th["filter_s"])
        if win:
            acc_excess = J.excess_over_baseline([r["_s"] for r in win], [r["_ay_f"] for r in win], b_s, b_a)
            jerk_excess = J.excess_over_baseline([r["_s"] for r in win], [r["_jerk"] for r in win], b_s, b_j)
        base_win = _window(baseline.ticks, eval_start, min(baseline.ticks[-1]["t"], eval_start + th.get("eval_window_s", 10.0)))
    margin_at_switch = None
    if switched:
        r_sw = next((r for r in ticks if r["t"] >= t_sw - 1e-9), None)
        if r_sw is not None and _finite(r_sw.get("lane_width_m")):
            margin_at_switch = J.lane_margin_m(r_sw["lane_width_m"], r_sw["lane_offset_m"], r_sw["bbox_half_width_m"])
    safe_stop = bool((run.summary.get("supervisor") or {}).get("safe_stop")) or ev["safe_stop_time"] is not None
    labels = J.recovery_labels(acc_excess=acc_excess, acc_total_max=acc_total, jerk_excess=jerk_excess,
                               jerk_total_max=jerk_total, margin_at_switch=margin_at_switch, safe_stop=safe_stop, th=th)
    final = J.final_safe_recovery(outcome, labels)
    category = J.classify_outcome(simulation_invalid=outcome == "simulation_invalid", triggered=triggered,
                                  switched=switched, final_safe=final)
    lrr = None
    if switched and outcome != "simulation_invalid" and margins:
        lrr = J.lane_recovery_robustness(margins, [float(r["lane_width_m"]) for r in win],
                                         float(win[0]["bbox_half_width_m"]), bool(col_in), th["collision_penalty"],
                                         (th.get("robustness_clip_min", -2.0), th.get("robustness_clip_max", 1.0)))
    # integration3 #9: signed safety margin over every final_safe_recovery criterion (GP target, judgment.safety_margin)
    hold = None
    sm = None
    if switched and outcome != "simulation_invalid" and margins:
        hold = J.settle_hold_max_s(margins=margins, e_y=e_y, e_psi=e_psi, dt=dt, th=th)
        need_s = max(1, round(th["settle_hold_s"] / dt)) * dt     # classify_recovery holds this many samples
        sm = J.safety_margin(lrr=lrr, collision=bool(col_in), acc_excess=acc_excess,
                             acc_limit=th["lat_acc_threshold_mps2"], jerk_excess=jerk_excess,
                             jerk_limit=th["lat_jerk_threshold_mps3"], settle_hold_s=hold, settle_need_s=need_s,
                             clip=(th.get("robustness_clip_min", -2.0), th.get("robustness_clip_max", 1.0)),
                             detail={"lane": {"min_lane_margin_m": min(margins)}})
    # P-01..03 over the evaluation window (planned onset .. onset + eval_window_s, as the SR monitor; BUG-022)
    eval_end = eval_start + th.get("eval_window_s", 10.0)
    ewin = _window(ticks, eval_start, min(end_t, eval_end))
    em = [footprint_margin_m(r) for r in ewin if all(_finite(r.get(k)) for k in ("lane_width_m", "lane_offset_m", "bbox_half_width_m"))]
    p01 = J.robustness_p01(em) if em else None
    p02 = J.robustness_p02([r["_ay_f"] for r in ewin], th["lat_acc_threshold_mps2"]) if ewin else None
    p03 = not any(_finite(c.get("t")) and eval_start - 1e-9 <= c["t"] <= eval_end + 1e-9 for c in run.collisions)
    mdelta = None
    if baseline is not None and base_win and p01 is not None:
        bm = [footprint_margin_m(r) for r in base_win
              if all(_finite(r.get(k)) for k in ("lane_width_m", "lane_offset_m", "bbox_half_width_m"))]
        if bm:
            mdelta = J.margin_delta_m(p01, J.robustness_p01(bm))
    # switch_state (S1 derived axes) + handover saturation
    max_steer = (run.config.get("geometry") or {}).get("max_steer_rad")
    rate_dps, rate_src = _rate_limit_dps(run)
    switch_state = handover = None
    command_jump = max_rate = None
    if switched:
        r_sw = next((r for r in ticks if r["t"] >= t_sw - 1e-9), {})
        ey_sw, epsi_sw = r_sw.get("ctrl_tracking_error"), r_sw.get("ctrl_heading_error")
        vx, vy = r_sw.get("vx"), r_sw.get("vy")
        eydot = (vx * math.sin(epsi_sw) + vy * math.cos(epsi_sw)) if all(_finite(x) for x in (vx, vy, epsi_sw)) else None
        ey_th, epsi_th = supervisor_thresholds(run)
        nse = max(abs(ey_sw) / ey_th, abs(epsi_sw) / epsi_th) if _finite(ey_sw) and _finite(epsi_sw) else None
        switch_state = dict(e_y_at_switch=ey_sw, e_psi_at_switch=epsi_sw, e_y_dot_at_switch=eydot,
                            e_psi_dot_at_switch=None, normalized_switch_error=nse, speed_at_switch_mps=vx,
                            kappa_at_switch=None, lane_offset_at_switch=r_sw.get("lane_offset_m"),
                            note="e_psi_dot/kappa need logging L-3")
        hw = _window(ticks, t_sw, t_sw + 1.0)
        steer = [r.get("ctrl_steer") for r in hw if _finite(r.get("ctrl_steer"))]
        if steer and _finite(max_steer):
            eps = 1e-3
            ang = [abs(x) >= 1 - eps for x in steer]
            rates = [abs(steer[i] - steer[i - 1]) * max_steer / dt for i in range(1, len(steer))]
            rate_sat = [math.degrees(x) >= rate_dps - 0.5 for x in rates]
            both = [a or (rs if i > 0 else False) for i, (a, rs) in enumerate(zip(ang, [False] + rate_sat))]
            handover = dict(window_s=1.0, max_steer_rad=max_steer, rate_limit_dps=rate_dps, rate_limit_source=rate_src,
                            angle_saturation_ratio=sum(ang) / len(ang),
                            rate_saturation_ratio=(sum(rate_sat) / len(rate_sat)) if rate_sat else None,
                            time_in_saturation_s=sum(both) * dt, first_lqr_command_unclipped_rad=None,
                            predicted_first_lqr_command_ratio=None, required_steer_rate_ratio=None,
                            note="unclipped LQR command needs logging L-1")
            prev = next((r for r in reversed(ticks) if r["t"] < t_sw - 1e-9 and _finite(r.get("ctrl_steer"))), None)
            if prev is not None:
                command_jump = abs(steer[0] - prev["ctrl_steer"]) * max_steer
            max_rate = math.degrees(max(rates)) if rates else None
    pre_sw = _window(ticks, ev["actual_fault_onset"] if ev["actual_fault_onset"] is not None else eval_start,
                     t_sw if switched else end_t)
    elapsed = [r.get("ctrl_primary_elapsed_ms") for r in ewin if r.get("active_controller") == "mpc"
               and _finite(r.get("ctrl_primary_elapsed_ms"))]
    deadline_ms = 1000.0 * dt  # supervised_lateral.py:105 deadline_ms = dt*1000
    metrics = {
        "recovery_success": None if final is None else float(final),
        "lane_recovery_robustness": lrr,
        "lat_acc_excess_mps2": acc_excess if switched else None,
        "lat_jerk_excess_mps3": jerk_excess if switched else None,
        "pre_switch_max_lateral_error_m": max((abs(r["lane_offset_m"]) for r in pre_sw if _finite(r.get("lane_offset_m"))), default=None),
        "switch_count": ev["switch_count"],
        "lateral_error_rmse_m": _rmse([r.get("lane_offset_m") for r in ewin]),
        "deadline_miss_rate": (sum(e > deadline_ms for e in elapsed) / len(elapsed)) if elapsed else None,
        "mpc_constraint_violation_rate": None,
        "post_switch_saturation_ratio": (handover["time_in_saturation_s"] / 1.0) if handover else None,
        "command_jump_rad": command_jump,
        "max_steering_rate_dps": max_rate,
        "margin_delta_m": mdelta,
        "recovery_robustness": None,  # SUPERSEDED (02 outcome_definitions)
        "settle_hold_max_s": hold,
        "safety_margin": None if sm is None else sm["value"],
        "safety_margin_binding": None if sm is None else sm["binding"],
        "safety_margin_detail": sm,
    }
    baseline_metrics = None
    if baseline is not None and base_win:
        baseline_metrics = {
            "pre_switch_max_lateral_error_m": max((abs(r["lane_offset_m"]) for r in base_win), default=None),
            "switch_count": compute_events(baseline, th)["switch_count"],
            "lateral_error_rmse_m": math.sqrt(sum(r["lane_offset_m"] ** 2 for r in base_win) / len(base_win)),
        }
    has_fault = ev["fault"]["mode"] != "none"
    gp_t = J.gp_targets(metric_rules, metrics, baseline_metrics, has_fault) if metric_rules else {}
    if not switched:  # absolute_post_switch targets are undefined without a switch (06 surrogate.yaml)
        for name, rule in (metric_rules or {}).items():
            if rule.get("gp", {}).get("form") == "absolute_post_switch":
                gp_t.pop(name, None)
    if outcome == "simulation_invalid":
        gp_t = {}
    attribution = "no_switch" if not switched else (
        "no_fault_configured" if not has_fault else (
            "inside_fault_window" if ev["fault"]["start"] is not None and
            ev["fault"]["start"] - 1e-9 <= t_sw < ev["fault"]["start"] + float(ev["fault"]["duration"] or 0) + 1e-9
            else "outside_fault_window"))
    record = J.build_run_record(
        run_id=run_id or run.run_dir.name, scenario_id=cand.get("scenario_id"),
        seed_scenario_id=cand.get("seed_scenario_id"), round=round_index, selector=selector or cand.get("source"),
        profile_id=cand.get("profile_id", "sp-av-stack-pdm-supervised"), source_commit=cand.get("source_commit"),
        feature_schema_version=cand.get("feature_schema_version"), hypothesis_ids=cand.get("hypothesis_ids"),
        parameters=cand.get("parameters") or {k: cand.get(k) for k in ("route_id", "environment", "fault", "onset_event", "onset_offset_s") if k in cand},
        planned_fault_onset=ev["planned_fault_onset"], actual_fault_onset=ev["actual_fault_onset"],
        trigger_time=ev["trigger_time"], trigger_source=ev["trigger_source"], trigger_attribution=attribution,
        switch_time=t_sw, switch_count=ev["switch_count"], switch_state=switch_state, handover_saturation=handover,
        outcome_category=category, recovery_outcome=outcome, recovery_labels=labels, final_safe_recovery=final,
        lane_recovery_robustness=lrr, lat_acc_excess_mps2=metrics["lat_acc_excess_mps2"],
        P01=p01, P02=p02, P03=p03, margin_delta_m=mdelta, switch_state_label=None, metrics=metrics,
        baseline_metrics=baseline_metrics, gp_targets=gp_t, gp_mean=cand.get("gp_mean"), gp_std=cand.get("gp_std"),
        sim_status="invalid" if outcome == "simulation_invalid" else "ok", invalid_reason=invalid_reason)
    record["supervisor_profile"] = supervisor_profile(run)
    return record


def monitor_timeline(run: RunData, record: dict, th: dict[str, float] | None = None) -> dict[str, Any]:
    """Compact, explainer-facing event list (inputs to 10_video_explainer, never labels it may change)."""
    ev = compute_events(run, th)
    hs = record.get("handover_saturation") or {}
    return {
        "fault_mode": ev["fault"]["mode"], "fault_magnitude": ev["fault"]["magnitude"],
        "planned_fault_onset_s": ev["planned_fault_onset"], "actual_fault_onset_s": ev["actual_fault_onset"],
        "trigger_time_s": ev["trigger_time"], "detection_time_s": ev.get("detection_time"),
        "trigger_source": ev["trigger_source"],
        "switch_time_s": ev["switch_time"], "lane_departure_intervals_s": ev["departure_intervals"],
        "lane_departure_depth_m": ev.get("departure_depth") or [],
        "first_departure_after_switch_s": ev["first_departure_after_switch"],
        "collision_times_s": ev["collision_times"], "safe_stop_time_s": ev["safe_stop_time"],
        "max_abs_lane_offset_m": ev.get("max_abs_lane_offset_m"), "max_abs_lane_offset_t_s": ev.get("max_abs_lane_offset_t"),
        "angle_saturation_ratio_1s": hs.get("angle_saturation_ratio"),
        "rate_saturation_ratio_1s": hs.get("rate_saturation_ratio"),
        "recovery_outcome_decided_by_monitor": record.get("recovery_outcome"),
        "recovery_labels_decided_by_monitor": record.get("recovery_labels"),
    }


SR_FIELDS = ("sr_outcome", "sr_failed_requirements", "sr_labels", "sr_scores", "detection_latency_s", "ftti_s",
             "sr_checks")


def attach_sr_checks(record: dict[str, Any], run: RunData, baseline: RunData | None,
                     no_reaction: RunData | None, spec: dict | None, friction_scale: float = 1.0) -> dict[str, Any]:
    """SR monitor v3.1 grading (pipeline/09_monitor/monitor.py via engine.sr_checks) on the same run, added as
    extra run_record fields. Our recovery v2 result is recorded next to SR5 for comparison.
    SR results never change our labels, outcome_category or GP targets."""
    from . import sr_checks as SR

    onset = record.get("planned_fault_onset")
    out = {k: None for k in SR_FIELDS}
    if record.get("sim_status") == "invalid" or onset is None:
        out.update(sr_outcome="invalid" if record.get("sim_status") == "invalid" else None,
                   sr_checks={"valid": False, "invalid_reasons": [record.get("invalid_reason") or "no fault onset"]})
        record.update(out)
        return record
    cfg = SR.config_from_spec(spec, dt=run.dt)
    if abs(friction_scale - 1.0) > 1e-9:   # integration3: SR P-02 aux limit follows the tyres too (reference only)
        cfg.mu = cfg.mu * friction_scale
        if cfg.a_y_max_mps2 is not None:
            cfg.a_y_max_mps2 = cfg.a_y_max_mps2 * friction_scale
    g = SR.grade(SR.trace_from_run(run), SR.trace_from_run(baseline) if baseline is not None and baseline.ticks else None,
                 float(onset), cfg, SR.trace_from_run(no_reaction) if no_reaction is not None and no_reaction.ticks else None,
                 recovery={"recovery_outcome": record.get("recovery_outcome"),
                           "final_safe_recovery": record.get("final_safe_recovery")})
    g["no_reaction_run"] = str(no_reaction.run_dir) if no_reaction is not None else None
    out.update(sr_outcome=g["sr_outcome"], sr_failed_requirements=g["failed_requirements"], sr_labels=g["sr_labels"],
               sr_scores=g["sr_scores"], detection_latency_s=g["detection_latency_s"], ftti_s=g["ftti_s"], sr_checks=g)
    record.update(out)
    return record


def load_safety_spec(spec: str | Path | dict | None = None) -> dict:
    if isinstance(spec, dict):
        return spec
    from .config import PIPELINE_DIR, load_yaml
    return load_yaml(Path(spec) if spec else PIPELINE_DIR / "09_monitor" / "safety_spec.yaml") or {}


def evaluate(run_dir: str | Path, spec: str | Path | dict | None = None, *, baseline_dir: str | Path | None = None,
             no_reaction_dir: str | Path | None = None, stage_cfg: dict | None = None, metric_rules: dict | None = None,
             candidate: dict | None = None, run_id: str | None = None, round_index: int | None = None,
             selector: str | None = None, timeline: bool = False) -> dict[str, Any]:
    """THE 09 monitor entry point: one av_stack run folder (+ baseline [+ no-reaction]) -> one run_record.

    spec        09 safety_spec.yaml (path or dict; default the repo's). Thresholds come from it (+ 02 recovery geometry).
    stage_cfg   engine.config.load_stage_configs() (loaded if omitted) for metric_rules / recovery thresholds.
    Our recovery v2 (engine.judgment) decides recovery_outcome / labels / final_safe_recovery / GP targets;
    engine.sr_checks adds sr_outcome (F1/F2/F3/pass), sr_* fields and sr_labels."""
    if stage_cfg is None:
        from .config import load_stage_configs
        stage_cfg = load_stage_configs()
    spec_d = load_safety_spec(spec if spec is not None else stage_cfg.get("safety_spec"))
    th = thresholds_from_config({**stage_cfg, "safety_spec": spec_d})
    if metric_rules is None:
        metric_rules = (stage_cfg.get("vulnerability_patterns") or {}).get("metric_rules")
    run = load_av_stack_run(run_dir)
    base = load_av_stack_run(baseline_dir) if baseline_dir else None
    nore = load_av_stack_run(no_reaction_dir) if no_reaction_dir else None
    fscale = run_friction_scale(run, candidate)
    th = scale_thresholds_for_friction(th, fscale)
    rec = compute_run_record(run, base, candidate=candidate, metric_rules=metric_rules, th=th, run_id=run_id,
                             round_index=round_index, selector=selector)
    if isinstance(rec.get("metrics"), dict):
        rec["metrics"]["tire_friction_scale"] = fscale
        rec["metrics"]["p02_lat_acc_threshold_mps2"] = th["lat_acc_threshold_mps2"]
    attach_sr_checks(rec, run, base, nore, spec_d, fscale)
    if timeline:
        rec["_timeline"] = monitor_timeline(run, rec, th)
    return rec


def evaluate_folder(run_dir: str | Path, baseline_dir: str | Path | None = None, **kw) -> dict[str, Any]:
    """Recovery-v2-only record (no SR fields). Prefer evaluate()."""
    run = load_av_stack_run(run_dir)
    base = load_av_stack_run(baseline_dir) if baseline_dir else None
    return compute_run_record(run, base, **kw)


def main(argv: list[str] | None = None) -> int:
    """09 Monitor CLI: one av_stack run folder (+ no-fault baseline [+ no-reaction run]) -> run_record JSON."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m engine.monitor", description=main.__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", type=Path, help="same scenario, no fault (needed for labels and SR checks)")
    ap.add_argument("--no-reaction", type=Path, help="same fault, --controller mpc_unprotected (SR1 FTTI)")
    ap.add_argument("--spec", type=Path, help="09 safety_spec.yaml (default: pipeline/09_monitor/safety_spec.yaml)")
    ap.add_argument("--out", type=Path, help="write the record here (.jsonl: append one line)")
    args = ap.parse_args(argv)
    rec = evaluate(args.run_dir, args.spec, baseline_dir=args.baseline, no_reaction_dir=args.no_reaction, timeline=True)
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    if args.out:
        if args.out.suffix == ".jsonl":
            with args.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        else:
            args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
