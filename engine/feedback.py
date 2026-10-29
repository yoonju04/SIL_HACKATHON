"""09/10 -> 03/06 feedback: failure table, few-shot, mechanism features, taxonomy proposals, counterfactual queue.

Only explanations whose verification passed (10 video_explainer.yaml verification) are used here.
Nothing in this module changes a run_record or a monitor rule: rule candidates are written to a file
for human review (auto_apply: false).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from .scenarios import faults_by_id, make_candidate, numeric_features, _param_unit, _param_from_unit

FAILURE_FIELDS = ("scenario_id", "violated_property", "originating_evidence_ids", "hypothesis_ids", "trigger_source",
                  "trigger_attribution", "recovery_outcome", "recovery_labels", "environment", "fault", "onset_event",
                  "observed_switch_behavior", "robustness", "reproduction_count")


def is_failure(rec: dict) -> bool:
    if rec.get("outcome_category") == "switched_unsafe":
        return True
    if rec.get("recovery_outcome") == "no_switch":
        return (rec.get("P01") is not None and rec["P01"] < 0) or rec.get("P03") is False
    return False


def visible_failure(rec: dict) -> bool:
    """integration4: a failure a camera can show - lane departure or collision (SR4), late / no switch (SR1 F1), or a
    recovery that left the lane. Excludes runs that fail only on lateral acceleration / jerk (most of result_8's
    'failures' with the tight supervisor), which kept the lane and gave the video explainer nothing to explain."""
    s4 = ((rec.get("sr_checks") or {}).get("sr") or {}).get("SR4") or {}
    return (s4.get("pass") is False or bool(s4.get("collision")) or rec.get("sr_outcome") == "F1_no_or_late_switch"
            or rec.get("recovery_outcome") in ("recovered_after_departure", "unrecovered"))


def failure_table(records: list[dict], candidates: dict[str, dict]) -> list[dict]:
    rows = []
    seen: Counter = Counter()
    for r in records:
        if not is_failure(r):
            continue
        c = candidates.get(r.get("scenario_id"), {})
        key = (c.get("route_id"), (c.get("fault") or {}).get("mode"), c.get("onset_event"))
        seen[key] += 1
        violated = [p for p, bad in (("P-01", r.get("P01") is not None and r["P01"] < 0), ("P-03", r.get("P03") is False),
                                     ("recovery", r.get("final_safe_recovery") is False and r.get("switch_time") is not None)) if bad]
        rows.append({"run_id": r["run_id"], "scenario_id": r.get("scenario_id"), "violated_property": violated,
                     "originating_evidence_ids": c.get("evidence_ids"), "hypothesis_ids": r.get("hypothesis_ids"),
                     "trigger_source": r.get("trigger_source"), "trigger_attribution": r.get("trigger_attribution"),
                     "recovery_outcome": r.get("recovery_outcome"), "recovery_labels": r.get("recovery_labels"),
                     "environment": c.get("environment"), "fault": c.get("fault"), "onset_event": c.get("onset_event"),
                     "observed_switch_behavior": {"switch_time": r.get("switch_time"), "switch_state": r.get("switch_state")},
                     "robustness": r.get("lane_recovery_robustness") if r.get("lane_recovery_robustness") is not None else r.get("P01"),
                     "lane_recovery_robustness": r.get("lane_recovery_robustness"), "lane_margin_min_m": r.get("P01"),
                     "safety_margin": safety_margin_view(r),
                     "reproduction_count": seen[key], "route_id": c.get("route_id")})
    order = {"unrecovered": 0, "recovered_after_departure": 1, "unsettled": 2}
    rows.sort(key=lambda x: (order.get(x["recovery_outcome"], 3), x["robustness"] if x["robustness"] is not None else 0))
    return rows


def few_shot(table: list[dict], explanations: dict[str, dict], k: int = 5) -> list[dict]:
    """integration3 #7: no unit-less 'robustness' (it mixed lane_recovery_robustness and P01 metres); both are given
    under their own names."""
    out = []
    for row in table[:k]:
        e = explanations.get(row["run_id"])
        item = {f: row.get(f) for f in ("route_id", "environment", "fault", "onset_event", "trigger_source",
                                        "recovery_outcome", "recovery_labels", "lane_recovery_robustness",
                                        "lane_margin_min_m", "safety_margin")}
        item["switch_state"] = (row.get("observed_switch_behavior") or {}).get("switch_state")
        if e and e.get("verification", {}).get("accepted"):
            item["verified_video_mechanism"] = {"tags": e["explanation"].get("mechanism_tags"),
                                                "text": e["explanation"].get("mechanism")}
        out.append(item)
    return out


def safety_margin_view(r: dict) -> dict | None:
    """integration3 #9: the run's GP target (judgment.safety_margin) for Nemotron A in real units: value, binding
    criterion, failing criteria, and every term as observed vs limit."""
    sm = (r.get("metrics") or {}).get("safety_margin_detail")
    if not isinstance(sm, dict):
        return None
    rnd = lambda v, n=3: None if v is None else round(float(v), n)   # noqa: E731
    obs = sm.get("observed") or {}
    terms = {}
    for k, t in (sm.get("terms") or {}).items():
        o = obs.get(k) or {}
        row = {"margin": rnd(t), "observed": rnd(o.get("value")), "limit": rnd(o.get("limit")), "unit": o.get("unit")}
        if o.get("min_lane_margin_m") is not None:
            row["min_lane_margin_m"] = rnd(o["min_lane_margin_m"])
        terms[k] = {a: b for a, b in row.items() if b is not None}
    return {"value": rnd(sm.get("value")), "binding": sm.get("binding"), "failing": sm.get("failing") or [], "terms": terms}


def failed_reason(r: dict) -> str:
    """Why a switched run is not final_safe_recovery (judgment.final_safe_recovery)."""
    oc = str(r.get("recovery_outcome"))
    if oc == "safe_recovery":
        return "lat_acc_or_jerk_excess"        # settled in the lane, but a P-02 / P-04 excess label is set
    return oc


def outcome_view(r: dict) -> dict:
    """What a past run showed, for Nemotron A (A may see results; B and Cosmos may not)."""
    sw = r.get("switch_time")
    onset = r.get("actual_fault_onset") if r.get("actual_fault_onset") is not None else r.get("planned_fault_onset")
    rnd = lambda v, n=3: None if v is None else round(float(v), n)   # noqa: E731
    out = {"sim_status": r.get("sim_status"), "switched": sw is not None,
           "switch_delay_s": rnd(sw - onset, 2) if sw is not None and onset is not None else None,
           "trigger": r.get("supervisor_reason") or r.get("trigger_source"),
           "recovery_outcome": r.get("recovery_outcome"), "final_safe_recovery": r.get("final_safe_recovery"),
           "lane_recovery_robustness": rnd(r.get("lane_recovery_robustness")),
           "lane_margin_min_m": rnd(r.get("P01")),
           "lat_acc_margin_mps2": rnd(r.get("P02"), 2),
           "safety_margin": safety_margin_view(r)}
    if out["final_safe_recovery"] is False and sw is not None:
        out["failed_reason"] = failed_reason(r)
        if out["safety_margin"] and out["safety_margin"]["failing"]:
            out["failed_criteria"] = out["safety_margin"]["failing"]
    return {k: v for k, v in out.items() if v is not None}


def outcome_label(r: dict | None) -> str:
    """integration3 #7: one word per run. never_switched is NOT failed (LQR never acted)."""
    if r is None:
        return "not_run"
    if r.get("sim_status") != "ok":
        return "invalid"
    if r.get("switch_time") is None:
        return "never_switched"
    fs = r.get("final_safe_recovery")
    return "recovered" if fs is True else (f"failed:{failed_reason(r)}" if fs is False else "unlabelled")


def tried_view(c: dict, r: dict | None) -> dict:
    """Compact already_tried row: the scenario plus its outcome label, lane_margin_min_m (P01, metres) and switched."""
    env, f = c.get("environment") or {}, c.get("fault") or {}
    out = {"route_id": c.get("route_id"), "max_speed_kmh": env.get("max_speed_kmh"),
           "tire_friction_scale": env.get("tire_friction_scale"), "fault_id": f.get("fault_id"), "params": f.get("params"),
           "onset_event": c.get("onset_event"), "onset_offset_s": c.get("onset_offset_s"), "outcome": outcome_label(r)}
    if r is not None and r.get("sim_status") == "ok":
        out["switched"] = r.get("switch_time") is not None
        if r.get("P01") is not None:
            out["lane_margin_min_m"] = round(float(r["P01"]), 3)
        if r.get("lane_recovery_robustness") is not None and out["switched"]:
            out["lane_recovery_robustness"] = round(float(r["lane_recovery_robustness"]), 3)
        m = r.get("metrics") or {}
        if m.get("safety_margin") is not None and out["switched"]:
            out["safety_margin"] = round(float(m["safety_margin"]), 3)
            out["safety_margin_binding"] = m.get("safety_margin_binding")
    return {k: v for k, v in out.items() if v is not None}


def _example(r: dict, c: dict) -> dict:
    out = {"run_id": r.get("run_id"), "fault_id": (c.get("fault") or {}).get("fault_id"),
           "fault_mode": (c.get("fault") or {}).get("mode"), "params": (c.get("fault") or {}).get("params"),
           "route_id": c.get("route_id"), "max_speed_kmh": (c.get("environment") or {}).get("max_speed_kmh"),
           "tire_friction_scale": (c.get("environment") or {}).get("tire_friction_scale"),
           "onset_event": c.get("onset_event"), "onset_offset_s": c.get("onset_offset_s"), "outcome": outcome_view(r)}
    return {k: v for k, v in out.items() if v is not None}


def outcome_examples(records: list[dict], candidates: dict[str, dict], space: dict, routes: list[str],
                     fault_pool: list[str], max_modes: int = 8) -> list[dict]:
    """integration3 #7: per fault mode the closest (recovered, failed) pair of switched runs in normalised parameter
    space (scenarios.numeric_features), so Nemotron A sees both sides of the boundary. A mode with only one side
    gets that side's run nearest to the mode's median and a note; never-switched runs are counted, not shown."""
    by: dict[str, dict] = {}
    for r in records:
        c = candidates.get(r.get("scenario_id"))
        if c is None or r.get("sim_status") != "ok":
            continue
        mode = (c.get("fault") or {}).get("mode")
        d = by.setdefault(mode, {"rec": [], "fail": [], "never_switched": 0})
        if r.get("switch_time") is None:
            d["never_switched"] += 1
        elif r.get("final_safe_recovery") is True:
            d["rec"].append((r, c))
        elif r.get("final_safe_recovery") is False:
            d["fail"].append((r, c))
    def x(c):
        return numeric_features(c, space, routes, fault_pool)[0]
    out = []
    for mode in sorted(by, key=lambda m: (not (by[m]["rec"] and by[m]["fail"]), str(m))):
        d = by[mode]
        e = {"fault_mode": mode, "never_switched_runs": d["never_switched"]}
        if d["rec"] and d["fail"]:
            dist, (rr, rc), (fr_, fc) = min(((math.dist(x(a[1]), x(b[1])), a, b) for a in d["rec"] for b in d["fail"]),
                                            key=lambda t: t[0])
            e.update(recovered=_example(rr, rc), failed=_example(fr_, fc), normalized_distance=round(dist, 3),
                     note="closest recovered/failed pair of switched runs: the boundary lies between them")
        elif d["rec"] or d["fail"]:
            side = "recovered" if d["rec"] else "failed"
            r, c = (d["rec"] or d["fail"])[-1]
            e[side] = _example(r, c)
            e["note"] = f"only {side} switched runs so far ({len(d['rec'] or d['fail'])})"
        else:
            e["note"] = "never switched: no recovery to judge yet"
        out.append(e)
    return out[:max_modes]


def accepted(explanations: dict[str, dict]) -> list[dict]:
    return [e for e in explanations.values() if (e.get("verification") or {}).get("accepted")]


def mechanism_summary(explanations: dict[str, dict], limit: int = 8) -> list[dict]:
    return [{"run_id": e["run_id"], "route_id": e.get("route_id"), "onset_event": e.get("onset_event"),
             "fault_mode": e.get("fault_mode"), "tags": e["explanation"].get("mechanism_tags"),
             "mechanism": e["explanation"].get("mechanism")} for e in accepted(explanations)[-limit:]]


def mechanism_features(c: dict, explanations: dict[str, dict], taxonomy: list[str], alpha: float = 1.0
                       ) -> tuple[list[float], list[str]]:
    """Context-level prevalence of each accepted mechanism tag among PAST runs with the same (route, onset_event).

    Known before the candidate runs (uses only earlier explanations), so it is a legal GP input.
    """
    ctx = [e for e in accepted(explanations) if e.get("route_id") == c.get("route_id") and e.get("onset_event") == c.get("onset_event")]
    n = len(ctx)
    counts = Counter(t for e in ctx for t in (e["explanation"].get("mechanism_tags") or []))
    return [counts[t] / (n + alpha) for t in taxonomy], [f"mech={t}" for t in taxonomy]


RULE_TEMPLATES = {
    "oscillation_after_switch": "label oscillation_after_switch: number of sign changes of lane_offset_m with |offset|>0.2 m in [t_sw, t_sw+window] >= 3",
    "post_switch_weave": "same signal as oscillation_after_switch (novel tag) - check whether it duplicates it",
    "overshoot_after_switch": "label overshoot_after_switch: after the first zero crossing of lane_offset_m post-switch, max |offset| on the opposite side > 0.3 m",
    "late_detection": "label late_detection: switch_time - trigger_time > T_detect (T_detect to be chosen from data)",
    "curve_entry_during_handover": "label handover_in_curve: |kappa(t_sw)| > kappa_on (needs logging L-3 kappa)",
    "steering_rate_limit": "existing field handover_saturation.rate_saturation_ratio; candidate label rate_limited_handover if > 0.2",
    "steering_angle_saturation": "existing field handover_saturation.angle_saturation_ratio; candidate label if > 0.3",
}
COVERED = {"departed_before_switch": "recovery_labels.departed_at_switch", "collision_after_departure": "P-03 / unrecovered",
           "stable_recovery": "recovery_outcome safe_recovery", "pre_switch_lateral_drift": "metrics.pre_switch_max_lateral_error_m"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_]{4,}", (text or "").lower()) if w not in {"synthetic", "stub", "explanation", "called"}}


def taxonomy_proposals(explanations: dict[str, dict], min_cluster_size: int = 2) -> dict[str, Any]:
    acc = accepted(explanations)
    tag_runs: dict[str, list[str]] = defaultdict(list)
    for e in acc:
        for t in (e["explanation"].get("mechanism_tags") or []) + (e["explanation"].get("novel_tags") or []):
            tag_runs[str(t)].append(e["run_id"])
    # text clusters (token Jaccard >= 0.5, single link) to surface wording that no tag captures
    clusters: list[list[dict]] = []
    for e in acc:
        tok = _tokens(e["explanation"].get("mechanism"))
        for cl in clusters:
            if any(len(tok & _tokens(x["explanation"].get("mechanism"))) / max(1, len(tok | _tokens(x["explanation"].get("mechanism")))) >= 0.5 for x in cl):
                cl.append(e); break
        else:
            clusters.append([e])
    proposals = []
    for tag, runs in sorted(tag_runs.items(), key=lambda kv: -len(kv[1])):
        if len(runs) < min_cluster_size or tag in COVERED:
            continue
        proposals.append({"tag": tag, "support_runs": sorted(set(runs)), "support": len(set(runs)),
                          "proposed_rule": RULE_TEMPLATES.get(tag, "NEW pattern: define a deterministic signal before use"),
                          "status": "proposed_for_human_review", "auto_applied": False})
    return {"meta": {"generated_by": "engine/feedback.py taxonomy_proposals", "auto_apply": False,
                     "note": "Candidates only. A human must define/approve a rule before 09 monitor uses it.",
                     "accepted_explanations": len(acc)},
            "already_covered_by_monitor": {t: COVERED[t] for t in tag_runs if t in COVERED},
            "rule_candidates": proposals,
            "text_clusters": [{"size": len(cl), "runs": [x["run_id"] for x in cl],
                               "example": cl[0]["explanation"].get("mechanism")} for cl in clusters if len(cl) >= min_cluster_size]}


def counterfactual_candidates(explanations: dict[str, dict], candidates: dict[str, dict], space: dict, round_index: int,
                              limit: int = 2) -> list[dict]:
    out = []
    env = space["environment"]["max_speed_kmh"]["range"]
    off = space["fault_timing"]["offset_s"]["range"]
    for e in accepted(explanations):
        cf = e["explanation"].get("counterfactual")
        base = candidates.get(e.get("scenario_id"))
        if not isinstance(cf, dict) or base is None or e.get("counterfactual_queued"):
            continue
        sign = -1.0 if cf.get("direction") == "decrease" else 1.0
        amt = cf.get("amount") if isinstance(cf.get("amount"), (int, float)) else 0.0
        speed, offset = base["environment"]["max_speed_kmh"], base["onset_offset_s"]
        params = dict(base["fault"]["params"])
        p = cf.get("parameter")
        if p == "onset_offset_s":
            offset = min(off[1], max(off[0], offset + sign * amt))
        elif p == "max_speed_kmh":
            speed = min(env[1], max(env[0], speed + sign * amt))
        elif p in ("fault_magnitude", "fault_duration_s"):
            f = faults_by_id(space)[base["fault"]["fault_id"]]
            key = "duration_s" if p == "fault_duration_s" else ("magnitude" if "magnitude" in params else
                                                                next((k for k in params if k != "duration_s"), None))
            if key is None or key not in (f.get("params") or {}):
                continue
            spec = f["params"][key]
            u = _param_unit(spec, params[key])
            params[key] = _param_from_unit(spec, (u or 0.5) + sign * 0.2)
        else:
            continue
        c = make_candidate(space, route_id=base["route_id"], speed=speed, fault_id=base["fault"]["fault_id"], params=params,
                           onset_event=base["onset_event"], offset=offset, source="counterfactual", round_index=round_index,
                           seed_scenario_id=base["seed_scenario_id"], expected_mechanism=cf.get("expected_effect"),
                           friction=(base.get("environment") or {}).get("tire_friction_scale"))
        c["counterfactual_of"] = e["run_id"]
        e["counterfactual_queued"] = c["scenario_id"]
        out.append(c)
        if len(out) >= limit:
            break
    return out
