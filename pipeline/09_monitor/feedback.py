"""09 Monitor feedback: run_records -> files for stages 2, 3 and 6.

  stage 2 (hypotheses)   hypothesis_tracking.json  was each hypothesis right? (supported / refuted / ...)
  stage 3 (Nemotron A/B) failure_table.json        top-5 verified failures: environment + fault plan
                                                   + which SR broke and by how much
  stage 6 (GP)           gp_training.jsonl         x = scenario parameters, y = run_record.gp_targets
                                                   (one z per team metric, 02 metric_rules) + metrics
  all                    run_records.jsonl         every graded pair in the runner's schema

Input records are in the runner schema (monitor.RUN_RECORD_FIELDS); our SR verdict sits in
metrics["monitor"] and is read through monitor.view(). Decision rules come from
02_hypotheses/vulnerability_patterns.yaml (metric_rules, decision_protocol).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from monitor import view

FAIL_OUTCOMES = ("F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered")

# decision protocol (02 vulnerability_patterns.yaml: decision_protocol)
DEFAULT_RULE = {
    "metric": "margin_delta_m",
    "direction": "negative",
    "level": 0.99,
    "resamples": 5000,
    "checkpoints": [10, 20, 30],
    "exclude_outcomes": ["invalid", "simulation_invalid"],
}
# minimum_effect per metric when metric_rules are not given (02 metric_rules values)
DEFAULT_DELTA_MIN = {"switch_count": 1.0, "max_steering_rate_dps": 10.0, "pre_switch_max_lateral_error_m": 0.1,
                     "recovery_success": 0.05, "lateral_error_rmse_m": 0.1, "deadline_miss_rate": 0.05,
                     "mpc_constraint_violation_rate": 0.05}


# ------------------------------------------------------------------ io
def read_records(path: str | Path) -> list[dict]:
    """Raw run_records (runner schema). Functions below take raw or viewed records."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _metric(r: dict, name: str):
    """A metric value: team metrics dict first, then a top-level field. bool -> 0/1."""
    m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
    v = m.get(name) if name in m else r.get(name)
    return float(v) if isinstance(v, bool) else v


def _valid(r: dict) -> bool:
    return r.get("sim_status") == "valid" and r.get("outcome_category") != "invalid" and r.get("assessment_complete") is not False


# ============================================================ stage 2
def hypotheses_from_yaml(path: str | Path, patterns_path: str | Path | None = None) -> dict:
    """Read 02_hypotheses/hypotheses.yaml (+ vulnerability_patterns.yaml metric_rules) into decision rules."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        y = yaml.safe_load(fh)
    rules = {}
    if patterns_path:
        with open(patterns_path, encoding="utf-8") as fh:
            rules = (yaml.safe_load(fh) or {}).get("metric_rules") or {}
    out = {}
    for h in y.get("hypotheses", []):
        pred = h["prediction"]
        mr = rules.get(pred["metric"], {})
        rule = {"metric": pred["metric"], "direction": pred.get("direction") or mr.get("support_direction"),
                "secondary": h.get("secondary_prediction", []) or h.get("optional_secondary_prediction", []),
                "hypothesis_status": h.get("status"), "claim": h.get("claim"), "pattern": h.get("from_pattern"),
                "phase": h.get("phase"), "condition": h.get("condition"), "control": h.get("control")}
        if pred.get("direction") == "non_inferior":
            rule["delta_min"] = float(pred.get("margin", 0.05))
        elif mr.get("minimum_effect") is not None:
            rule["delta_min"] = float(mr["minimum_effect"])
        if mr.get("ci_level"):
            rule["level"] = float(mr["ci_level"])
        pattern = (h.get("from_pattern") or "")
        pattern = next((k for k in ("P-DELAYED-SWITCH", "P-CHATTERING") if pattern.startswith(k)), pattern)
        if pattern == "P-DELAYED-SWITCH":   # "보조 지표가 반대 방향이면 supported 대신 inconclusive"
            rule["secondary_guard"] = True
        if pattern == "P-CHATTERING":       # "switch_count >= 3이 재현되지 않으면 refuted가 아니라 inconclusive"
            rule["require_in_condition"] = {"metric": "switch_count", "min": 3}
        out[h["id"]] = rule
    return out


def _bootstrap_ci(cond, ctrl, level, resamples, seed):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(cond, float), np.asarray(ctrl, float)
    ia = rng.integers(0, len(a), size=(resamples, len(a)))
    ib = rng.integers(0, len(b), size=(resamples, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def _decide(d_ci, direction, delta_min):
    lo, hi = d_ci
    if direction == "non_inferior":         # condition no worse than control by more than the margin
        if lo > -delta_min:
            return "supported"
        if hi < -delta_min:
            return "refuted"
        return None
    if direction == "negative":
        if hi < -delta_min:
            return "supported"
        if lo > -delta_min:
            return "refuted"
    else:
        if lo > delta_min:
            return "supported"
        if hi < delta_min:
            return "refuted"
    return None


def hypothesis_tracking(records: list[dict], hypotheses: dict | None = None,
                        delta_min_m: float | None = None, seed: int = 0) -> dict:
    """Stage 2: per hypothesis, d = mean(condition) - mean(control) with a bootstrap CI at each checkpoint.

    records: run_records; each carries hypothesis_arms {H-id: "condition" | "control"} set by the
             orchestrator for dedicated hypothesis runs (GP exploration runs carry none).
    hypotheses: {H-id: {metric, direction, secondary, secondary_guard, require_in_condition, delta_min}}
    delta_min_m: from monitor.noise_floor (calibration). None -> 0.1 m floor with a warning.
    """
    records = [view(r) for r in records]
    hypotheses = dict(hypotheses or {})
    for r in records:  # hypotheses seen only in records get the default rule
        for h in (r.get("hypothesis_arms") or {}):
            hypotheses.setdefault(h, {})
    out = {"rule": {k: v for k, v in DEFAULT_RULE.items()}, "delta_min_m": delta_min_m,
           "warnings": [], "hypotheses": {}}
    if delta_min_m is None:
        out["warnings"].append("noise floor not measured: delta_min_m = 0.1 (minimum) used")
    dm_m = delta_min_m if delta_min_m is not None else 0.1

    for hid, spec in sorted(hypotheses.items()):
        rule = {**DEFAULT_RULE, **spec}
        metric, direction = rule["metric"], rule["direction"]
        delta = rule.get("delta_min", dm_m if metric.endswith("_m") else DEFAULT_DELTA_MIN.get(metric, 0.0))
        arms = {"condition": [], "control": []}
        excluded = 0
        for r in records:
            arm = (r.get("hypothesis_arms") or {}).get(hid)
            if arm not in arms:
                continue
            value = _metric(r, metric)
            if (not _valid(r) or r.get("outcome_category") in rule["exclude_outcomes"]
                    or r.get("runner_outcome") in rule["exclude_outcomes"] or value is None
                    or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value))):
                excluded += 1
                continue
            arms[arm].append(r)
        n = min(len(arms["condition"]), len(arms["control"]))
        history, status = [], "testing"
        for cp in rule["checkpoints"]:
            if n < cp:
                break
            cond = [_metric(r, metric) for r in arms["condition"][:cp]]
            ctrl = [_metric(r, metric) for r in arms["control"][:cp]]
            d = float(np.mean(cond) - np.mean(ctrl))
            ci = _bootstrap_ci(cond, ctrl, rule["level"], rule["resamples"], seed)
            st = _decide(ci, direction, delta)
            guard = None
            if st == "supported" and rule.get("secondary_guard"):
                for sp in rule.get("secondary", []):
                    c2 = [_metric(r, sp["metric"]) for r in arms["condition"][:cp] if _metric(r, sp["metric"]) is not None]
                    k2 = [_metric(r, sp["metric"]) for r in arms["control"][:cp] if _metric(r, sp["metric"]) is not None]
                    if c2 and k2:
                        d2 = np.mean(c2) - np.mean(k2)
                        if (sp["direction"] == "positive" and d2 < 0) or (sp["direction"] == "negative" and d2 > 0):
                            st, guard = None, f"secondary {sp['metric']} went the other way"
            if st == "refuted" and rule.get("require_in_condition"):
                req = rule["require_in_condition"]
                if not any((_metric(r, req["metric"]) or 0) >= req["min"] for r in arms["condition"][:cp]):
                    st, guard = None, f"{req['metric']} >= {req['min']} never reproduced in condition arm"
            history.append({"checkpoint": cp, "d_estimate": d, "ci": list(ci), "status": st or "inconclusive",
                            "guard": guard})
            if st:
                status = st
                break
        else:
            if history:
                status = "inconclusive"
        last = history[-1] if history else {}
        observed = {}
        for arm, runs in arms.items():
            values = [float(_metric(r, metric)) for r in runs]
            observed[arm] = {"n": len(values), "mean": float(np.mean(values)) if values else None,
                             "min": min(values) if values else None, "max": max(values) if values else None,
                             "run_ids": [r.get("run_id") for r in runs]}
        a, b = observed["condition"]["mean"], observed["control"]["mean"]
        observed["mean_difference"] = a - b if a is not None and b is not None else None
        out["hypotheses"][hid] = {
            "metric": metric, "direction": direction, "delta_min": delta,
            "hypothesis_status": spec.get("hypothesis_status"), "claim": spec.get("claim"), "pattern": spec.get("pattern"), "phase": spec.get("phase"),
            "runs_condition": len(arms["condition"]), "runs_control": len(arms["control"]),
            "excluded_runs": excluded, "d_estimate": last.get("d_estimate"), "ci": last.get("ci"),
            "descriptive": observed, "ci_level": rule["level"],
            "condition_definition": spec.get("condition"), "control_definition": spec.get("control"),
            "status": status, "status_history": history,
            "next_checkpoint": next((c for c in rule["checkpoints"] if c > n), None) if status == "testing" else None,
        }
    return out


# ============================================================ stage 3
def _get(r, key):
    if r.get(key) is not None:
        return r[key]
    return (r.get("parameters") or {}).get(key)


def _how_much(r: dict) -> dict:
    """Which SR broke and by how much (the numbers Nemotron needs to understand the failure)."""
    sr = r.get("sr") or {}
    out = {}
    for k in r.get("failed_requirements") or []:
        v = sr.get(k, {})
        if k == "SR1":
            out[k] = {"latency_s": v.get("latency_s"), "limit_s": v.get("limit_s"),
                      "ftti_s": v.get("ftti_s"), "switched": v.get("latency_s") is not None}
        elif k == "SR2":
            out[k] = {"max_rate_dps": v.get("max_rate_dps"), "limit_dps": v.get("limit_dps")}
        elif k == "SR4":
            out[k] = {"margin_m": v.get("margin_m"), "nominal_margin_m": v.get("nominal_margin_m"),
                      "collision": v.get("collision"), "first_violation_t": v.get("first_violation_t")}
        else:
            out[k] = {kk: vv for kk, vv in v.items() if kk not in ("applies", "pass")}
        out[k]["robustness"] = v.get("robustness")
    return out


def _sentence(row: dict) -> str:
    fp = row["fault_plan"] or {}
    env = row["environment"] or {}
    sw = row["observed_switch_behavior"]
    mag = fp.get("magnitude", fp.get("strength"))
    dur = fp.get("duration_s")
    detail = ", ".join(x for x in (None if mag is None else str(mag), None if dur is None else f"{dur}s") if x) or "?"
    parts = [f"{env.get('route', env.get('town', '?'))}", f"고장 {fp.get('fault_type', '?')}"
             f"({detail})",
             f"시점 {row['onset_event'] or '?'}"]
    if sw["switch_time"] is None:
        what = "전환 없음"
    else:
        what = f"전환 지연 {sw['detection_latency_s']:.2f}s" if sw["detection_latency_s"] is not None else "전환"
    broke = ", ".join(row["violated_requirements"]) or row["outcome_category"]
    seen = f" / 영상: {row['vision']['text']}" if row.get("vision") else ""
    return (f"{' / '.join(parts)} → {what}, 위반 {broke}, 점수 {row['sr_overall']:.2f}, "
            f"재현 {row['reproduction_count']}회{seen}")


LANE_KO = {"centered": "중앙 유지", "drift_in_lane": "차선 안에서 밀림", "touches_line": "차선에 닿음",
           "crosses_line": "차선 넘음", "leaves_road": "도로 이탈"}
MOTION_KO = {"smooth": "부드러움", "slow_drift": "서서히 밀림", "weaving": "좌우로 흔들림", "sudden_swerve": "급조향",
             "overcorrection_after_switch": "전환 후 과잉 보정", "stopped": "정지"}
OBJ_KO = {"vehicle": "차량", "oncoming_vehicle": "맞은편 차량", "pedestrian": "보행자", "cyclist": "자전거",
          "road_edge": "도로 끝", "barrier": "방호벽", "parked_vehicle": "주차 차량"}
SIDE_KO = {"left": "왼쪽", "right": "오른쪽", "front": "앞", "rear": "뒤"}
NEAR_KO = {"far": "멀리", "near": "가까이", "very_near": "아주 가까이"}


def read_judgments(path: str | Path | None) -> dict:
    """run_id -> vision judgment row from explain.py (screen/commentary.jsonl). Missing file -> {}."""
    out: dict = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["run_id"]] = row
    return out


def vision_text(j: dict | None) -> str | None:
    """One Korean phrase of what the video showed (for summaries and Nemotron prompts)."""
    if not j:
        return None
    sur = ", ".join(f"{SIDE_KO.get(s['side'], '')} {OBJ_KO.get(s['type'], s['type'])} {NEAR_KO.get(s['proximity'], '')}"
                    .strip() for s in j.get("surroundings") or [])
    parts = [LANE_KO.get(j.get("lane")), MOTION_KO.get(j.get("motion")), sur or None,
             "아슬아슬한 근접" if j.get("near_miss") else None, f"위험도 {j['severity']}/3"]
    return ", ".join(x for x in parts if x)


def failure_table(records: list[dict], k: int = 5, min_reproduction: int = 2, judgments: dict | None = None) -> dict:
    """Stage 3: verified failures (same scenario failed >= 2 times), one per (fault_type, onset_event).

    With vision judgments (explain.py): each row also carries what the video showed (lane behaviour,
    motion pattern, surroundings, near miss, severity 0..3), rows are ranked by that severity first and
    the monitor's score second, and runs where the video and the monitor disagree go to `needs_review`.
    Returns {"verified": [...top k...], "candidates": [...failed once...], "needs_review": [...]}.
    """
    records = [view(r) for r in records]
    judgments = judgments or {}
    groups: dict[Any, list[dict]] = defaultdict(list)
    runs_total: dict[Any, int] = defaultdict(int)
    for r in records:
        if not _valid(r) or r.get("stress_test"):
            continue
        key = r.get("scenario_id") or r.get("run_id")
        runs_total[key] += 1
        if r.get("outcome_category") in FAIL_OUTCOMES and r.get("sr_overall") is not None:
            groups[key].append(r)

    rows = []
    for key, fails in groups.items():
        worst = min(fails, key=lambda r: r["sr_overall"])
        fp = _get(worst, "fault_plan") or {}
        js = [(judgments.get(r.get("run_id")) or {}).get("judgment") for r in fails]
        js = [j for j in js if j]
        vj = max(js, key=lambda j: j["severity"]) if js else None
        row = {
            "scenario_id": key,
            "outcome_category": worst["outcome_category"],
            "violated_requirements": worst.get("failed_requirements") or [],
            "how_much": _how_much(worst),
            "unsafe": worst.get("unsafe"),
            "environment": _get(worst, "environment"),     # set by Nemotron A (route, ODD)
            "fault_plan": fp,                               # set by Nemotron B (type, strength, duration, timing)
            "onset_event": fp.get("onset_event"),
            "observed_switch_behavior": {"switch_time": worst.get("switch_time"),
                                         "detection_latency_s": worst.get("detection_latency_s"),
                                         "switch_count": worst.get("switch_count")},
            "sr_overall": worst["sr_overall"],
            "sr_scores": worst.get("sr_scores"),
            "sr_overall_median": float(median(r["sr_overall"] for r in fails)),
            "reproduction_count": len(fails),
            "runs_total": runs_total[key],
            "run_ids": [r.get("run_id") for r in fails],
            "hypothesis_ids": worst.get("hypothesis_ids"),
            "reason_items": worst.get("reason_items"),
            "vision": None if vj is None else {
                "severity": vj["severity"], "lane": vj["lane"], "motion": vj["motion"],
                "surroundings": vj["surroundings"], "near_miss": vj["near_miss"],
                "near_miss_detail": vj.get("near_miss_detail"), "text": vision_text(vj)},
        }
        # names used by the team's error_table.yaml failure_table schema
        row.update(violated_property=row["violated_requirements"], fault=fp, robustness=row["sr_overall"],
                   originating_evidence_ids=(_get(worst, "parameters") or {}).get("originating_evidence_ids"))
        row["summary"] = _sentence(row)
        rows.append(row)

    # vision severity first (unknown last), then the monitor's score
    rows.sort(key=lambda x: (-(x["vision"]["severity"]) if x["vision"] else 1, x["sr_overall_median"], x["sr_overall"]))
    verified, seen = [], set()
    for row in rows:
        if row["reproduction_count"] < min_reproduction:
            continue
        div = (row["fault_plan"].get("fault_type"), row["onset_event"])
        if div in seen:
            continue
        seen.add(div)
        verified.append(row)
        if len(verified) == k:
            break
    candidates = [row for row in rows if row["reproduction_count"] < min_reproduction][:k]
    review = []
    for r in records:
        row = judgments.get(r.get("run_id")) or {}
        if row.get("agreement") in ("monitor_may_miss", "video_shows_no_risk") or row.get("switch_check") == "mismatch":
            review.append({"run_id": r.get("run_id"), "scenario_id": r.get("scenario_id"),
                           "monitor": r.get("outcome_category"), "agreement": row.get("agreement"),
                           "switch_check": row.get("switch_check"), "vision": vision_text(row.get("judgment")),
                           "fault_plan": _get(r, "fault_plan"), "environment": _get(r, "environment")})
    return {"rule": {"k": k, "min_reproduction": min_reproduction,
                     "rank_by": "vision severity descending (if judged), then sr_overall median ascending",
                     "diversity": "one per (fault_type, onset_event)", "exclude": ["invalid", "stress_test"]},
            "verified": verified, "candidates": candidates, "needs_review": review}


# ============================================================ stage 6
def _flatten(d: Any, prefix: str = "") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (list, tuple)):
        out[prefix[:-1]] = ",".join(map(str, d))
    elif d is not None:
        out[prefix[:-1]] = d
    return out


def gp_training_rows(records: list[dict]) -> list[dict]:
    """Stage 6 (06 surrogate.yaml feedback_in: features, metrics, baseline_metrics, gp_targets).

    One row per valid run that has at least one GP target.
    x = flattened scenario (A env + B fault plan + parameters [+ reason_items])
    y = run_record.gp_targets: z per team metric (02 metric_rules, z = sign * y / scale; lower = riskier).
        A metric without a value (e.g. recovery_robustness with no switch) is absent, never 0.
    sr_scores = our SR1..SR6 scores, kept alongside for the screen and failure table (not a GP target).
    """
    rows = []
    for raw in records:
        r = view(raw)
        if not _valid(r) or not r.get("gp_targets"):
            continue
        x = {}
        x.update(_flatten(r.get("environment"), "env."))
        x.update(_flatten(r.get("fault_plan"), "fault."))
        x.update(_flatten({k: v for k, v in (r.get("parameters") or {}).items()
                           if k not in ("environment", "fault_plan")}, "param."))
        x.update(_flatten(r.get("reason_items"), "reason."))
        metrics = {k: v for k, v in (r.get("metrics") or {}).items() if k != "monitor"}
        rows.append({"run_id": r.get("run_id"), "scenario_id": r.get("scenario_id"), "round": r.get("round"),
                     "feature_schema_version": r.get("feature_schema_version"), "x": x,
                     "gp_targets": r["gp_targets"], "metrics": metrics, "baseline_metrics": r.get("baseline_metrics"),
                     "sr_scores": r.get("sr_scores"), "outcome_category": r.get("runner_outcome", r.get("outcome_category")),
                     "failure_type": r.get("outcome_category"), "stress_test": bool(r.get("stress_test")),
                     "gp_mean_before": r.get("gp_mean"), "gp_std_before": r.get("gp_std")})
    return rows


# ============================================================ human view
def campaign_summary(records: list[dict]) -> str:
    """Short text for people (the full screen is dashboard.py)."""
    records = [view(r) for r in records]
    n = len(records)
    valid = [r for r in records if _valid(r)]
    cnt = defaultdict(int)
    for r in records:
        cnt[r.get("outcome_category") or "invalid"] += 1
    lines = [f"graded pairs: {n} (valid {len(valid)})"]
    for oc in ("pass", "no_trigger", "F1_no_or_late_switch", "F2_switch_transient", "F3_not_recovered",
               "invalid"):
        if cnt[oc]:
            lines.append(f"  {oc:22s} {cnt[oc]}")
    harmless = sum(1 for r in valid if r.get("note") == "undetected_harmless")
    if harmless:
        lines.append(f"  (pass, missed but harmless: {harmless})")
    gp = [r["sr_overall"] for r in valid if r.get("sr_overall") is not None]
    if gp:
        lines.append(f"sr_overall min {min(gp):.2f} / median {median(gp):.2f}")
    return "\n".join(lines)


def write_outputs(records: list[dict], out_dir: str | Path, hypotheses: dict | None = None,
                  delta_min_m: float | None = None, k: int = 5, judgments: dict | None = None) -> dict:
    """judgments: run_id -> explain.py row (vision judgment); optional."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    p = out / "run_records.jsonl"
    p.write_text("".join(json.dumps(r, default=str, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    paths["run_records"] = p
    p = out / "hypothesis_tracking.json"
    p.write_text(json.dumps(hypothesis_tracking(records, hypotheses, delta_min_m), indent=2, ensure_ascii=False),
                 encoding="utf-8")
    paths["stage2"] = p
    p = out / "failure_table.json"
    p.write_text(json.dumps(failure_table(records, k, judgments=judgments), indent=2, default=str, ensure_ascii=False),
                 encoding="utf-8")
    paths["stage3"] = p
    p = out / "gp_training.jsonl"
    p.write_text("".join(json.dumps(r, default=str, ensure_ascii=False) + "\n" for r in gp_training_rows(records)),
                 encoding="utf-8")
    paths["stage6"] = p
    return paths


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="09 Monitor feedback: run_records -> stage 2/3/6 files")
    ap.add_argument("--records", required=True, help="run_records.jsonl from monitor.py --out")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--hypotheses", default=None, help="02_hypotheses/hypotheses.yaml")
    ap.add_argument("--patterns", default=None, help="02_hypotheses/vulnerability_patterns.yaml (metric_rules)")
    ap.add_argument("--delta-min", type=float, default=None, help="from monitor.noise_floor (m)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--judgments", default=None, help="screen/commentary.jsonl from explain.py (vision judgment)")
    a = ap.parse_args(argv)
    recs = read_records(a.records)
    hyp = hypotheses_from_yaml(a.hypotheses, a.patterns) if a.hypotheses else None
    paths = write_outputs(recs, a.out, hyp, a.delta_min, a.k, read_judgments(a.judgments))
    print(campaign_summary(recs))
    for name, p in paths.items():
        print(f"{name:12s} -> {p}")
    return paths


if __name__ == "__main__":
    main()
