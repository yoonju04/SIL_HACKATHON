"""03 Nemotron A (scenario candidates) and 05b Nemotron B (code questions) on Token Factory.

A: Nemotron-3.5-Lightning, thinking OFF, response_format json_schema built from the scenario space
   (ranges as minimum/maximum, ids as enum), same schema also pasted into the prompt, English prompt.
   Optional refine pass (thinking ON, large max_tokens) for boundary-near candidates.
   A is the only role that sees failures (09 failure_table + accepted 10 explanations).
B: Nemotron-3-Ultra, thinking ON + medium_effort, whole av_stack source files (with line numbers)
   in context, answers each question from its option list with file:line evidence, repeated 2-3x for a
   consistency score. B never sees results, failures or GP predictions.
   integration3 #6: per_scenario questions (cached per b_key) vs static questions (once per campaign); only
   per_scenario + gp_feature questions become GP one-hot; majority answers are checked against expected_from_code
   (b_hallucination_rate) and citations against each question's evidence_regions (citations_relevant).
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import schema as S
from .llm import LLMError, extract_json, json_schema_format
from .feedback import failed_reason, outcome_view
from .scenarios import faults_by_id, friction_axis, make_candidate, sampling_ranges


# ----------------------------------------------------------------------------- A
def evidence_enum(profile: dict, prior: dict, space: dict, objectives: list[str], extra: list[str] = ()) -> list[str]:
    ids = [o["id"] for o in (profile.get("code_evidence") or {}).get("observations", [])]
    ids += [p["id"] for p in prior.get("principles", [])]
    ids += [f["id"] for f in space.get("faults", [])]
    ids += list(objectives) + ["E-01", "E-02", "E-03"] + list(extra)
    return sorted(dict.fromkeys(ids))


def param_schema(spec: dict) -> dict:
    if "values" in spec:
        vals = spec["values"]
        return {"enum": vals}
    lo, hi = spec["range"] if "range" in spec else (spec["ranges"][0][0], spec["ranges"][-1][1])
    return {"type": "number", "minimum": lo, "maximum": hi}


def candidate_schema(space: dict, routes: list[str], fault_pool: list[str], objectives: list[str],
                     evidence: list[str], n_max: int) -> dict:
    fax = friction_axis(space)
    branches = []
    for fid in fault_pool:
        f = faults_by_id(space)[fid]
        params = {k: param_schema(s) for k, s in (f.get("params") or {}).items()}
        rg = sampling_ranges(space, f)   # integration3 #3: control faults are proposed on their handover axes
        branches.append({
            "type": "object",
            "properties": {
                "fault_id": {"enum": [fid]},
                "params": {"type": "object", "properties": params, "required": list(params), "additionalProperties": False},
                "route_id": {"enum": routes},
                "max_speed_kmh": {"type": "number", "minimum": rg["max_speed_kmh"][0], "maximum": rg["max_speed_kmh"][1]},
                **({"tire_friction_scale": {"type": "number", "minimum": rg["tire_friction_scale"][0],
                                            "maximum": rg["tire_friction_scale"][1]}} if fax else {}),
                "onset_event": {"enum": rg["onset_event"]},
                "onset_offset_s": {"type": "number", "minimum": rg["offset_s"][0], "maximum": rg["offset_s"][1]},
                "objective_ids": {"type": "array", "items": {"enum": objectives}, "minItems": 1, "maxItems": len(objectives)},
                "evidence_ids": {"type": "array", "items": {"enum": evidence}, "minItems": 1, "maxItems": 6},
                "expected_mechanism": {"type": "string"},
                "expected_side": {"enum": ["recover", "fail", "boundary"]},
                "extreme_justification": {"type": "string"},
                "missing_information": {"type": "string"},
            },
            "required": ["fault_id", "params", "route_id", "max_speed_kmh"] + (["tire_friction_scale"] if fax else []) +
                        ["onset_event", "onset_offset_s",
                         "objective_ids", "evidence_ids", "expected_mechanism", "expected_side", "extreme_justification",
                         "missing_information"],
            "additionalProperties": False,
        })
    return {"type": "object",
            "properties": {"candidates": {"type": "array", "minItems": 1, "maxItems": n_max, "items": {"anyOf": branches}}},
            "required": ["candidates"], "additionalProperties": False}


A_SYSTEM = """You are Nemotron A, the scenario generator of a software-in-the-loop safety SEARCH for an MPC->LQR steering handover \
(CARLA 0.9.15, PDM-Lite planner, av_stack MPC primary + LQR backup + latched supervisor; faults are injected only at the MPC output).
Search objective S1: estimate WHERE LQR recovery after the switch stops succeeding, as a function of the error state at the \
switch moment and of steering angle / rate saturation. objective.judgment_criteria states exactly how a run is labelled.
Your job is NOT to find failures. Failures at the extreme corners of the space are already known (e.g. saturate at full \
steering for the longest duration fails every time; solver_fail / stale at low speed always recovers) and teach nothing.
Propose candidates that you expect to sit ON the success/failure boundary: conditions where the SAME fault could recover \
or fail depending on a small change (fault magnitude/duration, speed, tyre friction, onset relative to the curve).
Rules:
- A recovery boundary only exists where the supervisor actually switches to LQR. A run without a switch is not a failed \
recovery, it is a missing data point.
- surrogate_summary.boundary_regions are tried conditions the GP cannot yet classify, ranked with the same score the \
orchestrator uses (recovery level set penalised by 1 - P(switch), or switch probability near 0.5). Propose candidates in \
and around them, varying one or two fields. surrogate_summary.unexplored_regions are places the GP knows nothing about \
(uncertainty near its prior): useful to probe, but they are NOT evidence of a boundary.
- outcome_examples give, per fault mode, the closest recovered + failed pair seen so far: fill the gap between them.
- recent_outcomes_by_fault_mode[mode].status: "never_switched" -> make the fault stronger / longer or move the onset so \
the tracking error reaches the supervisor trigger (supervisor.tracking_divergence) and a switch happens; \
"failed_whenever_switched" -> make it milder (smaller magnitude, shorter, lower speed, higher friction, gentler curve); \
"recovered_whenever_switched" -> make it harder; "mixed" -> bracket the condition that separates the outcomes.
- route_geometry gives, per route and onset_event, the baseline time, curvature and speed and the latest onset offset that \
still leaves the evaluation window before the route ends (V-03). Do not propose an onset the route cannot grade. Use the \
nominal lateral acceleration v^2*|kappa| against the friction-scaled P-02 limit (route_geometry.rules) to reason about \
speed and tyre friction.
- Avoid the ends of the allowed ranges (within 5% of a minimum or maximum). If a value at a range end is genuinely needed, \
explain why in extreme_justification; otherwise leave extreme_justification as an empty string. Unjustified range-end \
values are down-ranked by the orchestrator.
- Set expected_side to "boundary" for boundary candidates; use "recover"/"fail" only for deliberate bracketing pairs.
- You never decide safety. A deterministic monitor labels every run. Do not claim a scenario is safe or unsafe as a fact.
- Cite only ids from the evidence enum. Unreviewed code observations are leads, not verified facts.
- Use only values allowed by the JSON schema (given as the response format). Do not invent controller parameters.
- Avoid repeating scenarios listed under already_tried (each entry shows its outcome).
- Output ONLY a JSON object that matches the schema. No prose."""

_HANGUL = re.compile(r"[\uac00-\ud7a3]")
_DROP = object()


def english_view(obj: Any, dropped: list[str] | None = None, path: str = "") -> Any:
    """integration3 #7: the prompt is English. A key `k_en` replaces `k`; keys ending in _ko are left out; a string
    with Hangul and no English twin is dropped (its path is appended to `dropped`). The YAML keeps the Korean."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.endswith("_ko") or (k.endswith("_en") and k[:-3] in obj):
                continue
            key = k[:-3] if k.endswith("_en") else k
            val = obj.get(f"{k}_en", v)
            r = english_view(val, dropped, f"{path}.{key}")
            if r is not _DROP:
                out[key] = r
        return out
    if isinstance(obj, list):
        items = [english_view(v, dropped, f"{path}[]") for v in obj]
        return [v for v in items if v is not _DROP]
    if isinstance(obj, str) and _HANGUL.search(obj):
        if dropped is not None:
            dropped.append(path)
        return _DROP
    return obj


def a_messages(*, schema: dict, n: int, context: dict, include_schema: bool = False) -> list[dict]:
    """integration3 #7: the schema goes in response_format (json_schema); it is pasted into the body only for the
    json_object fallback (call_a adds it there) or when include_schema is set."""
    user = {"task": f"Propose exactly {n} candidate scenarios for the active objectives.", **context}
    if include_schema:
        user["json_schema"] = schema
    return [{"role": "system", "content": A_SYSTEM}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, default=str)}]


def _with_schema(messages: list[dict], schema: dict) -> list[dict]:
    """json_object fallback: the model no longer gets the schema from response_format, so paste it into the body."""
    out = [dict(m) for m in messages]
    try:
        user = json.loads(out[-1]["content"])
    except (TypeError, ValueError):
        out[-1]["content"] = f"{out[-1]['content']}\n\nJSON SCHEMA\n{json.dumps(schema)}"
        return out
    user["json_schema"] = schema
    out[-1]["content"] = json.dumps(user, ensure_ascii=False, default=str)
    return out


def a_context(*, profile: dict, prior: dict, space: dict, objective_cfg: dict, routes: list[dict], few_shot: list[dict],
              mechanisms: list[dict], tried: list[dict], include_unreviewed: bool,
              surrogate_summary: dict | None = None, outcome_examples: list[dict] | None = None,
              route_geometry: dict | None = None) -> dict:
    obs = (profile.get("code_evidence") or {}).get("observations", [])
    verified = [o for o in obs if o.get("verified") is True]
    unreviewed = [] if not include_unreviewed else [
        {k: o.get(k) for k in ("id", "field", "observed_value", "note", "note_en") if o.get(k) is not None}
        for o in obs if o.get("verified") is not True]
    faults = [{k: f.get(k) for k in ("id", "name", "group", "injector_mode", "description", "description_en", "check",
                                     "check_en", "params", "handover_axes", "next_question", "next_question_en")
               if f.get(k) is not None}
              for f in space.get("faults", [])]
    ctx = {
        "objective": objective_cfg,
        "verified_code_observations": verified,
        "unreviewed_code_observations_static_extraction": unreviewed,
        "domain_prior_knowledge": prior.get("principles", []),
        "supervisor": space.get("supervisor"),
        "scenario_space": {"environment": space.get("environment"), "fault_timing": space.get("fault_timing"), "faults": faults},
        "routes": routes,
        "route_geometry": route_geometry or {"note": "no baseline run yet: route geometry unknown"},
        "outcome_examples": outcome_examples or [],
        "prior_failures_top_k": few_shot,
        "verified_video_mechanisms": mechanisms,
        "already_tried": tried,
        "surrogate_summary": surrogate_summary or {"gp": None, "boundary_regions": [], "unexplored_regions": [],
                                                    "recent_outcomes_by_fault_mode": {},
                                                    "note": "no runs yet (cold start): spread candidates over interior values"},
    }
    dropped: list[str] = []
    ctx = english_view(ctx, dropped)
    if dropped:
        ctx["untranslated_fields_omitted"] = len(dropped)
    return ctx


def a_to_candidates(obj: dict, space: dict, *, round_index: int, source: str = "nemotron_a") -> list[dict]:
    out = []
    for item in obj.get("candidates", []):
        try:
            c = make_candidate(space, route_id=item["route_id"], speed=item["max_speed_kmh"], fault_id=item["fault_id"],
                               friction=item.get("tire_friction_scale"),
                               params=item["params"], onset_event=item["onset_event"], offset=item["onset_offset_s"],
                               source=source, objective=(item.get("objective_ids") or ["S1"])[0], round_index=round_index,
                               evidence_ids=item.get("evidence_ids", []), expected_mechanism=item.get("expected_mechanism"),
                               missing_information=item.get("missing_information"))
            c["hypothesis_ids"] = item.get("objective_ids") or ["S1"]
            c["expected_side"] = item.get("expected_side")
            c["extreme_justification"] = (item.get("extreme_justification") or "").strip() or None
            ext = extreme_fields(c, space)
            c["range_extremes"] = ext
            if ext and not c["extreme_justification"]:
                c["tags"] = sorted(set(c.get("tags", []) + ["range_extreme_unjustified"]))
            out.append(c)
        except (KeyError, TypeError, ValueError):
            continue
    return out


EXTREME_EDGE = 0.05   # a continuous value within 5% of its range end counts as a range extreme


def continuous_units(c: dict, space: dict) -> dict[str, float]:
    """Unit position (0..1) of every continuous field of a candidate in its allowed range (global ranges)."""
    out: dict[str, float] = {}
    lo, hi = space["environment"]["max_speed_kmh"]["range"]
    out["max_speed_kmh"] = (float(c["environment"]["max_speed_kmh"]) - lo) / (hi - lo)
    fax = friction_axis(space)
    fr = (c.get("environment") or {}).get("tire_friction_scale")
    if fax and fr is not None and abs(float(fr) - float(fax.get("default", 1.0))) > 1e-9:  # nominal tyres are not "extreme"
        out["tire_friction_scale"] = (float(fr) - fax["range"][0]) / (fax["range"][1] - fax["range"][0])
    olo, ohi = space["fault_timing"]["offset_s"]["range"]
    out["onset_offset_s"] = (float(c["onset_offset_s"]) - olo) / (ohi - olo)
    f = faults_by_id(space).get((c.get("fault") or {}).get("fault_id"), {})
    for k, spec in (f.get("params") or {}).items():
        v = ((c.get("fault") or {}).get("params") or {}).get(k)
        if "range" in spec and isinstance(v, (int, float)):
            out[f"params.{k}"] = (float(v) - spec["range"][0]) / (spec["range"][1] - spec["range"][0])
    return out


def extreme_fields(c: dict, space: dict, edge: float = EXTREME_EDGE) -> list[str]:
    """Continuous fields at a range end (within `edge` of 0 or 1 in unit position)."""
    return sorted(k for k, u in continuous_units(c, space).items() if min(u, 1.0 - u) < edge)


def edge_distance(c: dict, space: dict) -> float | None:
    """Smallest unit distance of any continuous field to its range end (0 = at an end, 0.5 = all mid-range)."""
    us = continuous_units(c, space)
    return min(min(u, 1.0 - u) for u in us.values()) if us else None


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def outcome_status(d: dict) -> str:
    """integration3 #7: never_switched is not 'failed' (no LQR recovery exists to judge)."""
    if d["switched"] == 0:
        return "never_switched" if d["never_switched"] else "no_valid_runs"
    if d["recovered"] == 0 and d["failed"] > 0:
        return "failed_whenever_switched"
    if d["failed"] == 0 and d["recovered"] > 0:
        return "recovered_whenever_switched"
    return "mixed" if d["recovered"] and d["failed"] else "unknown"


def outcomes_by_fault_mode(records: list[dict]) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for r in records:
        mode = (((r.get("parameters") or {}).get("fault") or {}).get("mode")) or "?"
        d = per.setdefault(mode, {"n": 0, "switched": 0, "never_switched": 0, "recovered": 0, "failed": 0,
                                  "unlabelled": 0, "invalid": 0, "failed_reasons": {}})
        d["n"] += 1
        if r.get("sim_status") != "ok":
            d["invalid"] += 1
            continue
        if r.get("switch_time") is None:
            d["never_switched"] += 1
            continue
        d["switched"] += 1
        fs = r.get("final_safe_recovery")
        if fs is True:
            d["recovered"] += 1
        elif fs is False:
            d["failed"] += 1
            why = failed_reason(r)
            d["failed_reasons"][why] = d["failed_reasons"].get(why, 0) + 1
        else:
            d["unlabelled"] += 1
    for d in per.values():
        d["status"] = outcome_status(d)
        if not d["failed_reasons"]:
            d.pop("failed_reasons")
    return per


def surrogate_summary(*, regions: list[dict], records: list[dict], candidates: dict, space: dict, metric: str,
                      switch_metric: str | None, n_train: int | None, k: int = 6, recent: int = 30,
                      scorer=None, native=None, gp_info: dict | None = None, k_unexplored: int = 3,
                      unexplored_std_frac: float = 0.75) -> dict:
    """Compact GP view for Nemotron A every round (integration3 #4, LeGEND-style: the LLM proposes direction, the
    GP / search refines). integration3 #7:
    - values in the target's own units (`native(mean_z, std_z)` -> lane_recovery_robustness), not GP z units;
    - ranked by `scorer(c)` = engine.surrogate.boundary_score_parts (the real acquisition: level set penalised by
      1 - P(switch), or switch uncertainty); points whose GP std is >= unexplored_std_frac of the training std are
      'unexplored_regions' (prior-level uncertainty), the rest 'boundary_regions';
    - recent_outcomes_by_fault_mode separates never_switched from failed."""
    gi = dict(gp_info or {})
    y_scale = float(gi.get("y_scale_z") or 1.0)
    level_z = float(gi.get("level_z") or 0.0)
    if native is None:
        native = lambda m, sd: (m, sd)   # noqa: E731
    if scorer is None:
        from .surrogate import boundary_score_parts
        scorer = lambda c: boundary_score_parts(c, metric, switch_metric, beta=1.96, level_z=level_z, y_scale=y_scale)  # noqa: E731
    run_ids = {r.get("scenario_id"): r for r in records}

    def item(c: dict, parts: dict) -> dict:
        return region_item(c, parts, metric=metric, native=native, y_scale=y_scale, level_z=level_z, runs=run_ids)
    scored = []
    for c in regions:
        parts = scorer(c) or {}
        if parts.get("score") is None:
            continue
        sz = (c.get("gp_std") or {}).get(metric)
        frac = None if sz is None else sz / max(y_scale, 1e-9)
        scored.append((parts, frac, c))
    explored = [t for t in scored if t[1] is None or t[1] < unexplored_std_frac]
    unexplored = [t for t in scored if t[1] is not None and t[1] >= unexplored_std_frac]
    explored.sort(key=lambda t: t[0]["score"], reverse=True)
    unexplored.sort(key=lambda t: (t[0].get("p_switch") or 0.0, t[1]), reverse=True)
    per = outcomes_by_fault_mode(records[-recent:])
    gp = None
    if n_train:
        gp = {"target": gi.get("target", metric), "metric_id": metric, "n_train": n_train,
              "units": gi.get("units"), "level": gi.get("level", 0.0),
              "training_std": gi.get("training_std"),
              "meaning": gi.get("meaning"),
              "ranking": "acquisition_score = max(recovery level-set score in training-std units - switch_penalty * "
                         "(1 - p_switch), switch-classifier level-set score); the orchestrator picks runs with the "
                         "same score",
              "unexplored_rule": f"gp_uncertainty_vs_training_std >= {unexplored_std_frac}"}
    return {"gp": gp,
            "boundary_regions": [item(c, p) for p, _, c in explored[:k]],
            "unexplored_regions": [item(c, p) for p, _, c in unexplored[:k_unexplored]],
            "recent_outcomes_by_fault_mode": per}


def region_item(c: dict, parts: dict, *, metric: str, native, y_scale: float, level_z: float,
                runs: dict | None = None) -> dict:
    """One GP region / refine candidate in the candidate item shape plus the GP view in target units."""
    gm, gs = c.get("gp_mean") or {}, c.get("gp_std") or {}
    mz, sz = gm.get(metric), gs.get(metric)
    mean, sd = native(mz, sz) if mz is not None and sz is not None else (None, None)
    out = {"fault_id": c["fault"]["fault_id"], "fault_mode": c["fault"].get("mode"), "params": c["fault"]["params"],
           "route_id": c["route_id"], "max_speed_kmh": c["environment"]["max_speed_kmh"],
           "onset_event": c["onset_event"], "onset_offset_s": c["onset_offset_s"]}
    if "tire_friction_scale" in c["environment"]:
        out["tire_friction_scale"] = c["environment"]["tire_friction_scale"]
    out.update({
        "predicted_lane_recovery_robustness": None if mean is None else round(mean, 3),
        "predicted_lane_recovery_robustness_std": None if sd is None else round(sd, 3),
        "p_below_level": None if mz is None or not sz else round(_phi((level_z - mz) / sz), 3),
        "p_switch": None if parts.get("p_switch") is None else round(parts["p_switch"], 3),
        "gp_uncertainty_vs_training_std": None if sz is None else round(sz / max(y_scale, 1e-9), 2),
        "acquisition_score": None if parts.get("score") is None else round(parts["score"], 3),
        "ranked_by": parts.get("dominant")})
    r = (runs or {}).get(c.get("scenario_id"))
    if r is not None:
        out["observed"] = outcome_view(r)
    return out


def call_a(client, settings: dict, messages: list[dict], schema: dict, tag: str = "A") -> tuple[dict | None, dict]:
    """Returns (parsed_obj or None, stats). Falls back to json_object if json_schema is rejected (HTTP 400)."""
    stats = {"schema_errors": 0, "parse_error": None, "truncated": False, "response_format": "json_schema"}
    kw = dict(temperature=settings.get("temperature"), top_p=settings.get("top_p"), max_tokens=settings.get("max_tokens"),
              thinking=settings.get("thinking"), tag=tag)
    try:
        res = client.chat(settings["model"], messages, response_format=json_schema_format("scenario_candidates", schema), **kw)
    except LLMError as e:
        if "HTTP 400" not in str(e):
            stats["parse_error"] = str(e)[:200]
            stats["truncated"] = "finish_reason=length" in str(e)
            return None, stats
        stats["response_format"] = "json_object_fallback"
        res = client.chat(settings["model"], _with_schema(messages, schema), response_format={"type": "json_object"}, **kw)
    stats["usage"] = res.usage
    try:
        obj = extract_json(res.content)
    except ValueError as e:
        stats["parse_error"] = str(e)
        return None, stats
    errs = S.validate(schema, obj)
    stats["schema_errors"] = len(errs)
    stats["schema_error_examples"] = errs[:5]
    if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
        item_schema = schema["properties"]["candidates"]["items"]
        good = [c for c in obj["candidates"] if not S.validate(item_schema, c)]
        stats["items_returned"], stats["items_valid"] = len(obj["candidates"]), len(good)
        obj = {"candidates": good}
    return obj, stats


REFINE_INSTR = """Refine pass (thinking enabled). Each candidate below is a tried condition near the recovery boundary, \
with the GP prediction of lane_recovery_robustness in its own units (see surrogate.gp.units; level 0 = vehicle body at \
the lane edge), p_below_level, p_switch and the observed outcome if it ran. Propose ONE modified scenario per candidate \
that you expect to lie closer to the boundary (predicted robustness ~ 0 with the switch still happening), changing at \
most two fields and keeping route_id and fault_id. Respect route_geometry (V-03 latest onset offset) and \
objective.judgment_criteria. Output ONLY the JSON object of the response format."""

REFINE_CONTEXT_KEYS = ("objective", "supervisor", "scenario_space", "routes", "route_geometry", "outcome_examples",
                       "already_tried")


def refine_messages(schema: dict, near: list[dict], context: dict, include_schema: bool = False) -> list[dict]:
    """integration3 #7: refine gets the same compact context as the main call (faults, supervisor, criteria, route
    geometry, outcome pairs, already_tried, GP header + outcome mix) and `near` already in target units
    (region_item rows)."""
    summ = context.get("surrogate_summary") or {}
    user = {"task": REFINE_INSTR, "candidates": near,
            **{k: context.get(k) for k in REFINE_CONTEXT_KEYS if context.get(k) is not None},
            "surrogate": {"gp": summ.get("gp"), "recent_outcomes_by_fault_mode": summ.get("recent_outcomes_by_fault_mode")}}
    if include_schema:
        user["json_schema"] = schema
    return [{"role": "system", "content": A_SYSTEM}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, default=str)}]


# ----------------------------------------------------------------------------- B
def code_context(av_stack: Path | None, files: list[str]) -> tuple[str, dict[str, int]]:
    """Concatenate whole files with 1-based line numbers. Returns (text, {file: n_lines})."""
    if av_stack is None:
        return "", {}
    parts, sizes = [], {}
    for rel in files:
        p = av_stack / rel
        if not p.is_file():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        sizes[rel] = len(lines)
        parts.append(f"### FILE: {rel}\n" + "\n".join(f"{i + 1:4d}| {ln}" for i, ln in enumerate(lines)))
    return "\n\n".join(parts), sizes


def _yaml_answer(v: Any) -> str:
    """YAML 1.1 reads unquoted yes/no/on/off as booleans (the old Q-B2/Q-B3 bank). Map them back to strings."""
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return str(v)


def normalize_questions(questions: list[dict]) -> list[dict]:
    """integration3 #6: copy of the question bank with answer options / expected answers as strings and the
    scope / gp_feature defaults filled in (per_scenario + gp_feature true = the pre-integration3 behaviour)."""
    def norm_exp(e: Any) -> Any:
        if e is None:
            return None
        if isinstance(e, list):
            return [_yaml_answer(x) for x in e]
        if isinstance(e, dict):
            out = {}
            for k, v in e.items():
                if k == "cases":
                    out[k] = [dict(c, expect=norm_exp(c.get("expect"))) for c in (v or [])]
                elif k in ("by_fault_mode",):
                    out[k] = {m: norm_exp(x) for m, x in (v or {}).items()}
                elif k == "default":
                    out[k] = norm_exp(v)
                else:
                    out[k] = v
            return out
        return _yaml_answer(e)
    out = []
    for q in questions:
        q = dict(q)
        q["answer"] = [_yaml_answer(x) for x in q["answer"]]
        q["expected_from_code"] = norm_exp(q.get("expected_from_code"))
        q["scope"] = str(q.get("scope", "per_scenario"))
        if q["scope"] not in ("per_scenario", "static"):
            raise ValueError(f"{q['id']}: scope must be per_scenario or static")
        q["gp_feature"] = bool(q.get("gp_feature", q["scope"] == "per_scenario")) and q["scope"] == "per_scenario"
        out.append(q)
    return out


def questions_of_scope(questions: list[dict], scope: str) -> list[dict]:
    return [q for q in questions if q.get("scope", "per_scenario") == scope]


def param_bin_suffix(questions: list[dict], params: dict) -> str:
    """b_key suffix: bin index of every fault parameter a per_scenario question declares in param_bins
    ({duration_s: [0.15]} -> duration_s#0 for <= 0.15, #1 above). A fault without that parameter adds nothing
    (bias / drift / ... run until the route ends), so the cache stays bounded (mode x onset x bins)."""
    bins: dict[str, list[float]] = {}
    for q in questions_of_scope(questions, "per_scenario"):
        for name, edges in (q.get("param_bins") or {}).items():
            bins.setdefault(name, sorted(float(e) for e in edges))
    parts = []
    for name, edges in sorted(bins.items()):
        v = (params or {}).get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{name}#{sum(float(v) > e for e in edges)}")
    return ("|" + "|".join(parts)) if parts else ""


def _case_holds(when: dict, params: dict) -> bool:
    v = (params or {}).get(when.get("param"))
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    ops = {"le": lambda a, b: a <= b, "lt": lambda a, b: a < b, "ge": lambda a, b: a >= b, "gt": lambda a, b: a > b}
    checks = [(op, float(when[op])) for op in ops if op in when]
    return bool(checks) and all(ops[op](float(v), t) for op, t in checks)


def expected_answers(q: dict, ctx: dict | None) -> list[str] | None:
    """Acceptable answers the code decides for this question in this scenario context, or None (not decided).
    expected_from_code: 'x' | ['x', 'y'] | {by_fault_mode: {mode: spec}} with spec 'x' | [..] | {cases, default}."""
    e = q.get("expected_from_code")
    if isinstance(e, dict) and "by_fault_mode" in e:
        e = e["by_fault_mode"].get((ctx or {}).get("fault_mode"))
    if isinstance(e, dict):
        params = (ctx or {}).get("fault_params") or {}
        hit = next((c for c in e.get("cases") or [] if _case_holds(c.get("when") or {}, params)), None)
        e = hit.get("expect") if hit is not None else e.get("default")
    if e is None:
        return None
    return [_yaml_answer(x) for x in (e if isinstance(e, list) else [e])]


def _function_ranges(path: Path) -> dict[str, tuple[int, int]]:
    """{'Class': (a, b), 'Class.method': (a, b), 'func': (a, b)} with 1-based inclusive lines (ast)."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    out: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = (node.lineno, node.end_lineno or node.lineno)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[f"{node.name}.{sub.name}"] = (sub.lineno, sub.end_lineno or sub.lineno)
    return out


def resolve_regions(av_stack: Path | None, questions: list[dict]) -> dict[str, list[tuple[str, int, int]]]:
    """{question id: [(file, first, last), ...]} from evidence_regions ({file, function} or {file, lines: [a, b]}).
    A function that cannot be found (no av_stack, renamed) is skipped; a question with no resolvable region gets
    an empty list and citations_relevant None."""
    cache: dict[str, dict] = {}
    out: dict[str, list[tuple[str, int, int]]] = {}
    for q in questions:
        regs = []
        for r in q.get("evidence_regions") or []:
            f = r.get("file")
            if not f:
                continue
            if r.get("lines"):
                a, b = r["lines"]
                regs.append((f, int(a), int(b)))
            elif r.get("function") and av_stack is not None:
                if f not in cache:
                    cache[f] = _function_ranges(Path(av_stack) / f)
                rng = cache[f].get(r["function"])
                if rng:
                    regs.append((f, rng[0], rng[1]))
        out[q["id"]] = regs
    return out


def b_schema(questions: list[dict], files: list[str]) -> dict:
    items = []
    for q in questions:
        items.append({"type": "object", "properties": {
            "id": {"enum": [q["id"]]},
            "answer": {"enum": [_yaml_answer(x) for x in q["answer"]]},
            "evidence": {"type": "array", "minItems": 0, "maxItems": 6, "items": {
                "type": "object", "properties": {"file": {"enum": files or ["none"]}, "line": {"type": "integer", "minimum": 1, "maximum": 100000}},
                "required": ["file", "line"], "additionalProperties": False}},
            "rationale": {"type": "string"}},
            "required": ["id", "answer", "evidence", "rationale"], "additionalProperties": False})
    return {"type": "object", "properties": {"answers": {"type": "array", "minItems": len(questions), "maxItems": len(questions),
                                                         "items": {"anyOf": items}}},
            "required": ["answers"], "additionalProperties": False}


B_SYSTEM = """You are Nemotron B, a careful code reader. Answer questions about the av_stack MPC->LQR supervisor ONLY from \
the source files given (line-numbered). You are not shown any simulation results and must not guess them. For every \
answer cite the lines that decide it as {file, line}. Answer 'unknown' when the code does not decide the question; \
'unknown' is not 'no'. Output ONLY the JSON object matching the schema."""


def b_messages(code: str, questions: list[dict], scenario_ctx: dict, schema: dict) -> list[dict]:
    qs = []
    for q in questions:
        item = {"id": q["id"], "question": q.get("question_en") or q["question"], "allowed_answers": [_yaml_answer(x) for x in q["answer"]]}
        if q.get("answer_notes"):
            item["answer_meanings"] = q["answer_notes"]
        qs.append(item)
    # expected_from_code / evidence_regions are never sent: they are the answer key of the hallucination check
    user = f"SOURCE FILES\n{code}\n\nSCENARIO CONTEXT (inputs only, no results)\n{json.dumps(scenario_ctx)}\n\n" \
           f"QUESTIONS\n{json.dumps(qs, ensure_ascii=False)}\n\nJSON SCHEMA\n{json.dumps(schema)}"
    return [{"role": "system", "content": B_SYSTEM}, {"role": "user", "content": user}]


def check_citations(answer: dict, sizes: dict[str, int]) -> bool:
    ev = answer.get("evidence") or []
    return bool(ev) and all(e.get("file") in sizes and 1 <= int(e.get("line", 0)) <= sizes[e["file"]] for e in ev)


def citations_relevant(answer: dict, sizes: dict[str, int], regions: list[tuple[str, int, int]] | None) -> bool | None:
    """integration3 #6: valid (every line inside its file) AND at least one cited line inside the question's
    evidence_regions. None when the question has no resolvable region (cannot be judged)."""
    if not regions:
        return None
    if not check_citations(answer, sizes):
        return False
    return any(e.get("file") == f and a <= int(e.get("line", 0)) <= b for e in answer.get("evidence") or [] for f, a, b in regions)


def aggregate_b(samples: list[dict], questions: list[dict], sizes: dict[str, int], ctx: dict | None = None,
                regions: dict[str, list] | None = None) -> dict[str, dict]:
    """Majority answer per question + the hallucination check (integration3 #6):
    matches_code = majority answer is one the code allows (expected_answers), None when the code does not decide;
    hallucinated = matches_code False and the answer is not 'unknown' (unknown_is_not_false)."""
    out = {}
    for q in questions:
        answers = [dict(a, answer=_yaml_answer(a.get("answer", "unknown"))) for s in samples for a in s.get("answers", [])
                   if a.get("id") == q["id"]]
        votes = Counter(a["answer"] for a in answers)
        maj, cnt = votes.most_common(1)[0] if votes else ("unknown", 0)
        exp = expected_answers(q, ctx)
        matches = None if exp is None or not answers else maj in exp
        rel = [citations_relevant(a, sizes, (regions or {}).get(q["id"])) for a in answers]
        out[q["id"]] = {"answer": maj, "agreement": cnt / max(1, len(samples)), "votes": dict(votes),
                        "citations_valid": [check_citations(a, sizes) for a in answers],
                        "citations_relevant": rel,
                        "evidence": next((a.get("evidence") for a in answers if a.get("answer") == maj), []),
                        "scope": q.get("scope", "per_scenario"),
                        "expected_from_code": exp,
                        "matches_code": matches,
                        "hallucinated": bool(matches is False and maj != "unknown")}
    return out


def b_features(answers: dict[str, dict] | None, questions: list[dict]) -> tuple[list[float], list[str]]:
    """One-hot of the per_scenario questions with gp_feature true. Static questions (one fixed code fact per
    campaign) and gp_feature false questions are recorded for the hallucination check only: a constant column
    carries no information for the GP."""
    x, names = [], []
    for q in questions:
        if q.get("scope", "per_scenario") != "per_scenario" or not q.get("gp_feature", True):
            continue
        a = _yaml_answer((answers or {}).get(q["id"], {}).get("answer", "unknown"))
        for opt in (_yaml_answer(o) for o in q["answer"]):
            names.append(f"{q['id']}={opt}"); x.append(1.0 if a == opt else 0.0)
    return x, names


def b_hallucination_summary(entries: list[dict]) -> dict:
    """Campaign-level B check over aggregated answer dicts ({qid: aggregate_b row}, one per b_key + static)."""
    per_q: dict[str, dict] = {}
    for ent in entries:
        for qid, row in (ent or {}).items():
            d = per_q.setdefault(qid, {"asked": 0, "checked": 0, "matched": 0, "hallucinated": 0, "unknown_when_decided": 0,
                                       "answers": Counter(), "citations": 0, "citations_valid": 0, "citations_judged": 0,
                                       "citations_relevant": 0})
            d["asked"] += 1
            d["answers"][str(row.get("answer"))] += 1
            if row.get("matches_code") is not None:
                d["checked"] += 1
                d["matched"] += bool(row["matches_code"])
                d["hallucinated"] += bool(row.get("hallucinated"))
                d["unknown_when_decided"] += row.get("answer") == "unknown" and not row["matches_code"]
            cv = row.get("citations_valid") or []
            d["citations"] += len(cv); d["citations_valid"] += sum(bool(v) for v in cv)
            cr = [v for v in row.get("citations_relevant") or [] if v is not None]
            d["citations_judged"] += len(cr); d["citations_relevant"] += sum(bool(v) for v in cr)
    tot = {k: sum(d[k] for d in per_q.values()) for k in ("checked", "matched", "hallucinated", "unknown_when_decided",
                                                         "citations", "citations_valid", "citations_judged", "citations_relevant")}
    for d in per_q.values():
        d["answers"] = dict(d["answers"])
        d["distinct_answers"] = len(d["answers"])
        d["hallucination_rate"] = d["hallucinated"] / d["checked"] if d["checked"] else None
    return {"b_hallucination_rate": tot["hallucinated"] / tot["checked"] if tot["checked"] else None,
            "b_code_agreement_rate": tot["matched"] / tot["checked"] if tot["checked"] else None,
            "b_checked_answers": tot["checked"], "b_hallucinated_answers": tot["hallucinated"],
            "b_unknown_when_code_decides": tot["unknown_when_decided"],
            "citation_valid_rate": tot["citations_valid"] / tot["citations"] if tot["citations"] else None,
            "citation_relevant_rate": tot["citations_relevant"] / tot["citations_judged"] if tot["citations_judged"] else None,
            "per_question": per_q,
            "note": "hallucinated = majority answer outside expected_from_code (not 'unknown'); one row per b_key + static"}


# ----------------------------------------------------------------------------- stubs (--mock)
def _stub_near(region: dict, items_schema: dict, rng: random.Random) -> dict | None:
    """SYNTHETIC: a schema-valid copy of a boundary region with numeric fields jittered by ~8% of their range."""
    br = next((b for b in items_schema.get("anyOf", []) if b["properties"]["fault_id"]["enum"] == [region.get("fault_id")]), None)
    if br is None:
        return None
    c = S.sample(br, rng)
    props = br["properties"]

    def jitter(spec: dict, v):
        if not isinstance(v, (int, float)) or "minimum" not in spec:
            return None
        lo, hi = spec["minimum"], spec["maximum"]
        return min(hi, max(lo, float(v) + rng.gauss(0, 0.08) * (hi - lo)))
    for key in ("max_speed_kmh", "onset_offset_s", "tire_friction_scale"):
        if key in props and key in region:
            j = jitter(props[key], region[key])
            if j is not None:
                c[key] = j
    for key in ("route_id", "onset_event"):
        if region.get(key) in props[key].get("enum", []):
            c[key] = region[key]
    pprops = props["params"]["properties"]
    for k, v in (region.get("params") or {}).items():
        if k in pprops:
            j = jitter(pprops[k], v)
            c["params"][k] = j if j is not None else (v if v in pprops[k].get("enum", [v]) else c["params"][k])
    c["expected_side"] = "boundary"
    return c



def _stub_between(pair: dict, items_schema: dict, rng: random.Random) -> dict | None:
    """SYNTHETIC: a candidate halfway between a recovered and a failed run of the same fault (outcome_examples)."""
    rec, fail = pair["recovered"], pair["failed"]
    region = dict(fail)
    for key in ("max_speed_kmh", "onset_offset_s", "tire_friction_scale"):
        if isinstance(rec.get(key), (int, float)) and isinstance(fail.get(key), (int, float)):
            region[key] = 0.5 * (rec[key] + fail[key])
    region["params"] = {k: (0.5 * (v + rec["params"][k]) if isinstance(v, (int, float)) and not isinstance(v, bool)
                            and isinstance((rec.get("params") or {}).get(k), (int, float)) else v)
                        for k, v in (fail.get("params") or {}).items()}
    c = _stub_near(region, items_schema, rng)
    if c is not None:
        c["expected_side"] = "boundary"
    return c


def _stub_fit_route(c: dict, items_schema: dict, geom: dict) -> None:
    """SYNTHETIC: use route_geometry like a careful proposer would: pick an onset_event the route has at about this
    speed and keep the onset offset at or before the latest one V-03 can still grade."""
    g = geom.get(c.get("route_id"))
    if not g or not g.get("events"):
        return
    kmh = float(c.get("max_speed_kmh", 0))
    br = next((b for b in items_schema.get("anyOf", []) if b["properties"]["fault_id"]["enum"] == [c.get("fault_id")]), None)
    if br is None:
        return
    allowed = br["properties"]["onset_event"]["enum"]
    lo = br["properties"]["onset_offset_s"]["minimum"]
    hi = br["properties"]["onset_offset_s"]["maximum"]
    def latest(ev):   # route_geometry columns: [max_speed_kmh, t_s, kappa, speed, v2kappa, latest_offset]
        rows = g["events"].get(ev) or []
        return min(rows, key=lambda r: abs(float(r[0]) - kmh))[5] if rows else None
    if latest(c.get("onset_event")) is None or latest(c["onset_event"]) < lo:
        ok = [ev for ev in allowed if latest(ev) is not None and latest(ev) >= lo]
        if not ok:
            return
        c["onset_event"] = ok[0]
    c["onset_offset_s"] = max(lo, min(float(c["onset_offset_s"]), hi, latest(c["onset_event"]) - 0.05))


# SYNTHETIC: what the --mock B "reads" from the code. Mostly the real answer, with one deliberate mistake (stale ->
# stale_mpc for Q-B1, what the real Nemotron answered in result_1..5) so the hallucination metric has something to count.
_STUB_B_TABLE = {
    "Q-B1": {"solver_fail": "lqr_early", "stale": "stale_mpc", "output_delay": "stale_mpc"},
    "Q-B4": {"solver_fail": "solver_failure", "stale": "stale_sequence", "output_delay": "tracking_divergence"},
    "Q-B5": {"solver_fail": "ok_flag", "stale": "sequence_or_stamp", "output_delay": "steer_and_reported_state"},
    "Q-B2": {}, "Q-B3": {},
}
_STUB_B_DEFAULT = {"Q-B1": "corrupted_mpc", "Q-B4": "tracking_divergence", "Q-B5": "steer_value_only", "Q-B2": "no", "Q-B3": "no"}
_STUB_B_CITE = {"Q-B1": ("integration/fault_injection.py", 112), "Q-B4": ("integration/supervised_lateral.py", 74),
                "Q-B5": ("integration/fault_injection.py", 101), "Q-B2": ("integration/supervised_lateral.py", 117),
                "Q-B3": ("integration/supervised_lateral.py", 135)}


def _stub_b_answer(qid: str, mode: str | None, params: dict, opts: list[str]) -> str:
    ans = _STUB_B_TABLE.get(qid, {}).get(mode, _STUB_B_DEFAULT.get(qid, opts[-1]))
    if qid == "Q-B4" and mode == "saturate":
        # integration3 #8 (tight supervisor): a step (ramp 0) trips the 90 deg/s rate check at once; a ramped
        # saturate shorter than the 0.15 s divergence hold cannot fire during the fault
        if float(params.get("ramp_s", 0.0) or 0.0) <= 0.0:
            ans = "steer_rate_exceeded"
        elif float(params.get("duration_s", 1.0)) <= 0.15:
            ans = "none_during_fault"
    return ans if ans in opts else opts[-1]


def stub_handler(seed: int = 0):
    """SYNTHETIC answers for --mock. A: random schema-valid candidates; B: fixed answers with real file names."""
    rng = random.Random(seed)
    b_rng = random.Random(seed + 7919)   # B has its own stream: changing the B bank does not shift Nemotron A's

    def handler(tag: str, model: str, messages: list[dict], kw: dict) -> str:
        user = messages[-1]["content"]
        if tag.startswith("A"):
            payload = json.loads(user)
            # integration3 #7: the schema rides in response_format; the body has it only on the json_object fallback
            schema = payload.get("json_schema") or ((kw.get("response_format") or {}).get("json_schema") or {}).get("schema")
            refine = "candidates" in payload and "exactly" not in payload["task"]
            n = int(re.search(r"exactly (\d+)", payload["task"]).group(1)) if "exactly" in payload["task"] else \
                len(payload.get("candidates", [])) or 3
            items_schema = schema["properties"]["candidates"]["items"]
            summ = payload.get("surrogate_summary") or payload.get("surrogate") or {}
            regions = payload.get("candidates") if refine else (summ.get("boundary_regions") or [])
            pairs = [e for e in payload.get("outcome_examples") or [] if e.get("recovered") and e.get("failed")]
            geom = (payload.get("route_geometry") or {}).get("routes") or {}
            cands = []
            for i in range(n):
                # SYNTHETIC stub: refine jitters each given candidate; the main call jitters a GP boundary region for
                # half the items, takes the midpoint of a recovered/failed pair for some, and samples the rest
                if regions and (refine or i % 2 == 0):
                    c = _stub_near(regions[(i if refine else i // 2) % len(regions)], items_schema, rng)
                    if c is not None:
                        cands.append(c)
                        continue
                if pairs and i % 4 == 1:
                    c = _stub_between(pairs[(i // 4) % len(pairs)], items_schema, rng)
                    if c is not None:
                        cands.append(c)
                        continue
                cands.append(S.sample(items_schema, rng))
            for c in cands:
                _stub_fit_route(c, items_schema, geom)
                c["expected_mechanism"] = "SYNTHETIC stub: no model was called"
                c["extreme_justification"] = ""
            return json.dumps({"candidates": cands})
        if tag.startswith("B"):
            m = re.search(r"JSON SCHEMA\n(.*)$", user, re.DOTALL)
            schema = json.loads(m.group(1))
            cm = re.search(r"SCENARIO CONTEXT \(inputs only, no results\)\n(.*?)\n\nQUESTIONS", user, re.DOTALL)
            ctx = json.loads(cm.group(1)) if cm else {}
            mode = ctx.get("fault_mode")
            files = schema["properties"]["answers"]["items"]["anyOf"][0]["properties"]["evidence"]["items"]["properties"]["file"]["enum"]
            answers = []
            for br in schema["properties"]["answers"]["items"]["anyOf"]:
                qid = br["properties"]["id"]["enum"][0]
                opts = br["properties"]["answer"]["enum"]
                ans = _stub_b_answer(qid, mode, ctx.get("fault_params") or {}, opts)
                if b_rng.random() < 0.15:
                    ans = b_rng.choice(opts)
                f, line = _STUB_B_CITE.get(qid, ("integration/supervised_lateral.py", 113))
                if f not in files or b_rng.random() < 0.15:       # SYNTHETIC: some citations point at an unrelated line
                    f, line = files[-1], 1
                answers.append({"id": qid, "answer": ans, "evidence": [{"file": f, "line": line}],
                                "rationale": "SYNTHETIC stub answer"})
            return json.dumps({"answers": answers})
        raise ValueError(f"no stub for {tag}")
    return handler
