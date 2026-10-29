"""03 Nemotron A (scenario candidates) and 05b Nemotron B (code questions) on Token Factory.

A: Nemotron-3.5-Lightning, thinking OFF, response_format json_schema built from the scenario space
   (ranges as minimum/maximum, ids as enum), same schema also pasted into the prompt, English prompt.
   Optional refine pass (thinking ON, large max_tokens) for boundary-near candidates.
   A is the only role that sees failures (09 failure_table + accepted 10 explanations).
B: Nemotron-3-Ultra, thinking ON + medium_effort, whole av_stack source files (with line numbers)
   in context, answers yes/no/unknown(+Q-B1 options) with file:line evidence, repeated 2-3x for a
   consistency score. B never sees results, failures or GP predictions.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import schema as S
from .llm import LLMError, extract_json, json_schema_format
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
                "missing_information": {"type": "string"},
            },
            "required": ["fault_id", "params", "route_id", "max_speed_kmh"] + (["tire_friction_scale"] if fax else []) +
                        ["onset_event", "onset_offset_s",
                         "objective_ids", "evidence_ids", "expected_mechanism", "expected_side", "missing_information"],
            "additionalProperties": False,
        })
    return {"type": "object",
            "properties": {"candidates": {"type": "array", "minItems": 1, "maxItems": n_max, "items": {"anyOf": branches}}},
            "required": ["candidates"], "additionalProperties": False}


A_SYSTEM = """You are Nemotron A, the scenario generator of a software-in-the-loop safety SEARCH for an MPC->LQR steering handover \
(CARLA 0.9.15, PDM-Lite planner, av_stack MPC primary + LQR backup + latched supervisor; faults are injected only at the MPC output).
Search objective S1: estimate WHERE LQR recovery after the switch stops succeeding, as a function of the error state at the \
switch moment and of steering angle / rate saturation. Propose scenarios on BOTH sides of the boundary (expected recover, \
expected fail, and boundary) - a boundary cannot be estimated from failures only.
Rules:
- You never decide safety. A deterministic monitor labels every run. Do not claim a scenario is safe or unsafe as a fact.
- Cite only ids from the evidence enum. Unreviewed code observations are leads, not verified facts.
- Use only values allowed by the JSON schema. Do not invent controller parameters.
- Avoid repeating scenarios listed under already_tried.
- Output ONLY a JSON object that matches the schema below. No prose."""


def a_messages(*, schema: dict, n: int, context: dict) -> list[dict]:
    user = {"task": f"Propose exactly {n} candidate scenarios for the active objectives.",
            **context, "json_schema": schema}
    return [{"role": "system", "content": A_SYSTEM}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, default=str)}]


def a_context(*, profile: dict, prior: dict, space: dict, objective_cfg: dict, routes: list[dict], few_shot: list[dict],
              mechanisms: list[dict], tried: list[dict], include_unreviewed: bool) -> dict:
    obs = (profile.get("code_evidence") or {}).get("observations", [])
    verified = [o for o in obs if o.get("verified") is True]
    unreviewed = [] if not include_unreviewed else [
        {k: o.get(k) for k in ("id", "field", "observed_value", "note")} for o in obs if o.get("verified") is not True]
    faults = [{k: f.get(k) for k in ("id", "name", "group", "injector_mode", "description", "check", "params",
                                     "handover_axes", "next_question") if f.get(k) is not None}
              for f in space.get("faults", [])]
    return {
        "objective": objective_cfg,
        "verified_code_observations": verified,
        "unreviewed_code_observations_static_extraction": unreviewed,
        "domain_prior_knowledge": prior.get("principles", []),
        "supervisor": space.get("supervisor"),
        "scenario_space": {"environment": space.get("environment"), "fault_timing": space.get("fault_timing"), "faults": faults},
        "routes": routes,
        "prior_failures_top_k": few_shot,
        "verified_video_mechanisms": mechanisms,
        "already_tried": tried,
    }


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
            out.append(c)
        except (KeyError, TypeError, ValueError):
            continue
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
        res = client.chat(settings["model"], messages, response_format={"type": "json_object"}, **kw)
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


REFINE_INSTR = """Refine pass (thinking enabled). For each candidate below the surrogate predicts lane_recovery_robustness \
mean +- std (0 = lane boundary). Propose ONE modified scenario per candidate that you expect to lie closer to the \
recovery boundary (robustness ~ 0), changing at most two fields and keeping route_id and fault_id. Same JSON schema."""


def refine_messages(schema: dict, near: list[dict], context: dict) -> list[dict]:
    items = [{k: c.get(k) for k in ("route_id", "environment", "fault", "onset_event", "onset_offset_s")} |
             {"predicted_robustness_mean_z": (c.get("gp_mean") or {}).get("recovery_success"),
              "predicted_robustness_std_z": (c.get("gp_std") or {}).get("recovery_success")} for c in near]
    user = {"task": REFINE_INSTR, "candidates": items, "routes": context.get("routes"),
            "prior_failures_top_k": context.get("prior_failures_top_k"), "json_schema": schema}
    return [{"role": "system", "content": A_SYSTEM}, {"role": "user", "content": json.dumps(user, default=str)}]


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


def b_schema(questions: list[dict], files: list[str]) -> dict:
    items = []
    for q in questions:
        items.append({"type": "object", "properties": {
            "id": {"enum": [q["id"]]},
            "answer": {"enum": q["answer"]},
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
    qs = [{"id": q["id"], "question": q.get("question_en") or q["question"], "allowed_answers": q["answer"]} for q in questions]
    user = f"SOURCE FILES\n{code}\n\nSCENARIO CONTEXT (inputs only, no results)\n{json.dumps(scenario_ctx)}\n\n" \
           f"QUESTIONS\n{json.dumps(qs, ensure_ascii=False)}\n\nJSON SCHEMA\n{json.dumps(schema)}"
    return [{"role": "system", "content": B_SYSTEM}, {"role": "user", "content": user}]


def check_citations(answer: dict, sizes: dict[str, int]) -> bool:
    ev = answer.get("evidence") or []
    return bool(ev) and all(e.get("file") in sizes and 1 <= int(e.get("line", 0)) <= sizes[e["file"]] for e in ev)


def aggregate_b(samples: list[dict], questions: list[dict], sizes: dict[str, int]) -> dict[str, dict]:
    out = {}
    for q in questions:
        answers = [a for s in samples for a in s.get("answers", []) if a.get("id") == q["id"]]
        votes = Counter(a.get("answer", "unknown") for a in answers)
        maj, cnt = votes.most_common(1)[0] if votes else ("unknown", 0)
        out[q["id"]] = {"answer": maj, "agreement": cnt / max(1, len(samples)), "votes": dict(votes),
                        "citations_valid": [check_citations(a, sizes) for a in answers],
                        "evidence": next((a.get("evidence") for a in answers if a.get("answer") == maj), []),
                        "expected_from_code": q.get("expected_from_code")}
    return out


def b_features(answers: dict[str, dict] | None, questions: list[dict]) -> tuple[list[float], list[str]]:
    x, names = [], []
    for q in questions:
        a = (answers or {}).get(q["id"], {}).get("answer", "unknown")
        for opt in q["answer"]:
            names.append(f"{q['id']}={opt}"); x.append(1.0 if a == opt else 0.0)
    return x, names


# ----------------------------------------------------------------------------- stubs (--mock)
def stub_handler(seed: int = 0):
    """SYNTHETIC answers for --mock. A: random schema-valid candidates; B: fixed answers with real file names."""
    rng = random.Random(seed)

    def handler(tag: str, model: str, messages: list[dict], kw: dict) -> str:
        user = messages[-1]["content"]
        if tag.startswith("A"):
            payload = json.loads(user)
            schema = payload["json_schema"]
            n = int(re.search(r"exactly (\d+)", payload["task"]).group(1)) if "exactly" in payload["task"] else \
                len(payload.get("candidates", [])) or 3
            obj = {"candidates": [S.sample(schema["properties"]["candidates"]["items"], rng) for _ in range(n)]}
            for c in obj["candidates"]:
                c["expected_mechanism"] = "SYNTHETIC stub: no model was called"
            return json.dumps(obj)
        if tag.startswith("B"):
            m = re.search(r"JSON SCHEMA\n(.*)$", user, re.DOTALL)
            schema = json.loads(m.group(1))
            files = schema["properties"]["answers"]["items"]["anyOf"][0]["properties"]["evidence"]["items"]["properties"]["file"]["enum"]
            f0 = "integration/supervised_lateral.py" if "integration/supervised_lateral.py" in files else files[0]
            canned = {"Q-B1": "stale_mpc", "Q-B2": "no", "Q-B3": "yes"}
            answers = []
            for br in schema["properties"]["answers"]["items"]["anyOf"]:
                qid = br["properties"]["id"]["enum"][0]
                ans = canned.get(qid, br["properties"]["answer"]["enum"][-1])
                if rng.random() < 0.15:
                    ans = rng.choice(br["properties"]["answer"]["enum"])
                answers.append({"id": qid, "answer": ans, "evidence": [{"file": f0, "line": 113}],
                                "rationale": "SYNTHETIC stub answer"})
            return json.dumps({"answers": answers})
        raise ValueError(f"no stub for {tag}")
    return handler
