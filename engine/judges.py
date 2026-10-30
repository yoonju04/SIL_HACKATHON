"""05b B judge on the laya-judge branch: a "System One" typed-decision model as 25 judges per state + regression.

System One models do not generate text. One request = one `state` (string / JSON) and a map of named typed
questions answered in parallel and in isolation, over one wire protocol shared by two backends:
  laya (default) — open-source Laya (Apache-2.0, github.com/NandhaKishorM/laya, HF convaiinnovations/laya), served
                   locally by `laya-serve` (or in-process laya.Router). No key.
  jev            — TypeSafe's hosted Jev (docs.typesafe.ai, checked 2026-10-29), Authorization: Bearer $TYPESAFE_API_KEY.
  POST {base_url}/v1/systemone
  {"model": ..., "state": ..., "questions": {name: {"type": "choice", "instructions": str, "criteria": {option: description}}
                                              | {"type": "noul", "instructions": str, "criteria": {"true", "false"}}}}
  -> {"model", "answers": {name: {"type": "choice", "choice", "confidence", "probabilities": {option: p}}
                                 | {"type": "noul", "noul": P(yes)}}, "usage": {"input_tokens", "output_tokens"}}
Laya differences that matter here: context 512 (english) / 1024..8192 (multilingual) tokens per question, options
share a 192/256-token head budget, `confidence` is 1 - normalised entropy (not Jev's (K·pmax-1)/(K-1)), and
boolean-word choice labels (yes/no) are unreliable, so choice keys for yes/no questions are renamed here.
No rationale or citations, so the Nemotron B citation / hallucination checks are replaced by agreement with the
code-derived answer key (expected_from_code, supervisor-profile aware) and calibration (Brier, ECE) of out-of-fold
regression probabilities, plus low-confidence flags.

Pipeline here:
  judge_questions(bank, layout)  -> 25 questions generated from the Nemotron B bank (single question source)
  build_state / compact_state    -> state: code excerpts + scenario inputs + supervisor profile settings (NO answers)
  SystemOneClient / LayaInProcess / StubJudge -> raw answers (stub: deterministic SYNTHETIC, for --mock and tests)
  parse_answers                  -> per original question: one probability vector over its options per judge
  JudgeRegression                -> per question: L2 multinomial logistic regression from the judges' log-probs to
                                    the answer key (out-of-fold CV for every reported number); mean of judges when
                                    labels are too few (Quantitative LLM Judges, Sahoo et al. 2025, arXiv 2506.02945;
                                    Deusser et al. 2026 arXiv 2609.37647: Jev ranks well, a fixed 0.5 cut does not)
  jev_features                   -> regressed probabilities of the gp_feature questions as continuous GP inputs
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from engine import nemotron as NM

SYNTHETIC_NOTE = "SYNTHETIC stub judge: deterministic probabilities from the answer key; no model was called"
BACKENDS = ("laya", "jev", "nemotron")      # System One judges; "nemotron_b" (legacy generative B) is run.py-only
# Laya: boolean-word choice keys are unreliable (README "Honest limits"), so yes/no choice options get semantic keys
YN_KEYS = {"yes": "holds", "no": "ruled_out", "unknown": "undecided"}


# ---------------------------------------------------------------- questions (generated from the Nemotron B bank)
def _option_text(q: dict, opt: str, lang: str) -> str:
    notes = q.get("answer_notes") or {}
    if opt in notes and lang == "ko":
        return str(notes[opt])
    base = opt.replace("_", " ")
    if opt == "unknown":
        return "the code shown does not decide this"
    en = (q.get("answer_notes_en") or {}).get(opt)
    if en:
        return f"{base}: {en}"
    if opt in notes:
        return f"{base} ({notes[opt]})"
    return base


# ---------------------------------------------------------------- question bank (05b_judges/questions.yaml)
def load_bank(path: Path | None = None) -> list[dict]:
    """J1..J9 (laya-judge v2): the judges' single question source, normalised like the Nemotron B bank."""
    import yaml
    from engine.config import PIPELINE_DIR
    p = Path(path) if path else PIPELINE_DIR / "05b_judges" / "questions.yaml"
    return NM.normalize_questions(yaml.safe_load(p.read_text(encoding="utf-8"))["question_bank"])


def applies(q: dict, fault_mode: str | None) -> bool:
    modes = q.get("applies_to")
    return not modes or fault_mode in modes


def questions_for(questions: dict, meta: dict, bank: list[dict], fault_mode: str | None) -> dict:
    """The request's questions for this scenario: drop judges of questions whose applies_to excludes the fault mode."""
    bq = {q["id"]: q for q in bank}
    return {n: jq for n, jq in questions.items() if applies(bq[meta[n]["qid"]], fault_mode)}


def outcome_label(q: dict, rec: dict | None) -> str | None:
    """Regression label of an outcome question from a finished run record (never part of any judge state).
    detection_delay_bucket: switch_time - actual_fault_onset into edges_s; no switch -> last bucket.
    pre_switch_lateral_error_bucket: metrics.pre_switch_max_lateral_error_m (whole run if no switch) into edges_m;
      departed_at_switch -> last bucket. post_switch_rate_saturation: handover_saturation.rate_saturation_ratio > 0
      (only switched runs)."""
    spec = q.get("outcome_label") or {}
    if not rec or rec.get("sim_status") != "ok":
        return None
    opts = [str(o) for o in q["answer"]]
    kind = spec.get("kind")
    sw = rec.get("switch_time")
    if kind == "detection_delay_bucket":
        if sw is None:
            return opts[-1]
        on = rec.get("actual_fault_onset", rec.get("planned_fault_onset"))
        if on is None:
            return None
        lag = max(0.0, float(sw) - float(on))
        return opts[sum(lag >= e for e in spec.get("edges_s") or [])]
    if kind == "pre_switch_lateral_error_bucket":
        if (rec.get("recovery_labels") or {}).get("departed_at_switch"):
            return opts[-1]
        e = (rec.get("metrics") or {}).get("pre_switch_max_lateral_error_m")
        if e is None:
            return None
        return opts[sum(abs(float(e)) >= x for x in spec.get("edges_m") or [])]
    if kind == "post_switch_rate_saturation":
        if sw is None:
            return None
        r = (rec.get("handover_saturation") or {}).get("rate_saturation_ratio")
        return None if r is None else (opts[0] if float(r) > 0 else opts[1])
    return None


def labels_for(bank: list[dict], ctx: dict, rec: dict | None = None) -> dict[str, list[str] | None]:
    """{qid: allowed answers}. code questions: the profile-aware answer key from the scenario inputs; outcome questions:
    [label of the run] or None before the run. Questions that do not apply to the fault mode: None."""
    out = {}
    for q in bank:
        if not applies(q, ctx.get("fault_mode")):
            out[q["id"]] = None
        elif q.get("label_source") == "outcome":
            lab = outcome_label(q, rec)
            out[q["id"]] = [lab] if lab else None
        else:
            out[q["id"]] = NM.expected_answers(q, ctx)
    return out


def _focus_text(focus: list[str]) -> str:
    return "Use " + ", ".join(f"`{f}`" for f in focus) + "."


def judge_questions(bank: list[dict], layout: dict, views: list[str] | None = None
                    ) -> tuple[dict[str, dict], dict[str, dict]]:
    """(`questions` map, meta {name: {qid, variant, form, options, keys}}). One question per (question id, variant);
    names are '<qid>__<variant>' so answers map back. Only ids in layout['judges'] are asked. `views` = code sections
    present in the state; a focus on a missing section (compact Laya state) falls back to `code`. meta['keys'] maps
    the wire option key back to the bank option (yes/no choice options are sent as holds / ruled_out / undecided)."""
    qs = {q["id"]: q for q in bank}
    variants = layout["variants"]
    out, meta = {}, {}
    for qid, vs in (layout.get("judges") or {}).items():
        q = qs[qid]
        opts = [str(o) for o in q["answer"]]
        yes_no = set(opts) <= {"yes", "no", "unknown"}
        for v in vs:
            spec = variants[v]
            lang = spec.get("lang", "en")
            text = q.get("question_en") if lang == "en" else q.get("question")
            text = str(text or q.get("question"))
            focus = list(spec.get("focus") or ["scenario", "code"])
            if views is not None:
                focus = list(dict.fromkeys(f if not f.startswith("code.") or f[5:] in views else "code" for f in focus))
            instr = f"{text} {_focus_text(focus)}"
            name = f"{qid}__{v}"
            if spec["form"] == "noul":
                if not yes_no:
                    raise ValueError(f"{qid}: noul variant {v} needs a yes/no question")
                jq: dict[str, Any] = {"type": "noul", "instructions": instr}
                if spec.get("criteria"):
                    jq["criteria"] = {"true": "yes: the code shown allows it", "false": "no: the code shown rules it out"}
                meta[name] = {"qid": qid, "variant": v, "form": "noul", "options": opts}
            else:
                order = opts[::-1] if spec.get("order") == "reverse" else opts
                wire = {o: (YN_KEYS.get(o, o) if yes_no else o) for o in order}
                jq = {"type": "choice", "instructions": instr,
                      "criteria": {wire[o]: _option_text(q, o, lang) for o in order}}
                meta[name] = {"qid": qid, "variant": v, "form": "choice", "options": opts,
                              "keys": {w: o for o, w in wire.items()}}
            out[name] = jq
    return out, meta


# ---------------------------------------------------------------- state (inputs only)
def code_views(av_stack: Path | None, views: dict[str, list[dict]], max_chars: int = 9000) -> dict[str, str]:
    """{view: numbered source of the listed functions}. Missing av_stack / function -> a placeholder string."""
    out = {}
    for view, regs in (views or {}).items():
        parts = []
        for reg in regs:
            path = (av_stack / reg["file"]) if av_stack else None
            rng = NM._function_ranges(path).get(reg["function"]) if path and path.is_file() else None
            if not rng:
                parts.append(f"# {reg['file']} {reg['function']}: (source not available)")
                continue
            lines = path.read_text(encoding="utf-8").splitlines()[rng[0] - 1:rng[1]]
            parts.append(f"# {reg['file']} lines {rng[0]}-{rng[1]} ({reg['function']})\n" +
                         "\n".join(f"{rng[0] + i}: {ln}" for i, ln in enumerate(lines)))
        out[view] = "\n\n".join(parts)[:max_chars]
    return out


_COMMENT = re.compile(r"^\s*#")


def _strip(lines: list[str]) -> list[str]:
    """Drop blank lines, full-line comments and docstrings (compact Laya state; the code itself is untouched)."""
    out, in_doc = [], False
    for ln in lines:
        st = ln.strip()
        if in_doc:
            in_doc = not (st.endswith('"""') or st.endswith("'''"))
            continue
        if st.startswith(('"""', "'''")):
            in_doc = not (len(st) > 3 and (st.endswith('"""') or st.endswith("'''")))
            continue
        if not st or _COMMENT.match(ln):
            continue
        out.append(ln.split("  # ")[0].rstrip())
    return out


def compact_code(av_stack: Path | None, compact: dict, ctx: dict) -> dict[str, str]:
    """Laya state code: {view: text}. Each entry is a function (docstrings / comments / blank lines removed), optionally
    narrowed to lines matching `grep` regexes; `{fault_mode}` / `{fault_helper}` are filled from the scenario, so the
    fault-injection view shows only the injected mode's code. Entries that do not resolve are dropped."""
    mode = str(ctx.get("fault_mode") or "")
    helper = (compact.get("fault_helpers") or {}).get(mode, "")
    out = {}
    for view, regs in (compact.get("code_views") or {}).items():
        parts = []
        for reg in regs:
            fn = str(reg["function"]).format(fault_mode=mode, fault_helper=helper)
            if fn.endswith("."):
                continue
            path = (av_stack / reg["file"]) if av_stack else None
            rng = NM._function_ranges(path).get(fn) if path and path.is_file() else None
            if not rng:
                continue
            lines = _strip(path.read_text(encoding="utf-8").splitlines()[rng[0] - 1:rng[1]])
            pats = [re.compile(str(g).format(fault_mode=mode, fault_helper=helper).replace("\\\\", "\\"))
                    for g in reg.get("grep") or []]
            if pats:
                lines = [lines[0]] + [ln for ln in lines[1:] if any(pt.search(ln) for pt in pats)]
            if lines:
                ind = min(len(ln) - len(ln.lstrip()) for ln in lines)
                parts.append("\n".join(ln[ind:] for ln in lines))
        if parts:
            out[view] = "\n".join(parts)[:int(compact.get("max_chars_per_view", 2500))]
    return out


def state_text(state: dict) -> str:
    """Compact text rendering of a state ({scenario, code}) for short-context backends: '[scenario]' key: value lines,
    then one '[code.<view>]' section per view (what the questions' focus names)."""
    sc = state.get("scenario") or {}
    lines = ["[scenario]"]
    for k, v in sc.items():
        lines.append(f"{k}: {json.dumps(v, separators=(', ', ': ')) if isinstance(v, (dict, list)) else v}")
    for view, txt in (state.get("code") or {}).items():
        lines += [f"[code.{view}]", txt]
    return "\n".join(lines)


FORBIDDEN_STATE_KEYS = ("expected_from_code", "expected", "matches_code", "final_safe_recovery", "recovery_outcome",
                        "gp_mean", "gp_std", "outcome", "failure")


def build_state(code: dict[str, str], ctx: dict, profile: dict | None, tick_s: float = 0.05) -> dict:
    """Judge state: code excerpts + scenario inputs + supervisor profile settings. Unit conversions (rad -> deg) only."""
    params = dict(ctx.get("fault_params") or {})
    deg = {f"{k[:-4]}_deg": round(math.degrees(float(v)), 3) for k, v in params.items()
           if k.endswith("_rad") and isinstance(v, (int, float))}
    scen = {"fault_mode": ctx.get("fault_mode"), "fault_params": params, "fault_params_deg": deg,
            "onset_event": ctx.get("onset_event"), "planner": ctx.get("planner", "pdm"),
            "controller": ctx.get("controller", "supervised"), "control_tick_s": tick_s,
            "supervisor_profile": {"name": ctx.get("supervisor"), **(profile or {})}}
    bad = _keys(scen) & set(FORBIDDEN_STATE_KEYS)
    if bad:
        raise ValueError(f"judge state must not carry {sorted(bad)} (independence rule)")
    return {"scenario": scen, "code": code}


def state_views(cfg: dict, backend: str) -> list[str]:
    """Code sections the backend's state carries (judge_questions uses it to fall back focus to `code`)."""
    view = ((cfg.get("backends") or {}).get(backend) or {}).get("state_view", "full")
    return list(((cfg.get("state") or {}).get(view) or {}).get("code_views") or {})


def make_state(cfg: dict, backend: str, av_stack: Path | None, ctx: dict, profile: dict | None) -> Any:
    """State for one scenario on this backend. full (Jev, 64k context): {scenario, code: whole functions} JSON.
    compact (Laya, 1-2k tokens): text with the supervisor checks and only the injected fault mode's lines."""
    st = cfg.get("state") or {}
    view = ((cfg.get("backends") or {}).get(backend) or {}).get("state_view", "full")
    tick = float(st.get("tick_s", 0.05))
    if view == "compact":
        return state_text(build_state(compact_code(av_stack, st.get("compact") or {}, ctx), ctx, profile, tick))
    full = st.get("full") or {}
    return build_state(code_views(av_stack, full.get("code_views") or {}, int(full.get("max_chars_per_view", 9000))),
                       ctx, profile, tick)


def _keys(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set(map(str, obj)) | set().union(*(_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj)) if obj else set()
    return set()


# ---------------------------------------------------------------- clients
class JevError(RuntimeError):
    pass


JudgeError = JevError


class SystemOneClient:
    """POST {base_url}/v1/systemone (laya-serve or TypeSafe Jev: same wire protocol). An API key is optional for
    Laya (only when laya-serve sets LAYA_API_KEY) and required for Jev; it is read from the env and never logged.
    llm_calls.jsonl gets tag, backend, model, usage, latency and error only (role 'B_judge')."""

    def __init__(self, cfg: dict, log_path: Path | None = None, backend: str = "jev") -> None:
        self.backend = backend
        base = os.environ.get(str(cfg.get("base_url_env") or "")) if cfg.get("base_url_env") else None
        base = str(base or cfg.get("base_url") or "https://api.typesafe.ai").rstrip("/")
        self.endpoint = str(cfg.get("endpoint") or (base + str(cfg.get("path", "/v1/systemone"))))
        self.model = str(cfg.get("model", "jev-1.14.0"))
        self.timeout = float(cfg.get("timeout_s", 30))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.long_context = bool(cfg.get("long_context", False))
        self.key_required = bool(cfg.get("key_required", backend == "jev"))
        self.controls = {k: cfg[k] for k in ("max_len", "head_max_len") if cfg.get(k) is not None and backend == "laya"}
        env = cfg.get("api_key_env")
        self._key = os.environ.get(str(env)) if env else None
        self.log_path = log_path
        self.stub = False

    @property
    def has_key(self) -> bool:
        return bool(self._key) or not self.key_required

    def reachable(self) -> bool:
        """Laya: GET {base}/health answers. Jev: a key is set (no free probe endpoint)."""
        if self.backend != "laya":
            return self.has_key
        url = self.endpoint.rsplit("/v1/", 1)[0] + "/health"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

    def system_one(self, state: Any, questions: dict, tag: str = "B_judge") -> dict:
        if self.key_required and not self._key:
            raise JevError("TYPESAFE_API_KEY not set")
        body = json.dumps({"model": self.model, "state": state, "questions": questions, **self.controls}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        if self.long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                out["latency_s"] = round(time.time() - t0, 3)
                self._log(tag, out.get("model"), out.get("usage"), time.time() - t0, None)
                return out
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                err = f"HTTP {e.code}: {detail}"
                self._log(tag, self.model, None, time.time() - t0, err)
                if e.code not in (429, 500, 502, 503, 529) or attempt == self.max_retries:
                    break
                wait = float(e.headers.get("retry-after") or 2 ** attempt)
                time.sleep(min(wait, 30.0))
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
                err = f"{type(e).__name__}: {e}"[:300]
                self._log(tag, self.model, None, time.time() - t0, err)
                if attempt == self.max_retries:
                    break
                time.sleep(2 ** attempt)
        raise JevError(err or f"{self.backend} request failed")

    def _log(self, tag, model, usage, latency, error) -> None:
        if self.log_path is None:
            return
        u = usage or {}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(ts=time.time(), tag=tag, backend=self.backend, model=model, stub=self.stub,
                                     usage={"prompt_tokens": u.get("input_tokens", 0),
                                            "completion_tokens": u.get("output_tokens", 0)},
                                     latency_s=round(latency, 3), error=error)) + "\n")


JevClient = SystemOneClient


class LayaInProcess(SystemOneClient):
    """Laya without a server: laya.Router in this Python process (needs `pip install laya` + torch here; the
    project venv does not ship them, so run.py only uses this when the import works). Same answer schema."""

    def __init__(self, cfg: dict, log_path: Path | None = None) -> None:
        super().__init__(cfg, log_path, backend="laya")
        import laya  # noqa: F401  (ImportError -> caller falls back)
        from laya.router import Router
        self._router = Router(device=cfg.get("device") or None)

    def reachable(self) -> bool:
        return True

    def system_one(self, state: Any, questions: dict, tag: str = "B_judge") -> dict:
        t0 = time.time()
        try:
            out = self._router.predict(state, questions, model=self.model, **self.controls)
        except Exception as e:  # noqa: BLE001  (library errors -> JudgeError, logged)
            self._log(tag, self.model, None, time.time() - t0, f"{type(e).__name__}: {e}"[:300])
            raise JevError(f"laya in-process: {type(e).__name__}: {e}"[:300]) from e
        out = dict(out)
        out.setdefault("model", f"laya/{self.model}")
        out["latency_s"] = round(time.time() - t0, 3)
        self._log(tag, out.get("model"), out.get("usage"), time.time() - t0, None)
        return out


class NemotronSystemOne(SystemOneClient):
    """Nemotron (Token Factory chat) used as a System One judge: same state + 25 typed questions, the model returns a
    probability per option for every question (json_schema enforced), answered in the /v1/systemone answer shape.
    No rationale is requested or kept, so the regression sees the same kind of input as with Laya / Jev."""

    SYSTEM = ("You are a typed decision engine. You read one STATE (scenario inputs and source code) and answer every "
              "QUESTION independently by giving a probability for each of its options. Probabilities of one question sum "
              "to 1. Do not explain. Answer only from the state; if the code shown does not decide it, put the mass on "
              "'unknown' when that option exists.")

    def __init__(self, cfg: dict, llm_cfg: dict, log_path: Path | None = None, client: Any = None) -> None:
        super().__init__(cfg, log_path, backend="nemotron")
        from engine.llm import TokenFactoryClient, role_settings
        self.settings = role_settings(llm_cfg, str(cfg.get("role", "B")))
        self.model = str(self.settings.get("model"))
        self.client = client or TokenFactoryClient.from_config(llm_cfg, log_path=log_path)
        self.temperature = cfg.get("temperature", self.settings.get("temperature"))
        self.max_tokens = int(cfg.get("max_tokens", self.settings.get("max_tokens", 16000)))

    @property
    def has_key(self) -> bool:
        return bool(getattr(self.client, "has_key", True))

    def reachable(self) -> bool:
        return self.has_key

    @staticmethod
    def _opts(jq: dict) -> list[str]:
        return list(jq["criteria"]) if jq["type"] == "choice" else ["true", "false"]

    def messages(self, state: Any, questions: dict) -> list[dict]:
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        qs = []
        for name, jq in questions.items():
            if jq["type"] == "choice":
                opts = "\n".join(f"    - {k}: {v}" for k, v in jq["criteria"].items())
            else:
                opts = "    - true: yes\n    - false: no"
            qs.append(f"- {name}: {jq['instructions']}\n  options:\n{opts}")
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": f"STATE:\n{st}\n\nQUESTIONS:\n" + "\n".join(qs) +
                 "\n\nReturn JSON {question_name: {option: probability}} for every question."}]

    def schema(self, questions: dict) -> dict:
        props = {n: {"type": "object", "additionalProperties": False, "required": self._opts(jq),
                     "properties": {o: {"type": "number", "minimum": 0, "maximum": 1} for o in self._opts(jq)}}
                 for n, jq in questions.items()}
        return {"type": "object", "additionalProperties": False, "required": list(questions), "properties": props}

    def system_one(self, state: Any, questions: dict, tag: str = "B_judge") -> dict:
        from engine.llm import LLMError, extract_json
        t0 = time.time()
        try:
            res = self.client.chat(self.model, self.messages(state, questions), temperature=self.temperature,
                                   top_p=self.settings.get("top_p"), max_tokens=self.max_tokens,
                                   thinking=self.settings.get("thinking"),
                                   chat_template_kwargs=self.settings.get("chat_template_kwargs_extra"),
                                   response_format={"type": "json_schema",
                                                    "json_schema": {"name": "system_one", "schema": self.schema(questions)}},
                                   tag=tag)
            obj = extract_json(res.content) or {}
        except (LLMError, ValueError) as e:
            raise JevError(f"nemotron: {type(e).__name__}: {e}"[:300]) from e
        answers = {}
        for name, jq in questions.items():
            raw = {str(k): max(0.0, float(v)) for k, v in (obj.get(name) or {}).items() if isinstance(v, (int, float))}
            opts = self._opts(jq)
            tot = sum(raw.get(o, 0.0) for o in opts)
            if tot <= 0:
                continue
            pr = {o: raw.get(o, 0.0) / tot for o in opts}
            if jq["type"] == "noul":
                answers[name] = {"type": "noul", "noul": round(pr["true"], 4)}
            else:
                best = max(pr, key=pr.get)
                k = len(pr)
                answers[name] = {"type": "choice", "choice": best, "probabilities": {o: round(v, 4) for o, v in pr.items()},
                                 "confidence": round((k * pr[best] - 1) / (k - 1), 4) if k > 1 else 1.0}
        u = res.usage or {}
        return {"model": res.model, "answers": answers, "latency_s": round(time.time() - t0, 3),
                "usage": {"input_tokens": int(u.get("prompt_tokens") or 0), "output_tokens": int(u.get("completion_tokens") or 0)}}


def make_client(backend: str, cfg: dict, mode: str, bank: list[dict], meta: dict, seed: int = 0,
                log_path: Path | None = None, llm_cfg: dict | None = None) -> tuple[SystemOneClient, str]:
    """(client, backend label). --mock -> stub. laya: laya-serve if /health answers, else in-process laya if
    importable, else the stub (SYNTHETIC, said so in the label). jev: live with TYPESAFE_API_KEY, else the stub."""
    bcfg = (cfg.get("backends") or {})[backend]
    if mode == "mock":
        return StubJev(bcfg, bank, meta, seed=seed, log_path=log_path), "stub (mock)"
    if backend == "nemotron":
        if llm_cfg is None:
            import yaml
            from engine.config import PIPELINE_DIR
            llm_cfg = yaml.safe_load((PIPELINE_DIR / "llm_models.yaml").read_text(encoding="utf-8"))
        c = NemotronSystemOne(bcfg, llm_cfg, log_path)
        if c.has_key:
            return c, f"nemotron System One {c.model}"
        return StubJev(bcfg, bank, meta, seed=seed, log_path=log_path), "stub (no NEBIUS_API_KEY: SYNTHETIC answers)"
    live = SystemOneClient(bcfg, log_path, backend=backend)
    if backend == "laya":
        if live.reachable():
            return live, f"laya-serve {live.endpoint} model={live.model}"
        if bcfg.get("inprocess_fallback", True):
            try:
                c = LayaInProcess(bcfg, log_path)
                return c, f"laya in-process model={c.model}"
            except Exception:  # noqa: BLE001  (no laya / torch in this venv)
                pass
        return (StubJev(bcfg, bank, meta, seed=seed, log_path=log_path),
                f"stub (laya-serve not reachable at {live.endpoint}: SYNTHETIC answers)")
    if live.has_key:
        return live, f"jev live {live.model}"
    return StubJev(bcfg, bank, meta, seed=seed, log_path=log_path), "stub (no TYPESAFE_API_KEY: SYNTHETIC answers)"


# SYNTHETIC per-variant accuracy of the stub (Jev is weaker in Korean and with long indirect instructions)
_STUB_ACC = {"en_notes": 0.82, "en_notes_rev": 0.78, "ko": 0.6, "focus_fault": 0.75, "focus_supervisor": 0.75,
             "focus_switch_rev": 0.7, "focus_mpc_rev": 0.65, "noul_en": 0.85, "noul_criteria": 0.8, "noul_ko": 0.65,
             "choice_yn": 0.8}


class StubJev(SystemOneClient):
    """Deterministic SYNTHETIC Jev for --mock and runs without TYPESAFE_API_KEY. It reads the scenario from the
    state and looks the answer up in the bank's answer key (the stub's oracle; the key is never in the state), then
    emits a roughly calibrated distribution: top mass ~ U(0.45, 0.95), right with the variant's accuracy."""

    def __init__(self, cfg: dict, bank: list[dict], meta: dict[str, dict], seed: int = 0,
                 log_path: Path | None = None) -> None:
        super().__init__(cfg, log_path, backend="stub")
        self.bank = {q["id"]: q for q in bank}
        self.meta = meta
        self.seed = int(seed)
        self.stub = True
        self.model = "stub/judge"

    @property
    def has_key(self) -> bool:
        return True

    def reachable(self) -> bool:
        return True

    _OUTCOME_BY_PATH = {   # SYNTHETIC: outcome questions follow the detection path the code key gives (J1)
        "immediate_check": {"J2": ["under_0_5s"], "J5": ["low"], "J6": ["within_limit"]},
        "deviation_accumulation": {"J2": ["0_5_to_2s", "2_to_5s"], "J5": ["moderate", "high"], "J6": ["rate_limited"]},
        "no_dedicated_check": {"J2": ["over_5s_or_never"], "J5": ["low", "moderate"], "J6": ["within_limit"]}}

    def _outcome_oracle(self, q: dict, ctx: dict) -> list[str] | None:
        j1 = next((b for b in self.bank.values() if b.get("title") == "detection_path"), None)
        path = (NM.expected_answers(j1, ctx) or [None])[0] if j1 else None
        return self._OUTCOME_BY_PATH.get(path, {}).get(q["id"])

    def system_one(self, state: Any, questions: dict, tag: str = "B_judge") -> dict:
        scen = state.get("scenario") if isinstance(state, dict) else _scenario_from_text(str(state))
        ctx = {"fault_mode": scen.get("fault_mode"), "fault_params": scen.get("fault_params") or {},
               "supervisor": (scen.get("supervisor_profile") or {}).get("name")}
        answers = {}
        for name, jq in questions.items():
            m = self.meta[name]
            q = self.bank[m["qid"]]
            opts = m["options"]
            h = hashlib.sha256(f"{self.seed}|{tag}|{name}|{json.dumps(scen, sort_keys=True)}".encode()).digest()
            rng = random.Random(int.from_bytes(h[:8], "big"))
            exp = NM.expected_answers(q, ctx) if q.get("label_source") != "outcome" else self._outcome_oracle(q, ctx)
            if exp:
                pick = rng.choice(exp) if rng.random() < _STUB_ACC.get(m["variant"], 0.7) else \
                    rng.choice([o for o in opts if o not in exp and o != "unknown"] or opts)
            else:
                pick = rng.choice([o for o in opts if o != "unknown"] or opts)
            top = rng.uniform(0.45, 0.95)
            rest = [o for o in opts if o != pick]
            w = [rng.random() + 0.05 for _ in rest]
            probs = {pick: top, **{o: (1 - top) * wi / sum(w) for o, wi in zip(rest, w)}}
            if jq["type"] == "noul":
                answers[name] = {"type": "noul", "noul": round(probs.get("yes", 0.0) / max(1e-9, probs.get("yes", 0) + probs.get("no", 0)), 4)}
            else:
                k = len(jq["criteria"])
                back = m.get("keys") or {}
                pr = {w: round(probs[back.get(w, w)], 4) for w in jq["criteria"]}
                best = max(pr, key=pr.get)
                answers[name] = {"type": "choice", "choice": best, "probabilities": pr,
                                 "confidence": round((k * pr[best] - 1) / (k - 1), 4) if k > 1 else 1.0}
        n_tok = len(json.dumps({"state": state, "questions": questions})) // 4
        out = {"model": self.model, "answers": answers, "usage": {"input_tokens": n_tok, "output_tokens": len(answers) * 3}}
        self._log(tag, self.model, out["usage"], 0.0, None)
        return out


def _scenario_from_text(text: str) -> dict:
    """Inverse of state_text() for the '[scenario]' block (the stub reads the compact text state)."""
    sc, on = {}, False
    for ln in text.splitlines():
        if ln.startswith("["):
            on = ln == "[scenario]"
            continue
        if on and ": " in ln:
            k, v = ln.split(": ", 1)
            try:
                sc[k] = json.loads(v)
            except json.JSONDecodeError:
                sc[k] = v
    return sc


StubJudge = StubJev


# ---------------------------------------------------------------- answers -> per-question judge probabilities
def parse_answers(resp: dict, meta: dict[str, dict]) -> dict[str, list[dict]]:
    """{qid: [{judge, probs: {option: p} over the bank's options, confidence}]}; a missing / mistyped answer is
    skipped (Jev cannot return a type error, but a request can fail)."""
    out: dict[str, list[dict]] = {}
    for name, m in meta.items():
        a = (resp.get("answers") or {}).get(name)
        if not a:
            continue
        opts = m["options"]
        if a.get("type") == "noul" and a.get("noul") is not None:
            p = min(1.0, max(0.0, float(a["noul"])))
            probs = {o: 0.0 for o in opts}
            probs["yes"], probs["no"] = p, 1.0 - p
            conf = abs(2 * p - 1)
        elif a.get("type") == "choice":
            back = m.get("keys") or {}
            raw = {back.get(str(k), str(k)): float(v) for k, v in (a.get("probabilities") or {}).items()}
            tot = sum(raw.get(o, 0.0) for o in opts) or 1.0
            probs = {o: raw.get(o, 0.0) / tot for o in opts}
            conf = a.get("confidence")
        else:
            continue
        out.setdefault(m["qid"], []).append({"judge": m["variant"], "probs": probs, "confidence": conf})
    return out


def judge_confidence(probs: dict[str, float]) -> float:
    k = len(probs)
    pm = max(probs.values()) if probs else 0.0
    return (k * pm - 1) / (k - 1) if k > 1 else 1.0


# ---------------------------------------------------------------- regression (Quantitative LLM Judges)
def _x_row(judges: list[dict], variants: list[str], opts: list[str], clip: float) -> np.ndarray:
    by = {j["judge"]: j["probs"] for j in judges}
    uni = 1.0 / len(opts)
    row = []
    for v in variants:
        p = by.get(v)
        row += [math.log(max(clip, (p or {}).get(o, uni) if p is not None else uni)) for o in opts]
    return np.array(row, dtype=float)


def mean_probs(judges: list[dict], opts: list[str]) -> dict[str, float]:
    if not judges:
        return {o: 1.0 / len(opts) for o in opts}
    m = {o: sum(j["probs"].get(o, 0.0) for j in judges) / len(judges) for o in opts}
    s = sum(m.values()) or 1.0
    return {o: v / s for o, v in m.items()}


def brier(probs: list[dict[str, float]], truth: list[str]) -> float | None:
    if not probs:
        return None
    return float(np.mean([sum((p.get(o, 0.0) - (1.0 if o == t else 0.0)) ** 2 for o in p) for p, t in zip(probs, truth)]))


def ece(probs: list[dict[str, float]], truth: list[str], bins: int = 10) -> float | None:
    if not probs:
        return None
    conf = np.array([max(p.values()) for p in probs])
    hit = np.array([max(p, key=p.get) == t for p, t in zip(probs, truth)], dtype=float)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if sel.any():
            out += sel.mean() * abs(conf[sel].mean() - hit[sel].mean())
    return float(out)


class JudgeRegression:
    """Per original question: probabilities over its options from its judges. Fit on states whose answer key is a
    single option (lists = 'code allows several' and None = 'code does not decide' are not labels)."""

    def __init__(self, cfg: dict, variants_of: dict[str, list[str]], options_of: dict[str, list[str]], seed: int = 0,
                 report_only: set[str] | None = None):
        self.cfg = cfg
        self.variants_of = variants_of
        self.options_of = options_of
        self.seed = seed
        self.models: dict[str, Any] = {}
        self.report: dict[str, dict] = {}
        self.report_only = set(report_only or ())

    def _model(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=float(self.cfg.get("C", 0.5)), max_iter=2000)

    def _predict(self, model, x: np.ndarray, judges: list[dict], opts: list[str]) -> dict[str, float]:
        mean = mean_probs(judges, opts)
        if model is None:
            return mean
        pr = model.predict_proba(x.reshape(1, -1))[0]
        reg = {o: 0.0 for o in opts}
        for cls, p in zip(model.classes_, pr):
            reg[str(cls)] = float(p)
        a = float(self.cfg.get("blend_mean", 0.1))
        out = {o: (1 - a) * reg[o] + a * mean[o] for o in opts}
        s = sum(out.values()) or 1.0
        return {o: v / s for o, v in out.items()}

    def fit(self, rows: list[tuple[dict[str, list[dict]], dict[str, list[str] | None]]]) -> "JudgeRegression":
        """rows = [(parsed judges {qid: [...]}, answer key {qid: allowed options | None}) per state]."""
        from sklearn.model_selection import StratifiedKFold, KFold
        clip = float(self.cfg.get("prob_clip", 1e-3))
        for qid, variants in self.variants_of.items():
            opts = self.options_of[qid]
            lab = [(j[qid], key[qid][0]) for j, key in rows if j.get(qid) and key.get(qid) and len(key[qid]) == 1]
            allowed = [(j[qid], key[qid]) for j, key in rows if j.get(qid) and key.get(qid)]
            X = np.array([_x_row(js, variants, opts, clip) for js, _ in lab]) if lab else np.zeros((0, 0))
            y = np.array([t for _, t in lab])
            classes = sorted(set(y.tolist()))
            rep = {"n_states": sum(1 for j, _ in rows if j.get(qid)), "n_labels": len(lab), "classes": classes,
                   "n_judges": len(variants)}
            use_reg = len(lab) >= int(self.cfg.get("min_labels", 8)) and len(classes) >= int(self.cfg.get("min_classes", 2))
            use_reg = use_reg and qid not in self.report_only
            rep["method"] = "logistic_l2" if use_reg else ("report_only" if qid in self.report_only else "mean")
            self.models[qid] = self._model().fit(X, y) if use_reg else None
            # out-of-fold predictions for every reported number
            oof_reg, oof_mean = [], [mean_probs(js, opts) for js, _ in lab]
            if use_reg:
                counts = {c: int((y == c).sum()) for c in classes}
                k = min(int(self.cfg.get("cv_folds", 5)), min(counts.values()))
                splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed) if k >= 2 else \
                    KFold(n_splits=min(5, len(lab)), shuffle=True, random_state=self.seed)
                oof = [None] * len(lab)
                for tr, te in splitter.split(X, y):
                    m = self._model().fit(X[tr], y[tr]) if len(set(y[tr].tolist())) >= 2 else None
                    for i in te:
                        oof[i] = self._predict(m, X[i], lab[i][0], opts)
                oof_reg = oof
                rep["cv_folds"] = splitter.get_n_splits()
            truth = [t for _, t in lab]
            for name, pp in (("regression", oof_reg if use_reg else oof_mean), ("judge_mean", oof_mean)):
                rep[name] = {"accuracy": (float(np.mean([max(p, key=p.get) == t for p, t in zip(pp, truth)])) if pp else None),
                             "brier": brier(pp, truth), "ece": ece(pp, truth, int(self.cfg.get("ece_bins", 10)))}
            # single judges (out of the box, no fitting): does any one judge alone beat the combination?
            per_judge = {}
            for v in variants:
                pj = [next((j["probs"] for j in js if j["judge"] == v), None) for js, _ in lab]
                ok = [(p, t) for p, t in zip(pj, truth) if p]
                if ok:
                    per_judge[v] = round(float(np.mean([max(p, key=p.get) == t for p, t in ok])), 3)
            rep["single_judge_accuracy"] = per_judge
            # agreement with the key where the code allows several answers too (argmax in the allowed set)
            rep["agreement_allowed_set"] = (float(np.mean([max(p, key=p.get) in a for p, (_, a) in
                                                          zip([self.predict(qid, js) for js, _ in allowed], allowed)]))
                                            if allowed else None)
            rep["agreement_allowed_set_note"] = "in-sample (fit includes these states); out-of-fold numbers above"
            self.report[qid] = rep
        return self

    def predict(self, qid: str, judges: list[dict]) -> dict[str, float]:
        opts = self.options_of[qid]
        model = self.models.get(qid)
        x = _x_row(judges, self.variants_of[qid], opts, float(self.cfg.get("prob_clip", 1e-3)))
        return self._predict(model, x, judges, opts)


def aggregate(parsed: dict[str, list[dict]], reg: JudgeRegression, key: dict[str, list[str] | None],
              low_conf: float = 0.5, sources: dict[str, str] | None = None) -> dict[str, dict]:
    """Per question, the B-cache row shape the rest of the loop reads (answer = argmax, matches_code / hallucinated
    against the answer key) plus probs, confidence, low_confidence and the raw judge vectors."""
    out = {}
    for qid, opts in reg.options_of.items():
        js = parsed.get(qid) or []
        if not js:                                  # not asked (applies_to) -> no row; GP columns fall back to uniform
            continue
        p = reg.predict(qid, js)
        ans = max(p, key=p.get)
        exp = key.get(qid)
        conf = judge_confidence(p)
        matches = None if exp is None or not js else ans in exp
        out[qid] = {"answer": ans, "probs": {o: round(v, 4) for o, v in p.items()}, "confidence": round(conf, 4),
                    "low_confidence": conf < low_conf, "method": (reg.report.get(qid) or {}).get("method", "mean"),
                    "n_judges": len(js), "judges": js, "expected_from_code": exp, "matches_code": matches,
                    "hallucinated": bool(matches is False and ans != "unknown" and conf >= low_conf),
                    "scope": "per_scenario", "label_source": (sources or {}).get(qid, "code")}
    return out


def jev_features(answers: dict[str, dict] | None, questions: list[dict], drop: set[str] | None = None
                 ) -> tuple[list[float], list[str]]:
    """Continuous GP inputs from the gp_feature questions, kept small for a GP fit on tens of runs:
    gp_encoding ordinal (ordered outcome buckets J2 J5 J6) -> one column, expected bucket index / (n - 1);
    otherwise the regressed probability of every option except 'unknown'. Columns in `drop` (constant over the
    campaign's states) are left out. Missing answers -> uniform."""
    x, names = [], []
    for q in questions:
        if q.get("scope", "per_scenario") != "per_scenario" or not q.get("gp_feature", True):
            continue
        opts = [str(o) for o in q["answer"]]
        probs = ((answers or {}).get(q["id"]) or {}).get("probs") or {}
        p = {o: float(probs.get(o, 1.0 / len(opts))) for o in opts}
        if q.get("gp_encoding") == "ordinal":
            cols = [(f"{q['id']}=ordinal", sum(i * p[o] for i, o in enumerate(opts)) / max(1, len(opts) - 1))]
        else:
            nm = q.get("gp_names") or {}
            cols = [(nm.get(o) or f"{q['id']}=p({o})", p[o]) for o in opts if o != "unknown"]
        for n, v in cols:
            if drop and n in drop:
                continue
            names.append(n); x.append(v)
    return x, names


def constant_columns(entries: list[dict], questions: list[dict], tol: float = 1e-6) -> set[str]:
    rows = [jev_features(e.get("answers"), questions) for e in entries]
    if not rows:
        return set()
    names = rows[0][1]
    return {n for j, n in enumerate(names) if max(r[0][j] for r in rows) - min(r[0][j] for r in rows) <= tol}


def summary(entries: list[dict], reg: JudgeRegression | None) -> dict:
    """Campaign-level replacement of b_hallucination_summary: agreement with the code answer key (code questions) and
    with the run outcome (outcome questions, rows whose run finished), calibration (out-of-fold), low-confidence share
    and the regression report per question."""
    per_q: dict[str, dict] = {}
    for ent in entries:
        for qid, row in (ent.get("answers") or {}).items():
            d = per_q.setdefault(qid, {"asked": 0, "checked": 0, "matched": 0, "hallucinated": 0, "low_confidence": 0,
                                       "answers": {}, "label_source": row.get("label_source", "code")})
            d["asked"] += 1
            d["answers"][row["answer"]] = d["answers"].get(row["answer"], 0) + 1
            d["low_confidence"] += bool(row.get("low_confidence"))
            if row.get("matches_code") is not None:
                d["checked"] += 1
                d["matched"] += bool(row["matches_code"])
                d["hallucinated"] += bool(row.get("hallucinated")) and d["label_source"] == "code"
    def tot(src):
        return {k: sum(d[k] for d in per_q.values() if src is None or d["label_source"] == src)
                for k in ("checked", "matched", "hallucinated", "asked", "low_confidence")}
    tc, to, ta = tot("code"), tot("outcome"), tot(None)
    for qid, d in per_q.items():
        d["regression"] = (reg.report.get(qid) if reg else None)
    return {"judge": "system_one", "b_hallucination_rate": tc["hallucinated"] / tc["checked"] if tc["checked"] else None,
            "agreement_with_code": tc["matched"] / tc["checked"] if tc["checked"] else None,
            "agreement_with_outcome": to["matched"] / to["checked"] if to["checked"] else None,
            "low_confidence_rate": ta["low_confidence"] / ta["asked"] if ta["asked"] else None,
            "hallucination_definition": "code question answered outside the code's allowed set with confidence >= "
                                        "low_confidence (System One judges give no citations, so there is no citation check)",
            "per_question": per_q, **{f"n_{k}": v for k, v in ta.items()},
            "n_outcome_checked": to["checked"]}


def propose_scenarios(entries: list[tuple[dict, dict]], pcfg: dict, space: dict, make) -> list[dict]:
    """B scenario proposals (laya-judge): for cached judge states whose J1 says the fault can only be caught by
    accumulated deviation or by no check at all (p >= min_prob), propose a riskier neighbour of a scenario already in
    that state: bias / intermittent magnitude just under one rate-check tick (the code's 90 deg/s x 0.05 s) and a
    higher speed (more lateral error per second before tracking_divergence trips). `entries` = [(b_cache row, seed
    candidate)]; `make(seed, params, speed, reason)` builds the candidate. Code-grounded reason in expected_mechanism.
    Returns at most per_round, one per fault mode, highest probability first."""
    if not pcfg.get("enabled", True):
        return []
    rate_tick = math.radians(90.0) * 0.05
    env = space["environment"]["max_speed_kmh"]["range"]
    scored = []
    for row, seed in entries:
        pr = ((row.get("answers") or {}).get("J1") or {}).get("probs") or {}
        blind, acc = float(pr.get("no_dedicated_check", 0.0)), float(pr.get("deviation_accumulation", 0.0))
        if blind + acc >= float(pcfg.get("min_prob", 0.4)):
            scored.append((blind + acc, blind, acc, seed))
    out, modes = [], set()
    for tot, blind, acc, seed in sorted(scored, key=lambda t: (-t[0], t[3]["scenario_id"])):
        mode = seed["fault"]["mode"]
        if mode in modes:
            continue
        params = dict(seed["fault"].get("params") or {})
        if mode in ("bias", "intermittent_bias") and "magnitude_rad" in params:
            params["magnitude_rad"] = round(float(pcfg.get("below_rate_check_frac", 0.9)) * rate_tick, 4)
        speed = min(env[1], float(seed["environment"]["max_speed_kmh"]) + float(pcfg.get("speed_step_kmh", 5.0)))
        why = (f"B J1: p(no check)={blind:.2f}, p(accumulated deviation)={acc:.2f}. Code: MPCSupervisor.inspect only "
               f"catches a one-tick jump > {rate_tick:.4f} rad (90 deg/s); below that only tracking_divergence "
               f"(0.6 m / 0.35 rad held 0.15 s) or nothing trips, so a larger sub-threshold offset at higher speed "
               f"builds more lateral error before the switch.")
        c = make(seed, params, speed, why)
        if c is not None:
            out.append(c); modes.add(mode)
        if len(out) >= int(pcfg.get("per_round", 2)):
            break
    return out
