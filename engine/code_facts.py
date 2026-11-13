"""Deterministic 'code facts' for the 05b judges (laya-judge): --judge-context digest and short.

A short judge has trouble finding the 5 lines that decide a question inside ~1k tokens of code repeated for 25 judges.
This module reads the av_stack source with `ast` (nothing is hard-coded that the code or the profile defines) and
renders, per topic, a few factual lines: thresholds and check order of the supervisor profile, how the injected fault
mode rewrites MPCOutput, the order of calls in one supervised tick, the LQR takeover rate limit and what it is anchored
on, and every assignment of the controller latch.

Independence rule (docs/JUDGES.md): a digest or short state states what the code does, never what follows for this
scenario. It never names an answer option, never compares a fault parameter with a threshold, and never sees a run
record (the builder takes only the scenario inputs ctx and the av_stack path). tests/test_code_facts.py checks this.

Topics (judges.yaml state.digest.topics maps each question id to a list of them):
  supervisor  check order + conditions of MPCSupervisor.inspect with the profile's values substituted, MPC rate limit
  fault       injection window, the mode's rewrite of MPCOutput (fields replaced), the scenario's parameters
  switch      order of calls in SupervisedLateral.step and the switch-tick bookkeeping
  takeover    LQR takeover rate limit (value, per-tick size) and the steering it is clipped around
  latch       every `self.active = ...` assignment with the method and guard it sits in
  wiring      what the injector wraps and what the LQR call receives
"""
from __future__ import annotations

import ast
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SUP = "integration/supervised_lateral.py"
FAULT = "integration/fault_injection.py"
MPC = "mpc_controller/mpc.py"
TOPICS = ("supervisor", "fault", "switch", "takeover", "latch", "wiring")
# scenario parameter (fault catalog) -> FaultConfig field (engine/fault_args.py to_cli does the same mapping)
PARAM_TO_FIELD = {"magnitude_rad": "magnitude", "rate_rad_s": "magnitude", "delay_ticks": "magnitude",
                  "side": "magnitude (sign)", "magnitude": "level", "ramp_s": "ramp_s", "on_s": "on_s",
                  "off_s": "off_s", "duration_s": "duration_s"}
HELPERS = {"bias": "_shift", "drift": "_shift", "intermittent_bias": "_intermittent", "saturate": "_saturate",
           "output_delay": "_delayed"}


# ---------------------------------------------------------------- ast helpers
@lru_cache(maxsize=16)
def _tree(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)


def _method(cls: ast.ClassDef | None, name: str) -> ast.FunctionDef | None:
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _src(node: ast.AST) -> str:
    return ast.unparse(node).replace("math.radians", "radians").replace("np.clip", "clip")


def _const(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def dataclass_defaults(tree: ast.Module, cls_name: str) -> dict[str, Any]:
    """{field: default} of a dataclass, in declaration order (math.radians(x) defaults are evaluated)."""
    cls = _class(tree, cls_name)
    out: dict[str, Any] = {}
    for st in (cls.body if cls else []):
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            v = _const(st.value)
            if v is None and isinstance(st.value, ast.Call) and _src(st.value.func) == "radians" and st.value.args:
                a = _const(st.value.args[0])
                v = math.radians(a) if isinstance(a, (int, float)) else None
            out[st.target.id] = v
    return out


def supervisor_profile(av: Path, name: str | None) -> dict[str, Any]:
    """SupervisorConfig values of SUPERVISOR_PROFILES[name] (positional / keyword args mapped onto the fields)."""
    tree = _tree(str(av / SUP))
    fields = dataclass_defaults(tree, "SupervisorConfig")
    for st in tree.body:
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SUPERVISOR_PROFILES" for t in st.targets) \
                and isinstance(st.value, ast.Dict):
            for k, v in zip(st.value.keys, st.value.values):
                if _const(k) == (name or "tight") and isinstance(v, ast.Call):
                    prof = dict(fields)
                    for f, a in zip(list(fields), v.args):
                        prof[f] = _const(a)
                    for kw in v.keywords:
                        prof[kw.arg] = _const(kw.value)
                    return prof
    return dict(fields)


def _fmt(v: Any) -> str:
    return "None" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


# ---------------------------------------------------------------- topics
def facts_supervisor(av: Path, ctx: dict, tick_s: float) -> list[str]:
    name = ctx.get("supervisor") or "tight"
    prof = supervisor_profile(av, name)
    tree = _tree(str(av / SUP))
    insp = _method(_class(tree, "MPCSupervisor"), "inspect")
    out = [f"Supervisor profile '{name}': " + ", ".join(f"{k}={_fmt(v)}" for k, v in prof.items()) + "."]
    if insp is None:
        return out + ["MPCSupervisor.inspect: (source not available)"]
    subst = {f"cfg.{k}": _fmt(v) for k, v in prof.items()}

    def sub(s: str) -> str:
        for k, v in sorted(subst.items(), key=lambda kv: -len(kv[0])):
            s = s.replace(k, v)
        return s

    checks, guards = [], []
    none_fields = [k for k, v in prof.items() if v is None]

    def clean(cond: str) -> str:
        # `cfg.x is not None and ...` guards: drop when the profile sets x, mark the check disabled when it does not
        cond = re.sub(r"\b[0-9.e+-]+ is not None and ", "", cond)
        return cond

    def walk(body: list[ast.stmt], guard: list[str], raw_guard: str = "") -> None:
        for st in body:
            if isinstance(st, ast.If):
                ret = next((s for s in st.body if isinstance(s, ast.Return) and isinstance(_const(s.value), str)), None)
                raw = raw_guard + " " + _src(st.test)
                cond = clean(sub(_src(st.test)))
                if ret is not None:
                    off = [f for f in none_fields if f"cfg.{f} is not None" in raw]
                    checks.append((str(_const(ret.value)), " and ".join(guard + [cond]) if not off else
                                   f"(check disabled in this profile: {off[0]} is None)"))
                walk([s for s in st.body if not isinstance(s, ast.Return)], guard + [cond], raw)
                walk(st.orelse, guard + [f"not ({cond})"], raw_guard)
            elif isinstance(st, ast.Assign) and guard == [] and any(
                    isinstance(t, ast.Name) and t.id == "bad_tracking" for t in st.targets):
                guards.append(f"bad_tracking = {sub(_src(st.value))}")
    walk(insp.body, [])
    out.append("MPCSupervisor.inspect runs once per MPC tick and returns the first matching reason, in this order:")
    out += [f"  {i + 1}. '{r}' when {c}" for i, (r, c) in enumerate(checks)]
    out += [f"  where {g}; output.state[0] = lateral error e_y [m], output.state[2] = heading error e_psi [rad]"
            for g in guards]
    # hold-timer semantics: assignments to self.tracking_bad_since and the test they sit under
    for st in ast.walk(insp):
        if isinstance(st, ast.If):
            for s in st.body:
                if isinstance(s, ast.Assign) and any(_src(t) == "self.tracking_bad_since" for t in s.targets):
                    out.append(f"  self.tracking_bad_since = {_src(s.value)} when {_src(st.test)} (checked every tick)")
    # what check 8 compares against: the previous tick's inspected output (assignments in source order)
    for st in insp.body:
        if isinstance(st, ast.Assign) and "last_steer" in _src(st.targets[0]):
            out.append(f"  {_src(st.targets[0])} = {_src(st.value)}  (before check 8, every tick that reaches it)")
    rate = prof.get("max_steer_rate_dps")
    if rate is not None:
        out.append(f"  unit conversion: {_fmt(rate)} deg/s x control tick {tick_s:g} s = "
                   f"{math.radians(rate) * tick_s:.4f} rad of steer change per tick.")
    mpc = dataclass_defaults(_tree(str(av / MPC)), "MPCConfig") if (av / MPC).is_file() else {}
    if mpc.get("steer_rate_lim_dps") is not None:
        out.append(f"MPCConfig.steer_rate_lim_dps = {_fmt(mpc['steer_rate_lim_dps'])} (the MPC's own constraint on "
                   f"its own command, computed before MPCFaultInjector.apply).")
    return out


def _fault_lambda(av: Path, mode: str) -> str | None:
    apply_ = _method(_class(_tree(str(av / FAULT)), "MPCFaultInjector"), "apply")
    for n in ast.walk(apply_) if apply_ else []:
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if _const(k) == mode:
                    return _src(v)
    return None


def _replaced_fields(fn: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and _src(n.func) == "replace":
            out += [kw.arg for kw in n.keywords if kw.arg and kw.arg not in out]
    return out


def _body_lines(fn: ast.FunctionDef) -> list[str]:
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(_const(s.value), str))
            and not (isinstance(s, ast.If) and len(s.body) == 1 and isinstance(s.body[0], ast.Raise))]  # input guards
    return [ln for s in body for ln in _src(s).splitlines()]


def facts_fault(av: Path, ctx: dict, tick_s: float) -> list[str]:
    mode = str(ctx.get("fault_mode") or "none")
    tree = _tree(str(av / FAULT))
    cls = _class(tree, "MPCFaultInjector")
    fc = _class(tree, "FaultConfig")
    act = _method(fc, "active")
    out = [f"Injected fault mode: '{mode}'. Scenario parameters: " +
           (", ".join(f"{k}={v} (FaultConfig.{PARAM_TO_FIELD.get(k, k)})" for k, v in (ctx.get("fault_params") or {}).items())
            or "none") + "."]
    if act is not None:
        ret = next((s for s in act.body if isinstance(s, ast.Return)), None)
        if ret is not None:
            out.append(f"Fault window: FaultConfig.active(t) = {_src(ret.value)}; outside it apply() returns the MPC output unchanged.")
    lam = _fault_lambda(av, mode)
    out.append(f"MPCFaultInjector.apply for '{mode}': {lam}" if lam else f"MPCFaultInjector.apply has no entry for '{mode}'.")
    helper = _method(cls, HELPERS.get(mode, "")) if mode in HELPERS else None
    fields: list[str] = []
    if helper is not None:
        out.append(f"  {helper.name}: " + " | ".join(_body_lines(helper)))
        fields = _replaced_fields(helper)
        if helper.name == "_saturate":
            out.append("  max_steer = the vehicle's road-wheel steering limit (geometry['max_steer_rad'], read from the "
                       "vehicle at runtime).")
    elif lam:
        fields = _replaced_fields(ast.parse(lam))
    if mode == "stale":
        out.append("  self.previous = the last output apply() returned while the fault was inactive: the whole earlier "
                   "MPCOutput object is returned again, its seq and stamp_s included.")
        fields = []
    if fields:
        out.append("  MPCOutput fields replaced: " + ", ".join(fields) + "; all other fields (seq, stamp_s, ok, ...) "
                   "keep the current MPC values." if "ok" not in fields else
                   "  MPCOutput fields replaced: " + ", ".join(fields) + ".")
    return out


def _step(av: Path) -> ast.FunctionDef | None:
    return _method(_class(_tree(str(av / SUP)), "SupervisedLateral"), "step")


def facts_switch(av: Path, ctx: dict, tick_s: float) -> list[str]:
    step = _step(av)
    if step is None:
        return ["SupervisedLateral.step: (source not available)"]
    out = ["SupervisedLateral.step, one control tick, statements in source order:"]
    keep = ("output", "self.reason", "result", "self.active", "self.last")

    def visit(body: list[ast.stmt], depth: int, note: str = "") -> None:
        pad = "  " * depth
        for st in body:
            if isinstance(st, ast.If):
                out.append(f"{pad}if {_src(st.test)}:")
                visit(st.body, depth + 1)
                if st.orelse:
                    out.append(f"{pad}else:")
                    visit(st.orelse, depth + 1)
            elif isinstance(st, ast.Try):
                visit(st.body, depth)
                for h in st.handlers:
                    visit(h.body, depth, " (on exception)")
            elif isinstance(st, ast.Assign) and _src(st.targets[0]) in keep:
                tgt, val = _src(st.targets[0]), _src(st.value)
                if tgt == "result" and val.startswith("LateralResult("):
                    val = "LateralResult(output.steer, ...)"
                out.append(f"{pad}{tgt} = {val}{note}")
    visit(step.body, 1)
    out.append("  The `if self.active == 'lqr'` block runs in the same tick, after the MPC block.")
    return out


def facts_takeover(av: Path, ctx: dict, tick_s: float) -> list[str]:
    cls = _class(_tree(str(av / SUP)), "SupervisedLateral")
    init, step = _method(cls, "__init__"), _step(av)
    rate = None
    for n in ast.walk(init) if init else []:
        if isinstance(n, ast.Assign) and _src(n.targets[0]) == "self.rate_dps":
            rate = _const(n.value)
    out = [f"LQR takeover: self.rate_dps = {_fmt(rate)} deg/s (set in __init__)."]
    for n in ast.walk(step) if step else []:
        if isinstance(n, ast.Assign) and _src(n.targets[0]) in ("limit", "steer"):
            out.append(f"  {_src(n.targets[0])} = {_src(n.value)}")
    if isinstance(rate, (int, float)):
        out.append(f"  unit conversion: {_fmt(rate)} deg/s x {tick_s:g} s = {math.radians(rate) * tick_s:.4f} rad per tick.")
    out.append("  self.last is assigned at the end of every tick from the command sent to the vehicle; LQR's own "
               "controller state starts from the vehicle state passed to step().")
    return out


def facts_latch(av: Path, ctx: dict, tick_s: float) -> list[str]:
    cls = _class(_tree(str(av / SUP)), "SupervisedLateral")
    out = ["Every assignment to SupervisedLateral.active in the source:"]
    for fn in (cls.body if cls else []):
        if not isinstance(fn, ast.FunctionDef):
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for p in ast.walk(fn):
            for c in ast.iter_child_nodes(p):
                parents[c] = p
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and any(_src(t) == "self.active" for t in n.targets):
                g, p = [], parents.get(n)
                while p is not None and p is not fn:
                    if isinstance(p, ast.If):
                        g.append(_src(p.test))
                    p = parents.get(p)
                out.append(f"  {fn.name}(): self.active = {_src(n.value)}" + (f"  guarded by: {' / '.join(reversed(g))}" if g else ""))
    out.append("  The MPC block of step() runs only under `if self.active == 'mpc'`.")
    return out


def facts_wiring(av: Path, ctx: dict, tick_s: float) -> list[str]:
    step = _step(av)
    out = ["Data flow in SupervisedLateral.step:"]
    for n in ast.walk(step) if step else []:
        if isinstance(n, ast.Assign) and _src(n.targets[0]) == "output":
            out.append(f"  output = {_src(n.value)}")
        if isinstance(n, ast.Assign) and _src(n.targets[0]) == "result" and "step_controller" in _src(n.value):
            out.append(f"  result = {_src(n.value)}  (arguments are step()'s own inputs t, x, y, yaw, vx, vy, yaw_rate)")
    out.append("  The injector object is self.injector = MPCFaultInjector(...); it is called only on the line above.")
    return out


BUILDERS = {"supervisor": facts_supervisor, "fault": facts_fault, "switch": facts_switch, "takeover": facts_takeover,
            "latch": facts_latch, "wiring": facts_wiring}


def digest(av: Path, ctx: dict, topics: list[str], tick_s: float = 0.05) -> str:
    """Factual digest for one question: one '[facts.<topic>]' section per topic. `ctx` = judge scenario inputs only
    (fault_mode, fault_params, onset_event, supervisor); never a run record."""
    av = Path(av)
    parts = []
    for t in topics:
        if t not in BUILDERS:
            raise ValueError(f"unknown code-facts topic {t!r}")
        try:
            lines = BUILDERS[t](av, ctx, tick_s)
        except (OSError, SyntaxError) as e:
            lines = [f"(source not available: {type(e).__name__})"]
        parts.append(f"[facts.{t}]\n" + "\n".join(lines))
    return "\n".join(parts)


# ---------------------------------------------------------------- short context (--judge-context short)
# One compact state shared by every question in the request (same as full; unlike digest). Values come from the
# av_stack source and the supervisor profile (ast); the text only restates code facts, never a conclusion for this
# scenario. Section labels [code.supervisor] / [code.fault_injection] / [code.switch] match the compact state's
# views so focus variants keep pointing at a real section.
_TINY = re.compile(r"\s*\+\s*1e-[0-9]+")
_SELF = re.compile(r"\b(?:self|output|cfg|config)\.")


def inspect_checks(av: Path, ctx: dict) -> list[tuple[str, str]]:
    """Ordered (reason, condition) of MPCSupervisor.inspect. A disabled profile check still appears, with a
    '(check disabled in this profile: <field> is None)' condition so every fault mode sees every check."""
    name = ctx.get("supervisor") or "tight"
    prof = supervisor_profile(av, name)
    insp = _method(_class(_tree(str(av / SUP)), "MPCSupervisor"), "inspect")
    if insp is None:
        return []
    subst = {f"cfg.{k}": _fmt(v) for k, v in prof.items()}

    def sub(s: str) -> str:
        for k, v in sorted(subst.items(), key=lambda kv: -len(kv[0])):
            s = s.replace(k, v)
        return s

    def clean(cond: str) -> str:
        return re.sub(r"\b[0-9.e+-]+ is not None and ", "", cond)

    none_fields = [k for k, v in prof.items() if v is None]
    checks: list[tuple[str, str]] = []

    def walk(body: list[ast.stmt], guard: list[str], raw_guard: str = "") -> None:
        for st in body:
            if isinstance(st, ast.If):
                ret = next((s for s in st.body if isinstance(s, ast.Return) and isinstance(_const(s.value), str)), None)
                raw = raw_guard + " " + _src(st.test)
                cond = clean(sub(_src(st.test)))
                if ret is not None:
                    off = [f for f in none_fields if f"cfg.{f} is not None" in raw]
                    checks.append((str(_const(ret.value)), " and ".join(guard + [cond]) if not off else
                                   f"(check disabled in this profile: {off[0]} is None)"))
                walk([s for s in st.body if not isinstance(s, ast.Return)], guard + [cond], raw)
                walk(st.orelse, guard + [f"not ({cond})"], raw_guard)
    walk(insp.body, [])
    return checks


def _deadline_ms(av: Path, tick_s: float) -> float:
    """deadline_ms passed to MPCSupervisor in SupervisedLateral.__init__ (dt * 1000 -> tick_s * 1000)."""
    init = _method(_class(_tree(str(av / SUP)), "SupervisedLateral"), "__init__")
    for n in ast.walk(init) if init else []:
        if isinstance(n, ast.Call) and _src(n.func).endswith("MPCSupervisor"):
            for kw in n.keywords:
                if kw.arg == "deadline_ms":
                    expr = _src(kw.value)
                    if expr in ("dt * 1000", "1000 * dt"):
                        return tick_s * 1000.0
                    v = _const(kw.value)
                    if isinstance(v, (int, float)):
                        return float(v)
    insp_init = _method(_class(_tree(str(av / SUP)), "MPCSupervisor"), "__init__")
    if insp_init:
        for a, d in zip(reversed(insp_init.args.args), reversed(insp_init.args.defaults)):
            if a.arg == "deadline_ms":
                v = _const(d)
                if isinstance(v, (int, float)):
                    return float(v)
    return tick_s * 1000.0


def _literal_compare(av: Path, reason: str) -> float | None:
    """Numeric threshold compared in the inspect branch that returns `reason` (e.g. stale_output's 0.1 s)."""
    insp = _method(_class(_tree(str(av / SUP)), "MPCSupervisor"), "inspect")
    for st in ast.walk(insp) if insp else []:
        if isinstance(st, ast.If):
            ret = next((s for s in st.body if isinstance(s, ast.Return) and _const(s.value) == reason), None)
            if ret is None:
                continue
            nums = [n.value for n in ast.walk(st.test) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
            # skip tiny numeric tolerances (1e-6, 1e-9)
            nums = [x for x in nums if abs(x) >= 1e-4]
            if nums:
                return float(nums[-1])
    return None


def _tidy(s: str) -> str:
    s = _TINY.sub("", s)
    s = _SELF.sub("", s)
    s = s.replace("math.", "").replace("np.", "")
    s = re.sub(r"\bfloat\(([^)]+)\)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _short_check_phrase(reason: str, cond: str, prof: dict, tick_s: float, deadline_ms: float,
                        stale_s: float | None) -> str:
    if "check disabled" in cond:
        return f"{reason}: disabled in this profile ({cond[1:-1] if cond.startswith('(') else cond})"
    rate = prof.get("max_steer_rate_dps")
    if reason == "solver_failure":
        return f"{reason}: not ok"
    if reason == "missing_steer":
        return f"{reason}: steer None"
    if reason == "nonfinite_output":
        return f"{reason}: any of steer, s, kappa, stamp, state not finite"
    if reason == "steer_out_of_range":
        return f"{reason}: |steer| > max_steer"
    if reason == "deadline_miss":
        return f"{reason}: elapsed > {deadline_ms:g} ms"
    if reason == "stale_output":
        return f"{reason}: |t - stamp| > {stale_s if stale_s is not None else 0.1:g}"
    if reason == "stale_sequence":
        return f"{reason}: seq <= last"
    if reason == "steer_rate_exceeded" and isinstance(rate, (int, float)):
        per = math.radians(rate) * tick_s
        return (f"{reason}: |steer - last_steer| / dt > {rate:g} deg/s "
                f"(= {rate * tick_s:g} deg / {per:.4f} rad per tick)")
    if reason == "tracking_divergence":
        ey, ep, hold = prof.get("max_lateral_error_m"), prof.get("max_heading_error_rad"), prof.get("divergence_hold_s")
        return (f"{reason}: |e_y| > {_fmt(ey)} m or |e_psi| > {_fmt(ep)} rad held {_fmt(hold)} s; "
                "one good tick resets the timer")
    return f"{reason}: {_tidy(cond)}"


def _unchanged_fields(fields: list[str]) -> str:
    """Compact 'seq, stamp, ok, state unchanged', omitting any field the helper/lambda replaces."""
    replaced = {("stamp" if f == "stamp_s" else f) for f in fields}
    keep = [f for f in ("seq", "stamp", "ok", "state") if f not in replaced]
    return ("; " + ", ".join(keep) + " unchanged") if keep else ""


def _shift_offset(lam: str | None) -> str:
    """Second argument of `_shift(value, ...)` in the mode's apply lambda, tidied."""
    if not lam:
        return "magnitude"
    m = re.search(r"_shift\([^,]+,\s*(.+)\)\s*$", _tidy(lam))
    return m.group(1) if m else "magnitude"


def _short_fault_line(av: Path, ctx: dict) -> str:
    mode = str(ctx.get("fault_mode") or "none")
    lam = _fault_lambda(av, mode)
    tree = _tree(str(av / FAULT))
    cls = _class(tree, "MPCFaultInjector")
    helper = _method(cls, HELPERS.get(mode, "")) if mode in HELPERS else None
    fields: list[str] = []
    bits: list[str] = []
    if helper is not None:
        fields = _replaced_fields(helper)
        if helper.name == "_shift":
            bits.append(f"steer = mpc_steer + {_shift_offset(lam)}")
        elif helper.name == "_intermittent":
            bits.append("on_s / off_s cycle from at_s: during on, steer = mpc_steer + magnitude; during off, unchanged")
            fields = _replaced_fields(_method(cls, "_shift") or helper)
        elif helper.name == "_saturate":
            bits.append("steer goes to sign(magnitude) * level * max_steer, blending over ramp_s (0 = step)")
        elif helper.name == "_delayed":
            bits.append("steer, state, s, kappa taken from the command computed magnitude ticks ago; stamp and seq stay the current values")
            fields = []  # the sentence already names replaced vs kept fields
        else:
            bits.append(_tidy(lam or " ; ".join(_tidy(ln) for ln in _body_lines(helper))))
    elif lam:
        fields = _replaced_fields(ast.parse(lam))
        if mode == "solver_fail":
            bits.append("ok = False, steer = None")
        elif mode == "stale":
            bits.append("return the last output from outside the window (whole MPCOutput, seq and stamp included); if none, stamp = t-1")
            fields = []
        else:
            bits.append(_tidy(lam))
    else:
        bits.append(f"apply has no entry for {mode}")
    if fields:
        bits[-1] = bits[-1] + _unchanged_fields(fields)
    act = _method(_class(tree, "FaultConfig"), "active")
    window = "inside the fault window"
    if act is not None:
        ret = next((s for s in act.body if isinstance(s, ast.Return)), None)
        if ret is not None:
            window = "inside at_s <= t < at_s + duration_s:"
    return "fault code: " + window + " " + "; ".join(bits)


def _deg(v: float) -> str:
    return f"{math.degrees(float(v)):.1f} deg"


def _short_scenario(ctx: dict, tick_s: float) -> str:
    mode = str(ctx.get("fault_mode") or "none")
    p = dict(ctx.get("fault_params") or {})
    onset = ctx.get("onset_event") or ""
    parts = [mode]
    if "magnitude_rad" in p and isinstance(p["magnitude_rad"], (int, float)):
        parts.append(f"+{p['magnitude_rad']:g} rad ({_deg(p['magnitude_rad'])})")
    elif "rate_rad_s" in p and isinstance(p["rate_rad_s"], (int, float)):
        parts.append(f"+{p['rate_rad_s']:g} rad/s")
    if "on_s" in p or "off_s" in p:
        parts.append(f"on {_fmt(p.get('on_s'))} s / off {_fmt(p.get('off_s'))} s")
    if mode == "saturate":
        extra = []
        if "side" in p:
            extra.append(f"side {p['side']}")
        if "magnitude" in p:
            extra.append(f"magnitude {p['magnitude']:g}")
        if "level" in p:
            extra.append(f"level {p['level']:g}")
        if "ramp_s" in p:
            extra.append(f"ramp {_fmt(p['ramp_s'])} s")
        if "duration_s" in p:
            extra.append(f"duration {_fmt(p['duration_s'])} s")
        parts.extend(extra)
    if "delay_ticks" in p:
        parts.append(f"{p['delay_ticks']} ticks")
    where = "on MPC output" if mode in ("solver_fail", "stale") else "on MPC steer"
    line = "fault: " + " ".join(str(x) for x in parts) + f" {where}"
    if onset:
        line += f", onset {onset}"
    line += f", tick {tick_s:g} s"
    return line


def _takeover_rate_dps(av: Path) -> float | None:
    init = _method(_class(_tree(str(av / SUP)), "SupervisedLateral"), "__init__")
    for n in ast.walk(init) if init else []:
        if isinstance(n, ast.Assign) and _src(n.targets[0]) == "self.rate_dps":
            v = _const(n.value)
            return float(v) if isinstance(v, (int, float)) else None
    return None


def _active_assignments(av: Path) -> list[tuple[str, str]]:
    """(method, rhs) for every `self.active = ...` in SupervisedLateral."""
    cls = _class(_tree(str(av / SUP)), "SupervisedLateral")
    out = []
    for fn in (cls.body if cls else []):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and any(_src(t) == "self.active" for t in n.targets):
                out.append((fn.name, _src(n.value)))
    return out


def _short_switch(av: Path, tick_s: float) -> str:
    rate = _takeover_rate_dps(av)
    rate_s = f"{rate:g} deg/s * dt" if rate is not None else "rate_dps * dt"
    assigns = _active_assignments(av)
    back = [f"{m}: active = {v}" for m, v in assigns if m != "__init__" and v in ("'mpc'", '"mpc"')]
    latch = "never assigned back to mpc" if not back else "also assigned back to mpc in " + ", ".join(back)
    extras = [v for m, v in assigns if m != "__init__" and v not in ("'lqr'", '"lqr"', "'mpc'", '"mpc"')]
    extra = ("; if LQR invalid: active = " + str(extras[0]).strip(chr(39)+chr(34))) if extras else ""
    return ("tick loop: if active == mpc: out = mpc(); out = inject(out); reason = inspect(out); "
            f"if reason: active = lqr ({latch}{extra}); "
            f"if active == lqr: steer = clip(lqr_steer, last_applied ± {rate_s})")


def short(av: Path, ctx: dict, tick_s: float = 0.05) -> str:
    """One short factual state for every question of a scenario (~1k characters). `ctx` is judge scenario inputs
    only (fault_mode, fault_params, onset_event, supervisor); never a run record."""
    av = Path(av)
    name = ctx.get("supervisor") or "tight"
    prof = supervisor_profile(av, name)
    try:
        checks = inspect_checks(av, ctx)
        deadline = _deadline_ms(av, tick_s)
        stale = _literal_compare(av, "stale_output")
        phrases = [_short_check_phrase(r, c, prof, tick_s, deadline, stale) for r, c in checks]
        # first seven checks on one line, rate + tracking on their own (they carry numbers)
        head, tail = [], []
        for ph in phrases:
            name_r = ph.split(":", 1)[0]
            if name_r in ("steer_rate_exceeded", "tracking_divergence"):
                tail.append(ph)
            else:
                head.append(ph)
        sup_lines = ["supervisor checks per MPC tick, first match switches to LQR:"]
        if head:
            sup_lines.append(" " + " | ".join(head))
        sup_lines += [" " + t for t in tail]
        fault = _short_fault_line(av, ctx)
        switch = _short_switch(av, tick_s)
        scen = _short_scenario(ctx, tick_s)
    except (OSError, SyntaxError) as e:
        return f"[scenario]\n(source not available: {type(e).__name__})"
    return "\n".join(["[scenario]", scen, "[code.supervisor]", *sup_lines,
                      "[code.fault_injection]", fault, "[code.switch]", switch])
