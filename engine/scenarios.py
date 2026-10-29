"""04 scenario handling: candidate schema, space filling, local perturbation, feasibility gate, GP features.

The scenario space is sim-eval's scenario_space_av_stack.yaml (faults F-01..F-07, environment,
fault_timing). Candidates look like:
  {scenario_id, seed_scenario_id, source, route_id, environment: {max_speed_kmh},
   fault: {fault_id, mode, params}, onset_event, onset_offset_s, hypothesis_ids: ['S1'],
   evidence_ids, expected_mechanism, missing_information, tags}
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

from . import judgment as J

WHOLE_RUN_S = 500.0  # same as sim-eval scenario_space.py: faults without duration stay on until the route ends


def cand_id(c: dict) -> str:
    key = json.dumps({k: c.get(k) for k in ("route_id", "environment", "fault", "onset_event", "onset_offset_s")},
                     sort_keys=True, default=str)
    return "SC-" + hashlib.sha1(key.encode()).hexdigest()[:10]


def faults_by_id(space: dict) -> dict[str, dict]:
    return {f["id"]: f for f in space.get("faults", [])}


FRICTION = "tire_friction_scale"


def friction_axis(space: dict) -> dict | None:
    """environment.tire_friction_scale when run_benchmark.py can apply it (has a `cli:` flag; integration3), else None."""
    spec = (space.get("environment") or {}).get(FRICTION)
    return spec if isinstance(spec, dict) and spec.get("cli") and "range" in spec else None


def friction_of(c: dict) -> float:
    """Tire friction scale of a candidate (1.0 = CARLA default tyres, also for candidates made before integration3)."""
    v = (c.get("environment") or {}).get(FRICTION)
    return 1.0 if v is None else float(v)


def _param_unit(spec: dict, value: Any) -> float | None:
    if "range" in spec:
        lo, hi = spec["range"]
        return None if not isinstance(value, (int, float)) else (float(value) - lo) / (hi - lo)
    if "ranges" in spec:
        lo, hi = spec["ranges"][0][0], spec["ranges"][-1][1]
        return (float(value) - lo) / (hi - lo)
    vals = spec.get("values") or []
    if value in vals:
        return vals.index(value) / max(1, len(vals) - 1)
    return None


def _param_from_unit(spec: dict, u: float) -> Any:
    u = min(max(u, 0.0), 1.0)
    if "range" in spec:
        lo, hi = spec["range"]
        return lo + u * (hi - lo)
    if "ranges" in spec:
        lo, hi = spec["ranges"][0][0], spec["ranges"][-1][1]
        return lo + u * (hi - lo)
    vals = spec["values"]
    return vals[min(len(vals) - 1, int(round(u * (len(vals) - 1))))]


def make_candidate(space: dict, *, route_id: str, speed: float, fault_id: str, params: dict, onset_event: str,
                   offset: float, source: str, objective: str = "S1", round_index: int = 0,
                   seed_scenario_id: str | None = None, friction: float | None = None, **extra) -> dict:
    f = faults_by_id(space)[fault_id]
    env = {"max_speed_kmh": round(float(speed), 2)}
    if friction_axis(space) is not None:   # integration3: --tire-friction-scale (default 1.0 = CARLA tyres)
        env[FRICTION] = round(float(1.0 if friction is None else friction), 3)
    c = dict(route_id=route_id, environment=env,
             fault={"fault_id": fault_id, "mode": f["injector_mode"],
                    "params": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()}},
             onset_event=onset_event, onset_offset_s=round(float(offset), 3), hypothesis_ids=[objective],
             source=source, round=round_index, evidence_ids=extra.pop("evidence_ids", []),
             expected_mechanism=extra.pop("expected_mechanism", None),
             missing_information=extra.pop("missing_information", None), tags=[], **extra)
    c["scenario_id"] = cand_id(c)
    c["seed_scenario_id"] = seed_scenario_id or c["scenario_id"]
    return c


def unit_dims(space: dict, fault_pool: list[str]) -> int:
    return 6  # route, speed, fault, param-u, onset_event, offset (+ second param folded into same u)


def space_filling(space: dict, routes: list[str], fault_pool: list[str], n: int, rng: random.Random,
                  objective: str = "S1", round_index: int = 0, source: str = "space_filling") -> list[dict]:
    """Latin-hypercube style: every continuous dim is stratified into n bins; categoricals cycle."""
    if n <= 0:
        return []
    env = space["environment"]["max_speed_kmh"]["range"]
    events = space["fault_timing"]["onset_event"]["values"]
    off = space["fault_timing"]["offset_s"]["range"]
    fax = friction_axis(space)
    strata = [rng.sample(range(n), n) for _ in range(4 if fax else 3)]
    out = []
    for i in range(n):
        u_speed, u_par, u_off = ((strata[d][i] + rng.random()) / n for d in range(3))
        friction = None
        if fax:
            flo, fhi = fax["range"]
            friction = flo + (strata[3][i] + rng.random()) / n * (fhi - flo)
        fid = fault_pool[(i + rng.randrange(len(fault_pool))) % len(fault_pool)]
        f = faults_by_id(space)[fid]
        params = {k: _param_from_unit(spec, u_par if j == 0 else rng.random()) for j, (k, spec) in enumerate((f.get("params") or {}).items())}
        out.append(make_candidate(space, route_id=routes[i % len(routes)] if i < len(routes) else rng.choice(routes),
                                  speed=env[0] + u_speed * (env[1] - env[0]), fault_id=fid, params=params,
                                  onset_event=rng.choice(events), offset=off[0] + u_off * (off[1] - off[0]),
                                  source=source, objective=objective, round_index=round_index, friction=friction))
    return out


def perturb(space: dict, seed: dict, n: int, rng: random.Random, scale: float = 0.15) -> list[dict]:
    """04 local perturbation: route, fault type and onset_event stay fixed (05b answers reusable)."""
    env = space["environment"]["max_speed_kmh"]["range"]
    off = space["fault_timing"]["offset_s"]["range"]
    f = faults_by_id(space)[seed["fault"]["fault_id"]]
    out = []
    for _ in range(n):
        sp = min(env[1], max(env[0], seed["environment"]["max_speed_kmh"] + rng.gauss(0, scale) * (env[1] - env[0])))
        of = min(off[1], max(off[0], seed["onset_offset_s"] + rng.gauss(0, scale) * (off[1] - off[0])))
        params = {}
        for k, spec in (f.get("params") or {}).items():
            u = _param_unit(spec, seed["fault"]["params"].get(k, spec.get("default")))
            params[k] = _param_from_unit(spec, (u if u is not None else 0.5) + rng.gauss(0, scale))
        fr = None
        fax = friction_axis(space)
        if fax:
            flo, fhi = fax["range"]
            fr = min(fhi, max(flo, friction_of(seed) + rng.gauss(0, scale) * (fhi - flo)))
        out.append(make_candidate(space, route_id=seed["route_id"], speed=sp, fault_id=f["id"], params=params,
                                  onset_event=seed["onset_event"], offset=of, source="perturbation",
                                  objective=(seed.get("hypothesis_ids") or ["S1"])[0], round_index=seed.get("round", 0),
                                  seed_scenario_id=seed["seed_scenario_id"], friction=fr))
    return out


def validate_against_space(space: dict, c: dict, routes: list[str]) -> list[str]:
    """G-02: every parameter maps to a run_benchmark.py argument inside the declared range."""
    problems = []
    if c.get("route_id") not in routes:
        problems.append(f"unknown route {c.get('route_id')}")
    fid = (c.get("fault") or {}).get("fault_id")
    f = faults_by_id(space).get(fid)
    if f is None:
        return problems + [f"unknown fault {fid}"]
    if not f.get("implemented"):
        problems.append(f"{fid} not implemented in injector")
    lo, hi = space["environment"]["max_speed_kmh"]["range"]
    sp = (c.get("environment") or {}).get("max_speed_kmh")
    if not isinstance(sp, (int, float)) or not lo <= sp <= hi:
        problems.append(f"max_speed_kmh {sp} outside [{lo},{hi}]")
    fr = (c.get("environment") or {}).get(FRICTION)
    if fr is not None:
        fax = friction_axis(space)
        if fax is None:
            problems.append("tire_friction_scale given but run_benchmark has no flag for it in this space")
        elif not isinstance(fr, (int, float)) or not fax["range"][0] - 1e-9 <= fr <= fax["range"][1] + 1e-9:
            problems.append(f"tire_friction_scale {fr} outside {fax['range']}")
    if c.get("onset_event") not in space["fault_timing"]["onset_event"]["values"]:
        problems.append(f"onset_event {c.get('onset_event')}")
    olo, ohi = space["fault_timing"]["offset_s"]["range"]
    if not isinstance(c.get("onset_offset_s"), (int, float)) or not olo <= c["onset_offset_s"] <= ohi:
        problems.append("onset_offset_s outside range")
    params = (c.get("fault") or {}).get("params") or {}
    for k, spec in (f.get("params") or {}).items():
        v = params.get(k)
        if v is None:
            problems.append(f"missing param {k}")
        elif "range" in spec and not (spec["range"][0] - 1e-9 <= v <= spec["range"][1] + 1e-9):
            problems.append(f"{k}={v} outside {spec['range']}")
        elif "values" in spec and v not in spec["values"]:
            problems.append(f"{k}={v} not in {spec['values']}")
    extra = set(params) - set(f.get("params") or {})
    if extra:
        problems.append(f"unexpected params {sorted(extra)}")
    return problems


def gate(space: dict, c: dict, routes: list[str], route_events: dict | None, lat_acc_threshold: float | None,
         min_window_s: float | None = None) -> tuple[bool, list[str], list[str]]:
    """G-01..G-03 (04 feasibility_expansion.yaml), plus V-03 once the route's baseline is known. Returns (ok, tags, problems)."""
    problems = validate_against_space(space, c, routes)
    events = None
    if route_events is not None:
        events = [k for k in ("straight", "curve_entry", "curve_apex", "curve_exit") if route_events.get(k) is not None]
    nominal = None
    if route_events is not None and c.get("onset_event") in (route_events.get("kappa_at") or {}):
        k = route_events["kappa_at"][c["onset_event"]]
        v = min(c["environment"]["max_speed_kmh"] / 3.6, (route_events.get("speed_at") or {}).get(c["onset_event"], 1e9))
        nominal = v * v * abs(k)
    c["nominal_lateral_acc_mps2"] = nominal
    # integration3: P-02 limit 0.5*mu*g scales with the tyre friction of the candidate (mu = 0.9 * scale)
    lat_th = None if lat_acc_threshold is None else lat_acc_threshold * friction_of(c)
    ok, tags = J.feasibility_gate(dict(c, nominal_lateral_acc_mps2=nominal),
                                  events if events is not None else [c.get("onset_event")],
                                  lat_th, carla_mapping_ok=not problems)
    if events is None and ok:
        tags.append("G-01_pending_baseline")
    # BUG-028: an onset closer to the route end than the minimum evaluation window can never be graded (09 V-03);
    # e.g. 28035 has one curve at 46.5-49 s and ends at 52 s, so its curve_apex / curve_exit onsets are dead
    t_end = (route_events or {}).get("t_end")
    onset = onset_time(c, route_events) if ok and route_events and t_end is not None and min_window_s else None
    if onset is not None and t_end - onset < min_window_s:
        ok = False
        tags.append("V-03_reject")
        problems.append(f"onset {onset:.1f} s leaves {t_end - onset:.1f} s before the route ends ({min_window_s:.0f} s needed)")
    return ok, tags, problems


def dedup(cands: list[dict], space: dict, routes: list[str], fault_pool: list[str], min_distance: float) -> list[dict]:
    kept: list[dict] = []
    for c in cands:
        x = numeric_features(c, space, routes, fault_pool)[0]
        if all(math.dist(x, numeric_features(k, space, routes, fault_pool)[0]) >= min_distance for k in kept):
            kept.append(c)
    return kept


def numeric_features(c: dict, space: dict, routes: list[str], fault_pool: list[str]) -> tuple[list[float], list[str]]:
    """X-3 'numeric only' block: route one-hot, speed, fault one-hot, fault params (unit), onset one-hot, offset."""
    names, x = [], []
    for r in routes:
        names.append(f"route={r}"); x.append(1.0 if c.get("route_id") == r else 0.0)
    lo, hi = space["environment"]["max_speed_kmh"]["range"]
    names.append("max_speed_u"); x.append((c["environment"]["max_speed_kmh"] - lo) / (hi - lo))
    fax = friction_axis(space)
    if fax:
        flo, fhi = fax["range"]
        names.append("tire_friction_u"); x.append((friction_of(c) - flo) / (fhi - flo))
    for fid in fault_pool:
        names.append(f"fault={fid}"); x.append(1.0 if c["fault"]["fault_id"] == fid else 0.0)
    f = faults_by_id(space)[c["fault"]["fault_id"]]
    pu = [_param_unit(spec, c["fault"]["params"].get(k)) for k, spec in (f.get("params") or {}).items()]
    pu = [u for u in pu if u is not None]
    names += ["param_u0", "param_u1"]; x += [pu[0] if pu else 0.5, pu[1] if len(pu) > 1 else 0.5]
    for ev in space["fault_timing"]["onset_event"]["values"]:
        names.append(f"onset={ev}"); x.append(1.0 if c.get("onset_event") == ev else 0.0)
    olo, ohi = space["fault_timing"]["offset_s"]["range"]
    names.append("offset_u"); x.append((c["onset_offset_s"] - olo) / (ohi - olo))
    return x, names


def onset_time(c: dict, route_events: dict) -> float | None:
    t = route_events.get(c["onset_event"])
    return None if t is None else max(0.5, float(t) + float(c["onset_offset_s"]))


def fault_cli_local(space: dict, c: dict, onset_s: float) -> list[str]:
    """Same mapping as sim-eval scenario_space.py to_cli (used only when that file is not importable)."""
    f = faults_by_id(space)[c["fault"]["fault_id"]]
    p = c["fault"]["params"]
    if "delay_ticks" in p:
        mag = p["delay_ticks"]
    elif "side" in p:
        mag = 1.0 if p["side"] == "positive" else -1.0
    else:
        mag = p.get("magnitude_rad", p.get("rate_rad_s"))
    args = ["--mpc-fault", f["injector_mode"], "--fault-start", "%.2f" % onset_s,
            "--fault-duration", "%.2f" % p.get("duration_s", WHOLE_RUN_S)]
    if mag is not None:
        args += ["--fault-magnitude", "%.4f" % mag]
    if "on_s" in p:
        args += ["--fault-on", "%.3f" % p["on_s"], "--fault-off", "%.3f" % p["off_s"]]
    for name, spec in (f.get("params") or {}).items():   # integration3: --saturate-level / --fault-ramp
        if (spec or {}).get("cli") and p.get(name) is not None:
            args += [spec["cli"], "%.4f" % float(p[name])]
    for name, value in (c.get("environment") or {}).items():
        spec = space["environment"].get(name) or {}
        if spec.get("cli") and value is not None:
            args += [spec["cli"], spec.get("cli_format", "%.1f") % float(value)]
    return args


def fault_dict_for_sim(space: dict, c: dict, onset_s: float) -> dict:
    """Fault as run options (agent_config.json field names) — used by the synthetic simulator."""
    args = fault_cli_local(space, c, onset_s)
    get = lambda flag, d=None: float(args[args.index(flag) + 1]) if flag in args else d
    return dict(mode=c["fault"]["mode"], start=get("--fault-start"), duration=get("--fault-duration"),
                magnitude=get("--fault-magnitude", 0.05), on_s=get("--fault-on", 0.2), off_s=get("--fault-off", 0.2),
                level=get("--saturate-level", 1.0), ramp_s=get("--fault-ramp", 0.0))


def detect_route_events(ticks: list[dict], dt: float = 0.05, *, kappa_on: float = 1 / 60, kappa_off: float = 1 / 300,
                        min_curve_s: float = 1.0, min_straight_s: float = 3.0, t_min: float = 5.0,
                        event_index: int = 0, min_speed_mps: float = 2.0, straight_speed_frac: float = 0.6,
                        curve_speed_frac: float = 0.5) -> dict:
    """onset_event -> absolute time from a NO-FAULT baseline run (PROVISIONAL thresholds).

    Driven-path curvature kappa = yaw_rate / max(vx, 1) (steps.jsonl), 0.5 s moving average.
    Only ticks where the car is actually moving count (BUG-012): a stopped car has yaw_rate 0 and so reads
    as kappa 0 = "straight". v_ref = 90th percentile of the moving speed (cruise speed of this baseline);
    straight ticks need vx >= max(min_speed_mps, straight_speed_frac * v_ref), curve ticks need
    vx >= max(min_speed_mps, curve_speed_frac * v_ref) (kappa from yaw_rate/vx is noisy near standstill).
    The launch phase (before vx first reaches the straight speed) is skipped, as is t < t_min.
    curve_entry/apex/exit = start / |kappa| max / end of the event_index-th segment with |kappa| > kappa_on
    for >= min_curve_s; straight = middle of the first segment with |kappa| < kappa_off for >= min_straight_s.
    A stop ends a segment.
    """
    t = [r["t"] for r in ticks]
    vx = [float(r.get("vx") or 0.0) for r in ticks]
    k = [float(r.get("yaw_rate") or 0.0) / max(v, 1.0) for r, v in zip(ticks, vx)]
    k = J.moving_average(k, 0.5, dt)
    moving = sorted(v for v in vx if v > 0.5)
    v_ref = moving[int(0.9 * (len(moving) - 1))] if moving else 0.0
    v_s = max(min_speed_mps, straight_speed_frac * v_ref)
    v_c = max(min_speed_mps, curve_speed_frac * v_ref)
    t_launch = next((ti for ti, v in zip(t, vx) if v >= v_s), None)
    t0 = max(t_min, t_launch) if t_launch is not None else float("inf")
    cls: list[str | None] = []
    for ti, v, ki in zip(t, vx, k):
        if ti < t0:
            cls.append(None)
        elif abs(ki) > kappa_on and v >= v_c:
            cls.append("c")
        elif abs(ki) < kappa_off and v >= v_s:
            cls.append("s")
        else:
            cls.append(None)
    need_c, need_s = int(min_curve_s / dt), int(min_straight_s / dt)
    curves, straights = [], []
    i, n = 0, len(t)
    while i < n:
        c = cls[i]
        if c is None:
            i += 1; continue
        j = i
        while j < n and cls[j] == c:
            j += 1
        if c == "c" and j - i >= need_c:
            curves.append((i, j - 1))
        elif c == "s" and j - i >= need_s:
            straights.append((i, j - 1))
        i = j
    ev: dict[str, Any] = {"curve_entry": None, "curve_apex": None, "curve_exit": None, "straight": None,
                          "kappa_at": {}, "speed_at": {}, "n_curves": len(curves),
                          "method": "yaw_rate/vx, moving ticks only, PROVISIONAL thresholds", "kappa_on": kappa_on,
                          "v_ref_mps": v_ref, "min_straight_speed_mps": v_s, "min_curve_speed_mps": v_c,
                          "t_launch": t_launch, "t_end": t[-1] if t else None}
    if len(curves) > event_index:
        a, b = curves[event_index]
        apex = max(range(a, b + 1), key=lambda m: abs(k[m]))
        for name, idx in (("curve_entry", a), ("curve_apex", apex), ("curve_exit", b)):
            ev[name] = t[idx]; ev["kappa_at"][name] = k[idx]; ev["speed_at"][name] = vx[idx]
    if straights:
        a, b = straights[0]
        m = (a + b) // 2
        ev["straight"] = t[m]; ev["kappa_at"]["straight"] = k[m]; ev["speed_at"]["straight"] = vx[m]
    return ev
