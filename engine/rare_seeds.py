"""Rare failure-type seeds for 07 --rare-type-picks (laya-judge, mode quiet_seeds; config failure_types.yaml
acquisition.seeds).

Why: the 05c per-type classifier cannot rank lane_departure / unsettled / lat_acc (3-6 positives in result_8..12), so
picking 'near a rare type' with it is close to random. Instead the rare-type slots are filled from the region where a
fault is QUIET for the supervisor (no instant check trips, only accumulated tracking error can), which is where a
late switch hands LQR a degraded state:

  bias_sub_threshold  F-04 bias / F-07 intermittent_bias with magnitude just under one steer-rate tick
                      (supervisor.steer_rate_exceeded.max_dps x supervisor.period_s = 90 deg/s x 0.05 s = 0.0785 rad)
  slow_drift          F-05 drift whose per-tick step (rate x 0.05 s <= 0.0025 rad) never trips the rate check and whose
                      accumulated offset over the minimum evaluation window (09 safety_spec evaluation_window
                      min_length_s = 5 s) stays under that one tick: rate <= 0.0785 / 5 = 0.0157 rad/s
  curve_entry         a quiet fault (either of the two above) starting at onset_event curve_entry
  low_friction        a quiet fault with environment.tire_friction_scale in the low band (P-02 limit scales with it)
  rare_neighbours     parameter-jittered copies (scenarios.perturb: route, fault, onset event fixed) of known rare
                      failure runs (lane departure incl. SR4 footprint, unsettled, lat_acc) from saved run records and
                      this campaign
  b_proposals         the 05b rule-based B proposals already in the pool (source B: bias just under the rate check by
                      0.9 tick + 5 km/h) - reused, not regenerated

Only scenario-space parameters are used (no new physics). Candidates still pass stage 04 (gate, V-03, dedup)."""
from __future__ import annotations

import glob
import math
import random
from pathlib import Path
from typing import Any, Callable

from engine import failure_types as FT
from engine import scenarios as SC

SOURCES = ("bias_sub_threshold", "slow_drift", "curve_entry", "low_friction", "rare_neighbours", "b_proposals")
QUIET_MODES = ("bias", "intermittent_bias", "drift")


def rate_tick_rad(space: dict, supervisor: str = "tight") -> float | None:
    """One-tick steering jump the tight supervisor's steer_rate_exceeded check catches (rad); None under legacy (no
    rate check there) or when the space has no supervisor block."""
    sup = space.get("supervisor") or {}
    sr = sup.get("steer_rate_exceeded") or {}
    if supervisor == "legacy" or not sr.get("max_dps") or not sup.get("period_s"):
        return None
    return math.radians(float(sr["max_dps"])) * float(sup["period_s"])


def derived_bands(space: dict, scfg: dict, *, supervisor: str = "tight", eval_window_min_s: float = 5.0) -> dict:
    """The parameter bands each source samples, with where each bound came from (logged in loop_stats)."""
    faults = {f["injector_mode"]: f for f in space.get("faults", [])}
    tick = rate_tick_rad(space, supervisor)
    notes = []
    out: dict[str, Any] = {"rate_tick_rad": None if tick is None else round(tick, 5)}
    b = scfg.get("bias_sub_threshold") or {}
    lo, hi = [float(v) for v in b.get("magnitude_rad", [0.06, 0.078])]
    if tick is not None and hi >= tick:
        hi = round(tick * float(b.get("max_tick_frac", 0.995)), 4)
        notes.append(f"bias upper bound clipped under the rate tick {tick:.4f} rad")
    if tick is None:
        notes.append(f"supervisor {supervisor}: no steer-rate check, bias band kept from config (not code-derived)")
    out["bias_magnitude_rad"] = [lo, hi]
    d = scfg.get("slow_drift") or {}
    rng_d = ((faults.get("drift") or {}).get("params") or {}).get("rate_rad_s", {}).get("range", [0.005, 0.05])
    dlo = float(d.get("rate_rad_s_min") if d.get("rate_rad_s_min") is not None else rng_d[0])
    if d.get("rate_rad_s_max") is not None:
        dhi, why = float(d["rate_rad_s_max"]), "config"
    elif tick is not None:
        dhi, why = tick / float(eval_window_min_s), f"rate tick {tick:.4f} rad / evaluation window {eval_window_min_s:g} s"
    else:
        dhi, why = float(rng_d[1]), "drift range (no rate check under this supervisor)"
    out["drift_rate_rad_s"] = [dlo, round(min(dhi, float(rng_d[1])), 4)]
    out["drift_rate_max_from"] = why
    out["speed_kmh"] = [float(v) for v in scfg.get("speed_kmh", [25.0, 40.0])]
    out["low_friction"] = [float(v) for v in (scfg.get("low_friction") or {}).get("tire_friction_scale", [0.3, 0.6])]
    out["curve_entry_offset_s"] = [float(v) for v in (scfg.get("curve_entry") or {}).get("offset_s", [-0.5, 0.5])]
    out["notes"] = notes
    return out


def _uniform(rng: random.Random, band: list[float]) -> float:
    return band[0] + rng.random() * (band[1] - band[0])


def _quiet_params(space: dict, mode: str, bands: dict, rng: random.Random) -> tuple[str, dict] | None:
    f = next((f for f in space.get("faults", []) if f["injector_mode"] == mode and f.get("implemented")), None)
    if f is None:
        return None
    params = {}
    for k, spec in (f.get("params") or {}).items():
        if k == "magnitude_rad" and mode in ("bias", "intermittent_bias"):
            lo, hi = bands["bias_magnitude_rad"]
            r = spec.get("range") or [lo, hi]
            lo, hi = max(lo, r[0]), min(hi, r[1])
            if lo > hi:
                return None
            params[k] = _uniform(rng, [lo, hi])
        elif k == "rate_rad_s" and mode == "drift":
            params[k] = _uniform(rng, bands["drift_rate_rad_s"])
        else:
            params[k] = SC._param_from_unit(spec, rng.random())
    return f["id"], params


def _place(space: dict, routes: list[str], fault: dict, speed: float, rng: random.Random, event: str | None,
           speed_step: float) -> tuple[str, str, float] | None:
    """(route, onset_event, speed) the route can grade (route_onset_exclusions at the quantized speed); with `event`
    fixed, lower the speed in speed_step steps until some route allows it (never below the space minimum)."""
    lo = space["environment"]["max_speed_kmh"]["range"][0]
    q = lambda v: round(v / speed_step) * speed_step if speed_step > 0 else v
    sp = speed
    while sp >= lo - 1e-9:
        opts = [(r, ev) for r in routes for ev in SC.allowed_onset_events(space, r, fault, q(sp))
                if event is None or ev == event]
        if opts:
            r, ev = opts[rng.randrange(len(opts))]
            return r, ev, sp
        if event is None or speed_step <= 0:
            return None
        sp -= speed_step
    return None


def known_rare_runs(records: list[dict], scfg: dict, ft_cfg: dict, *, supervisor: str) -> list[dict]:
    """Run records of switched runs with a rare failure type (rare_neighbours.types) under the same supervisor
    profile. Labels via failure_types.type_labels (targets only); SR4 footprint departure counts when
    rare_neighbours.include_sr4_footprint (default true)."""
    import copy
    rn = scfg.get("rare_neighbours") or {}
    cfg = copy.deepcopy(ft_cfg)
    cfg.setdefault("types", {}).setdefault("lane_departure", {})["include_sr4_footprint"] = \
        bool(rn.get("include_sr4_footprint", True))
    types = list(rn.get("types") or ["lane_departure", "unsettled", "lat_acc"])
    out, seen = [], set()
    for r in records:
        if str(r.get("supervisor_profile") or "legacy") != supervisor:
            continue
        lab = FT.type_labels(r, cfg)
        if not lab:
            continue
        hit = [t for t in types if lab.get(t)]
        key = (r.get("run_id"), r.get("scenario_id"), r.get("_file"))
        if hit and key not in seen:
            seen.add(key)
            out.append({**r, "_rare_types": hit})
    return out


def load_records(patterns: list[str], root: Path) -> list[dict]:
    import json
    out = []
    files = sorted({f for p in patterns for f in glob.glob(str(root / p) if not Path(p).is_absolute() else p)})
    for fp in files:
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_file"] = str(Path(fp).relative_to(root)) if str(fp).startswith(str(root)) else fp
                out.append(r)
    return out


def record_candidate(space: dict, r: dict, routes: list[str], fault_pool: list[str]) -> dict | None:
    """A run record's plan as a candidate in THIS space (missing new params take catalog defaults); None if the route
    / fault is not in this campaign or the plan is outside the current ranges."""
    p = r.get("parameters") or {}
    fault = p.get("fault") or {}
    faults = SC.faults_by_id(space)
    if p.get("route_id") not in routes or fault.get("fault_id") not in fault_pool or fault.get("fault_id") not in faults:
        return None
    spec = faults[fault["fault_id"]].get("params") or {}
    params = {k: (fault.get("params") or {}).get(k, v.get("default")) for k, v in spec.items()}
    env = p.get("environment") or {}
    try:
        c = SC.make_candidate(space, route_id=p["route_id"], speed=env.get("max_speed_kmh", 30.0),
                              fault_id=fault["fault_id"], params=params, onset_event=p.get("onset_event"),
                              offset=p.get("onset_offset_s", 0.0), source="history", friction=env.get(SC.FRICTION))
    except (KeyError, ValueError, TypeError):
        return None
    return None if SC.validate_against_space(space, c, routes, check_exclusions=False) else c


def generate(space: dict, routes: list[str], fault_pool: list[str], scfg: dict, rng: random.Random, *,
             bands: dict, rare_runs: list[dict], round_index: int, speed_step: float = 0.0) -> list[dict]:
    """Seed candidates (source 'rare_seed', key rare_seed_source) for every enabled source except b_proposals (those
    are already in the pool). per_source candidates each; rare_neighbours: up to per_source known rare runs (rng
    order), neighbours_per_run jittered copies each."""
    enabled = [s for s in (scfg.get("sources") or SOURCES) if s in SOURCES]
    n = int(scfg.get("per_source", 4))
    pool_modes = {f["injector_mode"] for f in space.get("faults", []) if f["id"] in fault_pool}
    faults = SC.faults_by_id(space)
    out: list[dict] = []

    def quiet(src: str, modes: tuple[str, ...], *, event=None, friction_band=None, offset_band=None):
        modes = tuple(m for m in modes if m in pool_modes)
        for i in range(n if modes else 0):
            got = _quiet_params(space, modes[i % len(modes)], bands, rng)
            if got is None:
                continue
            fid, params = got
            pl = _place(space, routes, faults[fid], _uniform(rng, bands["speed_kmh"]), rng, event, speed_step)
            if pl is None:
                continue
            route, ev, sp = pl
            fax = SC.friction_axis(space)
            fr = None
            if fax:
                fr = _uniform(rng, friction_band or fax["range"])
            off_rng = offset_band or space["fault_timing"]["offset_s"]["range"]
            c = SC.make_candidate(space, route_id=route, speed=sp, fault_id=fid, params=params, onset_event=ev,
                                  offset=_uniform(rng, off_rng), source="rare_seed", round_index=round_index,
                                  friction=fr, expected_mechanism=_why(src, bands))
            c["rare_seed_source"] = src
            out.append(c)

    if "bias_sub_threshold" in enabled:
        quiet("bias_sub_threshold", ("bias", "intermittent_bias"))
    if "slow_drift" in enabled:
        quiet("slow_drift", ("drift",))
    if "curve_entry" in enabled:
        quiet("curve_entry", QUIET_MODES, event="curve_entry", offset_band=bands["curve_entry_offset_s"])
    if "low_friction" in enabled:
        if SC.friction_axis(space) is None:
            pass   # the space has no tire_friction_scale axis (no run_benchmark flag): skipped, see loop_stats note
        else:
            quiet("low_friction", QUIET_MODES, friction_band=bands["low_friction"])
    if "rare_neighbours" in enabled and rare_runs:
        rn = scfg.get("rare_neighbours") or {}
        runs = list(rare_runs)
        rng.shuffle(runs)
        for r in runs[:n]:
            base = record_candidate(space, r, routes, fault_pool)
            if base is None:
                continue
            for p in SC.perturb(space, base, int(rn.get("neighbours_per_run", 1)), rng, scale=float(rn.get("jitter", 0.08))):
                p["source"] = "rare_seed"
                p["round"] = round_index
                p["rare_seed_source"] = "rare_neighbours"
                p["rare_seed_of"] = {"run_id": r.get("run_id"), "file": r.get("_file"), "types": r.get("_rare_types")}
                p["expected_mechanism"] = (f"neighbour of {r.get('run_id')} ({', '.join(r.get('_rare_types') or [])}); "
                                           f"route, fault, onset event fixed, parameters jittered")
                p["scenario_id"] = SC.cand_id(p)
                p["seed_scenario_id"] = p["scenario_id"]
                out.append(p)
    return out


def _why(src: str, bands: dict) -> str:
    t = bands.get("rate_tick_rad")
    tick = f"{t:.4f} rad (90 deg/s x 0.05 s)" if t else "n/a (no rate check)"
    return {"bias_sub_threshold": f"quiet bias {bands['bias_magnitude_rad']} rad just under the steer-rate tick {tick}: "
                                  f"only tracking_divergence (0.6 m / 0.35 rad held 0.15 s) can switch",
            "slow_drift": f"slow drift {bands['drift_rate_rad_s']} rad/s ({bands['drift_rate_max_from']}): per-tick step "
                          f"far under the rate tick, switch only by accumulated tracking error",
            "curve_entry": "quiet fault starting at curve entry: tracking error builds while the curve demands steering",
            "low_friction": f"quiet fault at tire_friction_scale {bands['low_friction']}: lower P-02 limit and grip"}[src]


def nearest_rare_distance(x: list[float], rare_feats: list[list[float]]) -> float | None:
    return min((math.dist(x, r) for r in rare_feats), default=None)
