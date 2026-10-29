#!/usr/bin/env python3
"""SIL_HACKATHON one-command loop: 00 prior knowledge -> ... -> 09 monitor -> 10 video explainer -> feedback.

  python run.py --check                      # stage-YAML null report (exit 2 while required nulls remain)
  python run.py --mock                       # no CARLA, no network, synthetic fixtures + stub LLMs
  python run.py --mock --ablation-suite      # X-1..X-6 x seeds on the synthetic simulator
  python run.py --real --check               # preflight: av_stack, CARLA port, NEBIUS_API_KEY, models, ffmpeg
  python run.py --real                       # CARLA + Nebius Token Factory
  (python -m run ... works the same; 09 monitor alone: python -m engine.monitor RUN_DIR [--baseline DIR])

Outputs go to outputs/<mode>_<timestamp>/ (git-ignored). Thresholds are PROVISIONAL (02 outcome_definitions).
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from engine import judgment as J
from engine import nemotron as NM
from engine import scenarios as SC
from engine.carla_runner import CarlaRunner, MockRunner, carla_reachable
from engine.config import (REPO_ROOT, load_stage_configs, load_yaml, resolve_av_stack,
                                    resolve_scenario_space)
from engine.explainer import explain, stub_explanation, verify
from engine.feedback import (counterfactual_candidates, failure_table, few_shot, is_failure, outcome_examples, tried_view,
                                      mechanism_features, mechanism_summary, taxonomy_proposals)
from engine.llm import LLMError, StubClient, TokenFactoryClient, role_settings
from engine.monitor import (compute_run_record, evaluate as monitor_evaluate, load_av_stack_run, monitor_timeline,
                            scale_thresholds_for_friction, thresholds_from_config)
from engine.surrogate import MetricGPs, acquisition_score, boundary_score, boundary_score_parts, select_batch_scored
from engine.dashboard import refresh as refresh_dashboard
from engine.video import cover_times, extract_clip, ffmpeg_exe, key_times
from engine.prior_tools.prior_tools import make_nemotron_payload
from engine.cosmos_power import CosmosPower

ABLATIONS = {
    "X-1": dict(name="random_space_filling", use_a=False, use_gp=False, use_df=False, use_b=False),
    "X-2": dict(name="nemotron_a_only", use_a=True, use_gp=False, use_df=False, use_b=False),
    "X-3": dict(name="gp_numeric_only", use_a=True, use_gp=True, use_df=False, use_b=False),
    "X-4": dict(name="gp_plus_deterministic_features", use_a=True, use_gp=True, use_df=True, use_b=False),
    "X-5": dict(name="full_code_aware", use_a=True, use_gp=True, use_df=True, use_b=True),
    "X-6": dict(name="full_code_aware_plus_video_explainer", use_a=True, use_gp=True, use_df=True, use_b=True),
}


B_STATIC_KEY = "__static__"   # integration3 #6: 05b_answers.json row of the once-per-campaign static B questions
SWITCH_METRIC = "switched_indicator"   # +1 switched / -1 no switch; S1 boundary only exists where a switch happens
# integration3: modelled by a GP classifier (06 surrogate.yaml switch_model: classifier) -> gp_mean = P(switch) in [0, 1]


def switch_aware_score(c: dict, obj: dict, beta: float, weight: float = 2.0) -> float | None:
    """LEGACY (acquisition: straddle). S1 straddle on recovery_success, penalised where the switch GP predicts no
    switch (no S1 target there). Meant for switch_model: regression (mu_sw in [-1, 1]); the integration3 default is
    acquisition: level_set (engine.surrogate.boundary_score)."""
    s = acquisition_score(c, obj, beta)
    gm = c.get("gp_mean") or {}
    mu_sw = gm.get(SWITCH_METRIC)
    if mu_sw is None:
        return s
    if gm.get(SWITCH_METRIC + "__latent") is not None:   # classifier: P(switch) in [0, 1] -> [-1, 1] like the old regression
        mu_sw = 2.0 * mu_sw - 1.0
    if s is None:   # no recovery GP yet: prefer uncertain/likely-switch candidates
        return min(0.0, mu_sw) * weight + (c.get("gp_std") or {}).get(SWITCH_METRIC, 0.0)
    return s + weight * min(0.0, mu_sw)


def log(msg: str) -> None:
    print(msg, flush=True)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


class CosmosGuard:
    """--cosmos-power auto: the Cosmos dedicated endpoint is on only while a stage uses it.

    session(reason): ensure_on before, ensure_off after (unless keep_on). Yields True when Cosmos is usable.
    shutdown(): final off (try/finally + atexit in main). power None (manual mode / mock / not dedicated):
    no API calls, Cosmos assumed usable as before. After one failed start, later sessions skip Cosmos
    instead of waiting for the timeout again."""

    def __init__(self, power: Any = None, log=None) -> None:
        self.power = power
        self.log = log or (lambda m: None)
        self.failed = False
        self._shut = False

    @contextlib.contextmanager
    def session(self, reason: str, keep_on: bool = False):
        if self.power is None:
            yield True
            return
        if self.failed:
            self.log(f"[cosmos-power] skip start for {reason}: endpoint did not become ready earlier in this run")
            yield False
            return
        ok = self.power.ensure_on(reason)
        if not ok:
            self.failed = True
        try:
            yield ok
        finally:
            if not keep_on or not ok:
                self.power.ensure_off(reason)

    def shutdown(self, reason: str = "end of run") -> None:
        if self.power is None or self._shut:
            return
        self._shut = True
        init = getattr(self.power, "initial_state", None) or {}
        if init.get("enabled"):
            self.log("[cosmos-power] endpoint was already on when this run first checked it (manual start); "
                     "turning it off now (--cosmos-power auto)")
        self.power.ensure_off(reason)


class Loop:
    def __init__(self, *, mode: str, out: Path, ablation: str, video_on: bool, seed: int, cfg: dict | None = None,
                 cold_start: int | None = None, batch_size: int | None = None, rounds: int | None = None,
                 quiet: bool = False, no_reaction: bool | None = None, gp_history: list[str] | None = None,
                 supervisor: str | None = None):
        self.mode, self.out, self.ablation, self.video_on, self.seed = mode, Path(out), ablation, video_on, seed
        self.flags = ABLATIONS[ablation]
        if ablation == "X-6":
            self.video_on = True
        self.cfg = cfg or load_stage_configs()
        self.quiet = quiet
        self.rng = random.Random(seed)
        orch = self.cfg["orchestrator"]
        self.cold = cold_start if cold_start is not None else int(orch["cold_start"]["total_runs"])
        self.batch = batch_size if batch_size is not None else int(orch["acquisition"]["batch_size"])
        max_sims = (orch.get("budget") or {}).get("max_simulations") or (self.cold + 4 * self.batch)
        self.rounds = rounds if rounds is not None else max(0, math.ceil((max_sims - self.cold) / self.batch))
        space_path, self.space_synthetic = resolve_scenario_space(allow_fixture=(mode == "mock"))
        self.space = load_yaml(space_path)
        self.space_path = space_path
        pr = self.cfg["paired_runs"]
        self.routes = [r for r in pr["carla"]["routes"] if r.get("enabled", True)]
        self.route_ids = [r["route_id"] for r in self.routes]
        self.runner_cfg = pr["runner"]
        # integration3 #8: av_stack supervisor profile (run_benchmark.py --supervisor; mock: synthetic_sim.SUPERVISOR_PROFILES)
        self.supervisor = str(supervisor or self.runner_cfg.get("supervisor") or "tight")
        self.fault_pool = orch["fault_pool"]
        # BUG-025: control faults (solver_fail, stale) always switch at once; at most this many per round
        self.control_ids = {f["id"] for f in self.space.get("faults", []) if f.get("group") == "control"}
        self.control_cap = int(orch.get("control_faults_per_round_max", 1))
        self.objectives = orch["active_objectives"]
        self.obj_cfg = orch["objective_acquisition"]
        self.metric_rules = self.cfg["vulnerability_patterns"]["metric_rules"]
        self.gp_metrics = sorted({self.obj_cfg[o]["metric"] for o in self.objectives}) + [SWITCH_METRIC]
        self.th = thresholds_from_config(self.cfg)
        sur = (self.cfg.get("surrogate") or {}).get("surrogate") or {}
        self.switch_model = str(sur.get("switch_model", "classifier"))
        self.kernel_opts = sur.get("kernel_bounds") or {}
        self.prior_mean_mode = str(sur.get("prior_mean", "level"))   # level (integration3) | training_mean (legacy)
        self.min_train_points = int(sur.get("min_train_points", 4))
        self.min_train_for_acq = int(sur.get("min_train_runs_for_acquisition", 8))
        self.gps: MetricGPs | None = None
        self.gp_ready = False
        self.history_rows: list[tuple[dict, dict]] = []
        self.gp_history = list(gp_history or [])
        nr = (self.cfg["paired_runs"]["runner"].get("no_reaction_run") or {})
        self.no_reaction_on = bool(nr.get("mock" if mode == "mock" else "real", False)) if no_reaction is None else no_reaction
        self.vcfg = self.cfg["video_explainer"]
        self.taxonomy = self.vcfg["mechanism_taxonomy"]
        # integration3 #6: yes/no as strings, scope (per_scenario | static) and gp_feature defaults
        self.questions = NM.normalize_questions(self.cfg["nemotron_b"]["B_question_answering"]["question_bank"])
        self.av_stack = resolve_av_stack()
        self.records: list[dict] = []
        self.candidates: dict[str, dict] = {}
        self.features_of: dict[str, list[float]] = {}
        self.explanations: dict[str, dict] = {}
        self.b_cache: dict[str, dict] = {}
        self.events_cache: dict[str, dict] = {}
        self.cf_queue: list[dict] = []
        self.stats: dict[str, Any] = {"a_calls": [], "rejected": [], "llm_errors": []}
        if self.gp_history:
            self.load_gp_history(self.gp_history)
        self.cosmos = CosmosGuard()       # main() swaps in the auto guard for --real --cosmos-power auto
        self.insights_next = False        # campaign insights follow the last round -> keep Cosmos on
        self.open_screen = False          # main(): --open-screen on -> open screen/campaign.html once at start
        self._setup_clients()
        self._setup_runner()

    # ------------------------------------------------------------------ setup
    def say(self, msg: str) -> None:
        if not self.quiet:
            log(msg)

    def _setup_clients(self) -> None:
        llm_cfg = self.cfg["llm"]
        self.llm_log = self.out / "llm_calls.jsonl"
        if self.mode == "mock":
            stub = NM.stub_handler(self.seed)
            self.client_a = self.client_b = StubClient(stub, self.llm_log)
            self.client_c = None
            self.c_settings = {"model": "stub/explainer", "media": ["video"]}
        else:
            self.client_a = TokenFactoryClient.from_config(llm_cfg, self.llm_log, role="A")
            self.client_b = TokenFactoryClient.from_config(llm_cfg, self.llm_log, role="B")
            cos = role_settings(llm_cfg, "C_cosmos")
            if cos.get("model"):
                self.client_c, self.c_settings = TokenFactoryClient.from_config(llm_cfg, self.llm_log, role="C_cosmos"), cos
            else:
                self.client_c = TokenFactoryClient.from_config(llm_cfg, self.llm_log)
                self.c_settings = role_settings(llm_cfg, "C_fallback")
            self.c_fallback = role_settings(llm_cfg, "C_fallback")
        self.a_settings = role_settings(llm_cfg, "A")
        self.a_refine = role_settings(llm_cfg, "A_refine")
        self.b_settings = role_settings(llm_cfg, "B")

    def _setup_runner(self) -> None:
        rc = self.runner_cfg
        common = dict(out_root=self.out / "runs", routes=self.routes, seed=int(rc["seed"]), space=self.space,
                      stop_after_fault_s=float(rc["stop_after_fault_s"]), supervisor=self.supervisor)
        if self.mode == "mock":
            self.runner = MockRunner(**common)
        else:
            if self.av_stack is None:
                raise SystemExit("av_stack not found: merge junwoo/sim-eval (pipeline/08_carla_paired_run/av_stack) "
                                 "or set AV_STACK_ROOT / SIM_EVAL_ROOT")
            self.runner = CarlaRunner(**common, av_stack=self.av_stack, python_exe=os.environ.get("AV_STACK_PYTHON") or rc.get("python_exe"),
                                      host=os.environ.get("CARLA_HOST", rc["host"]), port=int(os.environ.get("CARLA_PORT", rc["port"])),
                                      tm_port=int(rc["traffic_manager_port"]), timeout_s=float(rc["rpc_timeout_s"]),
                                      wall_timeout_s=float(rc["wall_timeout_s"]), chase_view=bool(rc.get("chase_view")))

    # ------------------------------------------------------------------ live screen
    def screen(self, status: str | None = None) -> None:
        """Re-draw screen/campaign.html (the open page reloads itself every few seconds). Never fails the run."""
        self.status = status if status is not None else getattr(self, "status", None)
        try:
            refresh_dashboard(self.out / "run_records.jsonl", self.out / "screen",
                              explanations=self.out / "explanations.jsonl", spec=self.cfg["safety_spec"], status=self.status)
        except Exception as e:  # noqa: BLE001 - the screen is a view; results are already on disk
            self.say(f"  screen redraw failed: {type(e).__name__}: {e}")

    def open_browser(self) -> None:
        """Open the live page once (default browser). Skipped under pytest / SIL_NO_BROWSER=1."""
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("SIL_NO_BROWSER"):
            return
        page = (self.out / "screen" / "campaign.html").resolve()
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(page))          # Windows: default browser for .html
            else:
                webbrowser.open(page.as_uri())
            self.say(f"live screen opened: {page}")
        except Exception as e:  # noqa: BLE001
            self.say(f"could not open the browser ({type(e).__name__}); open {page} by hand")

    # ------------------------------------------------------------------ stages 00-02
    def stage_00_02(self) -> None:
        c = self.cfg
        payload = make_nemotron_payload(c["safety_profile"], {"hypotheses": []}, c["prior_knowledge"], self.space)
        dump(self.out / "00_prior_payload.json", payload)
        obs = (c["safety_profile"].get("code_evidence") or {}).get("observations", [])
        dump(self.out / "01_profile_summary.json", {
            "observations": len(obs), "verified": sum(o.get("verified") is True for o in obs),
            "triggers": [t["name"] for t in c["safety_profile"]["switching"]["triggers"]],
            "note": "unverified observations go to Nemotron A only as 'unreviewed leads' (nemotron_a.yaml include_unreviewed_observations)"})
        dump(self.out / "02_objectives.json", {
            "active": self.objectives, "acquisition": {o: self.obj_cfg[o] for o in self.objectives},
            "S1_objective": c["S1"]["objective"], "recovery_judgment_thresholds_PROVISIONAL": self.th})
        self.say(f"[00-02] prior payload ({payload['unverified_observations_omitted']} unverified obs omitted), "
                 f"objectives {self.objectives}, scenario space {'SYNTHETIC fixture' if self.space_synthetic else self.space_path}")

    # ------------------------------------------------------------------ helpers
    def quantize(self, c: dict) -> dict:
        step = float(self.runner_cfg.get("speed_step_kmh") or 0)
        if step > 0:
            lo, hi = self.space["environment"]["max_speed_kmh"]["range"]
            v = round(c["environment"]["max_speed_kmh"] / step) * step
            c["environment"]["max_speed_kmh"] = float(min(hi, max(lo, v)))
        fstep = float(self.runner_cfg.get("friction_step") or 0)
        fax = SC.friction_axis(self.space)
        if fstep > 0 and fax and SC.FRICTION in c["environment"]:
            flo, fhi = fax["range"]
            fv = round(round(c["environment"][SC.FRICTION] / fstep) * fstep, 3)
            c["environment"][SC.FRICTION] = float(min(fhi, max(flo, fv)))
        if step > 0 or (fstep > 0 and fax):
            c["scenario_id"] = SC.cand_id(c)
            if c.get("source") != "perturbation" and c.get("source") != "counterfactual":
                c["seed_scenario_id"] = c["scenario_id"]
        return c

    def route_events(self, c: dict) -> dict | None:
        d = self.runner.baseline_dir(c)
        k = str(d)
        if k in self.events_cache:
            return self.events_cache[k]
        if (d / ".complete").is_file():
            run = load_av_stack_run(d)
            ev = SC.detect_route_events(run.ticks, run.dt, **(self.cfg["paired_runs"].get("event_detection") or {}))
            self.events_cache[k] = ev
            return ev
        return None

    # ------------------------------------------------------------------ integration3 #7: what A needs to judge the boundary
    def judgment_criteria(self) -> dict:
        """The recovery judgment with the configured values (engine.monitor.thresholds_from_config + safety_spec P-02)."""
        th = self.th
        props = {q.get("id"): q for q in ((self.cfg["safety_spec"].get("safety_specification") or {}).get("properties") or [])
                 if isinstance(q, dict)}
        f = (props.get("P-02") or {}).get("lateral_acc_formula") or {"mu": 0.9, "g_mps2": 9.81, "factor": 0.5}
        lat = th["lat_acc_threshold_mps2"]
        return {
            "recovered": (f"final_safe_recovery: within {th['window_s']:g} s after the switch, |e_y| <= {th['e_settle_m']:g} m and "
                          f"|e_psi| <= {th['psi_settle_rad']:g} rad with lane margin >= 0 held for {th['settle_hold_s']:g} s, "
                          f"no lane departure, no collision, and neither lat_acc_excess nor lat_jerk_excess"),
            "lat_acc_limit": (f"lat_acc_excess: post-switch lateral acceleration (relative to the same-route baseline) > "
                              f"{lat:g} * tire_friction_scale m/s^2 = {f['factor']:g} * {f['mu']:g} * tire_friction_scale * "
                              f"{f['g_mps2']:g} (P-02); lat_jerk_excess: > {th['lat_jerk_threshold_mps3']:g} m/s^3 (P-04)"),
            "not_a_failure": ("a run with no switch is not a failed recovery (LQR never acted): it says the fault was too "
                              "mild / short or started too late for the supervisor to fire"),
            "gp_target": ("lane_recovery_robustness = min over the window of lane margin / (lane_width/2 - vehicle half "
                          f"width), dimensionless: 1 = centred, 0 = body at the lane edge (the boundary), < 0 departed, "
                          f"clipped to [{th['robustness_clip_min']:g}, {th['robustness_clip_max']:g}], collision <= "
                          f"{th['collision_penalty']:g}"),
            "evaluation_window": (f"P-01 lane margin is measured over onset .. onset + {th['eval_window_s']:g} s; an onset "
                                  f"less than {float(self.cfg['safety_spec']['evaluation_window']['min_length_s']):g} s "
                                  f"before the route end is rejected (V-03)"),
        }

    def _geom_candidate(self, rid: str, kmh: float, fr: float) -> dict:
        return {"route_id": rid, "environment": {"max_speed_kmh": kmh, "tire_friction_scale": fr},
                "fault": {"fault_id": self.fault_pool[0], "params": {}}, "onset_event": "curve_apex", "onset_offset_s": 0.0}

    def route_geometry(self) -> dict:
        """integration3 #7: per route and onset_event the baseline time, curvature, speed, v^2*kappa and the latest
        onset_offset_s that still passes V-03, from cached no-fault baselines (route_events). mock probes baselines at
        route_geometry.probe_speeds_kmh (synthetic fixture); real mode only reports baselines that already exist."""
        gcfg = self.cfg["nemotron_a"].get("route_geometry") or {}
        mw = float(self.cfg["safety_spec"]["evaluation_window"]["min_length_s"])
        if (gcfg.get("probe_baselines") or {}).get(self.mode, False) and not getattr(self, "_geom_probed", False):
            self._geom_probed = True
            for rid in self.route_ids:
                for kmh in gcfg.get("probe_speeds_kmh") or []:
                    self.runner.ensure_baseline(self._geom_candidate(rid, float(kmh), 1.0))
        found: dict[str, list[tuple[float, float, Path]]] = {}
        for comp in sorted((self.runner.out_root / "baselines").glob("*/.complete")):
            try:
                key = json.loads(comp.read_text())
            except (OSError, ValueError):
                continue
            if key.get("route_id") in self.route_ids and key.get("max_speed_kmh") is not None:
                found.setdefault(key["route_id"], []).append(
                    (float(key["max_speed_kmh"]), float(key.get("tire_friction_scale") or 1.0), comp.parent))
        events = [e for e in self.space["fault_timing"]["onset_event"]["values"]]
        olo = min(float(((f.get("handover_axes") or {}).get("offset_s") or {}).get("range", self.space["fault_timing"]["offset_s"]["range"])[0])
                  for f in self.space.get("faults", []))
        lat = self.th["lat_acc_threshold_mps2"]
        nmax = int(gcfg.get("max_baselines_per_route", 3))
        routes = {}
        for rid in self.route_ids:
            rows = sorted(set(found.get(rid, [])), key=lambda t: (abs(t[1] - 1.0), t[0]))
            nominal = [t for t in rows if abs(t[1] - 1.0) < 1e-9] or rows
            nominal.sort(key=lambda t: t[0])
            if len(nominal) > nmax:
                idx = sorted({round(i * (len(nominal) - 1) / (nmax - 1)) for i in range(nmax)}) if nmax > 1 else [0]
                nominal = [nominal[i] for i in idx]
            out_b, grad = [], {e: [] for e in events}
            rows_by_ev: dict[str, list] = {e: [] for e in events}
            ends = {}
            for kmh, fr, d in nominal:
                ev = self.route_events(self._geom_candidate(rid, kmh, fr))
                if ev is None:
                    continue
                t_end = ev.get("t_end")
                evs = {}
                for e in events:
                    t = ev.get(e)
                    if t is None:
                        evs[e] = None
                        continue
                    k = (ev.get("kappa_at") or {}).get(e)
                    v = min(kmh / 3.6, (ev.get("speed_at") or {}).get(e, 1e9))
                    latest = None if t_end is None else t_end - mw - float(t)
                    evs[e] = {"t_s": round(float(t), 2), "kappa_per_m": None if k is None else round(float(k), 4),
                              "speed_mps": round(v, 2),
                              "v2kappa_mps2": None if k is None else round(v * v * abs(float(k)), 2),
                              "latest_onset_offset_s_for_V03": None if latest is None else round(latest, 2)}
                    if latest is not None and latest >= olo:
                        grad[e].append(kmh)
                for e in events:
                    x = evs[e]
                    rows_by_ev[e].append([kmh] + ([None] * 5 if x is None else
                                                  [x["t_s"], x["kappa_per_m"], x["speed_mps"], x["v2kappa_mps2"],
                                                   x["latest_onset_offset_s_for_V03"]]))
                ends[f"{kmh:g}"] = None if t_end is None else round(float(t_end), 2)
                out_b.append(kmh)
            fr_used = sorted({t[1] for t in nominal})
            routes[rid] = {"town": next((r.get("town") for r in self.routes if r.get("route_id") == rid), None),
                           "baseline_tire_friction_scale": fr_used[0] if len(fr_used) == 1 else fr_used,
                           "route_end_s_by_kmh": ends, "events": rows_by_ev,
                           "events_gradable_at_kmh": grad,
                           **({} if out_b else {"note": "no completed baseline yet: geometry unknown, V-03 decided after the baseline run"})}
        return {"columns": ["max_speed_kmh", "t_s", "kappa_per_m", "speed_mps", "v2kappa_mps2", "latest_onset_offset_s_for_V03"],
                "rules": {
            "V03": f"onset = max(0.5, t_s + onset_offset_s); rejected when route_end_s - onset < {mw:g} s, i.e. "
                   f"onset_offset_s must be <= latest_onset_offset_s_for_V03 (null row = event not detected on that baseline = rejected); events_gradable_at_kmh lists the probed speeds with room for some allowed offset",
            "friction_demand": (f"v2kappa_mps2 = speed^2 * |kappa| is the nominal lateral demand; the P-02 limit is "
                                f"{lat:g} * tire_friction_scale m/s^2. DF-06 friction_utilization = v2kappa / "
                                f"(0.9 * tire_friction_scale * 9.81): near 1 the tyres are close to the grip limit and LQR "
                                f"has little margin left to recover; v2kappa above the P-02 limit only tags stress_test"),
            "speed": "speed_mps is the baseline speed at the event (the car may be slower than max_speed_kmh in curves); "
                     "it scales roughly with max_speed_kmh, the event times shift with speed"},
            "routes": routes}

    def features(self, c: dict, *, train_round: int | None = None) -> tuple[list[float], list[str]]:
        x, names = SC.numeric_features(c, self.space, self.route_ids, self.fault_pool)
        if self.flags["use_df"]:
            ev = self.route_events(c)
            profile = self.cfg["safety_profile"]
            kappa = None if ev is None else (ev.get("kappa_at") or {}).get(c["onset_event"])
            scen = {"speed_mps": c["environment"]["max_speed_kmh"] / 3.6, "road_curvature_per_m": None if kappa is None else abs(kappa),
                    "fault": {"delay_ms": 50.0 * c["fault"]["params"]["delay_ticks"]} if "delay_ticks" in c["fault"]["params"] else {}}
            scen["tire_friction_scale"] = (c.get("environment") or {}).get("tire_friction_scale", 1.0)
            df = J.compute_df(scen, profile).as_dict()
            # integration3 #6: DF-06 friction utilization v^2|kappa| / (0.9 * scale * g) next to DF-05
            for k, scale_by in (("DF-03", 100.0), ("DF-05", 5.0), ("DF-06", 2.0)):
                v = df.get(k)
                names += [k, f"{k}_missing"]
                x += [0.0 if v is None else float(v) / scale_by, 1.0 if v is None else 0.0]
        if self.flags["use_b"]:
            bx, bn = NM.b_features((self.b_cache.get(self.b_key(c)) or {}).get("answers"), self.questions)
            x += bx; names += bn
        if self.video_on and self.vcfg["feedback"]["gp_categorical_features"]["enabled"]:
            ex = self.explanations if train_round is None else {k: e for k, e in self.explanations.items() if e["round"] < train_round}
            mx, mn = mechanism_features(c, ex, self.taxonomy, self.vcfg["feedback"]["gp_categorical_features"]["smoothing_alpha"])
            x += mx; names += mn
        return x, names

    def b_key(self, c: dict) -> str:
        # integration3 #6/#8: + bins of the fault parameters a question depends on (Q-B4: saturate duration_s <= 0.15 s, ramp_s 0)
        return f"{c['fault']['mode']}|{c['onset_event']}" + NM.param_bin_suffix(self.questions, c["fault"].get("params") or {})

    # ------------------------------------------------------------------ 03 Nemotron A
    def stage_03(self, n: int, round_index: int, near: list[dict] | None = None,
                 gp_summary: dict | None = None) -> list[dict]:
        if n <= 0 or not self.flags["use_a"]:
            return []
        evidence = NM.evidence_enum(self.cfg["safety_profile"], self.cfg["prior_knowledge"], self.space, self.objectives)
        schema = NM.candidate_schema(self.space, self.route_ids, self.fault_pool, self.objectives, evidence, n)
        table = failure_table(self.records, self.candidates)
        acfg = self.cfg["nemotron_a"]
        s1 = self.cfg["S1"]
        # integration3 #7: judgment criteria with the configured values, route geometry (V-03 / v^2 kappa), per-mode
        # recovered/failed pairs and outcomes of already-tried scenarios: the material A needs to place a boundary
        geom = self.route_geometry()
        dump(self.out / f"round_{round_index:02d}" / "03_route_geometry.json", geom)
        by_sid = {r.get("scenario_id"): r for r in self.records}
        tried = [tried_view(c, by_sid.get(c.get("scenario_id")))
                 for c in list(self.candidates.values())[-int((acfg.get("already_tried") or {}).get("max_items", 40)):]]
        oe_cfg = acfg.get("outcome_examples") or {}
        examples = (outcome_examples(self.records, self.candidates, self.space, self.route_ids, self.fault_pool,
                                     int(oe_cfg.get("max_modes", 8))) if oe_cfg.get("enabled", True) else [])
        ctx = NM.a_context(profile=self.cfg["safety_profile"], prior=self.cfg["prior_knowledge"], space=self.space,
                           objective_cfg={"id": "S1", **s1["objective"],
                                          "natural_switch_bound": s1["switch_state_control"]["natural_switch_bound"],
                                          "natural_switch_bound_en": s1["switch_state_control"].get("natural_switch_bound_en"),
                                          "judgment_criteria": self.judgment_criteria()},
                           routes=[{k: r.get(k) for k in ("route_id", "town", "note", "note_en")} for r in self.routes],
                           few_shot=few_shot(table, self.explanations, int(acfg["feedback_in"]["k"])),
                           mechanisms=mechanism_summary(self.explanations) if self.video_on else [],
                           tried=tried,
                           include_unreviewed=bool(acfg.get("include_unreviewed_observations", True)),
                           surrogate_summary=gp_summary, outcome_examples=examples, route_geometry=geom)
        msgs = NM.a_messages(schema=schema, n=n, context=ctx)
        obj, st = NM.call_a(self.client_a, self.a_settings, msgs, schema, tag="A")
        st.update(round=round_index, requested=n)
        self.stats["a_calls"].append(st)
        cands = NM.a_to_candidates(obj or {}, self.space, round_index=round_index)
        st["boundary_regions_given"] = len((gp_summary or {}).get("boundary_regions") or [])
        st["range_extreme_unjustified"] = sum("range_extreme_unjustified" in c.get("tags", []) for c in cands)
        if near and self.a_refine.get("enabled", True) and self.cfg["nemotron_a"].get("refine", {}).get("enabled", True):
            schema_r = NM.candidate_schema(self.space, self.route_ids, self.fault_pool, self.objectives, evidence, len(near))
            near_items = [NM.region_item(c, self.summary_scorer(c) or {}, metric=self.obj_cfg[self.objectives[0]]["metric"],
                                         native=self.native, y_scale=self.target_y_scale(), level_z=self.level_z(self.obj_cfg[self.objectives[0]]),
                                         runs=by_sid) for c in near]
            obj_r, st_r = NM.call_a(self.client_a, self.a_refine, NM.refine_messages(schema_r, near_items, ctx), schema_r, tag="A_refine")
            st_r.update(round=round_index, requested=len(near), refine=True)
            self.stats["a_calls"].append(st_r)
            cands += NM.a_to_candidates(obj_r or {}, self.space, round_index=round_index, source="nemotron_a_refine")
        return [self.quantize(c) for c in cands]

    # ------------------------------------------------------------------ 04 feasibility + expansion
    def stage_04(self, cands: list[dict], round_index: int) -> list[dict]:
        lat = self.th["lat_acc_threshold_mps2"]
        mw = float(self.cfg["safety_spec"]["evaluation_window"]["min_length_s"])   # BUG-028
        ok_list = []
        for c in cands:
            ok, tags, problems = SC.gate(self.space, c, self.route_ids, self.route_events(c), lat, mw)
            c["tags"] = sorted(set(c.get("tags", []) + tags))
            if ok:
                ok_list.append(c)
            else:
                self.stats["rejected"].append({"scenario_id": c["scenario_id"], "source": c["source"], "tags": tags, "problems": problems})
        fe = self.cfg["feasibility_expansion"]["candidate_generation"]
        expanded = list(ok_list)
        if round_index > 0 and self.flags["use_gp"]:
            for c in [c for c in ok_list if c["source"].startswith("nemotron_a")]:
                for p in SC.perturb(self.space, c, int(fe["local_perturbation"]["per_seed"]), self.rng):
                    p = self.quantize(p)
                    ok, tags, _ = SC.gate(self.space, p, self.route_ids, self.route_events(p), lat, mw)
                    if ok:
                        p["tags"] = tags; expanded.append(p)
        seen = set(self.candidates) | {r["scenario_id"] for r in self.records}
        uniq, ids = [], set()
        for c in expanded:
            if c["scenario_id"] in seen or c["scenario_id"] in ids:
                continue
            ids.add(c["scenario_id"]); uniq.append(c)
        return SC.dedup(uniq, self.space, self.route_ids, self.fault_pool, float(fe["dedup"]["min_distance"]))

    # ------------------------------------------------------------------ 05b Nemotron B
    def stage_05b(self, cands: list[dict]) -> None:
        if not self.flags["use_b"]:
            return
        b_cfg = self.cfg["nemotron_b"]
        files = b_cfg["code_context"]["files"]
        code, sizes = NM.code_context(self.av_stack, files) if self.av_stack else ("", {})
        if not sizes:
            code = "(av_stack source not available on this machine; mock stub answers)" if self.mode == "mock" else code
            sizes = {f: 10_000 for f in files} if self.mode == "mock" else sizes
        regions = NM.resolve_regions(self.av_stack, self.questions)
        per_q, static_q = NM.questions_of_scope(self.questions, "per_scenario"), NM.questions_of_scope(self.questions, "static")
        jobs = []
        if static_q and B_STATIC_KEY not in self.b_cache:
            # static: one fixed code fact per campaign (Q-B2 latch, Q-B3 takeover), not per scenario
            jobs.append((B_STATIC_KEY, static_q, {"scope": "static (whole supervisor, any fault)", "planner": "pdm",
                                                  "controller": "supervised"}))
        seen = set()
        for c in cands:
            k = self.b_key(c)
            if k in self.b_cache or k in seen or not per_q:
                continue
            seen.add(k)
            jobs.append((k, per_q, {"fault_mode": c["fault"]["mode"], "fault_params": c["fault"]["params"],
                                    "onset_event": c["onset_event"], "planner": "pdm", "controller": "supervised"}))
        for k, qs, ctx in jobs:
            schema = NM.b_schema(qs, list(sizes))
            samples = []
            for i in range(int(self.b_settings.get("repeats", 3))):
                try:
                    res = self.client_b.chat(self.b_settings.get("model", "stub/B"), NM.b_messages(code, qs, ctx, schema),
                                             temperature=self.b_settings.get("temperature"), top_p=self.b_settings.get("top_p"),
                                             max_tokens=self.b_settings.get("max_tokens"), thinking=self.b_settings.get("thinking"),
                                             chat_template_kwargs=self.b_settings.get("chat_template_kwargs_extra"),
                                             response_format={"type": "json_schema", "json_schema": {"name": "b_answers", "schema": schema}},
                                             tag=f"B:{k}:{i}")
                    from engine.llm import extract_json
                    samples.append(extract_json(res.content))
                except (LLMError, ValueError) as e:
                    self.stats["llm_errors"].append({"role": "B", "key": k, "error": str(e)[:200]})
            self.b_cache[k] = {"answers": NM.aggregate_b(samples, qs, sizes, ctx=ctx, regions=regions),
                               "n_samples": len(samples), "context": ctx}
        self.stats["nemotron_b"] = NM.b_hallucination_summary([e.get("answers") for e in self.b_cache.values()])
        self.stats["b_hallucination_rate"] = self.stats["nemotron_b"]["b_hallucination_rate"]

    # ------------------------------------------------------------------ 06 + 07
    def load_gp_history(self, patterns: list[str]) -> None:
        """--gp-history: pool earlier campaigns' run_records.jsonl into GP training (warm start; result_5 had a
        recovery GP on 2 points). Runs are re-encoded with THIS space (missing new parameters take their catalog
        default, e.g. saturate magnitude 1.0 / ramp 0, friction 1.0). Skipped: invalid runs, routes / faults not in
        this campaign, plans outside the current ranges. They train the GP only; they are never re-labelled."""
        import glob
        files = sorted({f for pat in patterns for f in glob.glob(str(pat), recursive=True)})
        skipped: dict[str, int] = {}
        faults = SC.faults_by_id(self.space)
        for fpath in files:
            for line in Path(fpath).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                why = None
                p = r.get("parameters") or {}
                fault = p.get("fault") or {}
                if r.get("sim_status") != "ok":
                    why = "not sim_status ok"
                elif p.get("route_id") not in self.route_ids:
                    why = "route not in this campaign"
                elif fault.get("fault_id") not in self.fault_pool or fault.get("fault_id") not in faults:
                    why = "fault not in this pool"
                elif (self.runner_cfg.get("gp_history_same_supervisor", True)
                      and str(r.get("supervisor_profile") or "legacy") != self.supervisor):
                    # integration3 #8: a run graded under another supervisor switches at a different time (tight
                    # switches earlier), so its switch / recovery labels do not describe this campaign
                    why = f"supervisor {r.get('supervisor_profile') or 'legacy'} != {self.supervisor}"
                if why is None:
                    spec = faults[fault["fault_id"]].get("params") or {}
                    params = {k: (fault.get("params") or {}).get(k, v.get("default")) for k, v in spec.items()}
                    env = dict(p.get("environment") or {})
                    c = SC.make_candidate(self.space, route_id=p["route_id"], speed=env.get("max_speed_kmh", 30.0),
                                          fault_id=fault["fault_id"], params=params, onset_event=p.get("onset_event"),
                                          offset=p.get("onset_offset_s", 0.0), source="history",
                                          friction=env.get(SC.FRICTION))
                    probs = SC.validate_against_space(self.space, c, self.route_ids, check_exclusions=False)
                    if probs:
                        why = "outside current space: " + probs[0].split(" ")[0]
                if why is not None:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                tg = dict(r.get("gp_targets") or {})
                tg[SWITCH_METRIC] = 1.0 if r.get("switch_time") is not None else -1.0
                self.history_rows.append((c, tg))
        self.stats["gp_history"] = {"files": files, "loaded": len(self.history_rows), "skipped": skipped}
        self.say(f"[06] --gp-history: {len(self.history_rows)} earlier runs from {len(files)} file(s) pooled into GP training"
                 + (f", skipped {skipped}" if skipped else ""))

    def fit_gp(self, round_index: int) -> MetricGPs | None:
        if not self.flags["use_gp"]:
            return None
        rows = []
        for r in self.records:
            c = self.candidates.get(r["scenario_id"])
            if c is None or r.get("sim_status") != "ok":
                continue
            x, _ = self.features(c, train_round=r.get("round", 0) + 1)
            tg = dict(r.get("gp_targets") or {})
            tg[SWITCH_METRIC] = 1.0 if r.get("switch_time") is not None else -1.0
            rows.append((x, tg))
        n_own = len(rows)
        rows += [(self.features(c, train_round=0)[0], tg) for c, tg in self.history_rows]
        gps = MetricGPs(self.gp_metrics, self.seed, switch_metric=SWITCH_METRIC, switch_model=self.switch_model,
                        kernel_opts=self.kernel_opts, min_train_points=self.min_train_points,
                        prior_means=self.gp_prior_means())
        counts = gps.fit(rows)
        self.gps = gps
        # integration3: below this many training runs the GP share of the batch is space filling (06 surrogate.yaml)
        self.gp_ready = len(rows) >= self.min_train_for_acq
        self.stats.setdefault("gp_train_counts", []).append({"round": round_index, **counts, "rows": len(rows),
                                                             "own_runs": n_own, "history_runs": len(rows) - n_own,
                                                             "gp_acquisition_active": self.gp_ready})
        return gps

    def level_z(self, obj: dict) -> float:
        level = float(obj.get("level", 0.0))
        rule = (self.metric_rules or {}).get(obj["metric"])
        return J.orient(level, rule) if rule and level else 0.0

    def gp_prior_means(self) -> dict[str, float]:
        """06 surrogate.yaml prior_mean: level -> level-set objectives' GP prior mean = their level h (z units)."""
        if self.prior_mean_mode != "level":
            return {}
        out = {}
        for o in self.objectives:
            obj = self.obj_cfg[o]
            if obj.get("acquisition") == "level_set":
                out[obj["metric"]] = self.level_z(obj)
        return out

    def target_y_scale(self) -> float:
        m = self.obj_cfg[self.objectives[0]]["metric"]
        return self.gps.y_scale.get(m, 1.0) if self.gps else 1.0

    def summary_scorer(self, c: dict) -> dict | None:
        """integration3 #7: the same terms as acq_score (level_set) for Nemotron A's summary; None when not level_set
        or the GP is not ready (the summary then falls back to its own default score)."""
        obj = self.obj_cfg[self.objectives[0]]
        if not self.gp_ready or obj.get("acquisition") != "level_set":
            return None
        return boundary_score_parts(c, obj["metric"], SWITCH_METRIC, beta=float(obj.get("beta", 1.96)),
                                    level_z=self.level_z(obj), y_scale=self.target_y_scale(),
                                    switch_penalty=float(obj.get("switch_penalty", 1.0)))

    def native(self, mean_z: float, std_z: float) -> tuple[float, float]:
        """GP z -> lane_recovery_robustness (inverse of judgment.orient)."""
        rule = (self.metric_rules or {}).get(self.obj_cfg[self.objectives[0]]["metric"]) or {}
        sc = float((rule.get("gp") or {}).get("scale", 1.0))
        sign = 1.0 if rule.get("support_direction", "negative") == "negative" else -1.0
        return sign * mean_z * sc, std_z * sc

    def gp_info(self) -> dict:
        obj = self.obj_cfg[self.objectives[0]]
        rule = (self.metric_rules or {}).get(obj["metric"]) or {}
        ys = self.target_y_scale()
        return {"y_scale_z": ys, "level_z": self.level_z(obj), "target": (rule.get("gp") or {}).get("target", obj["metric"]),
                "units": "dimensionless lane_recovery_robustness (NOT metres): lane margin / (lane_width/2 - vehicle half width)",
                "level": float(obj.get("level", 0.0)), "training_std": round(self.native(0.0, ys)[1], 3),
                "meaning": self.judgment_criteria()["gp_target"]}

    def acq_score(self, c: dict, beta: float) -> float | None:
        """07 acquisition for the active objective. level_set (integration3 default): engine.surrogate.boundary_score;
        straddle (legacy): switch_aware_score; otherwise LCB. None until the GP has min_train_runs_for_acquisition."""
        if not self.gp_ready:
            return None
        obj = self.obj_cfg[self.objectives[0]]
        if obj.get("acquisition") == "level_set":
            metric = obj["metric"]
            level_z = self.level_z(obj)
            return boundary_score(c, metric, SWITCH_METRIC, beta=float(obj.get("beta", 1.96)), level_z=level_z,
                                  y_scale=(self.gps.y_scale.get(metric, 1.0) if self.gps else 1.0),
                                  switch_penalty=float(obj.get("switch_penalty", 1.0)))
        return switch_aware_score(c, obj, beta)

    def stage_07(self, pool: list[dict], gps: MetricGPs | None, round_index: int) -> list[dict]:
        if gps is not None and pool:
            X = [self.features(c)[0] for c in pool]
            means, stds = gps.predict(X)
            for c, m, s, x in zip(pool, means, stds, X):
                c["gp_mean"], c["gp_std"], c["normalized_features"] = m, s, x
        else:
            for c in pool:
                c["normalized_features"] = SC.numeric_features(c, self.space, self.route_ids, self.fault_pool)[0]
        orch = self.cfg["orchestrator"]
        forced = [c for c in pool if c.get("source") == "counterfactual"][: int(self.vcfg["feedback"]["counterfactual_queue"]["max_per_round"])]
        rest = [c for c in pool if c not in forced]
        if not self.flags["use_gp"]:
            if self.flags["use_a"]:
                ranked = [c for c in rest if c["source"].startswith("nemotron_a")] + [c for c in rest if not c["source"].startswith("nemotron_a")]
            else:
                ranked = rest[:]
                self.rng.shuffle(ranked)
            batch = ranked[: self.batch - len(forced)]
            for c in batch:
                c["selected_by"] = "nemotron_a_order" if self.flags["use_a"] else "random_space_filling"
        else:
            beta = J.beta_schedule(round_index, self.rounds + 1)
            obj = self.obj_cfg[self.objectives[0]]
            div = orch["acquisition"]["diversity"]
            share = self.nemotron_share(rest, self.batch - len(forced), lambda c: self.acq_score(c, beta))
            rest = [c for c in rest if c not in share]
            batch = share + select_batch_scored(rest, lambda c: self.acq_score(c, beta),
                                        batch_size=self.batch - len(forced) - len(share),
                                        acquisition_fraction=float(orch["acquisition"]["batch_composition"]["lcb"]),
                                        min_distance=float(div["min_normalized_distance"]), max_per_seed=int(div["max_per_seed"]),
                                        rng=self.rng, maximize=True)
        for c in forced:
            c["selected_by"] = "counterfactual_check"
        return forced + batch

    def nemotron_share(self, cands: list[dict], n_batch: int, score_of) -> list[dict]:
        """integration3 #4: reserve ceil(fraction * batch) slots for Nemotron A candidates every round (07
        orchestrator.yaml acquisition.batch_composition.nemotron_a, default 0.3). Before, GP + random took the whole
        batch after the cold start, so A's boundary proposals almost never ran. Order: justified / interior candidates
        first (range_extreme_unjustified last), then the acquisition score, then A's own order."""
        frac = float((self.cfg["orchestrator"]["acquisition"].get("batch_composition") or {}).get("nemotron_a", 0.3))
        if not self.flags["use_a"] or frac <= 0 or n_batch <= 0:
            return []
        a = [c for c in cands if c.get("source", "").startswith("nemotron_a")]
        def key(ic):
            i, c = ic
            sc = score_of(c)
            return ("range_extreme_unjustified" in c.get("tags", []), -(sc if sc is not None else -1e9), i)
        picked = [c for _, c in sorted(enumerate(a), key=key)][: math.ceil(n_batch * frac)]
        for c in picked:
            c["selected_by"] = "nemotron_a_share"
        return picked

    def reserve_order(self, pool: list[dict], batch: list[dict], round_index: int) -> list[dict]:
        """BUG-029: replacements for rejected batch candidates (08 baseline gate) in the same acquisition order stage 07
        used, not pool order (pool order puts Nemotron A first, so GP picks were swapped for unranked A candidates)."""
        reserve = [c for c in pool if c not in batch]
        if round_index == 0 or not self.flags["use_gp"]:
            return reserve
        beta = J.beta_schedule(round_index, self.rounds + 1)
        scored = [(self.acq_score(c, beta), c) for c in reserve]
        return [c for s, c in sorted(scored, key=lambda p: -math.inf if p[0] is None else p[0], reverse=True)]

    # ------------------------------------------------------------------ 08 + 09
    def is_control(self, c: dict) -> bool:
        return (c.get("fault") or {}).get("fault_id") in self.control_ids

    def cap_controls(self, batch: list[dict], pool: list[dict], round_index: int = 0) -> list[dict]:
        """BUG-025: keep at most control_cap control-fault candidates in a round's batch; the extras are swapped
        for the next non-control candidates of the pool. When the pool has none left (cold start runs the whole
        pool), new space-filling candidates without control faults are made, gated (04) and asked (05b)."""
        spare = [c for c in pool if c not in batch and not self.is_control(c)]
        extra = sum(self.is_control(c) for c in batch) - self.control_cap
        if extra > len(spare):
            risk_pool = [f for f in self.fault_pool if f not in self.control_ids]
            new = self.stage_04([self.quantize(c) for c in SC.space_filling(
                self.space, self.route_ids, risk_pool, 2 * (extra - len(spare)), self.rng, round_index=round_index)],
                round_index)
            new = [c for c in new if c not in pool and not self.is_control(c)]
            self.stage_05b(new)
            spare += new
        out, n_ctrl = [], 0
        for c in batch:
            if self.is_control(c):
                if n_ctrl >= self.control_cap and spare:
                    s = spare.pop(0)
                    s["selected_by"] = "control_cap_for_" + c["scenario_id"]
                    s.setdefault("normalized_features", SC.numeric_features(s, self.space, self.route_ids, self.fault_pool)[0])
                    out.append(s)
                    continue
                n_ctrl += 1
            out.append(c)
        return out

    def baseline_gate(self, base_dir: Path, onset: float) -> str | None:
        """Reasons the paired baseline cannot grade a fault at `onset` (BUG-003 / BUG-004), or None.

        Same checks the SR monitor applies afterwards: evaluation window long enough (V-03), and the no-fault
        baseline neither leaves the lane nor collides inside that window (G-04). Checking before the fault run
        saves the CARLA time of a pair that would be invalid anyway."""
        from engine import sr_checks as SRC
        import numpy as np
        run = load_av_stack_run(base_dir)
        tr = SRC.trace_from_run(run)
        chk = SRC.M.check_run(tr, onset, SRC.config_from_spec(self.cfg["safety_spec"], dt=run.dt), "baseline")
        if not chk.valid:
            return "; ".join(chk.reasons)
        if float(np.min(chk.margin[chk.win])) < 0:
            return "G-04 baseline footprint leaves the lane in the evaluation window"
        hits = [x for x in tr.collisions_t if chk.window[0] <= x <= chk.window[1]]
        return f"G-04 baseline collides at {hits[0]:.2f} s in the evaluation window" if hits else None

    def stage_08_09(self, batch: list[dict], round_index: int, reserve: list[dict] | None = None) -> list[dict]:
        """Paired runs for the batch. A candidate whose baseline cannot grade it is replaced by the next reserve one."""
        new = []
        mr = self.metric_rules
        queue, reserve = list(batch), list(reserve or [])
        n_ctrl = sum(self.is_control(c) for c in queue)
        while queue:
            c = queue.pop(0)

            def reject(tag: str, problem: str) -> None:
                self.stats["rejected"].append({"scenario_id": c["scenario_id"], "tags": [tag], "problems": [problem]})
                self.say(f"  [08] skip {c['scenario_id']} {c['route_id']} {c['environment']['max_speed_kmh']:.0f}km/h "
                         f"{c['fault']['mode']}: {problem}")
                nonlocal n_ctrl
                n_ctrl -= self.is_control(c)
                # a control-fault reserve only while the round is under its control cap (BUG-025)
                i = next((k for k, r in enumerate(reserve) if not self.is_control(r) or n_ctrl < self.control_cap), None)
                if i is not None:
                    nxt = reserve.pop(i)
                    n_ctrl += self.is_control(nxt)
                    nxt["selected_by"] = "replacement_for_" + c["scenario_id"]
                    nxt.setdefault("normalized_features",
                                   SC.numeric_features(nxt, self.space, self.route_ids, self.fault_pool)[0])
                    queue.append(nxt)

            base_dir, cached = self.runner.ensure_baseline(c)
            if not (base_dir / ".complete").is_file():
                reject("G-04_reject", "baseline failed")
                continue
            ev = self.route_events(c)
            onset = SC.onset_time(c, ev or {})
            if onset is None:
                reject("G-01_reject", f"route has no {c['onset_event']} (baseline event detection)")
                continue
            problem = self.baseline_gate(base_dir, onset)
            if problem:
                reject("G-04_reject" if "G-04" in problem else "V-03_reject", problem)
                continue
            run_id = f"R{round_index:02d}-{len(self.records):04d}-{c['fault']['mode']}"
            c["parameters"] = {"route_id": c["route_id"], "environment": c["environment"], "fault": c["fault"],
                               "onset_event": c["onset_event"], "onset_offset_s": c["onset_offset_s"], "onset_s": round(onset, 3)}
            res = self.runner.run_fault(c, self.out / "runs" / run_id, onset)
            nr_dir = None
            if self.no_reaction_on:
                d, _ = self.runner.ensure_no_reaction(c, onset)
                nr_dir = d if (d / ".complete").is_file() else None
            rec = monitor_evaluate(res.run_dir, self.cfg["safety_spec"], baseline_dir=base_dir, no_reaction_dir=nr_dir,
                                   stage_cfg=self.cfg, metric_rules=mr, candidate=c, run_id=run_id,
                                   round_index=round_index, selector=c.get("selected_by"))
            run = load_av_stack_run(res.run_dir)
            if res.returncode != 0 and rec["sim_status"] == "ok" and not run.ticks:
                rec["sim_status"], rec["invalid_reason"] = "invalid", f"run_benchmark exit {res.returncode}"
            rec["feature_schema_version"] = self.feature_schema()
            self.candidates[c["scenario_id"]] = c
            self.records.append(rec)
            new.append(rec)
            append_jsonl(self.out / "run_records.jsonl", rec)
            self.screen(f"라운드 {round_index}: {run_id} 끝 ({len(new)}/{len(batch)}) · 다음 시뮬레이션 중" if queue
                        else f"라운드 {round_index}: {run_id} 끝 ({len(new)}/{len(batch)})")
            self.say(f"  [08/09] {run_id} {c['route_id']} {c['environment']['max_speed_kmh']:.0f}km/h {c['fault']['mode']} "
                     f"@{c['onset_event']}{c['onset_offset_s']:+.2f}s -> {rec['recovery_outcome']} "
                     f"(lrr={_f(rec['lane_recovery_robustness'])}, trigger={_f(rec['trigger_time'])}s, "
                     f"detected={rec['trigger_source']}, "
                     f"sr={rec.get('sr_outcome')}, sel={c.get('selected_by')})")
        return new

    def feature_schema(self) -> str:
        parts = ["num"] + (["df"] if self.flags["use_df"] else []) + (["b"] if self.flags["use_b"] else []) + \
                (["mech"] if self.video_on else [])
        return "1.1+" + "+".join(parts)

    # ------------------------------------------------------------------ 10 video explainer
    def stage_10(self, new: list[dict], round_index: int) -> None:
        if not self.video_on:
            return
        sel = self.vcfg["select_runs"]
        picks = []
        for r in new:
            lrr = r.get("lane_recovery_robustness")
            near = lrr is not None and abs(lrr) < float(sel["boundary_near_abs_robustness"])
            if (sel["failed"] and is_failure(r)) or near:
                picks.append((0 if is_failure(r) else 1, r))
        picks = [r for _, r in sorted(picks, key=lambda p: p[0])][: int(sel["max_per_round"])]
        clip_cfg = self.vcfg["clip"]
        tol = float(self.vcfg["verification"]["time_tolerance_s"])
        use_cosmos = self.mode != "mock" and bool(picks) and self.c_settings is not self.c_fallback
        keep_on = round_index >= self.rounds and self.insights_next   # insights call Cosmos right after
        ctx = self.cosmos.session(f"stage_10 round {round_index}", keep_on=keep_on) if use_cosmos \
            else contextlib.nullcontext(True)
        with ctx as cosmos_ok:
            if not cosmos_ok:
                self.say("  [10] Cosmos endpoint not ready -> Super-VL fallback for this round")
            for r in picks:
                run = load_av_stack_run(self.out / "runs" / r["run_id"])
                timeline = monitor_timeline(run, r, self.th)
                anchor = r.get("switch_time")
                if anchor is None:
                    anchor = (timeline.get("lane_departure_intervals_s") or [[r.get("planned_fault_onset") or 0.0]])[0][0]
                clip = extract_clip(run.run_dir, self.out / "clips" / r["run_id"], anchor_s=float(anchor),
                                    front_frames=run.front_frames, tick_times=[t["t"] for t in run.ticks], dt=run.dt,
                                    before_s=float(clip_cfg["before_switch_s"]), after_s=float(clip_cfg["after_switch_s"]),
                                    fps=float(clip_cfg["fps"]), max_height=int(clip_cfg["max_height_px"]),
                                    prefer=tuple(clip_cfg["source_preference"]), sheet_cfg=clip_cfg.get("contact_sheet"),
                                    dense_times=key_times(timeline), dense_fps=clip_cfg.get("dense_fps"),
                                    dense_window_s=float(clip_cfg.get("dense_window_s", 1.0)),
                                    cover=cover_times(timeline) if clip_cfg.get("cover_events", True) else None)
                c = self.candidates[r["scenario_id"]]
                entry = {"run_id": r["run_id"], "scenario_id": r["scenario_id"], "round": round_index, "route_id": c["route_id"],
                         "onset_event": c["onset_event"], "fault_mode": c["fault"]["mode"], "timeline": timeline,
                         "monitor_labels_snapshot": {k: r[k] for k in ("recovery_outcome", "recovery_labels", "final_safe_recovery")}}
                if clip is None:
                    entry.update(explanation=None, verification={"accepted": False, "reason": "no_video_or_ffmpeg"}, clip=None)
                else:
                    entry["clip"] = clip.as_dict()
                    if self.mode == "mock":
                        expl = stub_explanation(timeline, clip, self.rng)
                        result = {"explanation": expl, "media_used": "video", "model": "stub/explainer (SYNTHETIC)", "errors": []}
                    else:
                        vis = tuple((self.vcfg.get("prompt") or {}).get("visible_timeline_keys") or ())
                        result = explain(self.client_c, self.c_settings, clip, timeline, self.taxonomy, tag="C", visible_keys=vis or None) \
                            if cosmos_ok else {"explanation": None, "errors": ["cosmos endpoint not ready (--cosmos-power auto)"]}
                        if result["explanation"] is None and self.c_settings is not self.c_fallback:
                            fb = TokenFactoryClient.from_config(self.cfg["llm"], self.llm_log)
                            result = {**explain(fb, self.c_fallback, clip, timeline, self.taxonomy, tag="C_fallback", visible_keys=vis or None),
                                      "primary_errors": result["errors"]}
                    entry.update(result)
                    entry["verification"] = verify(entry.get("explanation"), timeline, clip, self.taxonomy, tol,
                                                   float(self.vcfg["verification"].get("visible_departure_margin_m", 0.0)))
                assert entry["monitor_labels_snapshot"]["recovery_outcome"] == r["recovery_outcome"]  # explainer never edits labels
                self.explanations[r["run_id"]] = entry
                append_jsonl(self.out / "explanations.jsonl", entry)
                v = entry["verification"]
                self.say(f"  [10] {r['run_id']} explained via {entry.get('media_used')} -> "
                         f"{'ACCEPTED' if v.get('accepted') else 'rejected'} {[k for k, x in (v.get('checks') or {}).items() if not x['pass']]}")
        fb = self.vcfg["feedback"]
        if fb["counterfactual_queue"]["enabled"]:
            self.cf_queue += counterfactual_candidates(self.explanations, self.candidates, self.space, round_index + 1,
                                                       int(fb["counterfactual_queue"]["max_per_round"]))
        props = taxonomy_proposals(self.explanations, int(fb["taxonomy_builder"]["min_cluster_size"]))
        (self.out / fb["taxonomy_builder"]["output"]).write_text(yaml.safe_dump(props, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ------------------------------------------------------------------ loop
    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        dump(self.out / "run_config.json", {"mode": self.mode, "ablation": self.ablation, "flags": self.flags,
                                            "video_explainer": self.video_on, "seed": self.seed, "cold_start": self.cold,
                                            "supervisor": self.supervisor,
                                            "batch_size": self.batch, "rounds": self.rounds, "synthetic": self.mode == "mock",
                                            "scenario_space": str(self.space_path), "av_stack": str(self.av_stack)})
        self.screen("시작: 사전 지식 정리 중")
        if self.open_screen:
            self.open_browser()
        self.stage_00_02()
        mix = self.cfg["orchestrator"]["cold_start"]["mix"]
        for rnd in range(self.rounds + 1):
            rdir = self.out / f"round_{rnd:02d}"
            if rnd == 0:
                n_a = int(round(self.cold * float(mix["nemotron_a"]))) if self.flags["use_a"] else 0
                if self.ablation == "X-2":
                    n_a = self.cold
                self.screen(f"라운드 {rnd}: Nemotron A가 시나리오 만드는 중")
                a_c = self.stage_03(n_a, rnd)
                sf = SC.space_filling(self.space, self.route_ids, self.fault_pool, self.cold - len(a_c) + 6, self.rng, round_index=rnd)
                pool = self.stage_04([self.quantize(c) for c in a_c + sf], rnd)
                target = self.cold
                gps = None
            else:
                gps = self.fit_gp(rnd)
                near = []
                if gps is not None and self.records:
                    prev = [c for c in self.candidates.values()]
                    X = [self.features(c)[0] for c in prev]
                    means, stds = gps.predict(X)
                    m = self.obj_cfg[self.objectives[0]]["metric"]
                    beta = J.beta_schedule(rnd, self.rounds + 1)
                    # integration3: boundary candidates = highest level-set ambiguity (falls back to |mean| when the
                    # GP is not ready yet)
                    def near_key(mu: dict, sd: dict) -> float:
                        sc = self.acq_score({"gp_mean": mu, "gp_std": sd}, beta)
                        return -sc if sc is not None else abs(mu.get(m, 1e9))
                    scored = sorted(((near_key(mu, sd), c, mu, sd) for c, mu, sd in zip(prev, means, stds)),
                                    key=lambda p: p[0])
                    for _, c, mu, sd in scored[: int(self.cfg["nemotron_a"].get("refine", {}).get("max_candidates", 3))]:
                        near.append({**c, "gp_mean": mu, "gp_std": sd})
                    regions = [{**c, "gp_mean": mu, "gp_std": sd} for c, mu, sd in zip(prev, means, stds)]
                else:
                    regions = []
                # integration3 #4: Nemotron A gets a compact GP summary EVERY round (boundary regions + outcome mix)
                sv2 = self.cfg["nemotron_a"].get("summary_v2") or {}
                gp_summary = NM.surrogate_summary(
                    regions=regions, records=self.records, candidates=self.candidates, space=self.space,
                    metric=self.obj_cfg[self.objectives[0]]["metric"], switch_metric=SWITCH_METRIC,
                    n_train=sum(r.get("sim_status") == "ok" for r in self.records) if gps is not None else None,
                    k=int(self.cfg["nemotron_a"].get("surrogate_summary", {}).get("k", 6)),
                    scorer=self.summary_scorer if (self.gp_ready and self.obj_cfg[self.objectives[0]].get("acquisition") == "level_set") else None,
                    native=self.native, gp_info=self.gp_info(),
                    k_unexplored=int(sv2.get("k_unexplored", 3)), unexplored_std_frac=float(sv2.get("unexplored_std_frac", 0.75)))
                dump(rdir / "03_gp_summary.json", gp_summary)
                n_a = int(self.cfg["nemotron_a"]["A_scenario_generation"]["per_round"]) if self.flags["use_a"] else 0
                self.screen(f"라운드 {rnd}: Nemotron A가 시나리오 만드는 중")
                a_c = self.stage_03(n_a, rnd, near=near if self.flags["use_gp"] else None, gp_summary=gp_summary)
                sf = SC.space_filling(self.space, self.route_ids, self.fault_pool,
                                      int(self.cfg["orchestrator"]["candidate_pool"]["space_filling_per_round"]), self.rng, round_index=rnd)
                cf, self.cf_queue = self.cf_queue, []
                pool = self.stage_04([self.quantize(c) for c in a_c + sf] + cf, rnd)
                target = self.batch
            dump(rdir / "03_04_pool.json", [{k: c.get(k) for k in ("scenario_id", "source", "route_id", "environment", "fault",
                                                                    "onset_event", "onset_offset_s", "tags", "evidence_ids",
                                                                    "expected_mechanism", "expected_side", "range_extremes",
                                                                    "extreme_justification")} for c in pool])
            self.screen(f"라운드 {rnd}: Nemotron B가 후보 {len(pool)}개 검토 중")
            self.stage_05b(pool)
            dump(rdir / "05b_answers.json", self.b_cache)
            if rnd == 0:
                a_first = [c for c in pool if c["source"].startswith("nemotron_a")]
                others = [c for c in pool if not c["source"].startswith("nemotron_a")]
                batch = (a_first + others)[:target]
                for c in batch:
                    c["selected_by"] = "cold_start_" + ("nemotron_a" if c["source"].startswith("nemotron_a") else "space_filling")
                    c["normalized_features"] = SC.numeric_features(c, self.space, self.route_ids, self.fault_pool)[0]
            else:
                batch = self.stage_07(pool, gps, rnd)
            batch = self.cap_controls(batch, pool, rnd)
            dump(rdir / "07_batch.json", [{k: c.get(k) for k in ("scenario_id", "source", "selected_by", "gp_mean", "gp_std")} for c in batch])
            self.say(f"[round {rnd}] pool {len(pool)} -> batch {len(batch)} ({self.ablation}, video {'on' if self.video_on else 'off'})")
            self.screen(f"라운드 {rnd}: 시뮬레이션 {len(batch)}개 시작")
            new = self.stage_08_09(batch, rnd, reserve=self.reserve_order(pool, batch, rnd))
            if self.video_on and new:
                self.screen(f"라운드 {rnd}: 영상 해설 중")
            self.stage_10(new, rnd)
            self.screen(f"라운드 {rnd} 끝" + (" · 다음 라운드 준비 중" if rnd < self.rounds else ""))
        return self.finish()

    # ------------------------------------------------------------------ evaluation + report
    def finish(self) -> dict:
        table = failure_table(self.records, self.candidates)
        with (self.out / "failure_table.jsonl").open("w", encoding="utf-8") as fh:
            for row in table:
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        metrics = evaluate(self.records, self.candidates, self.explanations, self)
        dump(self.out / "evaluation_metrics.json", metrics)
        dump(self.out / "loop_stats.json", self.stats)
        write_report(self.out, metrics, self)
        return metrics


def _f(x):
    return "None" if x is None else f"{x:.2f}"


def evaluate(records: list[dict], candidates: dict, explanations: dict, loop: Loop | None = None) -> dict:
    """Boundary-estimation quality (evaluation/ablations.yaml metrics, v2)."""
    valid = [r for r in records if r.get("sim_status") == "ok"]
    switched = [r for r in valid if r.get("switch_time") is not None]
    safe = [r for r in switched if r.get("final_safe_recovery") is True]
    fail = [r for r in valid if is_failure(r)]
    first_fail = next((i + 1 for i, r in enumerate(records) if is_failure(r)), None)
    lrr = [r["lane_recovery_robustness"] for r in switched if r.get("lane_recovery_robustness") is not None]
    band = 0.2
    uniq = {(candidates[r["scenario_id"]]["route_id"], candidates[r["scenario_id"]]["fault"]["mode"],
             candidates[r["scenario_id"]]["onset_event"], int(candidates[r["scenario_id"]]["environment"]["max_speed_kmh"] // 10))
            for r in fail if r["scenario_id"] in candidates}
    expl = list(explanations.values())
    acc = [e for e in expl if (e.get("verification") or {}).get("accepted")]
    m = {
        "n_runs": len(records), "n_valid": len(valid), "n_switched": len(switched),
        "n_final_safe_recovery": len(safe), "n_failures": len(fail),
        "both_sides_observed": bool(safe) and any(r.get("final_safe_recovery") is False for r in switched),
        "boundary_band_runs": sum(abs(x) < band for x in lrr), "boundary_band": band,
        "failures_per_simulation": len(fail) / max(1, len(records)), "simulations_to_first_failure": first_fail,
        "unique_failure_conditions": len(uniq),
        "outcome_counts": _count(r.get("recovery_outcome") for r in records),
        "trigger_counts": _count(r.get("trigger_source") for r in records),
        "sr_outcome_counts": _count(r.get("sr_outcome") for r in records),
        "explanations": len(expl), "explanations_accepted": len(acc),
        "explanation_acceptance_rate": (len(acc) / len(expl)) if expl else None,
        "hallucination_rate": (1 - len(acc) / len(expl)) if expl else None,
        "boundary_metrics": boundary_metrics(records, candidates, band),
    }
    if loop is not None:
        m["gp_boundary_sign_accuracy"] = boundary_accuracy(loop)
        m["a_schema_valid_items"] = sum(s.get("items_valid", 0) for s in loop.stats["a_calls"])
        m["a_items_returned"] = sum(s.get("items_returned", 0) for s in loop.stats["a_calls"])
        m["rejected_candidates"] = len(loop.stats["rejected"])
        m["nemotron_a"] = nemotron_a_metrics(loop)
        m["nemotron_b"] = nemotron_b_metrics(loop)
        m["llm_usage"] = llm_usage(loop.llm_log)
    return m


def nemotron_b_metrics(loop: Loop) -> dict:
    """integration3 #6: B hallucination check (majority answer vs expected_from_code) and how much the B one-hot
    columns actually vary over the GP training runs (a column constant over all runs carries no information)."""
    if not loop.flags["use_b"]:
        return {"enabled": False}
    out = NM.b_hallucination_summary([e.get("answers") for e in loop.b_cache.values()])
    out["per_question"] = {q: {k: d[k] for k in ("asked", "checked", "hallucinated", "distinct_answers", "answers")}
                           for q, d in out["per_question"].items()}
    cols = None
    rows = []
    for r in loop.records:
        c = loop.candidates.get(r.get("scenario_id"))
        if r.get("sim_status") != "ok" or c is None:
            continue
        bx, bn = NM.b_features((loop.b_cache.get(loop.b_key(c)) or {}).get("answers"), loop.questions)
        cols = bn; rows.append(bx)
    if rows:
        const = [n for j, n in enumerate(cols) if len({row[j] for row in rows}) <= 1]
        out["gp_b_columns"] = len(cols)
        out["gp_b_constant_columns"] = len(const)
        out["gp_b_constant_fraction"] = len(const) / len(cols) if cols else None
        out["gp_b_questions"] = sorted({n.split("=")[0] for n in cols})
    out["b_keys"] = sum(1 for k in loop.b_cache if k != B_STATIC_KEY)
    return out


def nemotron_a_metrics(loop: Loop) -> dict:
    """integration3 #4: how much Nemotron A contributes and how close its proposals sit to the range ends.
    edge distance = smallest unit distance of any continuous field to its range end (0 = at an end, 0.5 = mid)."""
    a = [c for c in loop.candidates.values() if str(c.get("source", "")).startswith("nemotron_a")]
    ran = {r["scenario_id"] for r in loop.records}
    ex = [c for c in a if c["scenario_id"] in ran]
    d = [x for x in (NM.edge_distance(c, loop.space) for c in a) if x is not None]
    after = [r for r in loop.records if (r.get("round") or 0) > 0]
    return {"proposals_in_pool": len(a), "executed": len(ex),
            "share_of_executed_runs": len(ex) / len(loop.records) if loop.records else None,
            "share_after_cold_start": (sum(str((loop.candidates.get(r["scenario_id"]) or {}).get("source", "")).startswith("nemotron_a")
                                           for r in after) / len(after)) if after else None,
            "proposals_with_range_extreme": sum(bool(NM.extreme_fields(c, loop.space)) for c in a),
            "proposals_extreme_unjustified": sum("range_extreme_unjustified" in c.get("tags", []) for c in a),
            "mean_edge_distance": (sum(d) / len(d)) if d else None,
            "expected_side": _count(c.get("expected_side") for c in a)}


def boundary_metrics(records: list[dict], candidates: dict, band: float = 0.2) -> dict:
    """integration3: is the campaign finding a boundary at all? (result_1..5: every fault mode had a single outcome)

    boundary_band_runs: switched runs with |lane_recovery_robustness| < band (also for 0.5 m);
    per_fault_mode: n, recovery_outcome mix, final_safe_recovery count and whether both safe and unsafe occurred;
    gp_prediction_diversity: distinct gp_mean values the executed runs were chosen with (constant = GP not learning);
    selection_sources: candidate source of the executed runs (Nemotron A share)."""
    lrr = [r["lane_recovery_robustness"] for r in records if r.get("lane_recovery_robustness") is not None]
    per: dict[str, dict] = {}
    for r in records:
        if r.get("sim_status") != "ok":
            continue
        mode = (((r.get("parameters") or {}).get("fault") or {}).get("mode")) or "?"
        d = per.setdefault(mode, {"n": 0, "outcomes": {}, "final_safe": 0})
        d["n"] += 1
        d["outcomes"][str(r.get("recovery_outcome"))] = d["outcomes"].get(str(r.get("recovery_outcome")), 0) + 1
        d["final_safe"] += r.get("final_safe_recovery") is True
    for d in per.values():
        d["mixed"] = 0 < d["final_safe"] < d["n"]
    div = {}
    for metric in ("recovery_success", SWITCH_METRIC):
        v = [round(float((r.get("gp_mean") or {})[metric]), 6) for r in records
             if (r.get("gp_mean") or {}).get(metric) is not None]
        if v:
            mean = sum(v) / len(v)
            div[metric] = {"n": len(v), "distinct": len(set(v)), "min": min(v), "max": max(v),
                           "std": (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5}
    src = _count(str((candidates.get(r["scenario_id"]) or {}).get("source")) for r in records)
    n_a = sum(v for k, v in src.items() if k.startswith("nemotron_a"))
    return {"band": band, "boundary_band_runs": sum(abs(x) < band for x in lrr),
            "boundary_band_runs_0.5": sum(abs(x) < 0.5 for x in lrr), "n_lrr": len(lrr),
            "per_fault_mode": per, "modes_with_mixed_outcome": sorted(k for k, d in per.items() if d["mixed"]),
            "gp_prediction_diversity": div, "selection_sources": src,
            "nemotron_a_share": (n_a / len(records)) if records else None}


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def llm_usage(path: Path) -> dict:
    out: dict = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        role = (d.get("tag") or "?").split(":")[0]
        u = d.get("usage") or {}
        o = out.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "stub": bool(d.get("stub"))})
        o["calls"] += 1
        o["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        o["completion_tokens"] += int(u.get("completion_tokens") or 0)
    return out


def boundary_accuracy(loop: Loop) -> dict | None:
    """Sign accuracy of the final S1 GP (mean_z of lane_recovery_robustness) against held-out truth.

    mock: truth from the SYNTHETIC simulator on 40 fixed random scenarios. real: leave-one-out on observed runs.
    """
    if not loop.flags["use_gp"]:
        return None
    metric = loop.obj_cfg[loop.objectives[0]]["metric"]
    rows = [(loop.candidates[r["scenario_id"]], r) for r in loop.records
            if r.get("sim_status") == "ok" and metric in (r.get("gp_targets") or {})]
    if len(rows) < 6:
        return {"n": len(rows), "accuracy": None, "note": "too few switched runs"}
    if loop.mode == "mock":
        gps = loop.fit_gp(loop.rounds + 1)
        if metric not in gps.models:
            return None
        rng = random.Random(1234)
        test = [loop.quantize(c) for c in SC.space_filling(loop.space, loop.route_ids, loop.fault_pool, 40, rng)]
        tmp = loop.out / "_truth"
        ok = n = 0
        pairs: list[tuple[bool, bool]] = []
        for i, c in enumerate(test):
            base, _ = loop.runner.ensure_baseline(c)
            onset = SC.onset_time(c, loop.route_events(c) or {})
            if onset is None:
                continue
            from engine.fixtures.synthetic_sim import write_run_folder
            d = write_run_folder(tmp / f"t{i}", route=c["route_id"], max_speed_kmh=c["environment"]["max_speed_kmh"],
                                 fault=SC.fault_dict_for_sim(loop.space, c, onset), seed=loop.runner.seed,
                                 stop_after_fault_s=loop.runner.stop_after_fault_s, record_front=False,
                                 tire_friction_scale=SC.friction_of(c))
            rec = compute_run_record(load_av_stack_run(d), load_av_stack_run(base), candidate=c, metric_rules=loop.metric_rules,
                                     th=scale_thresholds_for_friction(loop.th, SC.friction_of(c)))
            y = (rec.get("gp_targets") or {}).get(metric)
            if y is None:
                continue
            mu = gps.predict([loop.features(c)[0]])[0][0].get(metric)
            n += 1; ok += int((mu >= 0) == (y >= 0))
            pairs.append((mu >= 0, y >= 0))
        shutil.rmtree(tmp, ignore_errors=True)
        return {"n": n, "accuracy": ok / n if n else None, "truth": "SYNTHETIC simulator (mock)", **_sign_detail(pairs)}
    ok = 0
    pairs = []
    for i in range(len(rows)):
        train = rows[:i] + rows[i + 1:]
        g = MetricGPs([metric], loop.seed, kernel_opts=loop.kernel_opts, prior_means=loop.gp_prior_means())
        g.fit([(loop.features(c)[0], r["gp_targets"]) for c, r in train])
        if metric not in g.models:
            return None
        mu = g.predict([loop.features(rows[i][0])[0]])[0][0][metric]
        ok += int((mu >= 0) == (rows[i][1]["gp_targets"][metric] >= 0))
        pairs.append((mu >= 0, rows[i][1]["gp_targets"][metric] >= 0))
    return {"n": len(rows), "accuracy": ok / len(rows), "truth": "leave-one-out on observed runs", **_sign_detail(pairs)}


def _sign_detail(pairs: list[tuple[bool, bool]]) -> dict:
    """integration3: plain sign accuracy is meaningless when the truth is one-sided (all failures -> 'always fail'
    scores 1.0). Report the truth balance, the majority-class baseline and balanced accuracy too."""
    if not pairs:
        return {}
    pos = [p for p, t in pairs if t]
    neg = [p for p, t in pairs if not t]
    tpr = (sum(pos) / len(pos)) if pos else None
    tnr = (sum(not p for p in neg) / len(neg)) if neg else None
    return {"truth_safe_fraction": len(pos) / len(pairs),
            "majority_baseline": max(len(pos), len(neg)) / len(pairs),
            "balanced_accuracy": ((tpr + tnr) / 2) if tpr is not None and tnr is not None else None,
            "predicted_safe_fraction": sum(p for p, _ in pairs) / len(pairs)}


def write_report(out: Path, m: dict, loop: Loop) -> None:
    syn = "SYNTHETIC (mock 모드: CARLA·LLM 호출 없음, 수치는 실측이 아님)" if loop.mode == "mock" else "실측 (CARLA + Token Factory)"
    lines = [f"# 파이프라인 실행 보고서 ({loop.ablation} {ABLATIONS[loop.ablation]['name']}, 영상 해설 {'on' if loop.video_on else 'off'})",
             "", f"- 데이터: **{syn}**", f"- 출력 폴더: `{out}`",
             f"- 런 {m['n_runs']}개 (유효 {m['n_valid']}, 전환 {m['n_switched']}), final_safe_recovery {m['n_final_safe_recovery']}개, 실패 {m['n_failures']}개",
             f"- 경계 양쪽 관측: {'예' if m['both_sides_observed'] else '아니오 (경계 미추정)'}; |lane_recovery_robustness| < {m['boundary_band']} 런 {m['boundary_band_runs']}개",
             f"- 첫 실패까지 런 수: {m['simulations_to_first_failure']}, 고유 실패 조건: {m['unique_failure_conditions']}",
             f"- GP 경계 부호 정확도: {m.get('gp_boundary_sign_accuracy')}",
             f"- 영상 해설: {m['explanations']}건, 검증 통과 {m['explanations_accepted']}건 (환각률 {m['hallucination_rate']})",
             f"- recovery_outcome 분포: {m['outcome_counts']}", f"- trigger_source 분포: {m['trigger_counts']}",
             f"- SR 검사 (engine/sr_checks.py, 참고) sr_outcome 분포: {m.get('sr_outcome_counts')}; 화면: `screen/campaign.html`",
             "", "## 경계 지표 (integration3)"] + _boundary_lines(m.get("boundary_metrics") or {}) + ["",
             "## 주의", f"- 판정 임계값({loop.th['e_settle_m']} m, {loop.th['psi_settle_rad']} rad, {loop.th['settle_hold_s']} s 유지, "
             f"{loop.th['window_s']} s 창, {loop.th['lat_acc_threshold_mps2']} m/s², {loop.th['lat_jerk_threshold_mps3']} m/s³, "
             f"충돌 {loop.th['collision_penalty']}; 09 safety_spec.yaml)은 PROVISIONAL 이다. 인증 기준이 아니다.",
             "- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.",
             "- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.",
             "- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함)."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _boundary_lines(b: dict) -> list[str]:
    if not b:
        return ["- (없음)"]
    lines = [f"- |lane_recovery_robustness| < {b['band']}: {b['boundary_band_runs']}개, < 0.5: {b['boundary_band_runs_0.5']}개 "
             f"(복구 지표가 있는 런 {b['n_lrr']}개)",
             f"- 같은 고장 안에서 안전·불안전이 둘 다 나온 고장: {b['modes_with_mixed_outcome'] or '없음 (결과가 고장 종류만으로 갈림)'}",
             f"- Nemotron A 후보 비율 (실행된 런): {b['nemotron_a_share']}; 출처: {b['selection_sources']}"]
    for mode, d in sorted((b.get("per_fault_mode") or {}).items()):
        lines.append(f"  - {mode}: {d['final_safe']}/{d['n']} final_safe, {d['outcomes']}")
    for metric, d in (b.get("gp_prediction_diversity") or {}).items():
        lines.append(f"- GP 예측 다양성 {metric}: 런 {d['n']}개 중 서로 다른 값 {d['distinct']}개, 범위 [{d['min']}, {d['max']}]")
    return lines


def preflight(cfg: dict) -> int:
    ok = True
    av = resolve_av_stack()
    log(f"av_stack: {av or 'NOT FOUND (merge junwoo/sim-eval or set AV_STACK_ROOT)'}")
    ok &= av is not None
    try:
        p, syn = resolve_scenario_space(allow_fixture=False)
        log(f"scenario space: {p}")
    except FileNotFoundError as e:
        log(f"scenario space: {e}"); ok = False
    rc = cfg["paired_runs"]["runner"]
    if av is not None:
        from engine.carla_runner import default_python
        py = os.environ.get("AV_STACK_PYTHON") or rc.get("python_exe") or default_python(av)
        log(f"av_stack python: {py} ({'exists' if Path(py).exists() else 'MISSING'})")
    host, port = os.environ.get("CARLA_HOST", rc["host"]), int(os.environ.get("CARLA_PORT", rc["port"]))
    reach = carla_reachable(host, port)
    log(f"CARLA {host}:{port}: {'reachable' if reach else 'NOT reachable (start_carla_stable.ps1)'}")
    ok &= reach
    log(f"ffmpeg: {ffmpeg_exe() or 'NOT FOUND (pip install imageio-ffmpeg)'}")
    llm_cfg = cfg["llm"]
    client = TokenFactoryClient.from_config(llm_cfg)
    if not client.has_key:
        log(f"{llm_cfg['provider']['api_key_env']}: NOT SET"); return 2
    try:
        models = set(client.list_models())
    except LLMError as e:
        log(f"model list failed: {e}"); return 2
    for role in ("A", "A_refine", "B", "C_cosmos", "C_fallback"):
        m = role_settings(llm_cfg, role).get("model")
        log(f"  role {role:10s} {m}: {'listed' if m in models else ('not configured' if not m else 'NOT in /v1/models (Dedicated endpoint?)')}")
    return 0 if ok else 2


def run_suite(args) -> int:
    cfg = load_stage_configs()
    suite = cfg["ablations"]["mock_suite"]
    conds = args.conditions or suite["conditions"]
    seeds = list(range(args.repeats or int(cfg["ablations"]["ablations"]["repeats_with_different_seeds"])))
    root = Path(args.out or REPO_ROOT / "outputs" / f"mock_suite_{time.strftime('%Y%m%d_%H%M%S')}")
    rows = []
    for cond in conds:
        for s in seeds:
            out = root / f"{cond}_seed{s}"
            loop = Loop(mode="mock", out=out, ablation=cond, video_on=(cond == "X-6"), seed=s, cfg=cfg,
                        cold_start=suite["cold_start"], batch_size=suite["batch_size"], rounds=suite["rounds"], quiet=True)
            m = loop.run()
            rows.append({"condition": cond, "seed": s, **{k: m.get(k) for k in suite["report_metrics"]},
                         "gp_boundary_sign_accuracy": (m.get("gp_boundary_sign_accuracy") or {}).get("accuracy")})
            log(f"{cond} seed {s}: " + ", ".join(f"{k}={rows[-1][k]}" for k in list(rows[-1])[2:]))
    dump(root / "ablation_summary.json", rows)
    log(f"wrote {root / 'ablation_summary.json'} (SYNTHETIC)")
    return 0


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win. Values are never printed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not v.startswith("replace-with") and k not in os.environ:
            os.environ[k] = v


def campaign_insights(out: Path, mode: str, cfg: dict, cosmos: CosmosGuard | None = None) -> dict | None:
    """09 monitor Cosmos campaign insights (pipeline/09_monitor/explain.py) after the loop -> out/insights/.
    Text-only (monitor numbers, no video). mock: no network. Never fails the run."""
    if mode != "mock" and cosmos is not None and cosmos.power is not None:
        with cosmos.session("campaign insights") as ok:
            if not ok:
                log("campaign insights skipped: Cosmos endpoint not ready (--cosmos-power auto); "
                    "later: python -m engine.insights " + str(out))
                return None
            return campaign_insights(out, mode, cfg, None)
    try:
        from engine.insights import run_insights
        res = run_insights(out, mode=mode, llm_cfg=cfg["llm"])
    except Exception as e:  # noqa: BLE001 - insights are a report add-on; run results are already written
        log(f"campaign insights failed (run results unaffected): {type(e).__name__}: {e}")
        return None
    log(f"campaign insights ({res.get('role') or mode}): {res['status']}, {res.get('n_insights', 0)} insights, "
        f"{res.get('model_calls', 0)} model calls -> {res['path']}" + (f"  error: {res['error']}" if res.get("error") else ""))
    try:
        with (out / "report.md").open("a", encoding="utf-8") as fh:
            fh.write(f"\n## 캠페인 인사이트 (09 monitor explain.py, Cosmos 텍스트 분석)\n"
                     f"- 상태: {res['status']}, 인사이트 {res.get('n_insights', 0)}개, 모델 호출 {res.get('model_calls', 0)}회"
                     + (f", 오류: {res['error']}" if res.get("error") else "") + "\n"
                     f"- 결과: `insights/insights.json`, 화면: `insights/campaign.html` (SR 모니터 기준 판정, 참고용)\n")
    except OSError as e:
        log(f"report.md insights line not written: {e}")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="synthetic fixtures + stub LLMs, no CARLA, no network")
    g.add_argument("--real", action="store_true", help="CARLA (sim-eval av_stack) + Nebius Token Factory")
    ap.add_argument("--check", action="store_true",
                    help="alone: stage-YAML null report (exit 2 while nulls remain); with --real: CARLA/API preflight only")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ablation", default="X-5", choices=sorted(ABLATIONS))
    ap.add_argument("--video-explainer", choices=("on", "off"), default="on")
    ap.add_argument("--insights", choices=("on", "off"), default="on",
                    help="09 monitor Cosmos campaign insights after the loop -> OUT/insights/ (mock: no network)")
    ap.add_argument("--cosmos-power", choices=("auto", "manual"), default="auto",
                    help="auto: start the Cosmos dedicated endpoint only for stage 10 / insights and stop it after "
                         "(and at exit); manual: never touch it (Nebius console). mock: no API calls")
    ap.add_argument("--cosmos-start-timeout", type=float, default=15.0, help="minutes to wait for Cosmos ready (auto)")
    ap.add_argument("--open-screen", choices=("on", "off"), default="on",
                    help="open OUT/screen/campaign.html in the default browser at start; it is re-drawn after every run "
                         "and stage, and the page reloads itself every 5 s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold-start", type=int, help="cold start runs (default: orchestrator.yaml cold_start rule)")
    ap.add_argument("--supervisor", choices=("tight", "legacy"),
                    help="av_stack supervisor profile (default: paired_runs.yaml runner.supervisor, tight)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--rounds", type=int, help="rounds after the cold start (default from orchestrator budget)")
    ap.add_argument("--ablation-suite", action="store_true", help="mock only: run evaluation/ablations.yaml mock_suite")
    ap.add_argument("--no-reaction-runs", choices=("on", "off"),
                    help="SR1 FTTI runs (same fault, --controller mpc_unprotected); default 08 paired_runs.yaml no_reaction_run")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--gp-history", action="append", metavar="GLOB",
                    help="pool earlier campaigns' run_records.jsonl (glob, repeatable) into GP training (warm start)")
    args = ap.parse_args(argv)
    if args.check and not args.real:
        from engine.config_check import cmd_check
        return cmd_check()
    if not (args.mock or args.real):
        ap.error("one of --mock / --real is required (or --check alone)")
    if args.real:
        load_dotenv()
    if args.real and args.check:
        return preflight(load_stage_configs())
    if args.ablation_suite:
        if not args.mock:
            ap.error("--ablation-suite is mock only (real runs cost CARLA time; run conditions one by one)")
        return run_suite(args)
    mode = "mock" if args.mock else "real"
    out = args.out or REPO_ROOT / "outputs" / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.real:
        cfg = load_stage_configs()
        if not TokenFactoryClient.from_config(cfg["llm"]).has_key:
            ap.error("NEBIUS_API_KEY is not set")
    loop = Loop(mode=mode, out=out, ablation=args.ablation, video_on=args.video_explainer == "on", seed=args.seed,
                cold_start=args.cold_start, batch_size=args.batch_size, rounds=args.rounds,
                no_reaction=None if args.no_reaction_runs is None else args.no_reaction_runs == "on",
                gp_history=args.gp_history, supervisor=args.supervisor)
    loop.insights_next = args.insights == "on"
    loop.open_screen = args.open_screen == "on"
    if args.real and args.cosmos_power == "auto":
        power = CosmosPower.from_env(role_settings(loop.cfg["llm"], "C_cosmos").get("model"), log=log,
                                     timeout_s=args.cosmos_start_timeout * 60)
        loop.cosmos = CosmosGuard(power, log=log)
        if power is None:
            log("[cosmos-power] auto: Cosmos model is not a dedicated endpoint (or no key) -> nothing to switch")
        else:
            atexit.register(loop.cosmos.shutdown, "process exit")
            log(f"[cosmos-power] auto: endpoint {power.routing_key} is started only for stage 10 / insights")
    try:
        m = loop.run()
        if args.insights == "on":
            loop.screen("캠페인 인사이트 만드는 중")
        ins = campaign_insights(out, mode, loop.cfg, loop.cosmos) if args.insights == "on" else None
    except BaseException:
        loop.screen("중단됨 (오류 또는 Ctrl+C)")
        raise
    finally:  # also on Ctrl+C / crash: never leave the paid endpoint running
        loop.cosmos.shutdown("end of run")
    loop.screen("완료")  # final redraw also carries the insights section
    log(json.dumps({k: m[k] for k in ("n_runs", "n_switched", "n_final_safe_recovery", "n_failures", "both_sides_observed",
                                      "boundary_band_runs", "explanations", "explanations_accepted")}, ensure_ascii=False))
    log(f"outputs: {out}  (report.md, run_records.jsonl, screen/campaign.html" + (", explanations.jsonl, monitor_rule_candidates.yaml" if loop.video_on else "")
        + (f", insights/{Path(ins['path']).name}, insights/campaign.html" if ins else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
