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
import json
import math
import os
import random
import shutil
import sys
import time
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
from engine.feedback import (counterfactual_candidates, failure_table, few_shot, is_failure,
                                      mechanism_features, mechanism_summary, taxonomy_proposals)
from engine.llm import LLMError, StubClient, TokenFactoryClient, role_settings
from engine.monitor import compute_run_record, load_av_stack_run, monitor_timeline, thresholds_from_config
from engine.surrogate import MetricGPs, acquisition_score, select_batch_scored
from engine.video import extract_clip, ffmpeg_exe
from engine.prior_tools.prior_tools import make_nemotron_payload

ABLATIONS = {
    "X-1": dict(name="random_space_filling", use_a=False, use_gp=False, use_df=False, use_b=False),
    "X-2": dict(name="nemotron_a_only", use_a=True, use_gp=False, use_df=False, use_b=False),
    "X-3": dict(name="gp_numeric_only", use_a=True, use_gp=True, use_df=False, use_b=False),
    "X-4": dict(name="gp_plus_deterministic_features", use_a=True, use_gp=True, use_df=True, use_b=False),
    "X-5": dict(name="full_code_aware", use_a=True, use_gp=True, use_df=True, use_b=True),
    "X-6": dict(name="full_code_aware_plus_video_explainer", use_a=True, use_gp=True, use_df=True, use_b=True),
}


SWITCH_METRIC = "switched_indicator"   # +1 switched / -1 no switch; S1 boundary only exists where a switch happens


def switch_aware_score(c: dict, obj: dict, beta: float, weight: float = 2.0) -> float | None:
    """S1 straddle on recovery_success, penalised where the switch GP predicts no switch (no S1 target there)."""
    s = acquisition_score(c, obj, beta)
    mu_sw = (c.get("gp_mean") or {}).get(SWITCH_METRIC)
    if mu_sw is None:
        return s
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


class Loop:
    def __init__(self, *, mode: str, out: Path, ablation: str, video_on: bool, seed: int, cfg: dict | None = None,
                 cold_start: int | None = None, batch_size: int | None = None, rounds: int | None = None,
                 quiet: bool = False):
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
        self.fault_pool = orch["fault_pool"]
        self.objectives = orch["active_objectives"]
        self.obj_cfg = orch["objective_acquisition"]
        self.metric_rules = self.cfg["vulnerability_patterns"]["metric_rules"]
        self.gp_metrics = sorted({self.obj_cfg[o]["metric"] for o in self.objectives}) + [SWITCH_METRIC]
        self.th = thresholds_from_config(self.cfg)
        self.vcfg = self.cfg["video_explainer"]
        self.taxonomy = self.vcfg["mechanism_taxonomy"]
        self.questions = self.cfg["nemotron_b"]["B_question_answering"]["question_bank"]
        self.av_stack = resolve_av_stack()
        self.records: list[dict] = []
        self.candidates: dict[str, dict] = {}
        self.features_of: dict[str, list[float]] = {}
        self.explanations: dict[str, dict] = {}
        self.b_cache: dict[str, dict] = {}
        self.events_cache: dict[str, dict] = {}
        self.cf_queue: list[dict] = []
        self.stats: dict[str, Any] = {"a_calls": [], "rejected": [], "llm_errors": []}
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
                      stop_after_fault_s=float(rc["stop_after_fault_s"]))
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

    def features(self, c: dict, *, train_round: int | None = None) -> tuple[list[float], list[str]]:
        x, names = SC.numeric_features(c, self.space, self.route_ids, self.fault_pool)
        if self.flags["use_df"]:
            ev = self.route_events(c)
            profile = self.cfg["safety_profile"]
            kappa = None if ev is None else (ev.get("kappa_at") or {}).get(c["onset_event"])
            scen = {"speed_mps": c["environment"]["max_speed_kmh"] / 3.6, "road_curvature_per_m": None if kappa is None else abs(kappa),
                    "fault": {"delay_ms": 50.0 * c["fault"]["params"]["delay_ticks"]} if "delay_ticks" in c["fault"]["params"] else {}}
            df = J.compute_df(scen, profile).as_dict()
            for k in ("DF-03", "DF-05"):
                v = df.get(k)
                names += [k, f"{k}_missing"]
                x += [0.0 if v is None else float(v) / (100.0 if k == "DF-03" else 5.0), 1.0 if v is None else 0.0]
        if self.flags["use_b"]:
            bx, bn = NM.b_features((self.b_cache.get(self.b_key(c)) or {}).get("answers"), self.questions)
            x += bx; names += bn
        if self.video_on and self.vcfg["feedback"]["gp_categorical_features"]["enabled"]:
            ex = self.explanations if train_round is None else {k: e for k, e in self.explanations.items() if e["round"] < train_round}
            mx, mn = mechanism_features(c, ex, self.taxonomy, self.vcfg["feedback"]["gp_categorical_features"]["smoothing_alpha"])
            x += mx; names += mn
        return x, names

    def b_key(self, c: dict) -> str:
        return f"{c['fault']['mode']}|{c['onset_event']}"

    # ------------------------------------------------------------------ 03 Nemotron A
    def stage_03(self, n: int, round_index: int, near: list[dict] | None = None) -> list[dict]:
        if n <= 0 or not self.flags["use_a"]:
            return []
        evidence = NM.evidence_enum(self.cfg["safety_profile"], self.cfg["prior_knowledge"], self.space, self.objectives)
        schema = NM.candidate_schema(self.space, self.route_ids, self.fault_pool, self.objectives, evidence, n)
        table = failure_table(self.records, self.candidates)
        ctx = NM.a_context(profile=self.cfg["safety_profile"], prior=self.cfg["prior_knowledge"], space=self.space,
                           objective_cfg={"id": "S1", **self.cfg["S1"]["objective"],
                                          "natural_switch_bound": self.cfg["S1"]["switch_state_control"]["natural_switch_bound"]},
                           routes=[{k: r.get(k) for k in ("route_id", "town", "note")} for r in self.routes],
                           few_shot=few_shot(table, self.explanations, int(self.cfg["nemotron_a"]["feedback_in"]["k"])),
                           mechanisms=mechanism_summary(self.explanations) if self.video_on else [],
                           tried=[{k: c.get(k) for k in ("route_id", "environment", "fault", "onset_event", "onset_offset_s")}
                                  for c in list(self.candidates.values())[-40:]],
                           include_unreviewed=bool(self.cfg["nemotron_a"].get("include_unreviewed_observations", True)))
        msgs = NM.a_messages(schema=schema, n=n, context=ctx)
        obj, st = NM.call_a(self.client_a, self.a_settings, msgs, schema, tag="A")
        st.update(round=round_index, requested=n)
        self.stats["a_calls"].append(st)
        cands = NM.a_to_candidates(obj or {}, self.space, round_index=round_index)
        if near and self.a_refine.get("enabled", True) and self.cfg["nemotron_a"].get("refine", {}).get("enabled", True):
            schema_r = NM.candidate_schema(self.space, self.route_ids, self.fault_pool, self.objectives, evidence, len(near))
            obj_r, st_r = NM.call_a(self.client_a, self.a_refine, NM.refine_messages(schema_r, near, ctx), schema_r, tag="A_refine")
            st_r.update(round=round_index, requested=len(near), refine=True)
            self.stats["a_calls"].append(st_r)
            cands += NM.a_to_candidates(obj_r or {}, self.space, round_index=round_index, source="nemotron_a_refine")
        return [self.quantize(c) for c in cands]

    # ------------------------------------------------------------------ 04 feasibility + expansion
    def stage_04(self, cands: list[dict], round_index: int) -> list[dict]:
        lat = self.th["lat_acc_threshold_mps2"]
        ok_list = []
        for c in cands:
            ok, tags, problems = SC.gate(self.space, c, self.route_ids, self.route_events(c), lat)
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
                    ok, tags, _ = SC.gate(self.space, p, self.route_ids, self.route_events(p), lat)
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
        schema = NM.b_schema(self.questions, list(sizes))
        for c in cands:
            k = self.b_key(c)
            if k in self.b_cache:
                continue
            ctx = {"fault_mode": c["fault"]["mode"], "fault_params": c["fault"]["params"], "onset_event": c["onset_event"],
                   "planner": "pdm", "controller": "supervised"}
            samples = []
            for i in range(int(self.b_settings.get("repeats", 3))):
                try:
                    res = self.client_b.chat(self.b_settings.get("model", "stub/B"), NM.b_messages(code, self.questions, ctx, schema),
                                             temperature=self.b_settings.get("temperature"), top_p=self.b_settings.get("top_p"),
                                             max_tokens=self.b_settings.get("max_tokens"), thinking=self.b_settings.get("thinking"),
                                             chat_template_kwargs=self.b_settings.get("chat_template_kwargs_extra"),
                                             response_format={"type": "json_schema", "json_schema": {"name": "b_answers", "schema": schema}},
                                             tag=f"B:{k}:{i}")
                    from engine.llm import extract_json
                    samples.append(extract_json(res.content))
                except (LLMError, ValueError) as e:
                    self.stats["llm_errors"].append({"role": "B", "key": k, "error": str(e)[:200]})
            self.b_cache[k] = {"answers": NM.aggregate_b(samples, self.questions, sizes), "n_samples": len(samples), "context": ctx}

    # ------------------------------------------------------------------ 06 + 07
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
        gps = MetricGPs(self.gp_metrics, self.seed)
        counts = gps.fit(rows)
        self.stats.setdefault("gp_train_counts", []).append({"round": round_index, **counts})
        return gps

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
            batch = select_batch_scored(rest, lambda c: switch_aware_score(c, obj, beta), batch_size=self.batch - len(forced),
                                        acquisition_fraction=float(orch["acquisition"]["batch_composition"]["lcb"]),
                                        min_distance=float(div["min_normalized_distance"]), max_per_seed=int(div["max_per_seed"]),
                                        rng=self.rng, maximize=True)
        for c in forced:
            c["selected_by"] = "counterfactual_check"
        return forced + batch

    # ------------------------------------------------------------------ 08 + 09
    def stage_08_09(self, batch: list[dict], round_index: int) -> list[dict]:
        new = []
        mr = self.metric_rules
        for c in batch:
            base_dir, cached = self.runner.ensure_baseline(c)
            if not (base_dir / ".complete").is_file():
                self.stats["rejected"].append({"scenario_id": c["scenario_id"], "tags": ["G-04_reject"], "problems": ["baseline failed"]})
                continue
            ev = self.route_events(c)
            onset = SC.onset_time(c, ev or {})
            if onset is None:
                self.stats["rejected"].append({"scenario_id": c["scenario_id"], "tags": ["G-01_reject"],
                                               "problems": [f"route has no {c['onset_event']} (baseline event detection)"]})
                continue
            run_id = f"R{round_index:02d}-{len(self.records):04d}-{c['fault']['mode']}"
            c["parameters"] = {"route_id": c["route_id"], "environment": c["environment"], "fault": c["fault"],
                               "onset_event": c["onset_event"], "onset_offset_s": c["onset_offset_s"], "onset_s": round(onset, 3)}
            res = self.runner.run_fault(c, self.out / "runs" / run_id, onset)
            run = load_av_stack_run(res.run_dir)
            base = load_av_stack_run(base_dir)
            rec = compute_run_record(run, base, candidate=c, metric_rules=mr, th=self.th, run_id=run_id,
                                     round_index=round_index, selector=c.get("selected_by"))
            if res.returncode != 0 and rec["sim_status"] == "ok" and not run.ticks:
                rec["sim_status"], rec["invalid_reason"] = "invalid", f"run_benchmark exit {res.returncode}"
            rec["feature_schema_version"] = self.feature_schema()
            self.candidates[c["scenario_id"]] = c
            self.records.append(rec)
            new.append(rec)
            append_jsonl(self.out / "run_records.jsonl", rec)
            self.say(f"  [08/09] {run_id} {c['route_id']} {c['environment']['max_speed_kmh']:.0f}km/h {c['fault']['mode']} "
                     f"@{c['onset_event']}{c['onset_offset_s']:+.2f}s -> {rec['recovery_outcome']} "
                     f"(lrr={_f(rec['lane_recovery_robustness'])}, trigger={rec['trigger_source']}, sel={c.get('selected_by')})")
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
                                prefer=tuple(clip_cfg["source_preference"]), sheet_cfg=clip_cfg.get("contact_sheet"))
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
                    result = explain(self.client_c, self.c_settings, clip, timeline, self.taxonomy, tag="C", visible_keys=vis or None)
                    if result["explanation"] is None and self.c_settings is not self.c_fallback:
                        fb = TokenFactoryClient.from_config(self.cfg["llm"], self.llm_log)
                        result = {**explain(fb, self.c_fallback, clip, timeline, self.taxonomy, tag="C_fallback", visible_keys=vis or None),
                                  "primary_errors": result["errors"]}
                entry.update(result)
                entry["verification"] = verify(entry.get("explanation"), timeline, clip, self.taxonomy, tol)
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
                                            "batch_size": self.batch, "rounds": self.rounds, "synthetic": self.mode == "mock",
                                            "scenario_space": str(self.space_path), "av_stack": str(self.av_stack)})
        self.stage_00_02()
        mix = self.cfg["orchestrator"]["cold_start"]["mix"]
        for rnd in range(self.rounds + 1):
            rdir = self.out / f"round_{rnd:02d}"
            if rnd == 0:
                n_a = int(round(self.cold * float(mix["nemotron_a"]))) if self.flags["use_a"] else 0
                if self.ablation == "X-2":
                    n_a = self.cold
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
                    scored = sorted(((abs(mu.get(m, 1e9)), c, mu, sd) for c, mu, sd in zip(prev, means, stds)), key=lambda p: p[0])
                    for _, c, mu, sd in scored[: int(self.cfg["nemotron_a"].get("refine", {}).get("max_candidates", 3))]:
                        near.append({**c, "gp_mean": mu, "gp_std": sd})
                n_a = int(self.cfg["nemotron_a"]["A_scenario_generation"]["per_round"]) if self.flags["use_a"] else 0
                a_c = self.stage_03(n_a, rnd, near=near if self.flags["use_gp"] else None)
                sf = SC.space_filling(self.space, self.route_ids, self.fault_pool,
                                      int(self.cfg["orchestrator"]["candidate_pool"]["space_filling_per_round"]), self.rng, round_index=rnd)
                cf, self.cf_queue = self.cf_queue, []
                pool = self.stage_04([self.quantize(c) for c in a_c + sf] + cf, rnd)
                target = self.batch
            dump(rdir / "03_04_pool.json", [{k: c.get(k) for k in ("scenario_id", "source", "route_id", "environment", "fault",
                                                                    "onset_event", "onset_offset_s", "tags", "evidence_ids",
                                                                    "expected_mechanism", "expected_side")} for c in pool])
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
            dump(rdir / "07_batch.json", [{k: c.get(k) for k in ("scenario_id", "source", "selected_by", "gp_mean", "gp_std")} for c in batch])
            self.say(f"[round {rnd}] pool {len(pool)} -> batch {len(batch)} ({self.ablation}, video {'on' if self.video_on else 'off'})")
            new = self.stage_08_09(batch, rnd)
            self.stage_10(new, rnd)
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
        "explanations": len(expl), "explanations_accepted": len(acc),
        "explanation_acceptance_rate": (len(acc) / len(expl)) if expl else None,
        "hallucination_rate": (1 - len(acc) / len(expl)) if expl else None,
    }
    if loop is not None:
        m["gp_boundary_sign_accuracy"] = boundary_accuracy(loop)
        m["a_schema_valid_items"] = sum(s.get("items_valid", 0) for s in loop.stats["a_calls"])
        m["a_items_returned"] = sum(s.get("items_returned", 0) for s in loop.stats["a_calls"])
        m["rejected_candidates"] = len(loop.stats["rejected"])
        m["llm_usage"] = llm_usage(loop.llm_log)
    return m


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
        for i, c in enumerate(test):
            base, _ = loop.runner.ensure_baseline(c)
            onset = SC.onset_time(c, loop.route_events(c) or {})
            if onset is None:
                continue
            from engine.fixtures.synthetic_sim import write_run_folder
            d = write_run_folder(tmp / f"t{i}", route=c["route_id"], max_speed_kmh=c["environment"]["max_speed_kmh"],
                                 fault=SC.fault_dict_for_sim(loop.space, c, onset), seed=loop.runner.seed,
                                 stop_after_fault_s=loop.runner.stop_after_fault_s, record_front=False)
            rec = compute_run_record(load_av_stack_run(d), load_av_stack_run(base), candidate=c, metric_rules=loop.metric_rules, th=loop.th)
            y = (rec.get("gp_targets") or {}).get(metric)
            if y is None:
                continue
            mu = gps.predict([loop.features(c)[0]])[0][0].get(metric)
            n += 1; ok += int((mu >= 0) == (y >= 0))
        shutil.rmtree(tmp, ignore_errors=True)
        return {"n": n, "accuracy": ok / n if n else None, "truth": "SYNTHETIC simulator (mock)"}
    ok = 0
    for i in range(len(rows)):
        train = rows[:i] + rows[i + 1:]
        g = MetricGPs([metric], loop.seed)
        g.fit([(loop.features(c)[0], r["gp_targets"]) for c, r in train])
        if metric not in g.models:
            return None
        mu = g.predict([loop.features(rows[i][0])[0]])[0][0][metric]
        ok += int((mu >= 0) == (rows[i][1]["gp_targets"][metric] >= 0))
    return {"n": len(rows), "accuracy": ok / len(rows), "truth": "leave-one-out on observed runs"}


def write_report(out: Path, m: dict, loop: Loop) -> None:
    syn = "SYNTHETIC (mock 모드: CARLA·LLM 호출 없음, 수치는 실측이 아님)" if loop.mode == "mock" else "실측 (CARLA + Token Factory)"
    lines = [f"# 파이프라인 실행 보고서 ({loop.ablation} {ABLATIONS[loop.ablation]['name']}, 영상 해설 {'on' if loop.video_on else 'off'})",
             "", f"- 데이터: **{syn}**", f"- 출력 폴더: `{out}`",
             f"- 런 {m['n_runs']}개 (유효 {m['n_valid']}, 전환 {m['n_switched']}), final_safe_recovery {m['n_final_safe_recovery']}개, 실패 {m['n_failures']}개",
             f"- 경계 양쪽 관측: {'예' if m['both_sides_observed'] else '아니오 (경계 미추정)'}; |lane_recovery_robustness| < {m['boundary_band']} 런 {m['boundary_band_runs']}개",
             f"- 첫 실패까지 런 수: {m['simulations_to_first_failure']}, 고유 실패 조건: {m['unique_failure_conditions']}",
             f"- GP 경계 부호 정확도: {m.get('gp_boundary_sign_accuracy')}",
             f"- 영상 해설: {m['explanations']}건, 검증 통과 {m['explanations_accepted']}건 (환각률 {m['hallucination_rate']})",
             f"- recovery_outcome 분포: {m['outcome_counts']}", f"- trigger_source 분포: {m['trigger_counts']}", "",
             "## 주의", "- 판정 임계값(0.3 m, 0.1 rad, 1 s, 5 s, 3 m/s², 5 m/s³)은 PROVISIONAL 이다. 인증 기준이 아니다.",
             "- 측정된 복구 성공 영역은 수학적 안전 보증이 아니다.",
             "- 영상 해설은 판정을 바꾸지 않는다. 검증을 통과한 해설만 피드백에 쓴다.",
             "- monitor_rule_candidates.yaml 은 사람 검토용 제안이다 (자동 적용 안 함)."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold-start", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--rounds", type=int, help="rounds after the cold start (default from orchestrator budget)")
    ap.add_argument("--ablation-suite", action="store_true", help="mock only: run evaluation/ablations.yaml mock_suite")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--repeats", type=int)
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
                cold_start=args.cold_start, batch_size=args.batch_size, rounds=args.rounds)
    m = loop.run()
    log(json.dumps({k: m[k] for k in ("n_runs", "n_switched", "n_final_safe_recovery", "n_failures", "both_sides_observed",
                                      "boundary_band_runs", "explanations", "explanations_accepted")}, ensure_ascii=False))
    log(f"outputs: {out}  (report.md, run_records.jsonl" + (", explanations.jsonl, monitor_rule_candidates.yaml" if loop.video_on else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
