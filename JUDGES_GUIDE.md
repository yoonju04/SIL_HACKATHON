# Judges' guide — AI-driven fault injection for MPC → LQR switching safety

**One line:** NVIDIA Nemotron proposes steering-controller faults, CARLA drives them, a deterministic monitor grades
every run, a Gaussian process steers the next runs towards the safe/unsafe boundary, and Cosmos Reason explains the
videos and summarises *which faults are critical and where their boundaries are*.

Branch `inter_final` = `integration4` (code) + all campaign results. The final campaign is **`result/result_12`**.

---

## 1. See the results in 2 minutes (no install)

Open these files from a checkout of this branch:

| What | File |
|---|---|
| Dashboard of the final campaign (32 CARLA runs) | `result/result_12/screen/campaign.html` (open in a browser) |
| Critical faults and boundary values | `result/result_12/insights/boundaries.md` |
| Campaign report (all numbers) | `result/result_12/report.md` |
| Per-run records (one JSON per line) | `result/result_12/run_records.jsonl` |
| Cosmos video explanations | `result/result_12/explanations.jsonl` |
| Console log of the run | `result/result_12/console.log` |

On the dashboard: left = the latest scenario with its verdict and front-camera video, middle = all runs with their
verdicts, right = the hypotheses and the **Cosmos Insights** panel (critical faults ranked, each with its boundary).

### Headline results (result_12 + earlier campaigns, 149 graded runs, tight supervisor)

| Rank | Fault (injected into the MPC output) | Unsafe runs | Late / missed switch | Boundary (unsafe = safety margin < 0) |
|---|---|---|---|---|
| 1 | Ramp bias / drift | 38 % | 100 % | 0.018–0.05 rad/s; at ≥ 35 km/h already from ~0.018, at ≤ 20 km/h only ~0.05 |
| 2 | Constant steering bias | 19 % | 23 % | ~0.043–0.075 rad (25 km/h and up); on a straight the speed boundary is 20–30 km/h |
| 3 | Steering saturation | 4 % | 0 % | detected at once; unsafe only when the handover happens while entering a curve |
| – | Intermittent bias, solver failure, stale output, output delay | 0 % | ≤ 5 % | no unsafe run in the tested range |

* Slowly growing faults (drift, bias) are the weak spot: the supervisor only sees them through the tracking error,
  so the switch comes late. Abrupt faults (saturation, intermittent bias) are caught on the first 50 ms sample by the
  steering-rate check.
* The GP-guided search worked: GP sign accuracy 0.73 vs 0.64 for always guessing the majority (balanced 0.76), and
  runs near the boundary (|safety margin| < 0.2) went from 0/8 in the random cold start to 2–3 of 6 per GP round.
* A Cosmos counterfactual (raise the intermittent-bias magnitude) was re-run automatically and gave the same margin:
  for faults caught instantly, *where* the handover happens (curve entry, speed) decides the outcome, not the fault size.

## 2. Run it yourself without CARLA (about 1 minute)

Synthetic simulator and stub LLMs, no GPU, no API key, no network. Python 3.10+:

```bash
pip install -r requirements-pipeline.txt
python run.py --mock --cold-start 6 --rounds 1 --batch-size 3 --open-screen off --out outputs/judge_demo
```

Open `outputs/judge_demo/screen/campaign.html` and `outputs/judge_demo/insights/boundaries.md`. Mock numbers are
synthetic (the report says so); the code path is the same as the real run.

Re-render the final campaign's pages from its stored analysis (no model call):

```bash
python -m engine.boundaries result/result_12 --publish-only
```

Tests (no pytest needed for the monitor; the engine tests use pytest):

```bash
python -m pytest tests -q
cd pipeline/09_monitor && python -m unittest discover -p "test_*.py"
```

## 3. Full run with CARLA and NVIDIA models

Requirements (Windows, what we used):

* CARLA 0.9.15 (Windows build) next to the repository or `CARLA_ROOT` set to its `WindowsNoEditor` folder
* av_stack environment (Python 3.10), see `pipeline/08_carla_paired_run/av_stack/README.md` → Setup:
  `tools\fetch_third_party.ps1`, then `uv venv .venv-tfpp --python 3.10` and
  `uv pip install --python .\.venv-tfpp\Scripts\python.exe -r requirements-tfpp-lock.txt`
* `.env` in the repository root from `.env.example`: `NEBIUS_API_KEY` (Nebius Token Factory: Nemotron 3.5 Lightning,
  Nemotron 3 Ultra, Nemotron 3.5 Super VL) and, for Cosmos Reason, `COSMOS_MODEL_ID` of a Nebius dedicated endpoint
  (it is started for the video step and stopped after it, `--cosmos-power auto`)
* ffmpeg on PATH (or `FFMPEG`)

```powershell
# 1) start CARLA
powershell -ExecutionPolicy Bypass -File .\pipeline\08_carla_paired_run\av_stack\start_carla_stable.ps1
# 2) preflight: CARLA port, models listed, ffmpeg
.\pipeline\08_carla_paired_run\av_stack\.venv-tfpp\Scripts\python.exe run.py --real --check
# 3) the final campaign's command (32 runs, ~3 h; drop --gp-history to start from scratch)
.\pipeline\08_carla_paired_run\av_stack\.venv-tfpp\Scripts\python.exe run.py --real --cold-start 8 --rounds 4 --batch-size 6 `
  --gp-history "result/result_8/run_records.jsonl" --gp-history "result/result_9/run_records.jsonl" `
  --gp-history "result/result_10/run_records.jsonl" --gp-history "result/result_11/run_records.jsonl" `
  --out results/my_campaign
```

Useful options: `--supervisor tight|legacy` (system under test; legacy = original thresholds), `--video-explainer off`
(no Cosmos cost), `--ablation X-1` (random selection baseline, for comparison), `CARLA_PORT` if CARLA is not on 2000.

## 4. How it works (pipeline stages, `run.py`)

| Stage | What it does | Code / config |
|---|---|---|
| 00–02 | Prior knowledge, code-verified safety profile of the system under test, hypotheses and the search objective | `pipeline/00..02_*`, `run.py stage_00_02` |
| 03 | **Nemotron A** proposes fault scenarios (route, speed, tyre friction, fault type and size, onset) aimed at the boundary, given the GP summary and past outcomes | `engine/nemotron.py`, `pipeline/03_nemotron_a` |
| 04 | Feasibility gate (scenario space, route events, evaluation window) + space-filling and perturbed candidates | `engine/scenarios.py`, `pipeline/04_feasibility_expansion/scenario_space_av_stack.yaml` |
| 05a / 05b | Deterministic features; **Nemotron B** answers code questions about the controller (cited lines, checked against a code answer key) | `pipeline/05a_*`, `pipeline/05b_nemotron_b` |
| 06 | GP surrogate: safety margin (regression) and switch probability (classifier) | `engine/surrogate.py` |
| 07 | Batch selection: level-set acquisition near the boundary, Nemotron share, diversity, control-fault cap | `run.py stage_07`, `pipeline/07_orchestrator` |
| 08 | CARLA paired runs: fault-free baseline (cached) and fault run (+ optional no-reaction run for FTTI) on Bench2Drive routes with the PDM-Lite planner and our MPC / supervisor / LQR | `engine/carla_runner.py`, `pipeline/08_carla_paired_run/av_stack` |
| 09 | **Deterministic monitor**: recovery verdict, safety margin (lane / settle / lateral accel. / jerk), SR1–SR6 safety requirements, F1/F2/F3/pass | `engine/monitor.py`, `pipeline/09_monitor/monitor.py`, `safety_spec.yaml` |
| 10 | **Cosmos Reason** explains each failed run's front video using the monitor's events as facts; accepted explanations feed back (examples for Nemotron A, counterfactual re-runs). After the campaign: critical faults and boundary values (code table + Cosmos reading) | `engine/explainer.py`, `engine/boundaries.py` |

Safety verdicts are always made by code (stage 09), never by an LLM; the models propose, explain and summarise.

System under test (`pipeline/08_carla_paired_run/av_stack/integration/supervised_lateral.py`): the MPC steers; the
supervisor switches to the LQR on solver failure, stale or out-of-range output, steering rate > 90 °/s, or tracking
error > 0.6 m / 0.35 rad held 0.15 s ("tight"; the original "legacy" thresholds were 1.0 m / 0.6 rad / 0.25 s with no
rate check, and every switch on a slowly growing fault came after the car had left the lane).

## 5. NVIDIA models and Nebius

See the README section *How we used NVIDIA models* (Nemotron 3.5 Lightning = scenario proposals, Nemotron 3 Ultra =
code questions, Cosmos 3 Super Reasoner = video explanations and the critical-fault / boundary summary, Nemotron 3.5
Super VL = fallback) and *Where Nebius Token Factory accelerated the work* (one OpenAI-compatible API for every model,
`json_schema` structured outputs, a 550B model on demand, a Cosmos dedicated endpoint started and stopped by code).
Every model call of a campaign is logged in `llm_calls.jsonl` (model, tokens; no keys).

## 6. License

MIT ([LICENSE](LICENSE)) for this repository's code; third-party components keep their own licenses.

## 7. Limitations (honest)

* Thresholds marked PROVISIONAL in `pipeline/09_monitor/safety_spec.yaml` (e.g. lateral jerk 15 m/s³, data-based) are
  our choices for this study, not certified limits. Simulation only; not a safety certification.
* Boundaries come from 149 runs over several routes and speeds; most are "mixed zones" whose exact value depends on the
  road geometry and speed. The run-level data behind every number are in `run_records.jsonl`.
* Two of the most critical (very late) drift / bias runs were graded *invalid* because the route or the 15 s run limit
  ended before the post-switch window; they are counted as invalid, not as failures.
* Bug reports and design notes: `docs/BUG_REPORTS.md`, `docs/INTEGRATION3.md`, `docs/INTEGRATION4.md`.
