# integration3 — boundary search fixes

Branch `integration3` (local), based on `integration2` @ `bd683b5`. Goal: the campaign should find the
**recovery success/failure boundary** instead of piling up failures (result_1..5: every fault mode had a single
outcome; Nemotron A proposed mostly range-extreme "make it fail" plans; GP predictions were constant).

Official judgment is still Recovery v2 (`engine/monitor.py`); SR checks remain reference only.

| # | Commit | Change |
|---|--------|--------|
| 1 | `c9a653a` | saturate (F-06) magnitude + linear ramp |
| 2 | `728252a` | tyre friction axis, P-02 scaled by friction |
| 3 | `a312c13` | control faults (solver_fail, stale) sampled on handover-state axes |
| 4 | `a4d7aec` | Nemotron A boundary-seeking proposals + per-round batch share |
| 5 | `651add7` | GP: unit features, level-set prior, switch classifier, level-set acquisition, `--gp-history`, metrics |

## 1. Saturate magnitude and ramp (DriveFI-style continuous fault magnitude)

* `FaultConfig.level` (fraction of max steer, 0.2–1.0, default 1.0 = old behaviour) and `ramp_s` (0–0.3 s linear
  blend from the MPC command, default 0).
* CLI: `run_benchmark --saturate-level --fault-ramp`; search space F-06 `magnitude` / `ramp_s`
  (`cli: --saturate-level / --fault-ramp`). PDM / TF++ agents, `fault_args`, `scenarios.fault_cli_local`,
  `fault_dict_for_sim` and the synthetic sim all honour them.

## 2. Tyre friction (CARLA `WheelPhysicsControl.tire_friction`)

* `carla_io.scale_tire_friction` (guarded; no-op if the API is missing) is applied after spawn by
  `tfpp_benchmark`; the scale is written to `agent_config.tire_friction` and the run manifest.
* CLI `run_benchmark --tire-friction-scale` (`%.2f`); environment axis in the search space; included in
  candidate generation, perturbation, validation, gate, features, baseline key (the no-fault baseline uses the same
  friction) and `paired_runs` quantisation (step 0.1).
* Monitor: `run_friction_scale` / `scale_thresholds_for_friction` — P-02 lateral-acceleration limit
  `0.5·0.9·g = 4.41 m/s²` and SR `mu`, `a_y_max` scale with friction.
* Synthetic sim: lateral acceleration capped at `0.9·scale·g` (understeer).

## 3. Handover axes for control faults (F-01 solver_fail, F-02 stale)

* `handover_axes` in the search space: onset `curve_entry` / `curve_apex`, offset ±0.5 s, speed 25–40 km/h,
  friction. `scenarios.sampling_ranges` uses them in space filling and in the Nemotron schema.
* Synthetic sim models `stale` (replays the last MPC sequence).
* TODO: there is no disturbance knob at the handover yet.

## 4. Nemotron A: boundary proposals

* New `A_SYSTEM` prompt: the goal is plans **on** the success/failure boundary, not failures.
* From round 1 on, A gets a **GP summary every round** (`nemotron.surrogate_summary`, saved as
  `round_XX/03_gp_summary.json`): the top-k (`nemotron_a.yaml surrogate_summary.k: 6`) most ambiguous pool
  candidates and the recent outcome mix per fault mode. Cold start gets a note instead.
* **Range extremes**: a continuous value within 5 % of its range end (`EXTREME_EDGE`) needs
  `extreme_justification` (required schema field). Unjustified ones are tagged `range_extreme_unjustified` and
  ranked last in the share.
* **Per-round share**: `orchestrator.yaml acquisition.batch_composition.nemotron_a: 0.3` reserves
  `ceil(0.3·batch)` slots for A candidates (`selected_by: nemotron_a_share`). 0 disables it.
* Offline stub (`stub_handler`) proposes jittered points around the summary's boundary regions.
* Metrics: `evaluation_metrics.nemotron_a` (share, extremes, mean edge distance, expected_side).

## 5. GP surrogate and acquisition

* **Features** (`scenarios.numeric_features`): every value on [0, 1]; one column per fault-parameter *name*
  (so F-06 `magnitude` / `ramp_s` reach the GP — the old two shared slots `param_u0/param_u1` silently dropped
  them), `tire_friction_u`, `max_speed_u`. A fault without a parameter gets 0.5.
* **Constant predictions fix**: no min/max re-normalisation on the training span; kernel bounds for unit features
  (`length_scale 0.1–10`, noise `1e-4–0.5`; `surrogate.yaml kernel_bounds`).
* **Prior mean at the level** (`surrogate.yaml prior_mean: level`): the level-set objective's GP uses a fixed prior
  mean h (0 = lane boundary) instead of the training mean. The campaign samples failure-heavy regions, so with the
  training mean every unexplored region was predicted "fail" (mock sign accuracy fell to 0.48–0.57 once the space
  grew with items 1–3). `training_mean` restores the old behaviour.
* **switched_indicator = GP classifier** (`switch_model: classifier`, Laplace): `gp_mean` = P(switch),
  `<metric>__latent` keeps the latent mean/std. The legacy straddle maps P to [−1, 1].
* **Level-set acquisition** (Gotovos et al. 2013) is the S1 default:
  `max((β·σ − |μ − h|)/std(y) − penalty·(1 − P(switch)), β·σ_f − |μ_f|)`, i.e. "recovery near the boundary where a
  switch is likely, OR switch itself uncertain". `acquisition: straddle` keeps the legacy score.
* **Minimum training count**: `min_train_runs_for_acquisition: 8` — below it the GP part of the batch is space
  filling (`gp_train_counts[].gp_acquisition_active`).
* **`--gp-history GLOB`** (repeatable): pools earlier `run_records.jsonl` into GP training; runs are re-encoded with
  the current space (missing new parameters take their defaults), invalid / out-of-campaign / out-of-range runs are
  skipped and counted in `loop_stats.gp_history`.
* **Metrics**: `evaluation_metrics.boundary_metrics` — boundary-band runs (|lane_recovery_robustness| < 0.2 and
  < 0.5 m), per-fault-mode outcome mix and `modes_with_mixed_outcome`, GP prediction diversity, selection sources;
  `gp_boundary_sign_accuracy` adds truth balance, majority baseline and balanced accuracy.

## 6. Nemotron B code questions + DF-06 (integration3 #6)

Problem (result_1..5, 224 answer sets): Q-B2 was 'no' 202× and Q-B3 'no' 194× (near-constant GP columns); Q-B1 only
mirrored the fault mode and had no option for "the corrupted MPC command reaches the vehicle" (bias / drift /
intermittent_bias / saturate → B answered `unknown`); `[yes, no, unknown]` was unquoted (YAML booleans), so
`expected_from_code: 'no'` never matched and was never compared; citations were only checked for being inside the
file; the cache key ignored fault parameters.

* **Question bank** (`pipeline/05b_nemotron_b/nemotron_b.yaml`): every question has `scope` (`per_scenario` = asked
  per `b_key`, `static` = asked once per campaign, stored as `05b_answers.json["__static__"]`) and `gp_feature`.
  Only per_scenario + gp_feature questions become GP one-hot (`engine.nemotron.b_features`).

  | id | scope | GP | options | expected_from_code (av_stack line refs) |
  |---|---|---|---|---|
  | Q-B1 | per_scenario | yes | hold_last, zero, stale_mpc, **corrupted_mpc**, lqr_early, unknown | solver_fail, stale → lqr_early (fault_injection.py:111-112, supervised_lateral.py:29-30, 42-43); bias/drift/intermittent_bias/saturate → corrupted_mpc (fault_injection.py:79-97); output_delay → stale_mpc (:99-102) |
  | Q-B4 | per_scenario | yes | solver_failure, missing_steer, nonfinite_output, steer_out_of_range, deadline_miss, stale_output, stale_sequence, tracking_divergence, controller_exception, none_during_fault, unknown | first check while the fault is active: solver_fail → solver_failure; stale → stale_sequence; saturate with duration_s ≤ 0.25 → none_during_fault (0.25 s hold, supervised_lateral.py:50), longer → {tracking_divergence, none_during_fault} (level ≤ 1 passes the strict range check :36); output_delay → {tracking_divergence, none_during_fault}; bias/drift/intermittent_bias → {steer_out_of_range, tracking_divergence, none_during_fault} |
  | Q-B5 | per_scenario | no | ok_flag, sequence_or_stamp, steer_value_only, steer_and_reported_state, none, unknown | solver_fail → ok_flag; stale → sequence_or_stamp; output_delay → steer_and_reported_state (fault_injection.py:101 also replaces `state`, so the tracking check sees k-tick-old errors); others → steer_value_only |
  | Q-B2 | static | no | yes, no, unknown | no — latched, no switch-back (supervised_lateral.py:99, 116-117) |
  | Q-B3 | static | no | yes, no, unknown | no — LQR has no internal state (lqr.py:158-174), takeover is rate-limited from the last applied steer (supervised_lateral.py:134-136) |

  A list means "the code rules out everything else"; which allowed answer happens depends on vehicle dynamics.
  `mpc_controller/mpc.py` was added to B's code context (seq increment :205, steering clamp :246).
* **YAML**: all options quoted; `normalize_questions` maps booleans back to 'yes'/'no' for old banks and old answers.
* **b_key** = `fault_mode|onset_event` + bins of the parameters a question declares in `param_bins`
  (Q-B4: `duration_s` at 0.25 s, so only saturate gets `|duration_s#0/1`). Cache stays bounded.
* **Hallucination check**: `aggregate_b` records `expected_from_code` (resolved for the scenario), `matches_code`
  (true / false / null when the code does not decide) and `hallucinated` (mismatch and not `unknown`).
  `loop_stats.json` and `evaluation_metrics.json["nemotron_b"]` carry `b_hallucination_rate`, code-agreement rate,
  per-question answer counts, and how many GP B columns are constant over the training runs.
* **Citation relevance**: each question lists `evidence_regions` (file + function or line range, resolved with
  `ast`). `citations_relevant` = in range AND at least one cited line inside a region (`citation_relevant_rate`).
* **Mock stub B** (SYNTHETIC): mostly the code answer, with a deliberate Q-B1 stale → stale_mpc mistake (what the
  real model said) plus 15 % random answers and 15 % unrelated citations, own RNG stream.
* **DF-06** `friction_utilization = v²|κ| / (0.9 · tire_friction_scale · 9.81)` (same μ as P-02; 1.0 = nominal
  tyre limit, 0.5 = P-02 limit). GP input next to DF-05 (/2.0) with `DF-06_missing`.
* **Video clip fix** (`engine/video.py`): the ffmpeg `select` sum of ~100 `eq()` terms failed with "Cannot allocate
  memory" on long dense clips (crashed a mock campaign); consecutive frames are now `between()` and terms are
  chunked (40 per ffmpeg call).

Offline check of the real answers in `result/result_1..5` against the new key (old options; 224 answer sets incl.
round-to-round repeats, 108 unique per campaign):
Q-B1 — 126 (62 unique) sets were bias/drift/intermittent_bias/saturate where the correct option did not exist
(all `unknown`); of the 98 (46) decidable, 70 (32) match and 28 (14) are wrong: stale → stale_mpc 25 (12),
stale → hold_last 3 (2) — the code catches a stale output on the first tick (stale_sequence), which the real
run records also show (5/5 stale runs switched by stale_sequence). Q-B2 — 202 match, 22 unknown, 0 wrong.
Q-B3 — 194 match, 28 unknown, 2 wrong. Overall hallucination rate on decidable answers: 30/546 = 5.5 %
(15/262 unique), concentrated in Q-B1 stale (28/98 = 29 %).

Mock (`python run.py --mock --ablation X-5 --seed S --cold-start 24 --batch-size 12 --rounds 4`, before = 09397ec
with only the video fix applied, SYNTHETIC simulator and stub LLMs):

| seed | GP sign acc. before → after | balanced before → after | B GP columns non-constant / total | B hallucination (mock) |
|---|---|---|---|---|
| 0 | 0.905 → 0.905 | 0.893 → 0.857 | 4/11 → 6/17 | – → 0.072 |
| 1 | 0.905 → 0.909 | 0.857 → 0.857 | 3/11 → 8/17 | – → 0.035 |
| 2 | 0.857 → 0.909 | 0.786 → 0.857 | 5/11 → 6/17 | – → 0.060 |

The remaining constant columns are options nobody chose (one-hot zeros). The mock B is synthetic, so these numbers
only show the plumbing works; real-model effects need a real run.

## 7. Nemotron A boundary context (integration3 #7)

Review finding: A is supposed to find the recovery boundary, but the information it needs to judge a boundary did
not reach it. Every point was checked in code first.

| # | finding | verdict | change |
|---|---|---|---|
| 1 | surrogate_summary values were GP z units labelled like metres | **confirmed, partly corrected**: values were z = lrr / 0.1 (10× lane_recovery_robustness), but lrr itself is dimensionless (margin / (lane_w/2 − half width)), not metres | values are converted to lrr (`Loop.native`), renamed `predicted_lane_recovery_robustness(_std)`, `gp.units/meaning/training_std` state the definition; P01 metres are shown separately as `lane_margin_min_m`; refine prompt now matches |
| 2 | summary "ambiguity" is not the acquisition | confirmed (summary used its own score with no 1 − P(switch) penalty) | `engine.surrogate.boundary_score_parts` = the exact terms of `boundary_score`; summary ranks by `Loop.summary_scorer` (same beta, level, y_scale, switch_penalty as `acq_score`). Points with GP std ≥ 0.75 × training std go to `unexplored_regions` (not boundary evidence) |
| 3 | no-switch runs counted as failed, prompt pushed toward milder faults | confirmed (`final_safe_recovery` is False without a switch) | `recent_outcomes_by_fault_mode` has switched / never_switched / recovered / failed / failed_reasons / status; prompt: never_switched → stronger/longer, failed_whenever_switched → milder, mixed → bracket |
| 4 | no route geometry → V-03 rejections | confirmed | `route_geometry` (round_XX/03_route_geometry.json): per route × onset_event rows `[kmh, t_s, kappa, speed, v²κ, latest_onset_offset_s_for_V03]` from completed baselines, rules for V-03 and v²κ vs 4.41·scale (DF-06). mock probes baselines at 20/30/40 km/h; **real mode does not probe** (`probe_baselines.real: false`, extra CARLA runs = user decision), it only shows baselines already run |
| 5 | few-shot was failures only, already_tried had no outcomes | confirmed | `outcome_examples`: per fault mode the closest recovered + failed pair (normalised parameter distance); `already_tried` rows carry outcome label, switched, `lane_margin_min_m`; `feedback_in.k` 5 → 3. B and Cosmos inputs unchanged (no results) |
| 6 | objective had no judgment criteria | confirmed | `objective.judgment_criteria` from `thresholds_from_config` + P-02 formula (3 s / 0.3 m / 0.1 rad / 1 s hold, 4.41·scale = 0.5·0.9·scale·9.81, jerk 5, window 10 s, V-03 5 s) |
| 7a | verified flags | not flipped; all 25 observations re-checked against av_stack (line drift noted in note_en, e.g. O-001 run_benchmark.py:114 not 105, O-017 rate 90°/s at pdm_agent.py:86/231) | – |
| 7b | Korean in the prompt | confirmed | `*_en` fields (faults, supervisor, handover, saturate params, S1 natural_switch_bound_en, route note_en, observation note_en); `english_view` uses them and drops Korean-only leaves; the captured prompt has no Hangul |
| 7c | schema sent twice | confirmed | schema only in `response_format` (json_schema); pasted into the body only on the json_object fallback |
| 7d | refine had less context | confirmed | refine gets objective, supervisor, scenario_space, routes, route_geometry, outcome_examples, already_tried, GP header + outcome mix; near rows in lrr units |
| 7e | two temperatures | confirmed: `nemotron_a.yaml temperature 0.7` and `nemotron_b.yaml 0.0` were never read; effective = `llm_models.yaml` A 1.0 / A_refine 1.0 / B 1.0 (top_p 0.95) | yaml keys removed with a comment; effective values unchanged |

Captured messages, mock seed 0, 2 rounds, `PYTHONHASHSEED=0` (before = 6fff06a; stub A, SYNTHETIC simulator):

| | before | after |
|---|---|---|
| user chars A r0 / r1 / refine1 / r2 / refine2 | 24,258 / 36,407 / 19,175 / 40,431 / 19,492 (incl. 11.8 k schema each) | 20,849 / 40,886 / 25,695 / 45,644 / 30,096 (no schema in body) |
| system prompt chars | 2,334 | 3,432 |
| boundary_regions r1 (mode, P(switch)) | F-03 .14, F-03 .16, F-03 .14, F-07 .15, F-04 .16, F-04 .16 | F-04 .75, F-04 .16, F-05 .16, F-02 .87, F-06 .88, F-05 .89 (+ unexplored F-03 .25/.19/.19) |
| boundary_regions r2 | F-03 ×6 (.10–.19) | F-04 .73, F-05 .86 ×3, F-04 .17 ×2 (+ unexplored F-03 .16 ×3) |
| output_delay (F-03) among 48 A items | 18 | 6 |
| V-03 rejections total / of A candidates | 13 / 5 | 5 / 2 |

GP sign / balanced accuracy, X-5 mock seeds 0–2 (`--cold-start 24 --batch-size 12 --rounds 4`, n = 21 each):
before 0.905/0.857, 0.905/0.857, 0.857/0.786 → after 0.857/0.786, 0.905/0.857, 0.905/0.857 (same mean; switched
runs 86 → 98 over the three seeds). The section-6 numbers (0.905/0.909/0.909) were without `PYTHONHASHSEED`.

Caveats: the stub A is SYNTHETIC (it reads the new fields mechanically), so these numbers only show the information
arrives; how the real Nemotron uses it is unmeasured. The mock route ends shrink with speed and V-03 is checked
against the nearest probed speed, so some rejections remain. In real mode route_geometry is empty until baselines
exist.

## Mock comparison (seed 0, `PYTHONHASHSEED=0`, SYNTHETIC simulator, stub LLMs)

`python run.py --mock --seed 0 --open-screen off`; before = `bd683b5`, after = `651add7`.

| | before (bd683b5) | after #1–4 (a4d7aec) | after #1–5 (651add7) |
|---|---|---|---|
| runs / switched / final safe | 66 / 29 / 5 | 62 / 25 / 3 | 62 / 27 / 3 |
| Nemotron A share of executed (after cold start) | 0.41 (0.44) | 0.58 (0.65) | 0.47 (0.50) |
| A proposals: mean / median edge distance | 0.130 / 0.130 | 0.114 / 0.143 | 0.108 / 0.143 |
| A proposals with a range extreme | 35 % | 33 % | 37 % |
| boundary-band runs <0.2 m / <0.5 m | 0 / 10 | 0 / 9 | 0 / 5 |
| GP sign accuracy (n) | 1.00 (18) | 0.48 (21) | 0.90 (21), balanced 0.89 |
| GP prediction diversity, recovery (distinct/n) | 48/48 | 42/48 | 37/48 |
| fault modes with both safe and unsafe | none | stale | stale |

Caveats: the mock simulator is synthetic and its lane margin is close to bimodal (collision clip or full recovery),
so boundary-band runs are near zero in every variant; the after space is larger (magnitude, ramp, friction,
handover axes), so the sign-accuracy test set differs (truth safe fraction 0.44 before, 0.67 after). The stub
Nemotron is half random; the effect of the new prompt on the real model has not been measured. Nothing here was
run on CARLA.

## Tests

`tests/test_integration3_faults.py`, `tests/test_integration3_nemotron.py`, `tests/test_integration3_gp.py`,
`tests/test_integration3_nemotron_b.py`, `tests/test_integration3_nemotron_a_context.py`
(plus updated existing tests). Root suite, `pipeline/09_monitor` and the av_stack suite (without GPU / reference
tests) pass.
