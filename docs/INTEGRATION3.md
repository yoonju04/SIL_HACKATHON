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

## 8. Route-end onsets, supervisor reaction, cold start (integration3 #8)

`d36b712` (cherry-pick of teammate `d26144d`, integration4), `fe2c28f` (#8a), `cf72c6b` (#8b), `85fd500` (#8c), #8d (this
section + frozen GP truth set). Every number below was re-measured on `result/` (result_1..6), not copied from the
proposal. / 아래 수치는 제안서 값을 옮긴 것이 아니라 result_1..6 에서 다시 잰 값이다.

### 8.1 route_onset_exclusions (#8b)

KO: 짧은 경로는 커브가 경로 끝에 붙어 있어, 그 이벤트에서 고장을 넣으면 평가창 5 s + 오프셋 최대 +1 s = 6 s 가
남지 않아 V-03 으로 항상 버려진다. `scenario_space_av_stack.yaml route_onset_exclusions` 에 (경로, 시작 이벤트) 를
적고, 문자열은 항상 제외, `{event, min_kmh}` 는 그 속도 이상에서만 제외한다. `SC.allowed_onset_events` 로
space_filling 이 제외 조합을 뽑지 않고, `validate_against_space` 가 G-02 로 거른다(Nemotron A / perturb /
counterfactual / quantize 후 후보 모두 04 단계에서 걸림, CARLA 미실행). `--gp-history` 는 예전 런이라 검사 안 함.
Nemotron A 에는 `scenario_space.route_onset_exclusions` + 규칙 문장을 넘긴다.

EN: onsets that leave < 6 s (5 s window + max +1 s offset) always fail V-03. String entries are always excluded,
`{event, min_kmh}` only at or above that speed (not applied when the speed is unknown). space_filling samples only
allowed onsets; validate_against_space rejects the rest (G-02) for every source; A is told.

Measured time left after the event (unique baselines in result_1..6) vs the proposal:

| route | event | proposal | measured | entry |
|---|---|---|---|---|
| 28035 (10 baselines) | curve_apex / curve_exit | 3.1–8.1 / 2.1–6.0 s | 3.1–8.1 / 2.1–6.0 s ✓ | excluded |
| 28035 | curve_entry | – | 4.2–10.9 s; < 6 s at ≥ 30 km/h (5.5/4.8/4.2), 25 km/h 6.4 | `{curve_entry, min_kmh: 30}` (added) |
| 28154 (12) | curve_apex / curve_exit | 1.1–11.4 / 0 s | 1.1–11.4 / 0.0 s ✓ | excluded |
| 28154 | curve_entry | – | 4.6–14.6 s; < 6 s at ≥ 25 km/h | `{curve_entry, min_kmh: 25}` (added) |
| 28048 (4, 15–25 km/h) | straight | 1.9–3.3 s | 1.9–3.3 s ✓ | excluded |
| 28048 | curve_exit | 4.4–7.6 s (excluded) | 15 km/h 7.6, 20 km/h 5.8, 25 km/h 4.4 | `{curve_exit, min_kmh: 20}` (**changed**: 15 km/h is gradable) |
| 28048 | curve_apex | – | 5.7–9.6 s (25 km/h 5.7) | `{curve_apex, min_kmh: 25}` (added) |
| 24206, town10_long | all | – | ≥ 14.9 s | none |

result_6: 15 candidates = 8 runs + 7 V-03 skips at stage 08 (after the baseline). The proposal's list catches 6 of 7
✓; the 7th (28035 curve_entry, 35 km/h, offset 0.0) is caught by the speed-conditional entry → 7/7. Cost of the
unconditional entries: a few low-speed combos that would have been gradable are no longer sampled (e.g. 28035
curve_apex at 15–20 km/h: 8.1 / 6.3 s left; 28154 curve_apex at 10 km/h: 11.4 s).

### 8.2 Tighter supervisor (d36b712 + #8a)

KO: 팀원 커밋 `d26144d` 가 제안과 같은 내용(0.6 m / 0.35 rad / 0.15 s, MPC 출력 조향 속도 > 90 °/s 즉시 전환,
`--supervisor tight|legacy`, 기본 tight)을 이미 av_stack 에 구현했다 → cherry-pick (`d36b712`). #8a 는 그 나머지:
mock 시뮬레이터(MPC 자체 60 °/s 제한 + 프로필), runner/Loop `--supervisor`, 런 기록 `supervisor_profile`,
monitor 의 전환 오차 정규화를 런 프로필로, `--gp-history` 는 다른 프로필 런을 건너뜀, Nemotron B Q-B4 에
`steer_rate_exceeded`, safety_profile T-08/T-10, 트리거 목록, 스펙 줄 번호. 판정 기준(4.41 m/s², 3 s, onset+10 s)은
그대로.

EN: the teammate's `d26144d` already implemented the proposal in av_stack; #8a wires it through the mock simulator,
runner, run records, monitor, gp-history filter, Nemotron B Q-B4 expectations and the specs. Judgment thresholds are
unchanged.

Fault-free baselines (unique; result_1..5 have 61 baseline folders but only 29 distinct hashes, 44 with result_6,
58,858 MPC ticks):

| item | proposal | measured | legacy | tight |
|---|---|---|---|---|
| max \|e_y\| | 0.46 m | 0.463 m ✓ | 1.0 m | 0.6 m |
| max \|e_psi\| | 0.23 rad | **0.242 rad** | 0.6 rad | 0.35 rad |
| hold | – | – | 0.25 s | 0.15 s |
| max MPC command rate | 61 °/s | 61.1 °/s ✓ | no check | > 90 °/s → switch |

Tight rules replayed on the 44 baselines: 0 fires; max pre-onset rate in the 44 risk-fault runs 49.2 °/s (0
pre-onset fires). Switched bias/drift/saturate/intermittent runs: result_2..5 15/15 crossed the lane before the
switch ✓ (18/18 with result_6). Open-loop replay of tight on lane-departure runs (result_2..5, incl. output_delay):
24 departures, tight detects 21 before (saturate 14, intermittent 3, drift 3, bias 1), 3 after (2 drift, 1 bias),
none missed; legacy 0 before. Teammate's INTEGRATION4 says 19/22 — not reproduced exactly (different set/definition).
The rate check fires at bias onset when \|magnitude\| ≳ 0.0785 rad (4.5°/tick), intermittent edges, saturate
ramp-0 steps and saturate end steps, possibly output_delay onset jumps; never on drift; a slow saturate ramp evades it.

### 8.3 Cold start rule (#8c)

KO: `cold_start.total_runs: 24 # 예시` → `total_runs: null` + `rule: {per_fault_type: 2, max_runs: 12}`:
max(min_train_runs_for_acquisition 8, 2 × 라운드 0 고장 종류(위험 5 + 대조군 cap 1)) = 12, 상한 12. `--cold-start`,
`Loop(cold_start=)`, 숫자 total_runs 는 그대로 덮어씀. space_filling 은 고장 종류를 실제로 순환(무작위 시작, 대조군은
라운드 cap 남은 만큼), 고장마다 시작 이벤트도 순환, 그 이벤트를 채점할 수 있는 경로를 고르고, 양자화 속도로 제외
검사. 라운드 0 은 A 후보를 먼저 거르고, 04 탈락분을 cold + 6 까지 채운다(이전: cold 24 중 21 만 실행).
mock 정확도 시험 세트는 예전 space_filling 을 그대로 얼린 `SC.truth_points` 로 고정하고(샘플러가 바뀌어도 잣대가
안 움직이게), 진실 런은 그 런의 감독기 프로필로 돌린다.

EN: rule-based cold start (12 here), real fault cycling, round-0 refill; the mock GP truth set is frozen
(`SC.truth_points` = space_filling as of 02832ea) and simulated with the run's supervisor profile.

Experiment (mock, SYNTHETIC simulator + stub LLMs, X-5, video off, budget 48 = cold + rounds × 4, seeds 0–9;
`/workspace/coldstart_sens`, before = `results.csv` b_total48 @ 02832ea, after = `results_after.csv` @ #8d):

| variant | cold | runs | r0 runs | r0 fault types | r0 fault×onset | r0 both sides | campaign both sides | GP first active round | balanced acc mean±sd [min,max] | majority | truth safe | final safe | rejected (V-03/G-02/G-04) | wall s |
| before (b_total48) | 4 | 48.0 | 4.0 | 3.4 | 3.9 | 5/10 | 10/10 | 2-3 | 0.836±0.033 [0.79,0.88] | 0.657 | 0.657 | 5.6 | n/a | 32 |
| before (b_total48) | 8 | 48.0 | 8.0 | 4.6 | 6.4 | 7/10 | 10/10 | 1-2 | 0.828±0.027 [0.79,0.86] | 0.649 | 0.649 | 3.9 | n/a | 32 |
| before (b_total48) | 12 | 47.1 | 11.1 | 5.5 | 8.9 | 9/10 | 10/10 | 1-1 | 0.825±0.039 [0.75,0.86] | 0.650 | 0.650 | 4.0 | n/a | 31 |
| before (b_total48) | 24 | 44.0 | 20.0 | 6.0 | 13.0 | 8/10 | 10/10 | 1-1 | 0.813±0.027 [0.78,0.86] | 0.644 | 0.644 | 3.9 | n/a | 28 |
| after tight | 4 | 48.0 | 4.0 | 3.5 | 3.8 | 5/10 | 10/10 | 2-3 | 0.510±0.076 [0.39,0.67] | 0.880 | 0.880 | 23.5 | 1.3/8.5/4.8 | 51 |
| after tight | 8 | 48.0 | 8.0 | 5.4 | 7.1 | 6/10 | 9/10 | 1-2 | 0.506±0.044 [0.46,0.62] | 0.872 | 0.872 | 24.1 | 2.0/5.2/3.3 | 48 |
| after tight | 12 | 48.0 | 12.0 | 6.0 | 10.2 | 9/10 | 10/10 | 1-1 | 0.577±0.109 [0.41,0.77] | 0.872 | 0.872 | 22.8 | 2.7/4.7/3.5 | 48 |
| after tight | 24 | 47.9 | 23.9 | 6.0 | 17.9 | 10/10 | 10/10 | 1-1 | 0.566±0.083 [0.45,0.67] | 0.862 | 0.862 | 23.0 | 2.2/4.2/3.5 | 45 |
| after legacy | 4 | 48.0 | 4.0 | 3.5 | 3.8 | 4/10 | 10/10 | 2-3 | 0.786±0.051 [0.69,0.86] | 0.649 | 0.649 | 4.8 | 2.6/8.8/3.5 | 48 |
| after legacy | 8 | 48.0 | 8.0 | 5.4 | 7.1 | 7/10 | 10/10 | 1-2 | 0.805±0.080 [0.61,0.88] | 0.652 | 0.652 | 5.3 | 1.9/9.0/3.5 | 47 |
| after legacy | 12 | 48.0 | 12.0 | 6.0 | 10.2 | 10/10 | 10/10 | 1-1 | 0.812±0.033 [0.78,0.86] | 0.656 | 0.656 | 5.1 | 2.5/8.8/3.5 | 44 |
| after legacy | 24 | 47.9 | 23.9 | 6.0 | 17.9 | 10/10 | 10/10 | 1-1 | 0.812±0.037 [0.75,0.86] | 0.654 | 0.654 | 3.5 | 1.9/6.8/4.3 | 44 |

Reading / 해석: with the legacy supervisor (same truth labels as before) the sampler + rule change mostly the cold
start itself: round 0 really runs its size (11.1 → 12.0 at 12, 20 → 23.9 at 24), sees more fault × onset combos and
both sides of the boundary more often (12: 9 → 10/10); campaign balanced accuracy is unchanged within seed spread
(0.825 → 0.812 at 12). The tight supervisor makes the synthetic simulator recover far more often (final safe ≈ 5 →
≈ 23 per campaign, truth safe 0.65 → 0.87), so the GP mostly predicts the majority class (balanced ≈ 0.5–0.58): in
the mock the boundary moves to stronger faults that the space barely contains. Cold 12 vs 24 is within noise in every
variant; 4 sees both sides in round 0 only 4–5/10 and starts GP acquisition a round later.

Risks / 위험: (1) `gp_history_same_supervisor: true` → the next campaign's `--gp-history` (48 legacy runs) loads 0
runs; set false to reuse them. (2) the baseline cache key ignores the profile (no fault-free baseline fires under
tight). (3) Q-B4 expectations are written for tight. (4) supervised_lateral.py line refs were remapped mechanically.
(5) the tight effect, the mock boundary shift and all accuracy numbers are mock-only; nothing here ran on CARLA.

## 9. GP target = signed safety margin over every final_safe_recovery criterion (integration3 #9)

KO: 예전 GP 타깃 `lane_recovery_robustness`(lrr)는 차선 여유만 보았다. 그런데 `final_safe_recovery` 는 차선 +
안정 유지 + 횡가속도(P-02) + 횡저크(P-04) 를 모두 본다. result_8 에서 전환된 50런 중 25런이 실패였는데 그중 23런은
`safe_recovery` (차선 안, 안정 유지) 이면서 저크 초과로만 실패했다. lrr 는 50런 모두 ≥ 0 이라 부호가 final_safe 와
25/50 만 맞았고, GP 는 "실제로 위험한" 런을 경계 밖으로 볼 수 없었다. 이제 타깃은 판정 코드와 같은 기준·임계값
(`engine.monitor.thresholds_from_config`, 09 safety_spec.yaml 단일 출처) 으로 만든 연속 부호 여유다:

EN: the old target (lrr) saw the lane only; final_safe_recovery also needs settling and the P-02 / P-04 excess
labels. On result_8, 23 of the 25 failed switched runs were `safe_recovery` failed only by lateral jerk, and lrr was
≥ 0 on all 50 switched runs (sign = final_safe on 25/50). The new target uses exactly the judgment's criteria and
thresholds:

```
safety_margin = clip( min(terms), robustness_clip [-2, 1] )                (engine/judgment.py safety_margin)
  lane     = lane_recovery_robustness  (min lane margin / (lane_width/2 - half width); collision -> collision_penalty)
  settle   = (longest settled hold in the 3 s window - settle_hold_s 1.0 s) / settle_hold_s
             settled = |e_y| <= 0.3 m, |e_psi| <= 0.1 rad, lane margin >= 0 (same test as classify_recovery)
  lat_acc  = (P-02 limit 4.41 m/s^2 x tire_friction_scale - lat_acc_excess) / limit
  lat_jerk = (P-04 limit 5.0 m/s^3 - lat_jerk_excess) / limit
safety_margin >= 0  <=>  final_safe_recovery   (labels use a strict '>', so exactly at a limit is safe: margin 0)
binding = argmin term ('collision' when the lane term binds on a collision); failing = terms < 0
```

- monitor (`compute_run_record`) writes `metrics.settle_hold_max_s`, `safety_margin`, `safety_margin_binding`,
  `safety_margin_detail` (terms + observed value / unit / limit per criterion). Undefined (None) exactly where
  final_safe_recovery is: invalid runs, never-switched runs (switch classifier unchanged), no baseline.
- 02 `metric_rules.recovery_success.gp.target: safety_margin` (`target_choices: {safety_margin, lrr}`); scale 0.1,
  level 0, prior_mean = level unchanged. `python run.py --gp-target lrr` restores the old target
  (`run.select_gp_target`); mock with `--gp-target lrr` reproduces the pre-#9 campaign exactly (table below).
- Nemotron A: `gp_info.target/units/meaning` name the margin; region items are `predicted_<target>(_std)`;
  `outcome_view` / failure table / few-shot carry `safety_margin {value, binding, failing, terms: {margin, observed,
  limit, unit}}` in real units (m/s³, m/s², s held, lrr) and `failed_criteria`; the refine prompt says level 0 = the
  judgment boundary.
- `--gp-history`: old records (no margin) are re-scored from their logged metrics (`judgment.rescore_safety_margin`;
  friction-scaled limit from `metrics.p02_lat_acc_threshold_mps2`). safe_recovery / departure / collision keep their
  sign without the settle term; `unsettled` runs logged before #9 and any run whose re-scored sign disagrees with the
  logged final_safe are excluded with the reason in `loop_stats.gp_history.skipped`.
- report / dashboard / video selection use the active target (`boundary_band_runs` on the margin; lrr kept as
  `boundary_band_runs_lrr`; dashboard row "GP 판정 여유"); `gp_boundary_sign_accuracy.vs_final_safe` gives the GP sign
  against final_safe for either target.

Agreement (new monitor re-run on the logged CARLA runs, `/workspace/gpt/rescore.py`) / 부호 일치:

| data | switched, labelled | sign(safety_margin) = final_safe | sign(lrr) = final_safe | logged final_safe reproduced | binding |
|---|---|---|---|---|---|
| result_8 (84 runs) | 50 | 50/50 | 25/50 | 84/84 | lat_jerk 50 |
| result_7 | 3 | 3/3 | 2/3 | 8/8 | lat_jerk 3 |
| result_6 | 4 | 4/4 | 4/4 | 8/8 | lat_jerk 4 |
| mock seeds 0/1/2 (after) | 43/45/46 | 43/43, 45/45, 46/46 | 34/43, 36/45, 33/46 | – | lat_jerk 27/25/35, lane 16/19/9 |

Offline GP on result_8 (same features: numeric + DF + B answers of the last round, same GP code, seed 6;
`/workspace/gpt/offline_gp.py`; truth = final_safe, 25 safe / 25 failed). Majority baseline balanced accuracy = 0.5
(LOO majority plain accuracy is 0.0 on a 25/25 split; round-by-round 0.405):

| target | LOO balanced acc | LOO AUC | round-by-round (n 42) balanced acc | round-by-round AUC | predicted safe (LOO) |
|---|---|---|---|---|---|
| lrr (old) | 0.50 | 0.795 | 0.50 | 0.739 | 50/50 |
| safety_margin (new) | **0.76** | **0.859** | **0.741** | **0.798** | 21/50 |

(The 0.57 AUC quoted for result_8 came from the campaign's own logged `gp_mean` with fewer training runs; the offline
lrr refit here ranks better but its sign never says "fail".)

Mock (`PYTHONHASHSEED=0`, SYNTHETIC simulator + stub LLMs; default seed = seed 0). `vs FS` = GP sign vs final_safe on
the frozen 26-point truth set (majority 0.808):

| seed | before 78f0169: switched / final safe / failures | after: switched / final safe / failures | after `--gp-target lrr` | GP vs FS balanced acc, lrr → safety_margin |
|---|---|---|---|---|
| 0 | 44 / 32 / 12 | 43 / 22 / 21 | 44 / 32 / 12 | 0.60 → 0.70 |
| 1 | 50 / 40 / 10 | 45 / 33 / 12 | 50 / 40 / 10 | 0.476 → 0.676 |
| 2 | 53 / 36 / 17 | 46 / 32 / 14 | 53 / 36 / 17 | 0.60 → 0.60 |

Jerk limit (data only, threshold NOT changed) / 저크 한계 (임계값 변경 없음): `safety_spec.yaml P-04
lateral_jerk_threshold_mps3: 5.0` ("PROVISIONAL. 사용자 제공 참고값 ISO 11270:2014 (E-01)"), default
`judgment.PROVISIONAL_RECOVERY`; jerk = d(a_y)/dt with a 0.5 s moving average, excess = max |j − j_baseline(s)| aligned
by distance in the 3 s post-switch window. result_8, 65 fault-free baselines (after the first 2 s): per-run max |j|
median 3.93, p90 11.04, max 18.23 m/s³, 26/65 exceed 5 somewhere; all samples p99 4.29, p99.9 8.56. Switched runs:
final_safe jerk_excess median 2.13 (max 4.87), failed median 11.41 (min 5.25, max 24.82); of 48 `safe_recovery` runs,
23 fail on jerk at 5 (19 at 7.5, 14 at 10, 2 at 15). The limit alone decides about half of the result_8 labels, and
the fault-free stack itself exceeds it in 40 % of baselines (the baseline-relative excess removes most of that), so the
provisional 5.0 deserves a reviewed source before conclusions are drawn.

Not changed here / 이번에 안 바꾼 것: switch classifier and switch-uncertainty gating, range widening, thresholds.
The new target makes the margin almost always jerk-bound on CARLA (50/50 runs), so the GP now models jerk excess;
whether the switch gate or range widening matter should be re-measured on the next CARLA campaign with this target.

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
`tests/test_integration3_nemotron_b.py`, `tests/test_integration3_nemotron_a_context.py`,
`tests/test_integration3_supervisor.py`, `tests/test_integration3_onset_exclusions.py`,
`tests/test_integration3_cold_start.py`, `tests/test_integration3_safety_margin.py` (plus updated existing tests). Root suite, `pipeline/09_monitor` and the av_stack suite (without GPU / reference
tests) pass.
