# Failure-type stacking, J-regression tidy-up and Laya calibration (laya-judge, 2026-11-10)

Offline only: no CARLA run and no new LLM call. Everything below is computed from saved run records
(`result/result_8..12/run_records.jsonl`) and saved judge answers (`/workspace/judges_eval/raw_nemotron.json`,
`raw_laya.json`, `/workspace/judges_eval_digest/raw_laya_digest.json`). Raw outputs of every table are in
`/workspace/stacking_eval/` on the box. **Sample sizes are very small (3 to 12 positives per failure type), so every
number here moves by a lot when one run changes.**

Independence rule (unchanged): run outcomes and answer keys are only labels. They never go into a judge state, and
the J regression is refit inside every evaluation fold, because J2, J5 and J6 are labelled by run outcomes.

## 1. Failure-type layer (05c, `engine/failure_types.py`, `pipeline/05c_failure_types/failure_types.yaml`)

**Labels.** Each switched, `sim_status ok` run gets four binary labels, and more than one can be on at once:

| type | rule (record fields) | pooled result_8..12 (97 switched runs) |
|---|---|---|
| jerk | `metrics.lat_jerk_excess_mps3 > 15` (P-04 after integration4) | 12 |
| lat_acc | `metrics.lat_acc_excess_mps2 > 4.41 × tire_friction_scale` (P-02) | 6 |
| lane_departure | `lane_recovery_robustness < 0`, `recovery_outcome ∈ {recovered_after_departure, unrecovered}` or `recovery_labels.departed_at_switch` | 3 (12 with `include_sr4_footprint: true`) |
| unsettled | `recovery_outcome == unsettled` | 3 |

Positives per campaign (jerk / lat_acc / departure / unsettled):

- result_8: 2/2/0/2 (50 runs)
- result_9: 2/1/0/0 (9)
- result_10: 1/0/0/0 (11)
- result_11: 1/1/1/0 (5)
- result_12: 6/2/2/1 (22)

80 runs have no failure type. Label combinations:

| combination | runs |
|---|---|
| jerk only | 8 |
| unsettled only | 3 |
| jerk + lat_acc | 2 |
| jerk + lat_acc + departure | 2 |
| lat_acc only | 1 |
| lat_acc + departure | 1 |

The jerk limit is applied as 15 m/s³ to every campaign, including result_8–10, which were recorded under the old limit of 5. result_8's
`lane_recovery_robustness` predates the integration4 footprint margin.

**Inputs** (`inputs:`): `df` is the GP's DF columns (DF-03, DF-05, DF-06, DF-07..09 and their `_missing` flags).
DF-01, DF-02 and DF-04 are not GP inputs, because their profile values are null. `judges` is the J regression probabilities the GP
already gets (`J1→B_*`, `J3/J4/J7=p(·)`, ordinal `J2/J5/J6`). `judge_uncertainty` is two columns: the mean pairwise TV
distance between variants (`judge_disagreement`) and the TV distance between the mean forward-order and mean reverse-order
variants (`judge_order_gap`). `scenario` is the rest of the GP vector (route, speed, friction, fault one-hot, parameters, onset).

**Models.** The baseline `logistic_balanced` is a per-type L2 logistic with class_weight='balanced'. Its logit is
prior-corrected, so the thresholds apply to calibrated-ish probabilities. `hier_bayes` is a hierarchical Bayesian
logistic: w_t = μ + δ_t with μ ~ N(0, 1) and δ_t ~ N(0, τ²), fitted as MAP plus Laplace, with τ chosen by Laplace
evidence on a grid. Both models give an epistemic logit variance per type. The decision for each type is
`p > c_fa / (c_fa + c_miss)`. The placeholder costs are false alarm 1 and misses jerk 2, lat_acc 5, departure 20 and
unsettled 5, so the departure threshold is 0.048. **The costs are a team decision.**

The optional ordinal severity model uses a cumulative logit over
`none < jerk_only < lat_acc < unsettled < departed_recovered < unrecovered`. **This order is a team decision, not
agreed.** The model is off by default.

**Results.** Each cell gives recall at the cost threshold / PR-AUC (chance = prevalence) / tp-fp-fn. LOO = leave one
run out. Forward = train on the earlier (campaign, round) blocks only and test the next block (≥ 20 training runs).

| data, inputs, model | eval | jerk | lat_acc | lane_departure | unsettled |
|---|---|---|---|---|---|
| pooled 97, df, logistic_balanced | LOO | 0.58 / 0.39 (0.12) / 7-5-5 | 0 / 0.10 (0.06) / 0-10-6 | 0.33 / 0.06 (0.03) / 1-25-2 | 0 / 0.03 (0.03) / 0-2-3 |
| | forward | 0.08 / 0.44 (0.16) / 1-3-11 | 0 / 0.13 (0.07) / 0-3-5 | 0 / 0.14 (0.04) / 0-5-3 | 0 / 0.05 (0.03) / 0-5-2 |
| pooled 97, df, hier_bayes | LOO | 0.42 / 0.38 / 5-5-7 | 0 / 0.09 / 0-11-6 | 0.33 / 0.05 / 1-38-2 | 0 / 0.06 / 0-3-3 |
| | forward | 0 / 0.49 / 0-1-12 | 0 / 0.11 / 0-14-5 | 0.33 / 0.07 / 1-21-2 | 0 / 0.06 / 0-9-2 |
| pooled 97, df+scenario, logistic_balanced | LOO | 0.58 / **0.60** / 7-3-5 | 0 / 0.07 / 0-13-6 | 0.33 / 0.11 / 1-22-2 | 0 / 0.05 / 0-5-3 |
| | forward | 0.25 / 0.56 / 3-0-9 | 0 / 0.07 / 0-6-5 | 0 / 0.12 / 0-6-3 | 0 / 0.04 / 0-5-2 |
| pooled 97, df+scenario, hier_bayes | LOO | 0.67 / 0.43 / 8-8-4 | 0.17 / 0.07 / 1-22-5 | 0.67 / 0.07 / 2-50-1 | 0.33 / **0.15** / 1-12-2 |
| | forward | 0.50 / **0.59** / 6-5-6 | 0.40 / 0.07 / 2-31-3 | 0.33 / 0.05 / 1-47-2 | 0.50 / 0.03 / 1-31-1 |
| result_8 50, df, logistic_balanced | LOO | 0 / 0.05 (0.04) / 0-0-2 | 0 / 0.11 (0.04) / 0-1-2 | no positives | 0 / 0.06 (0.04) / 0-3-2 |
| result_8 50, df+judges+uncertainty, logistic_balanced | LOO | 0 / 0.14 / 0-1-2 | 0 / 0.11 / 0-4-2 | no positives | 0 / 0.03 / 0-10-2 |
| result_8 50, df+judges+uncertainty, hier_bayes | LOO | 0 / 0.13 / 0-1-2 | 0.5 / 0.11 / 1-11-1 | 0-27-0 | 0 / 0.04 / 0-13-2 |
| | forward | 0.5 / 0.30 (0.07) / 1-3-1 | 0 / 0.06 / 0-13-1 | 0-22-0 | 0 / 0.05 / 0-11-1 |

The ordinal severity model (pooled, df, LOO) gets PR-AUC 0.43 for "any failure" (chance 0.18) and 0.06 for "departed or worse"
(chance 0.03).

With `include_sr4_footprint: true`, lane departure counts any footprint departure in onset..onset+10 s, pre-switch too.
That gives 12 positives, and df+scenario reaches LOO PR-AUC 0.90 (logistic) and 0.85 (hier) against chance 0.12. Under the
0.048 cost threshold it flags most runs, though (41 to 65 false alarms). These departures concentrate in a few routes and
speeds, so the scenario columns separate them.

**Reading.**

- **jerk is the only type with a usable signal.** PR-AUC is 0.4–0.6 against 0.12–0.16 chance, in both LOO and forward
  order, and scenario columns help. lat_acc, departure (post-switch) and unsettled are at or near chance. With 3–6
  positives that is the expected outcome, not a tuning problem.
- **hier_bayes vs the balanced logistic.** The shared prior raises rare-type recall (pooled df+scenario: unsettled
  0 → 0.33 LOO and 0 → 0.5 forward, lat_acc 0 → 0.4 forward, departure 0.33 → 0.67 LOO). It pays for that with many more
  false alarms at the low cost thresholds. Its PR-AUC is not better on the rare types. The shared prior also predicts
  departures in result_8, which has 0 positives (0-27-0). Recall goes up because the hierarchical model is less sure,
  not because it ranks better.
- **The judge features cannot be assessed for failure types.** Judge answers exist only for result_8, which has 0–2
  positives per type. PR-AUC moves by about ±0.08 between input sets, which is noise at n = 2.
- As with DF-07..09 before, LOO and forward order disagree in places (jerk recall 0.58 LOO vs 0.08 forward with df only).
  Report the forward numbers.

## 2. J regression tidy-up (`engine/judges.py`, `judges.yaml regression.*`)

Here is what changed:

- **Compressed inputs** are the default (`inputs: compressed`). Each question gets K mean log-probs across variants, the std
  across variants and the forward−reverse gap (J1: 6 inputs instead of 16, J4: 8 instead of 18). The old layout stays behind
  `inputs: per_judge`.
- **Ordinal J2 and J5** use a cumulative logit with one shared slope (`ordinal: [J2, J5]`).
- **`model: bayes_laplace`** is optional. It is a Laplace Bayesian multinomial whose answer rows carry `epistemic` (the
  mutual information of the posterior predictive). `judges.judge_epistemic()` turns that into a state-level signal.

Tool: `tools/j_regression_offline_eval.py`.

Nemotron, 82 result_8 states (n = 74–81 per question, 50 for J6). Each cell gives accuracy / Brier. The fault-mode lookup is judge-free (LOO), and the LOMO
majority is the majority label of the training fault modes.

| q | per_judge LOO | compressed LOO | compressed+ordinal LOO | lookup | per_judge LOMO | compressed LOMO | +ordinal LOMO | Bayes LOMO | LOMO majority |
|---|---|---|---|---|---|---|---|---|---|
| J1 | 0.797 / 0.310 | 0.797 / 0.304 | same | 0.743 | 0.784 / 0.514 | 0.784 / 0.512 | same | 0.784 / 0.498 | 0.297 |
| J4 | 0.784 / 0.324 | 0.784 / 0.313 | same | 0.743 | 0.743 / 0.547 | 0.743 / 0.528 | same | 0.757 / 0.529 | 0.297 |
| J7 | 0.743 / 0.361 | 0.770 / 0.356 | same | 0.743 | 0.757 / 0.467 | 0.757 / 0.479 | same | 0.757 / 0.474 | 0.622 |
| J2 | 0.544 / 0.595 | 0.544 / 0.593 | 0.532 / 0.611 | 0.747 | 0.456 / 0.803 | 0.494 / 0.784 | **0.544 / 0.739** | 0.544 / 0.739 | 0.152 |
| J5 | 0.519 / 0.640 | 0.519 / 0.637 | 0.543 / 0.611 | 0.556 | 0.519 / 0.718 | 0.519 / 0.714 | **0.543 / 0.655** | 0.543 / 0.655 | 0.457 |
| J6 | 0.760 / 0.369 | 0.760 / 0.369 | same | 0.740 | 0.680 / 0.439 | 0.680 / 0.438 | same | 0.680 / 0.434 | 0.760 |

Laya digest, 42 states. Compressed inputs leave Laya at the LOMO majority, about 0.5 for J1 and J4, where per_judge gave
0.34 and 0.37. That is not an improvement in what Laya knows: the compressed regression simply stops over-fitting
Laya's noise. LOO falls from 0.61 to 0.50 (J1) and from 0.76 to 0.58 (J4). All of these are below the judge-free lookup (0.84).

**Reading.**

- Compression keeps the same accuracy with a third of the inputs and a slightly better Brier, so it is the default.
- The ordinal regression helps J2 and J5 under LOMO (+0.05 and +0.02 accuracy, Brier 0.78 → 0.74 and 0.71 → 0.66). The
  outcome questions still stay below the judge-free fault-mode lookup in LOO.
- **The epistemic value does not flag the regression's errors.** The LOMO AUROC of epistemic MI for "this answer is
  wrong" is 0.26 (J1), 0.43 (J4), 0.49 (J7), 0.49 (J2), 0.55 (J5) and 0.36 (J6); 0.5 is chance. Under a new fault mode the
  errors are confident and in-distribution for a linear model on 6 inputs. It stays as an exposed signal
  (`epistemic` in b_cache rows, `judge_epistemic`), but it is not evidence that "the judge does not know here".

### Ridge option (`model: ridge`)

`--judge-regression-model ridge` (or `regression.model: ridge`) is an L2 least-squares model: RidgeCV picks alpha by
generalized cross-validation, inputs are z-scored on the training fold, and scores are clipped to [0, 1] then
renormalised (one-vs-rest; two classes reduce to one 0/1 ridge). There is no epistemic value. Ordinal-coded ridge
(targets as 0..K−1, a triangular kernel around the clipped prediction, argmax ≈ rounding) exists but is **opt-in**
(`ridge_ordinal: true`). It is not used just because a question is in `ordinal: [J2, J5]`, so flipping the model flag
does not silently replace the cumulative logit.

Same Nemotron / result_8 data, compressed inputs. LOMO accuracy / Brier, next to the current default (compressed L2,
with cumulative-logit J2/J5) and the judge-free baselines. J3 / J8 / J9 have no regression (J3 has 15 single-answer
labels, all `caught_by_rate_check`; J8/J9 are constant / report-only).

| q | lookup (LOO) | LOMO majority | L2 (J2/J5 ordinal) LOMO | ridge OvR LOMO | ridge ordinal-coded LOMO |
|---|---|---|---|---|---|
| J1 | 0.743 | 0.297 | 0.784 / 0.512 | 0.784 / 0.492 | (same as OvR) |
| J4 | 0.743 | 0.297 | 0.743 / 0.528 | 0.743 / 0.554 | (same as OvR) |
| J7 | 0.743 | 0.622 | 0.757 / 0.479 | **0.622 / 0.499** | (same as OvR) |
| J2 | 0.747 | 0.152 | **0.544 / 0.739** | 0.443 / 0.752 | 0.063 / 1.273 |
| J5 | 0.556 | 0.457 | **0.543 / 0.655** | 0.519 / 0.633 | 0.346 / 0.757 |
| J6 | 0.740 | 0.760 | 0.680 / 0.438 | 0.680 / 0.456 | (same as OvR) |

Round-order / forward (train on earlier rounds only; n = 42–69) does not change the reading: ridge OvR is a little
better on J5 (0.551 / 0.592 vs L2 0.507 / 0.644) and J6 accuracy (0.762 vs 0.714), and a little worse on J1 and J4.
One-vs-rest AUROC under LOMO is similar for J1/J4/J7 and worse for J6 (ridge 0.311 vs L2 0.480).

**Reading.** The default stays `logistic_l2` with ordinal J2/J5. Ridge OvR is not clearly better: J1's LOMO Brier
improves slightly (0.512 → 0.492) and J5's Brier improves without a change in accuracy, but J7's LOMO accuracy falls
from 0.757 to the training majority (0.622). Ordinal-coded ridge is harmful on J2: the labels are U-shaped (34
`under_0_5s`, 33 `over_5s_or_never`, 6+6 in the middle), least squares predicts the mean of the two ends, and that
lands on the rare middle buckets (LOMO 0.063, Brier 1.273). The cumulative logit does not have that failure mode.

## 3. Laya calibration and fine-tuning

`.venv-laya` ships laya 0.4.17 with a real fine-tune path (`laya-train`, `laya.train.finetune`). It supports
`--loss soft-ce` (soft targets), `--freeze-encoder` (head only), `--shuffle-options` (option-order augmentation) and
`--eval` for a before/after check. Calibration is per-type temperature (`laya.calibrate`).

**Temperature scaling** (`tools/laya_calibration_eval.py`): one T per code question (J1 J3 J4 J7 J8 J9; outcome
questions excluded), fitted on held-out folds against the code answer key. List answers use a soft target, uniform over
the allowed set.

| Laya digest, 42 states | T | k-fold ECE before → after | Brier | LOMO ECE | raw accuracy |
|---|---|---|---|---|---|
| J1 | 2.5 | 0.16 → 0.07 | 0.68 → 0.66 | 0.16 → 0.14 | 0.46 |
| J4 | 1.3 | 0.25 → 0.25 | 0.69 → 0.69 | 0.25 → 0.27 | 0.42 |
| J7 | 50 (bound) | 0.41 → 0.15 | 0.92 → 0.70 | 0.41 → 0.15 | 0.10 |
| J8 | 50 (bound) | 0.54 → 0.22 | 1.21 → 0.68 | 0.54 → 0.22 | 0.12 |
| J9 | 5.0 | 0.39 → 0.32 | 0.85 → 0.66 | 0.39 → 0.35 | 0.44 |

- Temperature does not change any judge's argmax, so raw accuracy is identical by construction.
- Where T hits the upper bound (J3, J7, J8), the best-calibrated Laya answer is uniform: the probabilities carry no
  usable information.
- With the regression on top, LOMO goes from 0.34 to 0.50 (J1), from 0.37 to 0.47 (J4) and from 0.34 to 0.16 (J7). The
  LOMO majority is 0.50, 0.50 and 0.16. Calibration moves Laya onto the majority answer, not above it.
- Nemotron for reference is over-confident: mean confidence 0.96 against accuracy about 0.8, with T about 5–6. NLL drops
  from 2.8 to 0.9, but LOMO ECE gets worse (0.22 → 0.42).

**Fine-tuning** (soft targets, code questions only) is set up but not evaluated. `tools/laya_finetune_data.py` writes
laya-train JSONL from result_8 scenario inputs. The labels come from `expected_from_code`, uniform over the allowed set for
list answers, and only J1 J3 J4 J7 J8 J9 are included. `--holdout-mode` writes one fault mode to eval.jsonl. For drift
held out that is 385 train rows (1177 items) and 25 eval rows (70 items).

The recipe, CPU, head only:

```bash
python tools/laya_finetune_data.py --out /workspace/laya_ft --holdout-mode drift      # one run per held-out mode = LOMO
laya-train --data /workspace/laya_ft/train.jsonl --eval /workspace/laya_ft/eval.jsonl \
  --base ~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/<rev>/multilingual \
  --out /workspace/laya_ft/ckpt_drift_head --loss soft-ce --shuffle-options --freeze-encoder --epochs 2 \
  --grad-accum 2 --device cpu
# then: LAYA_MODELS=/workspace/laya_ft/ckpt_drift_head laya-serve, and tools/judges_offline_eval.py --backend laya
```

Training the full encoder (drop `--freeze-encoder`) needs a GPU. On this 8-core CPU, the head-only pilot above took more than
25 minutes before its first log line. 82 states are also few: the answer key is a function of fault mode and parameters,
so a fine-tune can memorise the key per mode. Only a held-out-mode check tells whether it learned to read the code.

## 4. Search hook (07, `run.py Loop.rare_type_picks`)

It is off by default. `--rare-type-picks N` (1–2 suggested) reserves N batch slots per round, taken before the
LCB/level-set slots like the Nemotron A share. `acquisition.mode` (or `--rare-type-mode`) picks how they are filled.

### Mode `quiet_seeds` (default, `engine/rare_seeds.py`, config `acquisition.seeds`)

The 05c classifier cannot rank the rare types (3–6 positives each), so "near a rare type" under it is close to random.
The rare-type slots are therefore seeded where a fault is **quiet for the tight supervisor**: no instant check trips,
only accumulated tracking error (`tracking_divergence` 0.6 m / 0.35 rad held 0.15 s) can switch, so a late switch hands
LQR a degraded state. Every bound comes from the code values in `scenario_space_av_stack.yaml supervisor` and
`09 safety_spec.yaml`, not from guesses:

| source | what it generates | bound and where it comes from |
|---|---|---|
| `bias_sub_threshold` | F-04 bias / F-07 intermittent_bias | magnitude 0.06–0.078 rad; upper bound kept under one steer-rate tick = 90 °/s × 0.05 s = 0.0785 rad (clipped there if configured higher) |
| `slow_drift` | F-05 drift | rate 0.005–0.0157 rad/s; 0.0157 = rate tick / evaluation window min length 5 s, so the offset over the shortest graded window stays under one tick (the per-tick step, ≤ 0.0025 rad, never trips the rate check for any drift in range) |
| `curve_entry` | either quiet fault at `onset_event: curve_entry`, offset −0.5…0.5 s | `route_onset_exclusions` apply; speed is lowered in 5 km/h steps until a route can grade curve_entry |
| `low_friction` | either quiet fault at `tire_friction_scale` 0.3–0.6 | friction is a scenario parameter (`run_benchmark --tire-friction-scale`); skipped with a note if the space has no friction axis |
| `rare_neighbours` | `scenarios.perturb` copies (scale 0.08; route, fault, onset event fixed) of known rare failure runs | lane departure incl. SR4 footprint, unsettled, lat_acc; from `result_8..12/run_records.jsonl` + this campaign, same supervisor profile only |
| `b_proposals` | the 05b B proposals already in the pool (source B: bias at 0.9 × rate tick, +5 km/h) | reused, not regenerated |

Quiet sources sample speed 25–40 km/h: in result_8..12 (tight), bias 0.06–0.078 rad switched 1/13 times at ≤ 20 km/h
and 6/7 at ≥ 25 km/h. A run that never switches gets no failure-type label (LQR never acted).

Each source makes `per_source` (4) candidates per round. They join the pool before stage 04, so they pass the same
gate, V-03 and dedup (0.05) and get judge answers and GP predictions. The N picks are at most one per source
(`max_picks_per_source`), skip points within `min_distance_to_run` (0.05) of a run point or an earlier pick, and then go
through `enforce_diversity` with the rest of the batch (the 50 % per-fault cap). Ranking (`seeds.ranking`):

- `gp_boundary` (default): the 07 level-set boundary score on the **active recovery GP target** once the GP is ready.
  That is `safety_margin` (or `lane_recovery_robustness` with `--gp-target lrr`). `safety_margin` is the minimum over
  the lane, settle, lat_acc and jerk terms, so its boundary covers the rare types, but jerk can dominate it.
  `lat_acc_excess_mps2` has a `metric_rules` entry with a GP target, but it is not an active objective, so the loop
  fits no GP for it. No separate lane-margin or lat_acc GP is fitted for this hook.
- Fallback and tie-break: the normalized feature distance to the nearest known rare failure run.

Picks are labelled `selected_by: rare_type_seed:<source>`. `loop_stats.json` logs `rare_seeds` (generated per source,
bands, notes) and `rare_type_picks` (ranking used and each pick's source, fault, speed, friction, onset and scores).
`07_batch.json` carries `rare_seed_source` / `rare_seed_of`. Under `--real`, seeds with new judge keys add 05b judge
calls (cached per key).

### Mode `classifier` (117a724)

Each round the 05c model (`model.method`, default hier_bayes) is fitted on the campaign's labelled switched runs once
there are at least 12 (`min_train_runs`). N slots go to the pool candidates with the highest
`weight × max over rare types of P_t(x) × epistemic_t(x)` (lane_departure, unsettled, lat_acc; epistemic = logit
variance), labelled `selected_by: rare_type_epistemic`. Kept for comparison only.

### Jerk labels across campaigns

`type_labels` re-labels every campaign from the raw `metrics.lat_jerk_excess_mps3` with the one limit in this file
(15 m/s³), so result_8..12 are consistent here: 12 jerk positives among the 97 switched runs (2/2/1/1/6). Never-switched
runs have no `lat_jerk_excess_mps3`, so counting all `ok` runs also gives 12. The `recovery_labels.lat_jerk_excess`
stored in the run records were written at run time and are **mixed**. result_8–10 used the old limit of 5 (stored
positives start at 5.25, 5.78 and 11.95; negatives reach 4.87, 3.13 and 4.37). result_11–12 used 15 (negatives reach
5.65 and 12.31; positives start at 24.2 and 16.4). Stored primary positives are 42 among the switched runs and 52 among
the ok runs. Re-labelled at 5, the switched count would be 49. Anything that reads the stored labels mixes the two
limits; the 05c layer does not.

## How to enable in a CARLA run

```bash
python run.py --real --judge-b nemotron \
  --judge-regression-inputs compressed      # already the judges.yaml default; per_judge = old layout
  --judge-regression-model bayes_laplace    # optional: per-answer epistemic in b_cache
  --rare-type-picks 2                       # optional: 2 slots/round for rare failure types (default off)
  --rare-type-mode quiet_seeds              # default; classifier = the 117a724 model-based picks
```

The failure-type costs, the label rules and the ordinal order are in `pipeline/05c_failure_types/failure_types.yaml`.
The costs and the order are placeholders for the team to decide.
