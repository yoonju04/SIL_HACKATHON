# integration4

`integration4` = `integration3` @ `c03dd95` (everything except `result/`) + PR #4 (dashboard redesign, English UI)
+ log-grounded video explanation (below). `result/` keeps integration4's own campaigns (result_1..5, result_7,
result_8; result_6 lives on integration3). The tight supervisor, route onset exclusions and the safety-margin GP
target come from integration3 (#8, #9; see `docs/INTEGRATION3.md`).

## Merge note (PR #4)

PR #4 is based on integration3 `78f0169`; integration3 #9 later edited `engine/dashboard.py`. The PR #4 version was
kept and #9's additions ported in English: a "GP margin (safety_margin)" row with the tightest criterion
(`SM_EN`), and the "Actual (margin)" column of the hypothesis table.

## Video explanation: log_grounded (stage 10)

**Problem (result_8, 84 runs, tight supervisor):** 23 of 24 explanations were rejected, all on
`departure_time_s`. The monitor logged no lane departure in any of the 24 runs; the model claimed one in 23
(12 times at the switch time it was given, often "every frame after the switch is outside the lane").
1. A front camera does not show the wheels against the lane line, yet the model had to find departures in it.
2. The questions and most tags presume a failure.
3. The selected "failed" runs were mostly lateral-jerk-only failures that kept the lane: nothing to see.

**Change**
* `pipeline/10_video_explainer/video_explainer.yaml prompt.mode: log_grounded` (default; `pixel_check` = before).
  `engine/explainer.py`: the monitor's lane-departure intervals, first departure after the switch, collisions,
  max offset, angle / rate saturation and safe stop are given as facts (`GROUNDED_TIMELINE_KEYS`; the final label
  stays hidden). The model is asked what the clip shows around those events and why (`SYSTEM_GROUNDED`, neutral
  questions, "say so (stable_recovery)" if nothing went wrong), and answers only mechanism, tags, novel tags,
  evidence frames, counterfactual and uncertainty (`GROUNDED_FIELDS`).
* `verify_grounded`: tags exist and do not contradict the log (`tag_conflicts`: departed_before_switch needs a
  departure before the switch, collision_after_departure a collision, stable_recovery no departure after the switch
  and no collision, saturation tags a saturation ratio > 0), evidence frames are in the clip and at least one sits
  at a given event (switch, departure interval, collision), and a mechanism of at least 20 characters is written.
  Whether the mechanism text is right is not machine-checked: spot-check the first explanations by hand.
* Run selection `select_runs.failed_rule: visible` (default; `any` = before): `engine.feedback.visible_failure` =
  lane departure or collision (SR4), late / no switch (F1), recovered after departure or not recovered.
  Boundary-near runs whose tightest criterion is lateral jerk / acceleration are skipped too.
* Cosmos stays the model (Dedicated Endpoint, `--cosmos-power auto`); the Super-VL fallback uses the same mode.
  Fewer, more relevant runs also keep the endpoint on for less time.

Tests: `tests/test_engine_video.py` (grounded prompt, grounded verify), `tests/test_engine_feedback_visible.py`.
Mock check (`run.py --mock --ablation X-6`, SYNTHETIC stub): the two rejections were the stub's deliberate
contradictions (`collision_after_departure` without a collision).

Not measured yet: how Cosmos does with the new prompt on real clips (needs a real run with the endpoint on).

## Judgment fixes after result_9 / result_10

Three changes, checked by re-grading result_8..10 (113 valid runs, 70 switched, tight supervisor; no CARLA).

1. **P-04 lateral-jerk limit 5 -> 15 m/s^3** (`pipeline/09_monitor/safety_spec.yaml`; decides both `lat_jerk_excess` /
   `final_safe_recovery` and the GP `safety_margin`). Every switched run of result_9/10 had `lat_jerk` as the binding
   term (bias/drift at -2.0, even a normal solver_fail at -1.17), so the GP learnt "how abrupt the handover was".
   Measured: fault-free baselines reach |jerk| p90 9.5 and max 18.2 m/s^3 in curves (n = 91), so 5 (ISO 11270
   reference, E-01) is below normal driving. Pre-onset excess (fault and baseline identical there) is ~0 (median 0.1),
   so the distance alignment is fine. Prompt switches: excess mostly <= 5, up to ~13 in curves where the baseline
   itself was over 5 (13 of 18). Late `tracking_divergence` switches (bias/drift): 11..25 for all 17 - LQR pulling a
   car that is already 0.6-0.7 m off back to the path. 15 keeps the latter and drops the curve noise.
2. **Never-switched runs get a GP target** (`judgment.no_switch_safety_margin`, `rescore_no_switch` for
   `--gp-history`): lane term only = min footprint margin / (lane_width/2 - half width) over the evaluation window,
   collision <= penalty. Before, they had none, so result_9 round 1 (4/4 no switch) added nothing to the target GP,
   while the bias boundary (0.035 rad undetected and harmless vs 0.045..0.055 rad late) lies exactly there.
   `orchestrator.yaml switch_penalty` 1.0 -> 0.0 for the same reason. History re-scoring always recomputes (the
   stored margin and label may use older thresholds) and checks the sign against the label re-derived now.
3. **One lane-departure definition** (`engine/monitor.py`): departure intervals, recovery margins and
   `lane_recovery_robustness` use the vehicle footprint (4 corners incl. heading error), as SR4 / P-01. Before, the
   centre-line margin (no heading) gave result_10's drift runs "no departure" in the timeline while SR4 had a
   corner 0.04 m over the line, and the video explanation said "did not cross the lane line" and passed.

Re-grade result_8 + result_9 + result_10:
* final_safe_recovery 35 -> 56 of 70 switched runs.
* GP target defined for 20 -> 113 runs (43 never-switched runs now included; none of them left the lane).
* Switched runs with margin < 0: only bias (8) and drift (6) - lane 6, lane+settle 2, lane+lat_acc 1,
  lat_acc+lat_jerk 2, lat_jerk only 3. No control fault, saturate or intermittent-bias run fails any more.
* SR4 "out of the lane" and the timeline departure now agree in 9 of 9 runs.

Tests: `tests/test_integration3_safety_margin.py` updated (limit 15, no-switch re-scoring, re-derived labels) plus
`test_no_switch_margin_and_footprint_departure`.

## Critical faults and boundary values (`engine/boundaries.py`)

Goal of the campaign insights: which faults are a problem and where their boundary is. The old Cosmos campaign
insights (`pipeline/09_monitor/explain.py`, kept as is) only quoted group means (result_11: "failed runs had 4.2 s
detection latency"), and no boundary.
* Code builds the table (deterministic): the campaign's runs plus `--gp-history` runs, all re-scored with the current
  thresholds (`judgment.rescore_safety_margin`). Per fault: runs, harmful rate (safety_margin < 0), late/missed
  detection rate (SR F1), worst margin, detection sources; ranking = harmful rate, then late rate, then worst margin.
  Per fault parameter and condition group (all, speed band, onset event, friction < / >= 0.7): the bracket for the harm
  and the detection boundary - bracketed [max OK, min BAD], overlap [min BAD, max OK], all_ok, all_bad.
* Cosmos (C_cosmos, text only) reads only that table and answers critical_faults, boundaries (copying a bracket, or
  `estimated` approximate ranges inside the space range when the data do not bracket it), condition_effects and
  next experiments that narrow the boundary. `check_answer` keeps only items that agree with the table (names,
  ranges, run ids, no Han / kana) and re-asks with the problems; the code summary is written either way.
* Runs after the explain.py insights in `run.py campaign_insights` (`--insights on`, history = `--gp-history`);
  standalone: `python -m engine.boundaries OUT --history "result/result_*/run_records.jsonl" [--mock]`.
  Outputs `insights/boundaries.md` and `insights/boundaries.json`, plus a line in `report.md`.
* Code table on result_8..11 (119 runs): drift (late 100 %, harmful 20 %), bias (harmful 14 %, late 19 %), then
  intermittent_bias; saturate, solver_fail, output_delay, stale never harmful.

Video explanation (result_11): the prompt now says that trigger_time_s is when the fault became effective (not its
detection), what the angle vs rate saturation ratios mean, that fault names are not tags; `novel_tags` must be
English snake_case (new check `novel_tags_snake_case`).
