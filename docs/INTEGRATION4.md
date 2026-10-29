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
