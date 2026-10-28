# 09 monitor compatibility audit (2026-10-27)

Target branch: `junwoo/sim-eval`. Only `pipeline/09_monitor/` is changed.
Runner reference: `new_pipeline` commit `d743fac6ec11b5f8c1ba53033dc2523483db59d1`.

## Findings and changes

- The branch-local monitor uses the earlier 28-field record. The latest runner exports a
  37-field contract from `pipeline/engine/judgment.py` through `scripts/runner_skeleton.py`.
  `runner_adapter.py` loads the actual engine sources and preserves its complete output,
  adding only `metrics.monitor` with the versioned SR assessment.
- The latest runner uses post-switch recovery outcomes and baseline-relative acceleration/
  jerk labels. They must not be replaced with the legacy P01/P02 safe label or equated with
  the SR verdict. All runner labels and GP targets are preserved exactly.
- Normalized applied steering in `agent_controls.jsonl` now converts with the logged
  `agent_config.json.geometry.max_steer_rad`. Missing commands/scales stay unassessed.
- v3 enforces SR6 when enabled, respects sub-tick FTTI budgets, includes the rate interval
  entering the evaluation window, and prevents SR2/SR5 from reading beyond that window.
  Configured acceleration limits are consistent between auxiliary and legacy metrics.
- SR3 and comfort/TLC diagnostics do not become unvalidated safety labels. Missing required
  observations are explicit and excluded from complete-case numerical insights/feedback.

## Validation

87 synthetic/local tests passed: monitor 41, spec v3 16, numerical insights 17,
replay 5, probe 1, runner compatibility 7. The actual runner skeleton self-test also passed.
Runner integration compares all 37 fields, including actual GP targets, against original
engine output after removing only `metrics.monitor`. Cases cover safe recovery, SR6 failure,
missing applied commands, short recovery windows and collision. No API inference or CARLA
campaign was run. Integration source files were exported to a scratch directory from the
reference commit; no runner source was copied into this commit.

## Integration boundary

`pipeline/run.py` currently calls `pipeline.engine.monitor` directly. This 09-only change
cannot replace that call. Use the documented `runner_adapter.py` postprocessing entrypoint
on saved fault/baseline folders (point `--runner-root` at the current runner checkout).
The original 28-field `monitor.py` API remains available on this branch. The adapter records
hashes of the actual engine sources used, so later runner changes are traceable.
