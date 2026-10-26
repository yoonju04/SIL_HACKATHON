# MPC→LQR hypothesis-driven scenario research pipeline (v2.0)

Based on the supplied `hypotheses.yaml v1.1` example. This package is a **research draft**: it does not prove control safety and is not yet a CARLA worker, Nemotron client, GP learner, or statistical evaluator.

## Files

- `generate_hypotheses.py`: safe YAML profile + catalog + scenario-space → evidence-linked hypotheses; blocks unknown values; no Python `eval()`.
- `vulnerability_patterns.yaml`: 11 pattern templates (H-01…H-11) and metric-specific statistical decision specifications.
- `safety_profile.example.yaml`: **fictional/example-only**, not verified from any real repository. `null` means not observed.
- `scenario_space.example.yaml`: illustration bounds only, not verified ODD or physical operating envelope.
- `hypotheses.generated.yaml`: actual generator output from the example inputs.
- `build_scenario_candidates.py`: conservative scenario specification sampler. Unknown runtime selectors are preserved as `unbound_conditions` and `executable:false`.
- `scenario_candidates.example.yaml`: sampled candidate specifications, not CARLA runs.
- `test_pipeline.py`: unit checks.
- `build_assets.py`: optional file-builder from user attachment; **not** required for runtime generation, and references original chat mount. Do not run outside this container; catalog is already generated.

## Commands

```bash
pip install PyYAML
python generate_hypotheses.py safety_profile.example.yaml vulnerability_patterns.yaml \
  --scenario-space scenario_space.example.yaml -o hypotheses.yaml
python build_scenario_candidates.py hypotheses.yaml scenario_space.example.yaml \
  --per-arm 8 --seed 7 -o candidate_scenarios.yaml
python -m unittest -v test_pipeline.py
```

## Important limitations

1. `proposed` means **testable candidate**, never an established hazard. `blocked` indicates unresolved profile values or oracle artifacts. `applies_if=false` omits pattern entirely. The example has 4 proposed / 7 blocked.
2. H-08's stress settings and H-11's lateral acceleration screening are proposed experimental conditions, not failure thresholds established by citations.
3. `R_REACH_ESTIMATED` and `NOT_CERTIFIED` do **not** assert formal reachability or nonrecoverability. Sampled simulation cannot certify irrecoverability. To use reachability hypotheses, compute and store model/version-specific artifacts, and verify their provenance.
4. The condition/control YAML mixes declarative target constraints and scenario variables. The sampler only binds supported scalars/inequalities; symbolic expressions, recovery labels, switch triggers, and actuator command conditions require explicit scenario adaptors. Even bound candidates require CARLA-specific implementation and validation.
5. Metric rules describe the appropriate effect type/direction, but `generate_hypotheses.py` does **not** compute bootstrap intervals, run checkpoints, or infer support/refutation. Non-inferiority H-07 must be evaluated separately.
6. `scenario_candidates.example.yaml` is an **exploratory set**, not a statistically valid randomized paired test. Dedicated condition/control sampling, matched nuisance distributions, run validity, seed tracking, and GP exploration/hold-out separation belong in a separate orchestrator.
7. The example observations (O-xxx) are tagged `example_only_unverified`. An observation ID alone does not establish actual code evidence. Populate the profile with values confirmed from current code and source revision.
8. `llm_proposals` gate conservatively refuses automatic installation into the catalog even after basic checks; manual review is required to ensure semantically meaningful comparison and physical plausibility.
9. The `recovery_success` definition requiring an invariant set is **undefined** unless a valid certificate exists. This is intentional; do not mark failure when measurement is unavailable.

## Next implementation steps

- Wire a code extractor and observation provenance to the actual MPC/LQR repository.
- Add a scenario compiler for each pattern (`near`, onset events, recovery boundary, fault timing), using CARLA runner interfaces.
- Implement scenario validity checks with road geometry, friction, actuator limits and mutually matched control conditions.
- Add metric collectors (MPC constraint violations, solver timing, lane margins, LQR handover).
- Build a separate evaluator for binary/count/continuous outcomes and non-inferiority, with fixed checkpoints and corrected multiple testing as appropriate.
