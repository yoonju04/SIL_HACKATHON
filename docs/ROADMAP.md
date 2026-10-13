# Development Roadmap

**Stage:** proposal, pending implementation. Tasks below are checklists, not completed functionality.

## Milestone 1 — Controllable dynamics
- [ ] Define reference path and lateral bicycle dynamics with units.
- [ ] Add simplified SbW actuator model (first-order lag, rate limit, saturation) between supervisor and plant.
- [ ] Add LQR baseline and choose/control-check state-space model.
- [ ] Implement MPC with constraints, deadline/failure hooks.
- [ ] Implement supervisor (fault criteria, debounce/hysteresis, safe fallback policy).
- [ ] Implement deterministic injection, logging and reproducible seeds.

**Exit criterion:** baseline completes no-fault tracks, force MPC failure and observe a logged LQR handover in SIL.

## Milestone 2 — Evaluator / baselines
- [ ] Per-step logs and event timestamps, stable schema.
- [ ] Safety evaluator with unit tests on synthetic traces.
- [ ] Freeze safety criteria and scenario bounds.
- [ ] Random and grid/stratified exploration; run reproducible budgets.
- [ ] Report failures and limitations without overstating safety.

**Exit criterion:** given identical logs/criteria, evaluator reproduces identical verdicts.

## Milestone 3 — Agent loop (Nemotron / Token Factory)
- [ ] Confirm available model ID via model catalog/API.
- [ ] Prompt: propose scenarios from failure summaries + coverage gaps.
- [ ] Structured JSON output with hard range/dedup validation.
- [ ] Strict separation: LLM proposes, evaluator judges.
- [ ] Ablate feedback (one-shot vs iterative) with matched budgets.

**Exit criterion:** at least two full iterations of proposal → simulation → objective score → feedback.

## Milestone 4 — Scale / cloud
- [ ] Queue and worker contracts, idempotent result writes.
- [ ] Parallel SIL runners; optional Nebius Cloud workers.
- [ ] Record compute utilization, LLM tokens/cost and failure/retry rates.
- [ ] Container versions and reproducibility manifest.

**Exit criterion:** two or more distinct scenarios complete concurrently without result collisions.

## Milestone 5 — Combined faults / demonstration
- [ ] Inject SbW actuator faults (lag increase, rate limit, saturation) together with MPC faults.
- [ ] Inject SbW angle-sensor faults (noise, dropout, stuck) affecting both controllers.
- [ ] Capture SIL animation, telemetry plot, failure replay and summary report.
- [ ] (Optional) CARLA / Isaac Sim integration with synchronized ticks and correct command mappings.
- [ ] Run evaluation protocol and publish only measured results.

**Exit criterion:** demonstrable failure scenario + telemetry + comparison figure (if successful).

## Non-goals for first demo
- SbW redundant motor/ECU verification; physical actuator diagnosis/redundancy certification; real-car deployment; validating ISO 26262 compliance; claiming collision avoidance guarantees; comprehensive multi-sensor perception failures.
