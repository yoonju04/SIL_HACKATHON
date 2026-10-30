# Research Design and Evaluation

## Research question

Can a feedback-guided Nemotron scenario agent identify **more unique safety-critical MPC→LQR handover failures per fixed simulation budget** than random and grid/stratified test selection?

### Hypotheses (not yet verified)

- H1: Feedback-guided agent increases **unique failure discovery** under matched trials/time/cost.
- H2: The agent reduces **trials to first reproducible failure**.
- H3: Found scenarios remain diverse rather than clustering around the same single failure.

So far H1–H3 are **hypotheses**, not research results.

## Controlled experiment protocol

1. Fix the plant / dynamics, controller gains, initial-state envelope, simulator and criteria versions.
2. Pre-register the injection ranges and variables: speed, curvature, onset, duration, detection delay, magnitude, supervisor threshold / debounce. Faults are injected into the MPC only.
3. **Identical query budget** (e.g. simulation calls); time and GPU use are measured separately. LLM inference cost is reported as well.
4. Compare random, grid / stratified, one-shot agent (optional) and feedback agent.
5. Repeat over several pre-fixed seeds and report mean, variance / confidence intervals; no post-hoc adjustment of criteria or thresholds from test results.
6. Report the share of invalid candidate outputs, the duplicate rate and the retry cost.
7. Re-run important failures independently with the same input and record whether they reproduce.

## Metrics and formulas

### Safety metrics (per run)

- `max lateral deviation`: \(e_{max}=\max_{t\in[0,T]} |e_y(t)|\) [m]. `e_y` is signed path-relative lateral error.
- `lane boundary violation`: vehicle footprint intersects/exceeds specified drivable lane boundary. If lane geometry is absent, label as **not evaluated**, not `false`.
- `steering discontinuity`: \(J_\delta=\max_{k\ge1}|\delta_{applied}[k]-\delta_{applied}[k-1]|\) [rad]; include sampling period and also report rate \(J_\delta/\Delta t\) [rad/s]. Define whether actuator filtering is included.
- `detection latency`: \(t_{detected}-t_{fault-onset}\) [s]. If undetected, mark censored/not detected.
- `handover latency`: \(t_{first LQR-applied}-t_{fault-onset}\) [s]; also report detection-to-handover interval separately.
- `recovery time`: first time after fault at which \(|e_y|\le e_{tol}\) and other predefined constraints stay satisfied continuously for `hold_s`. If no such interval exists, `not recovered`.

### Search metrics (per method, fixed budget N)

- `Failure discovery rate = valid safety-violating runs / valid evaluated runs`. Always provide numerator and denominator; excludes invalid scenarios but count them separately.
- `Unique failure coverage = count of predeclared failure categories / clusters detected`. Define taxonomy **before** experiments.
- `Trials to first failure = smallest valid run index yielding a verified failure`; if none, right-censored at budget.
- `Simulation efficiency = verified unique failures / simulation-hour` (also GPU-hour and full inference cost if available).
- `State-at-switch coverage`: coverage of predefined bins over vehicle state at handover (yaw rate, steering angle, lateral error, MPC−LQR command gap).
- `Diversity`: coverage over predefined bins of speed, curvature, onset, fault type; this differs from count of invalid/duplicated candidates.

## What qualifies as a failure?

A safety violation is produced by **deterministic predicates**, e.g. lane-boundary violation, `max |e_y| > allowed_lateral_error_m`, `max steering rate > threshold`, or failure to recover in defined horizon. Thresholds depend on plant model, road width, maneuver and sampling. No universally safe threshold is implied here; freeze before testing and justify independently.

Other validation dimensions: baseline controller tracking performance without faults, false-positive switching, maximum control saturation, scenario feasibility, dynamics parameter sensitivity, open-loop/closed-loop confusion, common input faults that break both controllers.

## Internal validity / limitations

- Search agent may exploit simulator artifacts. Reconfirm key failures under more than one map/vehicle parameter set where possible.
- Grid and random must receive equivalent valid ranges and budget. Agent's invalid proposals count as cost.
- LLM does not itself estimate vehicle safety metrics or replace certified safety assessment.
- MPC timeout generated in simulated time does not prove worst-case execution time of a real CPU/ECU.
- Only MPC computation/execution faults are injected; sensor, actuator and communication faults are out of scope. The steering actuator is a simplified plant model (lag, rate limit, saturation), so results cannot assert SbW hardware redundancy or real actuator behavior.
- Single failure discovery does not prove production prevalence. No failures discovered does not prove absence of failures.

## Planned figures in final experiment report

1. **Unique failures vs. number of simulator runs** (all methods, multiple seeds).
2. **Time/trials-to-first-failure** (distributions, censoring indicated).
3. **Speed × curvature** heatmap: failure locations / scenario coverage.
4. **State-at-switch failure map**: e.g. yaw rate × MPC−LQR command gap at handover, colored by violation.
5. **Handover trace**: MPC/LQR/applied command, lateral error, fault onset, detection, switch timestamps.
6. **Failure replay** synchronized with trace and scenario manifest.

**No measured plots are included in the current repository.**
