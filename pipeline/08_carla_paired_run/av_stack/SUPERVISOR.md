# MPC health supervisor

Use `run_benchmark.py --planner pdm --controller supervised` to start on MPC with a latched LQR
fallback. Fault injection is opt-in: `--mpc-fault solver_fail --fault-start 11.5 --fault-duration 500`.
Modes: `solver_fail`, `stale`, `bias`, `drift`, `intermittent_bias`, `saturate`, `output_delay`, `none`
(see README "Faults you can inject"). Plain `mpc` and `lqr` remain available.
Injecting a fault outside supervised / `mpc_unprotected` mode is rejected before connecting to CARLA.

The injector modifies only MPC output. `MPCSupervisor` consumes no injection flag:
it inspects solver health, finite output/state, hardware steering bounds, sample
timestamp/sequence, elapsed wall time, and persistent tracking divergence. A bad
sample selects LQR within that same loop iteration and latches until a new run.
The 50ms default is the 20Hz control period; a deadline miss is checked after the
call returns. This is not a process watchdog for a hung or dead Python process.

Both controllers receive route/speed-reference updates. LQR steering changes are
rate-limited after takeover. An invalid LQR result latches a full-brake safe stop,
holding the previous valid steering command. Core MPC and LQR packages are unchanged.

Every PDM run writes the 09 Monitor inputs (`steps.jsonl`, `behavior.jsonl`, `monitor_probe.jsonl`,
`collisions.json`, `summary.json`, see `integration/monitor_logs.py`).

Tests: `tests/test_supervisor.py` (real MPC/LQR, nonlinear local physics, faults, turning takeover)
and `tests/test_pdm_reference.py`. They do not substitute for a live CARLA validation.

## Unprotected comparison

`--controller mpc_unprotected` instantiates only MPC, with the same optional fault
injector and no health supervisor or LQR fallback. An unusable command (failure or
missing steering) sends neutral steering 0 for that cycle. Valid stale/late results are
not rejected in this baseline. The fault window ends normally; no latch is added.
The 08 loop uses it as the no-reaction run that measures FTTI for SR1.
