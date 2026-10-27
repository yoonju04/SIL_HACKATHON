# MPC health supervisor

Use `run_urban.py --controller supervised` to start on MPC with a latched LQR
fallback. Fault injection is opt-in: `--mpc-fault solver_fail --fault-at 150
--fault-duration 2`. Other modes: `nan`, `stale`, `exception`, `delay`, `none`.
These options also work in `run_autonomy.py`. Plain `mpc` and `lqr` remain available.
Injecting a fault outside supervised mode is rejected before connecting to CARLA.

The injector modifies only MPC output. `MPCSupervisor` consumes no injection flag:
it inspects solver health, finite output/state, hardware steering bounds, sample
timestamp/sequence, elapsed wall time, and persistent tracking divergence. A bad
sample selects LQR within that same loop iteration and latches until a new run.
The 50ms default is the 20Hz control period; a deadline miss is checked after the
call returns. This is not a process watchdog for a hung or dead Python process.

Both controllers receive route/speed-reference updates. LQR steering changes are
limited to 60 degrees/second after takeover. BehaviorAgent stop intent continues
to override PI acceleration. An invalid LQR result latches a full-brake safe stop,
holding the previous valid steering command. Core MPC and LQR packages are unchanged.

`supervisor.jsonl` records state transitions immediately. `behavior.jsonl` records
the selected steering source and supervisor state each tick. `summary.json`
includes the final latch, reason, switch time/count, and the separate fault setup.
`primary_elapsed_ms` after takeover retains the final MPC observation.

Tests: `tests/test_supervisor.py`, `tests/test_supervised_episode.py`. They exercise
real MPC/LQR, nonlinear local physics, faults, turning takeover, and hazard stops.
They do not substitute for a live CARLA validation of the new supervisor.

## Unprotected comparison

`--controller mpc_unprotected` instantiates only MPC, with the same optional fault
injector and no health supervisor or LQR fallback. An unusable command (failure,
missing steering, NaN or calculation exception) sends neutral steering 0 for that
cycle. Speed PI and behavior stop intent continue. Valid stale/late results are
not rejected in this baseline. The fault window ends normally; no latch is added.
To observe extended steering loss, use `--mpc-fault solver_fail --fault-at 165
--fault-duration 500 --duration 220`. Do not interpret this deliberate failure
experiment as a route-completion test. See `tests/test_unprotected.py`.
