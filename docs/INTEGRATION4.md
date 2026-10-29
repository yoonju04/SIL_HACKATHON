# integration4 — integration3 + tighter supervisor

`integration4` = `integration3` (merged as is, see `docs/INTEGRATION3.md`) + one change: the av_stack supervisor.
The GP rework tried here before (`bf7f590`) was reverted (`f860b38`); integration3 covers its items 2-5.

## Why

In result_2..5 all 15 risk-fault runs that switched (bias, drift, saturate, intermittent bias) had already left
the lane before the switch. The old divergence rule (|e_y| > 1.0 m or |e_psi| > 0.6 rad, held 0.25 s) is wider
than the in-lane margin (0.83-1.08 m from the car side to the lane edge), and there was no steering-rate check,
so saturate steps (1000+ deg/s) and intermittent-bias edges (107-161 deg/s) passed the instant checks.

## Change (`pipeline/08_carla_paired_run/av_stack/integration/supervised_lateral.py`)

`SupervisorConfig` / `SUPERVISOR_PROFILES`, chosen with `run_benchmark.py --supervisor tight|legacy` (default tight;
written to `agent_config.json`; the PDM agent uses it, TF++ uses the default):

| | tight (default) | legacy |
|---|---|---|
| tracking_divergence | 0.6 m or 0.35 rad, held 0.15 s | 1.0 m or 0.6 rad, held 0.25 s |
| steer_rate_exceeded (new) | MPC command rate > 90 deg/s | none |

Thresholds from 61 fault-free CARLA baselines (result_2..5, 48k MPC ticks): |e_y| max 0.46 m (p99.9 0.31),
|e_psi| max 0.23 rad, command rate max 61 deg/s; the MPC itself limits its rate to 60 deg/s (`mpc.py`), so a
healthy MPC cannot trip the rate check.

## Offline estimate (logged runs, no CARLA)

Replaying the tight rules on the MPC rows of the result_2..5 risk-fault runs (valid up to the real switch):
of 22 runs with a lane departure, tight detects 19 before the departure (saturate / intermittent bias by the rate
check, 0.25-0.65 s earlier; drift / bias by divergence) and 3 after it (2 drift, 1 bias); none missed. Legacy:
0 of 22 before. Detecting before the departure does not by itself mean the LQR recovers; that needs CARLA runs.

## Not changed

* `pipeline/01_safety_profile/safety_profile.yaml` and `02_hypotheses/.../S1_*.yaml` (team files) still describe
  the 1.0 m / 0.6 rad / 0.25 s rule, and Nemotron A reads the safety profile.
* The mock simulator (`engine/fixtures/synthetic_sim.py`) still uses the legacy rule.
* Low friction (integration3 #2) was not in the baselines used for the thresholds: check the baseline gate
  (baseline switches) on the first low-friction runs.
