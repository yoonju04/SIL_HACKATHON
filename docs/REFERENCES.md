# Sources & Evidence Boundaries

Checked for document preparation on **2026-10-12**. Official pages may change; recheck technical compatibility before implementation.

## Vendor and tool documentation

1. **NVIDIA — Nemotron foundation models**. Describes model family and agent-oriented positioning. Does **not** substantiate an advantage for steering-fault scenario generation. https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/
2. **Nebius — Token Factory**. Describes OpenAI-compatible inference API and open-model service. https://dev.nebius.com/token-factory
3. **Nebius — Token Factory API reference**. Documents `/v1/models` and `/v1/chat/completions`; confirm currently available model IDs in account. https://api.tokenfactory.nebius.com/docs
4. **CARLA — Synchrony and time-step**. Describes synchronous world mode/fixed time step; relevant for reproducible experiments. https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
5. **CARLA — official repository**. Simulator code, releases, issues and Python API docs. https://github.com/carla-simulator/carla
6. **Nexteer — Steer-by-Wire**. Describes redundancy/high availability and other SbW safety features; an example of commercial system-level redundancy, **not proof of MPC→LQR standardization**. https://www.nexteer.com/electric-power-steering/steer-by-wire/

## Evidence classifications

| Statement | Classification |
|---|---|
| Steering systems can employ redundant hardware/signaling to tolerate selected faults | Vendor-documented industry feature (Nexteer) |
| Nemotron is an agent-oriented model family | Vendor-documented model positioning (NVIDIA) |
| Token Factory exposes OpenAI-compatible inference endpoints | Vendor-documented API (Nebius) |
| CARLA supports synchronous/fixed-time-step simulation | Official simulator documentation |
| Our feedback-guided agent discovers failures more efficiently than baselines | **Hypothesis, not yet tested** |
| MPC→LQR is typical production steering failover | **Not established** |
| Our architecture fulfills a particular hackathon's official evaluation criteria | **Not established by these sources** |
| The proposed design proves real-world vehicle safety | **False/not claimed** |

## Additional reading (not required dependencies)

- ISO 26262 series (road-vehicle functional safety) — consult the licensed standard for exact requirements; do not infer certification from simulated tests.
- Fault-Tolerant Control and Model Predictive Control textbooks/papers — select plant- and control-specific sources before claims of stability.
- CARLA Python API/recorder version-specific documentation for the installed release.
