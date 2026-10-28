# CARLA agents source

- Repository: https://github.com/carla-simulator/carla
- Release: 0.9.15
- Commit: `d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18`
- Source directory: https://github.com/carla-simulator/carla/tree/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/PythonAPI/carla/agents
- License: MIT; upstream LICENSE and file copyright headers retained.
- Downloaded navigation files: basic_agent.py, behavior_agent.py, behavior_types.py,
  controller.py, global_route_planner.py, local_planner.py, package initializers.
- Downloaded tools: misc.py and package initializer.
- Local source patch: see PATCHES.md (inclusive lower angular bound).
- Integration extensions are outside this directory, under integration/behavior_*.py.

`controller.py` is retained because upstream imports it. The integration replaces
the local planner's controller endpoint with SpeedCapture before any run_step;
the upstream PID never supplies steering, throttle, or brake to the ego vehicle.

Official behavior documentation: https://carla.readthedocs.io/en/0.9.15/adv_agents/
This is a rule-based, simulator-ground-truth baseline, not a camera/LiDAR perception model.
