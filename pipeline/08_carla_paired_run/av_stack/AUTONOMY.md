# Official CARLA behavior planning + av_stack MPC/LQR

Implemented 2026-10-22 for the Windows CARLA 0.9.15 / Python 3.8 environment.

## Selection and scope

Chosen baseline: CARLA's official BehaviorAgent, release 0.9.15, commit
`d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18`.
It matches the installed simulator, has no ROS requirement, and exposes the
behavior/local-plan boundary needed to preserve the existing steering controllers.
Autoware is a broader ROS-based stack; installing it would introduce a separate
runtime and simulator bridge for this Windows setup. This selection is a practical
CARLA research baseline, not a claim that BehaviorAgent is an industry production stack.

- Official agent docs: https://carla.readthedocs.io/en/0.9.15/adv_agents/
- Pinned source: https://github.com/carla-simulator/carla/tree/d7b45c1e159e6d13296f7a3a4e8b13e6c2d62c18/PythonAPI/carla/agents
- Autoware architecture: https://docs.autoware.org/main/design/
- Source/license inventory: `third_party/carla_0915/SOURCE.md` and `LICENSE`.
- One local forward-angle boundary patch: `third_party/carla_0915/PATCHES.md`.

## Ownership and behavior

```text
CARLA ground-truth actors/map/ego state
  -> official BehaviorAgent: speed limits, red lights, pedestrians, front vehicles
  -> official LocalPlanner route queue + speed capture (upstream PID never stepped)
  -> ReferencePlanner: smooth map route + curvature-limited speed profile
  -> MPC OR LQR: steering
  -> av_stack PI: throttle/brake; emergency intent overrides throttle
  -> CARLA VehicleControl
```

The upstream controller module remains vendored because official imports require
it. A non-actuating SpeedCapture replaces the endpoint before any agent run_step.
No steering/throttle computed by an upstream PID is applied to the ego vehicle.
The bridge converts the official km/h target into m/s, then caps it by `--speed`
and the geometric speed profile. Stop intent applies full brake and clears PI
integration, preventing acceleration windup on resume.

With `--destination INDEX`, the official GlobalRoutePlanner computes the route.
Without it, the bridge walks the current lane's waypoints for `--route-length`
and selects the first branch at junctions, matching the old deterministic route
convention. The official queue advances each tick; this alone does not rebuild
the reference. Explicit upstream plan changes create a new reference revision.
Both controllers can consume these revisions without resetting MPC filter/history.

Current responses are the upstream rule-based responses. Pedestrian handling is
primarily stopping; front-vehicle handling includes following and emergency braking.
Tailgating/lane decisions remain upstream behavior, with cautious/normal/aggressive
profiles available. No general-purpose static-prop detour planner, camera/LiDAR
recognition, learned motion prediction, stop-sign enforcement, or MPC-to-LQR
Supervisor switching was added. The official policy's scene/detection limits remain.

## Run yourself (PowerShell)

Start `C:\Users\dongh\Desktop\junwoo\CALRA 0.9.15\WindowsNoEditor\CarlaUE4.exe`
yourself. Keep SimLingo and other synchronous tick-driving clients stopped.

```powershell
cd "C:\Users\dongh\Desktop\junwoo\SIL_HACKATHON\pipeline\08_carla_paired_run\av_stack"
$PY = "C:\Users\dongh\.conda\envs\carla\python.exe"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& $PY tools\wait_carla.py

# Clear route, LQR. End the run before starting another.
& $PY run_autonomy.py --controller lqr --town Town04 --spawn 10 --speed 5 --route-length 300 --duration 90 --follow --plot --out runs\autonomy_lqr
Invoke-Item .\runs\autonomy_lqr\episode.png

# Clear route, MPC. Already-loaded Town04 is reused even if --town is supplied.
& $PY run_autonomy.py --controller mpc --town Town04 --spawn 10 --speed 5 --route-length 300 --duration 90 --follow --plot --out runs\autonomy_mpc
Invoke-Item .\runs\autonomy_mpc\episode.png

# Stationary pedestrian 30 m along the planned path; expect braking and waiting.
& $PY run_autonomy.py --controller lqr --town Town04 --spawn 10 --speed 5 --route-length 120 --duration 30 --demo pedestrian --follow --plot --out runs\autonomy_lqr_pedestrian
& $PY run_autonomy.py --controller mpc --town Town04 --spawn 10 --speed 5 --route-length 120 --duration 30 --demo pedestrian --follow --plot --out runs\autonomy_mpc_pedestrian

# Stationary front vehicle instead of a pedestrian.
& $PY run_autonomy.py --controller lqr --town Town04 --spawn 10 --speed 5 --route-length 120 --duration 30 --demo vehicle --follow --plot --out runs\autonomy_lqr_vehicle
```

Demo actors stay in the lane for the whole episode. `route_completed=false` and
`termination=time_limit` are expected if the agent waits behind one. These demo
actors, the ego vehicle, and its collision sensor are cleaned up on normal exit
or Ctrl+C; existing external actors are not deleted. Simulator settings are restored.
If a spawn is occupied, select another spawn or clear that scene yourself.

`--destination` is a spawn-point index; it is optional and must be valid for the
loaded map. `--speed` is m/s, `--duration` is simulation seconds, and
`--route-length` is metres. Live braking at arbitrary speeds/scenes is not certified.
Begin with the explicit 5 m/s commands above.

Dependencies `networkx==2.8.8` and `shapely==1.8.5.post1` have been installed in the
existing carla Python environment. For another environment, use
`uv pip install --python "$PY" -r .\requirements-autonomy.txt`.

## Evidence files

- `steps.jsonl`: actual state, lateral error, steering, solver status, speed target.
- `behavior.jsonl`: behavior reason, stop intent, target speed, applied pedals,
  steering owner, reference revision, and distance to goal.
- `trajectory_0001.json`, etc.: each complete map-anchored reference revision.
- `trajectory.json`: final reference; `episode.png`: final reference against whole drive.
  If route changes, use revision files to interpret earlier path segments.
- `summary.json`: episode metadata and tracking metrics.
- `collisions.json`: collision-sensor events received during this episode. An empty
  list is observed sensor evidence, not a general proof of collision avoidance.

Completion in this mode requires projection in the final 3 m of the reference,
distance to its end <= 3 m, and absolute longitudinal speed < 0.3 m/s.
This differs from the old runner's 15/20 m end-of-route tolerance.
Logs label perception as `simulator_ground_truth` and retain the chosen controller.
Use distinct output directories if you need to preserve separate trials.

## Verification completed without a CARLA server

The implementation did not launch/connect to CARLA or operate the UI.

| Check | Result |
|---|---|
| Original planner/controller/fake-CARLA suite | 22 passed |
| Official behavior branches with real native Map/math | 15 passed |
| Stop override and PI windup/resume | 3 passed |
| Actual MPC/LQR + nonlinear local plant + behavior, and CLI help | 7 passed |
| Moving red-light/pedestrian hazard, both actual controllers | 4 passed |
| Python 3.8 compilation / critical Ruff checks | Passed |

The native map uses two linked OpenDRIVE roads and real CARLA waypoint/transform
objects; only actor/world RPC access is replaced by in-memory fixtures. The
official BehaviorAgent/GlobalRoutePlanner and actual MPC/LQR implementations run.
A tripwire fails the test if upstream PID.run_step or bridge.apply_control executes.
Moving hazard tests introduce a hazard at 5.11 m/s, observe braking to standstill,
remove the hazard, and observe renewed acceleration and arrival. The local plant
has no collision engine; positive geometric clearance is the test's independent
collision proxy. These tests do not establish live Unreal vehicle stopping behavior,
sensor delivery, demo spawning, curve tracking with behavior changes, arbitrary
obstacle avoidance, or full autonomous-driving safety. Live checks remain for the
user to run with the commands above.

Run the suites in separate fresh processes because the old suite substitutes a
fake `carla` module, whereas the new tests require native CARLA bindings:

```powershell
& $PY tests\run_all.py
& $PY tests\test_behavior_bridge.py
& $PY tests\test_autonomy_control.py
& $PY tests\test_autonomy_episode.py
& $PY tests\test_autonomy_moving_hazards.py
```
