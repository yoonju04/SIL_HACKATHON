# Long Town05 urban scenario

`run_urban.py` runs one long, continuous city trip using the real CARLA server.
It uses the official BehaviorAgent for ego behavior and the existing MPC or LQR
for ego steering. The ego is never placed on CARLA autopilot. NPC cars use Traffic
Manager; the scripted lead car uses the official behavior/PID agent for its own
controls, so it can follow the same route and stop/resume without controlling ego.

## Environment and route

- Town05; seed 17 chooses spawn 30 and a 1,474.815 m continuous route.
- Two left and five right junction maneuvers; eight route-relevant signal actors.
- Thirty moving background passenger cars, sixty AI pedestrians, one moving lead
  vehicle and two staged crossing pedestrians, plus the ego vehicle.
- Native navigation-mesh pedestrians walk at 1–1.5 m/s. Traffic and pedestrian
  placement are seeded; CARLA runtime scheduling can still affect exact interactions.
- The first route crossing may clear before ego arrives; the second crossing
  produces a pedestrian stop in the recorded MPC run.
- A lead vehicle brakes, waits for ego to stop behind it, then departs. Its route
  continues 120 m beyond ego's destination so it does not park on ego's goal.
- One native signal group is held red on approach, changed to green after an
  observed wait, then its saved state/freeze setting is restored. Other signals
  continue their native cycles, so the trip includes additional natural red waits.

These are controlled environmental events layered over actual background traffic,
not a real-world dataset or uncontrolled emergent events. Perception remains
CARLA ground truth. No learned visual perception or general obstacle detour planner
was introduced. All ego steering remains MPC/LQR; braking remains PI plus stop intent.

## Run (PowerShell)

Use the existing CARLA Python environment. The RGB recording dependency
`opencv-python==4.8.1.78` has been installed and pinned in requirements-autonomy.txt.
The validated rendered server uses **Epic** quality. Low quality crashed this
Windows installation in skeletal-mesh rendering; Epic completed the 600 s RGB run.

Start a fresh server yourself if one is not already running:

```powershell
Start-Process -FilePath "C:\Users\dongh\Desktop\junwoo\CALRA 0.9.15\WindowsNoEditor\CarlaUE4.exe" -ArgumentList "-windowed -ResX=960 -ResY=540 -quality-level=Epic -nosound"
```

Then, after the server is ready:

```powershell
cd "C:\Users\dongh\Desktop\junwoo\SIL_HACKATHON\av_stack"
$PY = "C:\Users\dongh\.conda\envs\carla\python.exe"
& $PY tools\wait_carla.py

# Visible city drive with an RGB movie and native replay recording.
& $PY run_urban.py --controller mpc --duration 650 --vehicles 30 --walkers 60 --record-video --out runs\urban_mpc

# Run sequentially, after MPC exits. LQR owns ego steering in this version.
& $PY run_urban.py --controller lqr --duration 650 --vehicles 30 --walkers 60 --record-video --out runs\urban_lqr
```

For server physics without Unreal image rendering, replace `--record-video` with
`--no-render`. Native recorder, actors, behavior, signals, controllers, and collision
sensor still run. The headless run leaves image rendering disabled; a subsequent
normal urban run enables it again. Do not run two tick-owning clients together.

## Files and checks

- `drive.mp4`: optional RGB camera behind ego, 960×540 at 5 fps.
- `replay.log`: native CARLA recorder file, always enabled.
- `scenario_events.jsonl`: staged events, actual progress and observed responses.
- `scenario.json`: actual spawn counts and event coverage, including ego actor ID.
- `behavior.jsonl`: actual behavior reason, hazard actor ID, speed target and control.
- `steps.jsonl`, `summary.json`, `episode.png`: actual tracking results.
- `collision_events_live.jsonl`: appended immediately when a collision arrives;
  `collisions.json` contains the final received sensor events.

Coverage is checked independently of spawning an event. A successful wait requires
the correct behavior reason, matching hazard actor ID, low measured ego speed,
and proximity; lead waiting also requires the same native road/lane as the lead.
Turn completion is recorded only after the full contiguous maneuver run is passed.
`all_events_observed` does not imply route completion or prove general driving safety.
`route_completed` retains the separate final 3 m / speed < 0.3 m/s requirement.

On exit, only episode-owned actors are removed. Walker controllers stop before
destruction; actors already removed by the simulator are excluded from cleanup.
Signal states/freeze settings, Traffic Manager synchronization, and world settings
are restored. The explicit `--no-render` choice remains enabled to avoid turning
the Windows renderer back on during actor teardown.

The adapter also guards CARLA's empty pedestrian-list fallback: an empty nearby
walker list must not be silently replaced by the vehicle list. A stopped front car
now has reason `vehicle`, locked by a regression that failed before the fix.

An additional behavior guard projects nearby walkers onto the upcoming route and
uses their current lateral velocity to anticipate a crossing before they enter
the lane. Its forward range includes a speed-dependent stopping allowance. It
requests a stop through the same behavior interface; MPC/LQR still owns steering.
This guard addresses a late-detection contact found during an actual crowded MPC
run. Constant-velocity anticipation does not cover every sudden movement and is
not a general collision-free guarantee.

## Replay a completed run

The stock tool accepts the recorded filename and an actor ID to follow. Use the
`ego_actor_id` in that run's scenario.json, not a hardcoded ID from another run:

```powershell
$RUN = "C:\Users\dongh\Desktop\junwoo\SIL_HACKATHON\av_stack\runs\urban_mpc"
$EGO_ID = (Get-Content "$RUN\scenario.json" -Raw | ConvertFrom-Json).ego_actor_id
& $PY "..\..\CALRA 0.9.15\WindowsNoEditor\PythonAPI\examples\start_replaying.py" -f "$RUN\replay.log" -c $EGO_ID --move-spectator
```

Replay after the driving script has exited. It reproduces recorded actor motion;
it does not rerun MPC/LQR as a new experiment. The original run's video is directly
viewable with `Invoke-Item "$RUN\drive.mp4"`.
