# Local patch to CARLA 0.9.15 agents

The upstream sources retain their MIT copyright and license notices. One line is
changed in `agents/tools/misc.py`, function `is_within_distance`:

```diff
-    return min_angle < angle < max_angle
+    return min_angle <= angle < max_angle
```

The official forward hazard sector uses `[0, 90]` (or another positive upper
angle), but its strict lower comparison excluded a target exactly straight ahead
at zero degrees. This can suppress red-light, pedestrian, and front-vehicle
responses on an exactly straight lane. Include the lower bound; retain the upper
angle and distance limits. This makes all supplied sectors lower-inclusive,
including the rear sector used by the upstream tailgating check.

## Reproduction and evidence

`tests/behavior_fixture.py` creates a real native `carla.Map` from two linked
OpenDRIVE roads. Actor poses, vectors, boxes, and waypoint transforms use the real
CARLA 0.9.15 bindings. Only world/actor RPC access is replaced by in-memory actors;
no server, client connection, autopilot, or vehicle control application is used.

Three possible causes were distinguished at runtime:

- Map/lane mismatch: rejected; ego and light were both `(road_id=1, lane_id=-1)`.
- Distance cutoff: rejected; trigger distance was `4.0 m`, threshold `5.0 m`.
- Angle boundary: confirmed; the same transforms returned `False` for `[0, 90]`
  and `True` for `[-0.001, 90]` before the patch.

Before the patch, the zero-angle helper regression and four actual BehaviorAgent
hazard scenarios failed: red light, pedestrian, stopped vehicle, and slow lead
vehicle. After this one-line patch, the standalone suite reports
`15 server-free checks passed`, including side/rear/out-of-range rejection,
km/h conversion, configured cruise cap, official global routing, reference
revision handling, and a tripwire that raises if the upstream PID executes.

Run from the `av_stack` directory in a fresh CARLA Python environment:

```powershell
& 'C:\Users\dongh\.conda\envs\carla\python.exe' tests\test_behavior_bridge.py
```

These tests verify planning decisions with native geometry. They do not verify
camera/LiDAR perception, vehicle stopping dynamics, CARLA-server interaction,
road-scene generalization, or full autonomous-driving safety. A live scenario
run remains required before claiming collision avoidance works in simulation.
