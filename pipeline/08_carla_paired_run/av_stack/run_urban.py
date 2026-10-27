from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
from pathlib import Path

import carla

from integration.autonomy_episode import EpisodeSettings, run_episode
from integration.autonomy_scene import AutonomyScene
from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
from integration.carla_io import flatten_steering_curve
from integration.urban_route import choose_urban_route, quick_fault_route
from integration.urban_scenario import UrbanScenario
from integration.urban_traffic import UrbanTraffic, UrbanTrafficConfig
from integration.supervisor_cli import add_supervisor_arguments, parse_fault


def main() -> None:
    parser = argparse.ArgumentParser(description="Long urban drive with real traffic, crossings, signals and MPC/LQR steering")
    parser.add_argument("--controller", choices=("mpc", "lqr", "supervised", "mpc_unprotected"), default="lqr")
    add_supervisor_arguments(parser)
    parser.add_argument("--town", default="Town05")
    parser.add_argument("--duration", type=float, default=650.0)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--vehicles", type=int, default=30)
    parser.add_argument("--walkers", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument('--quick-fault-demo', action='store_true', help='start 110m before the first turn; no scripted waits')
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--record-video", action="store_true", help="optional RGB sensor; native recorder is always enabled")
    parser.add_argument("--no-render", action="store_true", help="server physics and traffic without Unreal image rendering")
    args = parser.parse_args()
    fault = parse_fault(parser, args)
    if args.no_render and args.record_video:
        parser.error("RGB recording requires rendering")
    if not 0.0 < args.duration < float("inf") or args.vehicles < 0 or args.walkers < 0:
        parser.error("duration must be positive and finite; actor counts must be nonnegative")
    behavior_config = BehaviorConfig(speed_mps=args.speed, route_length_m=80.0)
    out = args.out or Path("runs") / f"urban_{args.controller}"
    out.mkdir(parents=True, exist_ok=True)
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if world.get_map().name.rsplit("/", 1)[-1] != args.town:
        world = client.load_world(args.town)
    original = world.get_settings()
    ego = None
    traffic = None
    scenario = None
    collision = None
    recording = False
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = args.no_render
        world.apply_settings(settings)
        lights = list(world.get_actors().filter("traffic.traffic_light"))
        route = choose_urban_route(world.get_map(), lights, args.seed)
        if args.quick_fault_demo:
            route = quick_fault_route(route)
        blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", "hero")
        spawn = world.get_map().get_spawn_points()[route.spawn_index]
        if args.quick_fault_demo:
            spawn = route.plan[0][0].transform
            spawn.location.z += 0.5
        ego = world.spawn_actor(blueprint, spawn)
        for _ in range(20):
            world.tick()
        flatten_steering_curve(ego, carla)
        bridge = BehaviorBridge(ego, behavior_config)
        bridge.agent.set_global_plan(route.plan)
        bridge.trajectory = bridge._reference()
        if args.quick_fault_demo:
            bridge.agent.ignore_traffic_lights(True)
            print('[quick-demo] scripted waits OFF; traffic lights ignored for isolated controller comparison', flush=True)
        traffic = UrbanTraffic(client, UrbanTrafficConfig(args.vehicles, args.walkers, args.seed))
        scenario = UrbanScenario(ego, route, out)
        scenario.setup(traffic.tm, bridge.agent, args.record_video, scripted_events=not args.quick_fault_demo)
        collision = AutonomyScene(ego, out)
        traffic.spawn(ego, route.plan)
        for _ in range(20):
            world.tick()
        print(client.start_recorder(str((out / "replay.log").resolve()), True), flush=True)
        recording = True
        client.set_timeout(10.0)
        settings = EpisodeSettings(args.controller, args.duration, 0.05, out,
                                   follow=True, plot=True, observer=scenario, fault=fault)
        print(f"[urban] {route.length_m:.0f}m, {route.left_turns} left / {route.right_turns} right, {route.signal_count} signals", flush=True)
        run_episode(ego, bridge, settings)
    except RuntimeError as error:
        failure = dict(error_type=type(error).__name__, message=str(error), controller=args.controller,
                       fault_at_s=fault.at_s, quick_fault_demo=args.quick_fault_demo)
        (out/'runtime_failure.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        print('[runtime-failure] ' + str(error), flush=True)
        raise
    finally:
        try:
            if recording:
                client.stop_recorder()
            if scenario is not None:
                try:
                    scenario.save(traffic.metadata() if traffic is not None else {})
                finally:
                    scenario.close()
        finally:
            try:
                if collision is not None:
                    collision.close()
                    collision.save(out)
            finally:
                try:
                    if traffic is not None:
                        traffic.close()
                finally:
                    try:
                        if ego is not None:
                            ego.destroy()
                    finally:
                        if args.no_render:
                            original.no_rendering_mode = True
                        world.apply_settings(original)


if __name__ == "__main__":
    main()
