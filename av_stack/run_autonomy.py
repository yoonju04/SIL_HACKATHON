#!/usr/bin/env python3
"""CARLA BehaviorAgent planning with av_stack MPC/LQR steering and PI speed control.

Run in the existing CARLA Python 3.8 environment. This entrypoint drives a simulator.
Example: python run_autonomy.py --controller lqr --town Town04 --spawn 10 --follow --plot
"""
from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from pathlib import Path

import carla

from integration.autonomy_episode import EpisodeSettings, run_episode
from integration.autonomy_scene import AutonomyScene
from integration.behavior_bridge import BehaviorBridge, BehaviorConfig
from integration.carla_io import flatten_steering_curve
from integration.supervisor_cli import add_supervisor_arguments, parse_fault


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("mpc", "lqr", "supervised", "mpc_unprotected"), default="lqr")
    add_supervisor_arguments(parser)
    parser.add_argument("--behavior", choices=("cautious", "normal", "aggressive"), default="cautious")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--spawn", type=int, default=10)
    parser.add_argument("--destination", type=int, default=None, help="destination spawn index; otherwise follow the lane")
    parser.add_argument("--vehicle", default="vehicle.tesla.model3")
    parser.add_argument("--speed", type=float, default=8.0, help="speed cap in m/s, also limited by behavior and map")
    parser.add_argument("--route-length", type=float, default=500.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--demo", choices=("pedestrian", "vehicle"), default=None,
                        help="place a stationary obstacle 30 m along the route; expect a stop, not completion")
    args = parser.parse_args(argv)
    fault = parse_fault(parser, args)
    if not 0.0 < args.duration < float("inf"):
        parser.error("--duration must be positive and finite")
    config = BehaviorConfig(speed_mps=args.speed, dt=args.dt, behavior=args.behavior,
                            route_length_m=args.route_length)
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    if client.get_server_version() != "0.9.15":
        parser.error("This integration is pinned to CARLA server 0.9.15")
    world = client.get_world()
    if args.town and world.get_map().name.rsplit("/", 1)[-1] != args.town:
        world = client.load_world(args.town)
    spawns = world.get_map().get_spawn_points()
    for index in (args.spawn, args.destination):
        if index is not None and not 0 <= index < len(spawns):
            parser.error(f"spawn index must be between 0 and {len(spawns) - 1}")
    if args.destination is not None:
        from dataclasses import replace
        config = replace(config, destination=spawns[args.destination].location)
    original = world.get_settings()
    vehicle = None
    scene = None
    out = args.out or Path("runs") / f"autonomy_{args.controller}"
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.dt
        world.apply_settings(settings)
        blueprint = world.get_blueprint_library().find(args.vehicle)
        blueprint.set_attribute("role_name", "hero")
        vehicle = world.spawn_actor(blueprint, spawns[args.spawn])
        for _ in range(10):
            world.tick()
        flatten_steering_curve(vehicle, carla)
        bridge = BehaviorBridge(vehicle, config)
        scene = AutonomyScene(vehicle)
        if args.demo:
            scene.spawn_demo(bridge.trajectory, args.demo)
            for _ in range(10):
                world.tick()
        episode = EpisodeSettings(args.controller, args.duration, args.dt, out, args.follow, args.plot, fault=fault)
        print(f"[autonomy] ground-truth behavior + {args.controller.upper()}, route={bridge.trajectory.length:.1f}m")
        run_episode(vehicle, bridge, episode)
    finally:
        try:
            if scene is not None:
                try:
                    scene.close()
                finally:
                    scene.save(out)
        finally:
            try:
                if vehicle is not None:
                    vehicle.destroy()
            finally:
                world.apply_settings(original)


if __name__ == "__main__":
    main()
