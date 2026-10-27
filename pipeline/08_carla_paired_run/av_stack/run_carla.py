#!/usr/bin/env python3
"""Drive a CARLA vehicle with: planner (trajectory only) -> MPC or LQR (steering).

Start the CARLA server first (e.g. ./CarlaUE4.sh -RenderOffScreen), then:
  python run_carla.py --controller mpc --town Town04 --spawn 10 --follow
  python run_carla.py --controller lqr --town Town04 --spawn 10 --speed 12 --plot
  python run_carla.py --controller mpc --lane-change 120:60:-3.5     # lane change to the right lane
Outputs (--out): steps.jsonl, trajectory.json, summary.json, episode.png (--plot).

Data flow per tick (synchronous mode, fixed dt):
  CARLA vehicle state --(frame conversion)--> controller(trajectory) --> steer
  planner speed profile --> PI speed loop --> throttle / brake
The planner runs once, before the loop (no re-planning from the ego pose).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # deterministic timing: no BLAS thread contention

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

try:
    import carla  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("CARLA Python API not found: pip install carla==<server version> "
             "(or add PythonAPI/carla/dist/*.egg / *.whl to PYTHONPATH)")

from integration.carla_io import (follow_camera, flatten_steering_curve, read_state,  # noqa: E402
                                  to_carla_steer, vehicle_geometry)
from integration.cli import common_args  # noqa: E402
from integration.episode import (EpisodeLog, LaneMonitor, StepRow, build_controller,  # noqa: E402
                                 step_controller)
from integration.speed import SpeedPI, speed_reference  # noqa: E402
from planner import PlannerConfig, ReferencePlanner, route_from_carla_map  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default=None, help="e.g. Town04 (default: keep the loaded map)")
    ap.add_argument("--vehicle", default="vehicle.tesla.model3")
    ap.add_argument("--spawn", type=int, default=0, help="spawn point index")
    ap.add_argument("--route-length", type=float, default=None, help="default: speed*duration + 60 m")
    ap.add_argument("--cf", type=float, default=None, help="front axle cornering stiffness [N/rad]")
    ap.add_argument("--cr", type=float, default=None, help="rear axle cornering stiffness [N/rad]")
    ap.add_argument("--follow", action="store_true", help="spectator camera follows the car")
    ap.add_argument("--no-render", action="store_true", help="CARLA no_rendering_mode (faster)")
    a = ap.parse_args(argv)
    out_dir = a.out or os.path.join("runs", f"carla_{a.controller}")

    client = carla.Client(a.host, a.port)
    client.set_timeout(30.0)
    world = client.load_world(a.town) if a.town else client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = a.dt
    if a.no_render:
        settings.no_rendering_mode = True
    world.apply_settings(settings)
    try:
        client.get_trafficmanager().set_synchronous_mode(True)
    except Exception:
        pass

    vehicle = None
    try:
        bp = world.get_blueprint_library().find(a.vehicle)
        spawn = world.get_map().get_spawn_points()[a.spawn]
        vehicle = world.spawn_actor(bp, spawn)
        for _ in range(10):  # let the car settle on the ground
            world.tick()
        flatten_steering_curve(vehicle, carla)
        geom = vehicle_geometry(vehicle)
        if a.cf:
            geom["cf"] = a.cf
        if a.cr:
            geom["cr"] = a.cr
        print("[vehicle] " + ", ".join(f"{k}={v:.4g}" for k, v in geom.items()))

        # 1) planner: trajectory only, once
        length = a.route_length or a.speed * a.duration + 60.0
        rx, ry = route_from_carla_map(world.get_map(), vehicle.get_transform().location,
                                      length_m=length, step_m=1.0, lane_type=carla.LaneType.Driving)
        traj = ReferencePlanner(PlannerConfig(v_max=a.speed, a_lat_max=a.a_lat, mu=a.mu)).plan(
            rx, ry, tuple(a.lane_change))
        print(f"[planner] {traj.length:.0f} m, max |kappa| {traj.meta['max_abs_kappa']:.4f} 1/m, "
              f"max a_lat {traj.meta['max_lat_acc']:.2f} m/s^2, feasible={traj.meta['feasible']}")

        # 2) controller: MPC or LQR
        ctrl = build_controller(a.controller, geom, a.dt, traj)
        lane = LaneMonitor(traj.lane_x, traj.lane_y)
        speed_pi = SpeedPI()
        spectator = world.get_spectator()
        log = EpisodeLog()

        t = 0.0
        while t < a.duration:
            x, y, yaw, vx, vy, r = read_state(vehicle)
            res = step_controller(a.controller, ctrl, t, x, y, yaw, vx, vy, r)
            if res.s > traj.length - 15.0:
                print("[info] end of route")
                break
            steer = res.steer if res.steer is not None else (log.rows[-1].steer_rad if log.rows else 0.0)
            v_ref = speed_reference(traj, res.s, vx)
            throttle, brake = speed_pi(vx, v_ref, a.dt)
            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, brake=brake, steer=to_carla_steer(steer, geom["max_steer_rad"])))
            log.add(StepRow(t=round(t, 4), x=x, y=y, yaw=yaw, vx=vx, vy=vy, yaw_rate=r, s=res.s,
                            v_ref=v_ref, e_y=res.e_y, e_psi=res.e_psi,
                            e_y_lane=lane.lateral_offset(x, y), kappa=res.kappa, steer_rad=steer,
                            ok=res.ok, status=res.status, solve_ms=res.solve_ms,
                            qp_iterations=res.qp_iterations))
            if a.follow:
                follow_camera(spectator, vehicle, carla)
            world.tick()
            t = round(t + a.dt, 6)

        summary = log.summary(a.dt, traj.length)
        meta = {"simulator": "carla", "town": a.town, "vehicle": a.vehicle, "spawn": a.spawn,
                "controller": a.controller, "dt": a.dt, "vehicle_params": geom,
                "planner": traj.meta}
        log.save(out_dir, traj, meta, summary)
        if a.plot:
            log.plot(out_dir, traj, f"CARLA {a.town or ''} spawn {a.spawn} - {a.controller.upper()}")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original)


if __name__ == "__main__":
    main()
