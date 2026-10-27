#!/usr/bin/env python3
"""Same pipeline as run_carla.py without CARLA: planner -> MPC/LQR -> offline vehicle.

  python run_offline.py --controller mpc --plot
  python run_offline.py --controller lqr --track arc --speed 12
  python run_offline.py --controller mpc --track straight --lane-change 100:60:3.5 --plot
The offline vehicle (sim/plant.py) is a nonlinear single-track model with tyre
saturation and a steering actuator lag, i.e. NOT the controllers' linear model.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402

from integration.cli import common_args  # noqa: E402
from integration.episode import EpisodeLog, LaneMonitor, StepRow, build_controller, step_controller  # noqa: E402
from integration.speed import accel_command, speed_reference  # noqa: E402
from planner import PlannerConfig, ReferencePlanner  # noqa: E402
from sim import BicyclePlant, PlantParams, arc_route, demo_route, straight_route  # noqa: E402

TRACKS = {"demo": demo_route, "arc": arc_route, "straight": straight_route}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--track", choices=sorted(TRACKS), default="demo")
    ap.add_argument("--plant-mu", type=float, default=0.9, help="road friction of the offline vehicle")
    a = ap.parse_args(argv)
    out_dir = a.out or os.path.join("runs", f"offline_{a.controller}_{a.track}")

    pp = PlantParams()
    geom = {"mass": pp.mass, "iz": pp.iz, "lf": pp.lf, "lr": pp.lr, "cf": pp.cf, "cr": pp.cr,
            "max_steer_rad": pp.max_steer_rad}
    rx, ry = TRACKS[a.track]()
    traj = ReferencePlanner(PlannerConfig(v_max=a.speed, a_lat_max=a.a_lat, mu=a.mu)).plan(
        rx, ry, tuple(a.lane_change))
    ctrl = build_controller(a.controller, geom, a.dt, traj)
    lane = LaneMonitor(traj.lane_x, traj.lane_y)
    plant = BicyclePlant(pp, mu=a.plant_mu)
    st = plant.s
    st.x, st.y, st.yaw, st.vx = float(traj.x[0]), float(traj.y[0]), float(traj.yaw[0]), 0.0
    log = EpisodeLog()
    t = 0.0
    while t < a.duration:
        res = step_controller(a.controller, ctrl, t, st.x, st.y, st.yaw, st.vx, st.vy, st.r)
        if res.s > traj.length - 5.0:
            break
        steer = res.steer if res.steer is not None else (log.rows[-1].steer_rad if log.rows else 0.0)
        v_ref = speed_reference(traj, res.s, st.vx)
        log.add(StepRow(t=round(t, 4), x=st.x, y=st.y, yaw=st.yaw, vx=st.vx, vy=st.vy, yaw_rate=st.r,
                        s=res.s, v_ref=v_ref, e_y=res.e_y, e_psi=res.e_psi,
                        e_y_lane=lane.lateral_offset(st.x, st.y), kappa=res.kappa, steer_rad=steer,
                        ok=res.ok, status=res.status, solve_ms=res.solve_ms,
                        qp_iterations=res.qp_iterations))
        plant.step(steer, accel_command(st.vx, v_ref), a.dt)
        t = round(t + a.dt, 6)
    summary = log.summary(a.dt, traj.length)
    meta = {"simulator": "offline", "track": a.track, "controller": a.controller, "dt": a.dt,
            "vehicle_params": geom, "planner": traj.meta}
    log.save(out_dir, traj, meta, summary)
    if a.plot:
        log.plot(out_dir, traj, f"offline {a.track} - {a.controller.upper()}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
