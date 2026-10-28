"""Small closed-loop helper for tests: planner -> controller -> offline plant."""
import numpy as np

import lqr_controller as L
import mpc_controller as M
from planner import PlannerConfig, ReferencePlanner
from sim import BicyclePlant, PlantParams, demo_route


def drive(kind, route=None, cfg=None, v_max=15.0, a_lat=3.0, dt=0.05, mu=0.9, t_max=80.0,
          lane_changes=(), hook=None):
    pp = PlantParams()
    geom = dict(mass=pp.mass, iz=pp.iz, lf=pp.lf, lr=pp.lr, cf=pp.cf, cr=pp.cr, max_steer_rad=pp.max_steer_rad)
    rx, ry = route if route is not None else demo_route()
    traj = ReferencePlanner(PlannerConfig(v_max=v_max, a_lat_max=a_lat)).plan(rx, ry, lane_changes)
    if kind == "lqr":
        ctrl = L.LQRLateralController(L.VehicleParams(**geom), dt, cfg)
        ctrl.set_reference(traj.x, traj.y, traj.yaw, traj.kappa, traj.s)
    else:
        ctrl = M.MPCLateralController(M.VehicleParams(**geom), dt, cfg)
        ctrl.set_reference(traj.x, traj.y, traj.yaw, traj.kappa, traj.s, traj.v, traj.t)
    plant = BicyclePlant(pp, mu=mu)
    s = plant.s
    s.x, s.y, s.yaw = float(traj.x[0]), float(traj.y[0]), float(traj.yaw[0])
    rows, t = [], 0.0
    while t < t_max:
        o = ctrl.compute(s.x, s.y, s.yaw, s.vx, s.vy, s.r) if kind == "lqr" else \
            ctrl.compute(s.x, s.y, s.yaw, s.vx, s.vy, s.r, t)
        if o.s > traj.length - 5.0:
            break
        steer = o.steer if hook is None else hook(t, o)
        v_ref = float(np.interp(o.s + max(2.0, 0.5 * s.vx), traj.s, traj.v))
        plant.step(steer, max(-4.0, min(2.0, v_ref - s.vx)), dt)
        rows.append((t, float(o.state[0]), steer, s.vx, o))
        t += dt
    e = np.array([r[1] for r in rows if r[3] > 1.0])
    st = np.array([r[2] for r in rows])
    return {"max_e_y": float(np.abs(e).max()), "rms_e_y": float(np.sqrt((e**2).mean())),
            "max_rate": float(np.abs(np.diff(st)).max() / dt), "t_end": t, "rows": rows,
            "traj": traj, "ctrl": ctrl}
