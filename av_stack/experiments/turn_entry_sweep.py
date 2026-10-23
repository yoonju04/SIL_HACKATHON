"""Closed-loop junction-turn check: straight road -> R=8 m right turn, TF++-style moving checkpoints.

Each tick the controller receives 10 path points 2.5..11.5 m ahead in the ego frame (TF++ format),
exactly like the live agent. Reports the max distance to the true path during and after the turn.
"""
import json, math, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import BicyclePlant, PlantParams
import integration.tfpp_reference as R
from integration.tfpp_control import PredictionController, SensorState
from integration.fault_injection import FaultConfig
from integration.episode import build_controller
from mpc_controller import MPCConfig

RADIUS, STRAIGHT, SIDE = 8.0, 25.0, -1.0  # SIDE -1 = right turn, +1 = left turn
def true_path(ds=0.1):
    s = np.arange(0, STRAIGHT + RADIUS*math.pi/2 + 30, ds)
    x, y = [], []
    for si in s:
        if si < STRAIGHT: x.append(si); y.append(0.)
        elif si < STRAIGHT + RADIUS*math.pi/2:
            a = (si-STRAIGHT)/RADIUS; x.append(STRAIGHT+RADIUS*math.sin(a)); y.append(SIDE*RADIUS*(1-math.cos(a)))
        else:
            x.append(STRAIGHT+RADIUS); y.append(SIDE*(RADIUS+(si-STRAIGHT-RADIUS*math.pi/2)))
    return s, np.array(x), np.array(y)

EXT = {'quad': 'quadratic', 'linear': 'linear'}

def run(mode, ext, speed, mpc_cfg=None, geo=None, lqr_cfg=None, fault=FaultConfig()):
    S, PX, PY = true_path()
    p = PlantParams(**{k: geo[k] for k in ('mass','iz','lf','lr','cf','cr','max_steer_rad')}, max_steer_rate_rad_s=math.radians(360))
    plant = BicyclePlant(p); plant.s.vx = speed
    ctrl, worst, i_hint = None, 0., 0
    for frame in range(int((STRAIGHT + RADIUS*math.pi/2 + 15)/speed/0.05)):
        st = plant.s
        d = np.hypot(PX-st.x, PY-st.y); i = int(np.argmin(d)); worst = max(worst, d[i]) if S[i] > STRAIGHT-5 else worst
        target = S[i] + np.arange(2.5, 12.5, 1.0)
        tx, ty = np.interp(target, S, PX), np.interp(target, S, PY)
        dx, dy = tx-st.x, ty-st.y
        # TF++ ego frame: x forward, y right
        pts = np.column_stack((math.cos(st.yaw)*dx+math.sin(st.yaw)*dy, math.sin(st.yaw)*dx-math.cos(st.yaw)*dy))
        ref = R.checkpoints_to_reference(pts, (st.x, -st.y, -st.yaw), speed, st.vx, EXT[ext])
        lqr_ref = R.checkpoints_to_reference(pts, (st.x, -st.y, -st.yaw), speed, st.vx, 'quadratic')
        if ctrl is None:
            ctrl = PredictionController(mode, geo, .05, ref, fault)
            if mpc_cfg is not None:
                ctrl.controller = build_controller('mpc', geo, .05, ref, mpc_cfg=mpc_cfg)
            if lqr_cfg is not None:
                ctrl.controller = build_controller('lqr', geo, .05, ref, lqr_cfg=lqr_cfg)
        cmd = ctrl.command(ref, speed, SensorState(st.x, st.y, st.yaw, st.vx, st.r), frame*.05, lqr_ref)
        plant.step(-cmd.steer*p.max_steer_rad, 0., .05)
        plant.s.vx = speed
    return worst

if __name__ == '__main__':
    geo = json.load(open(Path(__file__).resolve().parents[1]/'runs/bench_mpc_v4_01/agent_config.json'))['geometry']
    speeds, cases = (4, 6, 8), [(8., -1.), (8., 1.), (15., -1.)]
    def entry_fault(v):  # MPC fails 1 s before the turn starts -> LQR takes the whole turn
        return FaultConfig('solver_fail', max(0., (STRAIGHT-v)/v), 500.)
    variants = [('MPC old (quadratic)', 'mpc', 'quad', None),
                ('MPC new (linear)', 'mpc', 'linear', None),
                ('LQR (quadratic)', 'lqr', 'quad', None),
                ('supervised, no fault', 'supervised', 'linear', None),
                ('supervised, fault at entry', 'supervised', 'linear', entry_fault)]
    print('max distance to true path [m]; columns = radius/side@speed (straight -> turn)')
    print('%-28s' % 'variant' + ''.join('%8s' % f"{int(r)}{'R' if sd<0 else 'L'}@{v}" for r, sd in cases for v in speeds) + '   worst')
    for name, mode, ext, fault in variants:
        vals = []
        for r, sd in cases:
            RADIUS, SIDE = r, sd
            vals += [run(mode, ext, v, geo=geo, fault=fault(v) if fault else FaultConfig()) for v in speeds]
        print('%-28s' % name + ''.join('%8.2f' % x for x in vals) + '%8.2f' % max(vals))
