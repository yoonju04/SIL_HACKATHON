"""LQR takeover on a PDM-style junction turn at PDM-Lite speeds (offline, Lincoln geometry)."""
import json, math, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
import test_pdm_reference as T
from sim import BicyclePlant, PlantParams
from integration.pdm_agent import route_to_reference
from integration.supervised_lateral import SupervisedLateral
from integration.fault_injection import FaultConfig
from lqr_controller import LQRConfig
import integration.supervised_lateral as SL

geo = json.load(open(Path(__file__).resolve().parents[1]/'runs/pdm26956_sup_01/agent_config.json'))['geometry'] \
    if Path(Path(__file__).resolve().parents[1]/'runs/pdm26956_sup_01/agent_config.json').exists() else None

def drive(speed, radius, lqr_cfg=None, rate_dps=60., fault=True):
    T.RADIUS = radius
    p = PlantParams(**{k: geo[k] for k in ('mass','iz','lf','lr','cf','cr','max_steer_rad')}, max_steer_rate_rad_s=math.radians(360))
    plant = BicyclePlant(p); plant.s.vx = speed
    route = T.carla_route(); rh = np.column_stack((route[:, 0], -route[:, 1])); ctl = None; worst = 0.
    for f in range(int((T.STRAIGHT + radius*math.pi/2 + 25)/speed/.05)):
        st = plant.s; d = np.hypot(rh[:, 0]-st.x, rh[:, 1]-st.y); i = int(np.argmin(d)); worst = max(worst, d[i])
        b = min(i, 30); ref = route_to_reference(route[i-b:], st.vx, b)
        if ctl is None:
            ctl = SupervisedLateral(geo, .05, ref, FaultConfig('solver_fail', (T.STRAIGHT-1.5*speed)/speed, 500.) if fault else FaultConfig())
            if lqr_cfg is not None:
                ctl.lqr = SL.build_controller('lqr', geo, .05, ref, lqr_cfg=lqr_cfg)
            ctl.rate_dps = rate_dps
        ctl.set_reference(ref, st.vx); r = ctl.step(f*.05, st.x, st.y, st.yaw, st.vx, st.vy, st.r)
        plant.step(r.steer, 0., .05); plant.s.vx = speed
    return worst

if __name__ == '__main__':
    cases = [(10, 10.), (12, 10.), (10, 15.), (13, 15.)]
    print('max path error [m], MPC fails 1.5 s before the turn; columns = speed@radius')
    print('%-30s' % '' + ''.join('%9s' % f'{v}@R{int(r)}' for v, r in cases))
    print('%-30s' % 'MPC only (no fault)' + ''.join('%9.2f' % drive(v, r, fault=False) for v, r in cases))
    for name, cfg, rate in [('LQR current (60 dps)', None, 60.), ('LQR 120 dps', None, 120.),
                            ('LQR preview 0.3 s', LQRConfig(preview_time_s=.3), 60.),
                            ('LQR preview 0.5 s', LQRConfig(preview_time_s=.5), 60.),
                            ('LQR preview 0.3 s, 120 dps', LQRConfig(preview_time_s=.3), 120.),
                            ('LQR preview 0.5 s, 120 dps', LQRConfig(preview_time_s=.5), 120.)]:
        print('%-30s' % name + ''.join('%9.2f' % drive(v, r, cfg, rate) for v, r in cases))
