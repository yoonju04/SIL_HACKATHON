"""09 Monitor input files from the PDM agent (integration/monitor_logs.py), with fake CARLA objects.

Checks the intended-lane offset sign (left +, CARLA's y flipped), that the probe keeps matching
the intended route when the car has drifted, that a route passing the same place twice is
matched by progress, and the file layout the monitor's load_av_stack_run reads.
"""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.monitor_logs import MonitorLogs, MonitorProbe


class Location:
    def __init__(self, x, y, z=0.):
        self.x, self.y, self.z = x, y, z


class Vehicle:
    bounding_box = NS(extent=NS(x=2.4, y=1.0))

    def __init__(self):
        self.x, self.y, self.yaw_deg = 0., 0., 0.

    def get_transform(self):
        return NS(location=Location(self.x, self.y), rotation=NS(yaw=self.yaw_deg))

    def get_acceleration(self):
        return NS(x=0., y=0.)


class Map:
    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        return NS(lane_width=3.5, road_id=1, lane_id=-1)


class Sensor:
    def listen(self, callback):
        self.callback = callback

    def stop(self):
        pass

    def destroy(self):
        self.destroyed = True


class World:
    def __init__(self):
        self.sensor = Sensor()

    def get_map(self):
        return Map()

    def get_blueprint_library(self):
        return NS(find=lambda name: name)

    def spawn_actor(self, blueprint, transform, attach_to=None):
        return self.sensor


FAKE_CARLA = NS(LaneType=NS(Driving='driving'), Transform=lambda: None)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def test_offset_sign_and_files():
    route = np.column_stack((np.arange(0., 200.), np.zeros(200)))  # CARLA frame, east
    vehicle, world = Vehicle(), World()
    with tempfile.TemporaryDirectory() as tmp:
        logs = MonitorLogs(FAKE_CARLA, world, vehicle, route, Path(tmp), meta=dict(controller='supervised'))
        for k in range(40):
            t = .05 * k
            vehicle.x, vehicle.y = 10. + .5 * k, 1.2  # CARLA +y = right of the route
            logs.record(t, 10., 0., 0., -1.2, 0., 'mpc' if k < 20 else 'lqr',
                        None if k < 20 else 'tracking_divergence', k >= 5, False, 5 <= k < 20)
        world.sensor.callback(NS(other_actor=NS(type_id='vehicle.tesla.model3')))
        logs.close()
        probe = _rows(Path(tmp, 'monitor_probe.jsonl'))
        steps = _rows(Path(tmp, 'steps.jsonl'))
        behavior = _rows(Path(tmp, 'behavior.jsonl'))
        assert len(probe) == len(steps) == len(behavior) == 40
        assert [r['t'] for r in probe] == [r['t'] for r in steps] == [r['t'] for r in behavior]
        assert all(abs(r['lane_offset_m'] + 1.2) < 1e-9 for r in probe)  # right of the lane = negative
        assert probe[0]['lane_width_m'] == 3.5 and probe[0]['bbox_half_width_m'] == 1.0
        assert {'vx', 'vy', 'yaw_rate', 'e_y_lane', 'e_psi'} <= set(steps[0])
        assert behavior[25]['active_controller'] == 'lqr' and behavior[4]['fault_injected'] is False
        # BUG-002: whether the fault changed the MPC output is logged per tick in both files
        assert [r['fault_effective'] for r in probe] == [5 <= k < 20 for k in range(40)]
        assert behavior[10]['fault_effective'] is True and behavior[30]['fault_effective'] is False
        collisions = json.loads(Path(tmp, 'collisions.json').read_text())
        assert collisions == [dict(t=1.95, other='vehicle.tesla.model3')]
        summary = json.loads(Path(tmp, 'summary.json').read_text())
        assert summary['supervisor']['safe_stop'] is False and summary['meta']['controller'] == 'supervised'
        assert world.sensor.destroyed


def test_route_passing_the_same_place_twice():
    """Out east along y=0, U-turn (R=10 m), back west along y=-20 (CARLA): a 25 m drift south
    from the outbound leg must still read as 25 m off that leg, not 5 m off the return leg."""
    out = [(x, 0.) for x in np.arange(0., 100.)]
    turn = [(100 + 10 * math.sin(a), -10 + 10 * math.cos(a)) for a in np.linspace(0, math.pi, 32)[1:-1]]
    back = [(x, -20.) for x in np.arange(100., -1., -1.)]
    route = np.array(out + turn + back)
    vehicle = Vehicle()
    with tempfile.TemporaryDirectory() as tmp:
        probe = MonitorProbe(Map(), vehicle, route[:, 0], -route[:, 1], Path(tmp, 'p.jsonl'))
        for k in range(30):
            vehicle.x, vehicle.y = 20. + k, -k * 25. / 29  # CARLA -y = left, drifting onto the return leg
            probe.record(.05 * k)
        probe.close()
        last = _rows(Path(tmp, 'p.jsonl'))[-1]
        assert abs(last['lane_offset_m'] - 25.) < 1e-6, last


if __name__ == '__main__':
    test_offset_sign_and_files()
    test_route_passing_the_same_place_twice()
    print('ok')
