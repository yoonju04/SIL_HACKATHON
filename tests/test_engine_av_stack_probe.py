"""Probe test with minimal duck-typed CARLA objects (left-handed frame)."""
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

from engine.av_stack_probe import MonitorProbe  # noqa: E402


class Loc:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Veh:
    def __init__(self, x, y_rh, yaw_rh):
        self.bounding_box = NS(extent=NS(x=2.4, y=0.95, z=0.7))
        self.set(x, y_rh, yaw_rh)
        self.acc = NS(x=0.0, y=0.0, z=0.0)

    def set(self, x, y_rh, yaw_rh):
        self.tf = NS(location=Loc(x, -y_rh), rotation=NS(yaw=-math.degrees(yaw_rh)))

    def get_transform(self):
        return self.tf

    def get_acceleration(self):
        return self.acc


class Map:  # straight lane along x, lane 1 for |y|<1.75, lane 2 to the left
    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        y_rh = -loc.y
        return NS(lane_width=3.5, road_id=7, lane_id=1 if y_rh < 1.75 else 2)


def test_offset_heading_ay_and_intended_lane():
    xs = np.arange(0.0, 100.0, 1.0)
    veh = Veh(50.0, 0.0, 0.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "monitor_probe.jsonl")
        pr = MonitorProbe(Map(), veh, xs, 0 * xs, p)
        pr.record(0.0)
        veh.set(50.0, 2.5, math.radians(3))  # drifted into the left lane
        veh.acc = NS(x=0.0, y=-2.0, z=0.0)  # CARLA y right -> 2 m/s^2 to the LEFT
        pr.record(0.05, fault_effective=True)
        pr.close()
        rows = [json.loads(line) for line in open(p)]
    assert abs(rows[0]["lane_offset_m"]) < 1e-9
    r = rows[1]
    assert abs(r["lane_offset_m"] - 2.5) < 1e-9  # measured against the INTENDED lane
    assert r["intended_lane"] == [7, 1] and r["map_lane_now"] == [7, 2]
    assert abs(r["lane_heading_err_rad"] - math.radians(3)) < 1e-9
    assert abs(r["ay_mps2"] - 2.0 * math.cos(math.radians(3))) < 1e-9
    assert r["lane_width_m"] == 3.5 and r["bbox_half_width_m"] == 0.95


if __name__ == "__main__":
    test_offset_heading_ay_and_intended_lane()
    print("PASS test_offset_heading_ay_and_intended_lane")
