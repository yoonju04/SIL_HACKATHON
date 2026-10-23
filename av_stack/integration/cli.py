"""Command-line options shared by run_carla.py and run_offline.py."""
from __future__ import annotations

import argparse

from planner import LaneChange

from .episode import CONTROLLERS


def parse_lane_change(text: str) -> LaneChange:
    s0, length, offset = (float(v) for v in text.split(":"))
    return LaneChange(s0, length, offset)


def common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--controller", choices=CONTROLLERS, default="mpc")
    ap.add_argument("--speed", type=float, default=15.0, help="planner cruise speed v_max [m/s]")
    ap.add_argument("--a-lat", type=float, default=3.0, help="planner lateral acceleration limit [m/s^2]")
    ap.add_argument("--mu", type=float, default=0.9, help="friction assumed by the planner")
    ap.add_argument("--lane-change", action="append", default=[], type=parse_lane_change,
                    metavar="S0:LEN:OFFSET", help="quintic lane change, lateral offset left + [m]")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.05, help="control period [s]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
