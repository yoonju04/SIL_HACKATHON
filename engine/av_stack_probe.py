"""Ground-truth probe for the 09 Monitor, to be called once per tick by an av_stack runner.
(teammate's pipeline/09_monitor/av_stack_probe.py, sim-eval a5242b2; moved to engine/. The copy that actually
runs inside CARLA is pipeline/08_carla_paired_run/av_stack/integration/monitor_logs.py MonitorProbe.)

Writes monitor_probe.jsonl next to steps.jsonl. It only READS simulator state
(no commands), and it does not use any controller output, so P-01..P-03 do
not depend on what the system under test believes.

Why not simply map.get_waypoint(vehicle_location)? That returns the NEAREST
lane: once the car has drifted across the line, the adjacent lane becomes
"nearest" and the offset looks small again, hiding the violation. So the
probe measures against the INTENDED lane: the scenario's lane-centre polyline
(fixed before the run) with the lane width read from the CARLA map at each
route point. The map-projected lane id is logged only as a cross-check.

Usage in a runner (duck-typed: pass the carla module's LaneType.Driving):
    probe = MonitorProbe(world.get_map(), vehicle, traj.lane_x, traj.lane_y,
                         out_dir / "monitor_probe.jsonl", lane_type=carla.LaneType.Driving)
    ... every tick, after reading the state and before world.tick():
    probe.record(t, fault_effective=injector_changed_output_this_tick)
    ... at the end: probe.close()
Frames: CARLA is left-handed; everything logged here is right-handed (y left).
"""
from __future__ import annotations

import json
import math

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MonitorProbe:
    def __init__(self, world_map, vehicle, lane_x, lane_y, out_path, lane_type=None):
        self.vehicle = vehicle
        self.map = world_map
        self.lane_type = lane_type
        self.x = np.asarray(lane_x, float)
        self.y = np.asarray(lane_y, float)  # right-handed
        self.dx, self.dy = np.diff(self.x), np.diff(self.y)
        self.l2 = np.maximum(self.dx**2 + self.dy**2, 1e-12)
        self.yaw = np.arctan2(self.dy, self.dx)
        # intended lane width / id along the route, read from the map once
        self.width, self.lane_key = [], []
        for xi, yi in zip(self.x[:-1], self.y[:-1]):
            wp = self._waypoint(xi, -yi)
            self.width.append(float(wp.lane_width) if wp is not None else float("nan"))
            self.lane_key.append(None if wp is None else (wp.road_id, wp.lane_id))
        self.width = np.asarray(self.width)
        ext = vehicle.bounding_box.extent  # half sizes [m]
        self.half_l, self.half_w = float(ext.x), float(ext.y)
        self.fh = open(out_path, "w", encoding="utf-8")

    def _waypoint(self, x_carla, y_carla):
        loc = type(self.vehicle.get_transform().location)(x_carla, y_carla, 0.0)
        kw = {"project_to_road": True}
        if self.lane_type is not None:
            kw["lane_type"] = self.lane_type
        try:
            return self.map.get_waypoint(loc, **kw)
        except Exception:
            return None

    def record(self, t: float, fault_effective: bool | None = None):
        tf = self.vehicle.get_transform()
        px, py = tf.location.x, -tf.location.y
        yaw = -math.radians(tf.rotation.yaw)
        a = np.clip(((px - self.x[:-1]) * self.dx + (py - self.y[:-1]) * self.dy) / self.l2, 0.0, 1.0)
        qx, qy = self.x[:-1] + a * self.dx, self.y[:-1] + a * self.dy
        i = int(np.argmin((px - qx) ** 2 + (py - qy) ** 2))
        lane_yaw = float(self.yaw[i])
        offset = -math.sin(lane_yaw) * (px - qx[i]) + math.cos(lane_yaw) * (py - qy[i])
        acc = self.vehicle.get_acceleration()  # world frame, left-handed [m/s^2]
        aX, aY = acc.x, -acc.y
        ay = -math.sin(yaw) * aX + math.cos(yaw) * aY
        wp_now = self._waypoint(tf.location.x, tf.location.y)
        now_key = None if wp_now is None else (wp_now.road_id, wp_now.lane_id)
        row = {
            "t": round(float(t), 4),
            "lane_offset_m": offset,  # signed, left +, reference point = actor origin
            "lane_heading_err_rad": _wrap(yaw - lane_yaw),
            "lane_width_m": float(self.width[i]),
            "intended_lane": self.lane_key[i],
            "map_lane_now": now_key,  # cross-check only (nearest lane)
            "ay_mps2": ay,
            "bbox_half_length_m": self.half_l,
            "bbox_half_width_m": self.half_w,
            "fault_effective": fault_effective,
        }
        self.fh.write(json.dumps(row) + "\n")

    def close(self):
        self.fh.close()
