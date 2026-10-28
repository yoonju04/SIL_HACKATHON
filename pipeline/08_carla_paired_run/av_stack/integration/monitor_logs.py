"""Input files of the pipeline's 09 Monitor (pipeline/09_monitor/monitor.py), written by the PDM agent.

The monitor's loader (load_av_stack_run) reads, joined on t:
  steps.jsonl          t, vx, vy, yaw_rate, e_y_lane, e_psi            (vehicle state, controller errors)
  behavior.jsonl       t, active_controller, supervisor_reason, fault_injected
  monitor_probe.jsonl  intended-lane offset / heading / width, a_y, bounding box (ground truth)
  collisions.json      [{t, other}]   summary.json  {supervisor: {safe_stop}, meta}

The probe follows the team's av_stack_probe.MonitorProbe: it only reads simulator state and
measures against the INTENDED lane (the route fixed at the start), not the nearest lane, so a
car that has drifted into the next lane does not look centred again. One change: the closest
route segment is searched in a window around the previous one, because Bench2Drive routes can
pass the same place twice.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Route segments (1 m) searched around the previous match; 20 m per 50 ms tick is far above any
# real speed, and a short window cannot jump to a later pass of the same road.
SEARCH_BEHIND, SEARCH_AHEAD = 5, 20


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
        self.width, self.lane_key = [], []
        for xi, yi in zip(self.x[:-1], self.y[:-1]):
            wp = self._waypoint(xi, -yi)
            self.width.append(float(wp.lane_width) if wp is not None else float('nan'))
            self.lane_key.append(None if wp is None else (wp.road_id, wp.lane_id))
        self.width = np.asarray(self.width)
        ext = vehicle.bounding_box.extent  # half sizes [m]
        self.half_l, self.half_w = float(ext.x), float(ext.y)
        self.index = None  # first tick: whole route
        self.fh = open(out_path, 'w', encoding='utf-8')

    def _waypoint(self, x_carla, y_carla):
        loc = type(self.vehicle.get_transform().location)(x_carla, y_carla, 0.0)
        kw = {'project_to_road': True}
        if self.lane_type is not None:
            kw['lane_type'] = self.lane_type
        try:
            return self.map.get_waypoint(loc, **kw)
        except Exception:
            return None

    def record(self, t: float, fault_effective: bool | None = None):
        tf = self.vehicle.get_transform()
        px, py = tf.location.x, -tf.location.y
        yaw = -math.radians(tf.rotation.yaw)
        lo, hi = (0, len(self.dx)) if self.index is None else             (max(0, self.index - SEARCH_BEHIND), min(len(self.dx), self.index + SEARCH_AHEAD))
        x0, y0, dx, dy = self.x[lo:hi], self.y[lo:hi], self.dx[lo:hi], self.dy[lo:hi]
        a = np.clip(((px - x0) * dx + (py - y0) * dy) / self.l2[lo:hi], 0.0, 1.0)
        qx, qy = x0 + a * dx, y0 + a * dy
        j = int(np.argmin((px - qx) ** 2 + (py - qy) ** 2))
        i = self.index = lo + j
        lane_yaw = float(self.yaw[i])
        offset = -math.sin(lane_yaw) * (px - qx[j]) + math.cos(lane_yaw) * (py - qy[j])
        acc = self.vehicle.get_acceleration()  # world frame, left-handed [m/s^2]
        aX, aY = acc.x, -acc.y
        ay = -math.sin(yaw) * aX + math.cos(yaw) * aY
        wp_now = self._waypoint(tf.location.x, tf.location.y)
        now_key = None if wp_now is None else (wp_now.road_id, wp_now.lane_id)
        row = {
            't': round(float(t), 4),
            'lane_offset_m': offset,  # signed, left +, reference point = actor origin
            'lane_heading_err_rad': _wrap(yaw - lane_yaw),
            'lane_width_m': float(self.width[i]),
            'intended_lane': self.lane_key[i],
            'map_lane_now': now_key,  # cross-check only (nearest lane)
            'ay_mps2': ay,
            'bbox_half_length_m': self.half_l,
            'bbox_half_width_m': self.half_w,
            'fault_effective': fault_effective,
        }
        self.fh.write(json.dumps(row) + '\n')

    def close(self):
        self.fh.close()


class MonitorLogs:
    """All monitor input files of one run. route_xy: intended route, CARLA frame, 1 m spacing."""

    def __init__(self, carla, world, vehicle, route_xy, out: Path, meta: dict):
        self.out = Path(out)
        route_xy = np.asarray(route_xy, float)
        self.probe = MonitorProbe(world.get_map(), vehicle, route_xy[:, 0], -route_xy[:, 1],
                                  self.out / 'monitor_probe.jsonl', lane_type=carla.LaneType.Driving)
        self.steps = (self.out / 'steps.jsonl').open('w', encoding='utf-8')
        self.behavior = (self.out / 'behavior.jsonl').open('w', encoding='utf-8')
        self.meta = meta
        self.safe_stop = False
        self.collisions = []
        self.t = 0.
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        # Sensor callbacks fire during the tick; stamp them with the agent clock of that tick.
        self.sensor.listen(lambda event: self.collisions.append(
            dict(t=round(self.t, 4), other=event.other_actor.type_id)))

    def record(self, t, vx, vy, yaw_rate, e_y, e_psi, active, reason, fault_injected, safe_stop,
               fault_effective=None):
        self.t = t
        key = round(float(t), 4)
        self.steps.write(json.dumps(dict(t=key, vx=vx, vy=vy, yaw_rate=yaw_rate,
                                         e_y_lane=e_y, e_psi=e_psi)) + '\n')
        self.behavior.write(json.dumps(dict(t=key, active_controller=active, supervisor_reason=reason,
                                            fault_injected=bool(fault_injected),
                                            fault_effective=fault_effective)) + '\n')
        self.probe.record(t, fault_effective)
        self.safe_stop = self.safe_stop or bool(safe_stop)

    def close(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
        for fh in (self.steps, self.behavior):
            fh.close()
        self.probe.close()
        (self.out / 'collisions.json').write_text(json.dumps(self.collisions), encoding='utf-8')
        (self.out / 'summary.json').write_text(json.dumps(dict(
            supervisor=dict(safe_stop=self.safe_stop), meta=self.meta), indent=2), encoding='utf-8')
