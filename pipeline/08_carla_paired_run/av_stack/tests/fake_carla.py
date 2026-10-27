"""Minimal stand-in for the `carla` module, to test run_carla.py without a server.

Reproduces CARLA's LEFT-HANDED conventions (y right, yaw clockwise in degrees,
angular velocity in deg/s, positive steer = right, wheel positions in world
cm, Waypoint.next(d) returning a list) on top of the right-handed offline
plant, so the frame conversions in integration/carla_io.py are exercised.
Only the API calls used by run_carla.py are implemented.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.plant import BicyclePlant, PlantParams  # noqa: E402
from sim.tracks import demo_route  # noqa: E402

ROUTE = {"fn": demo_route}  # tests may swap the route


class Vector2D:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


Location = Vector3D


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def get_forward_vector(self):
        y = math.radians(self.rotation.yaw)
        return Vector3D(math.cos(y), math.sin(y), 0.0)


class LaneType:
    Driving = 2


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0):
        self.throttle, self.steer, self.brake = throttle, steer, brake


class _Wheel:
    def __init__(self, max_steer_angle, position):
        self.max_steer_angle, self.position = max_steer_angle, position


class _Physics:
    def __init__(self, veh):
        p = veh.p
        tf = veh.get_transform()
        f = tf.get_forward_vector()
        L = tf.location

        def pos(d):  # world position (cm) of a point d metres ahead of the CoG
            return Vector3D((L.x + d * f.x) * 100.0, (L.y + d * f.y) * 100.0, 0.0)

        ms = math.degrees(p.max_steer_rad)
        self.mass = p.mass
        self.center_of_mass = Vector3D(0.0, 0.0, 0.0)
        self.wheels = [_Wheel(ms, pos(p.lf)), _Wheel(ms, pos(p.lf)), _Wheel(0.0, pos(-p.lr)),
                       _Wheel(0.0, pos(-p.lr))]
        self.steering_curve = [Vector2D(0.0, 1.0), Vector2D(120.0, 0.7)]


class _Route:
    def __init__(self):
        x, y = ROUTE["fn"]()
        self.x, self.y = np.asarray(x), np.asarray(y)  # right-handed, ~1 m spacing
        self.yaw = np.unwrap(np.arctan2(np.gradient(self.y), np.gradient(self.x)))


class _Waypoint:
    def __init__(self, route, i):
        self._r, self._i = route, i
        self.transform = Transform(Location(route.x[i], -route.y[i], 0.0),
                                   Rotation(yaw=-math.degrees(route.yaw[i])))

    def next(self, step):
        j = self._i + max(1, int(round(step)))
        return [_Waypoint(self._r, j)] if j < len(self._r.x) else []


class _Map:
    def __init__(self, route):
        self._r = route

    def get_spawn_points(self):
        r = self._r
        return [Transform(Location(r.x[0], -r.y[0], 0.5), Rotation(yaw=-math.degrees(r.yaw[0])))]

    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        d = (self._r.x - loc.x) ** 2 + (-self._r.y - loc.y) ** 2
        return _Waypoint(self._r, int(np.argmin(d)))


class _Vehicle:
    def __init__(self, world, tf):
        self.world = world
        self.p = PlantParams()
        self.plant = BicyclePlant(self.p)
        s = self.plant.s
        s.x, s.y, s.yaw = tf.location.x, -tf.location.y, -math.radians(tf.rotation.yaw)
        self.control = VehicleControl()
        self.physics_applied = None

    def get_transform(self):
        s = self.plant.s
        return Transform(Location(s.x, -s.y, 0.0), Rotation(yaw=-math.degrees(s.yaw)))

    def get_velocity(self):
        s = self.plant.s
        vX = s.vx * math.cos(s.yaw) - s.vy * math.sin(s.yaw)
        vY = s.vx * math.sin(s.yaw) + s.vy * math.cos(s.yaw)
        return Vector3D(vX, -vY, 0.0)

    def get_angular_velocity(self):
        return Vector3D(0.0, 0.0, -math.degrees(self.plant.s.r))

    def get_physics_control(self):
        return _Physics(self)

    def apply_physics_control(self, pc):
        self.physics_applied = pc

    def apply_control(self, c):
        assert -1.0 <= c.steer <= 1.0 and 0.0 <= c.throttle <= 1.0 and 0.0 <= c.brake <= 1.0
        self.control = c

    def destroy(self):
        self.world._vehicle = None

    def _tick(self, dt):
        c = self.control
        delta = -c.steer * self.p.max_steer_rad  # CARLA steer right+ -> right-handed left+
        accel = 4.0 * c.throttle - 8.0 * c.brake - 0.01 * self.plant.s.vx ** 2 / 10.0
        self.plant.step(delta, accel, dt)


class _Settings:
    def __init__(self):
        self.synchronous_mode = False
        self.fixed_delta_seconds = None
        self.no_rendering_mode = False


class _World:
    def __init__(self):
        self._route = _Route()
        self._settings = _Settings()
        self._vehicle = None
        self.ticks = 0

    def get_settings(self):
        s = _Settings()
        s.__dict__.update(self._settings.__dict__)
        return s

    def apply_settings(self, s):
        self._settings = s

    def get_map(self):
        return _Map(self._route)

    def get_blueprint_library(self):
        return type("BPL", (), {"find": lambda self_, name: name})()

    def spawn_actor(self, bp, tf):
        self._vehicle = _Vehicle(self, tf)
        return self._vehicle

    def get_spectator(self):
        return type("Spec", (), {"set_transform": lambda self_, tf: None})()

    def tick(self):
        assert self._settings.synchronous_mode, "run_carla must enable synchronous mode"
        if self._vehicle is not None:
            self._vehicle._tick(self._settings.fixed_delta_seconds or 0.05)
        self.ticks += 1


class Client:
    world = None

    def __init__(self, host, port):
        Client.world = _World()

    def set_timeout(self, t):
        pass

    def get_world(self):
        return Client.world

    def load_world(self, town):
        return Client.world

    def get_trafficmanager(self):
        return type("TM", (), {"set_synchronous_mode": lambda self_, v: None})()
