"""CARLA <-> stack conversions (the only place that knows CARLA's frame).

CARLA is left-handed: x forward, y RIGHT, yaw clockwise [deg], angular
velocity [deg/s], positive steer = turn RIGHT, wheel positions in world
coordinates [cm]. The planner and both controllers use a right-handed frame:
x forward, y LEFT, yaw counter-clockwise [rad], positive steer = LEFT.
"""
from __future__ import annotations

import math


def read_state(vehicle):
    """Vehicle state in the right-handed frame: x, y, yaw, vx, vy, yaw_rate.

    vx / vy are body-frame velocities (forward / left)."""
    tf = vehicle.get_transform()
    v = vehicle.get_velocity()
    w = vehicle.get_angular_velocity()
    x, y = tf.location.x, -tf.location.y
    yaw = -math.radians(tf.rotation.yaw)
    vX, vY = v.x, -v.y
    vx = math.cos(yaw) * vX + math.sin(yaw) * vY
    vy = -math.sin(yaw) * vX + math.cos(yaw) * vY
    r = -math.radians(w.z)
    return x, y, yaw, vx, vy, r


def to_carla_steer(delta_rad: float, max_steer_rad: float) -> float:
    """Road-wheel angle (left +) -> CARLA normalised steer in [-1, 1] (right +)."""
    return float(max(-1.0, min(1.0, -delta_rad / max_steer_rad)))


def vehicle_geometry(vehicle, defaults: dict | None = None) -> dict:
    """Mass, axle distances and steering limit from CARLA's physics control.

    Returns a dict with the field names of both controllers' VehicleParams
    (mass, iz, lf, lr, cf, cr, max_steer_rad). CARLA does not expose tyre
    cornering stiffness, so cf / cr are the defaults scaled by mass; identify
    them for your blueprint if tracking is poor. iz is approximated by
    m * lf * lr (common estimate for passenger cars).
    """
    d = {"mass": 1800.0, "lf": 1.35, "lr": 1.55, "cf": 1.2e5, "cr": 1.4e5,
         "max_steer_rad": math.radians(30.0)}
    d.update(defaults or {})
    pc = vehicle.get_physics_control()
    m = float(pc.mass)
    front_wheels = pc.wheels[:2]
    max_steer = math.radians(sum(float(w.max_steer_angle) for w in front_wheels) / 2.0)
    lf, lr = d["lf"], d["lr"]
    try:  # wheel positions are world coordinates in cm
        tf = vehicle.get_transform()
        f = tf.get_forward_vector()
        loc = tf.location

        def along(wh):
            return (wh.position.x / 100.0 - loc.x) * f.x + (wh.position.y / 100.0 - loc.y) * f.y

        front = 0.5 * (along(pc.wheels[0]) + along(pc.wheels[1]))
        rear = 0.5 * (along(pc.wheels[2]) + along(pc.wheels[3]))
        com_x = float(pc.center_of_mass.x)  # local, metres, relative to the actor origin
        wheelbase = front - rear
        if 1.5 < wheelbase < 5.0:
            lf, lr = front - com_x, com_x - rear
            if not (0.3 < lf < wheelbase - 0.3):  # implausible CoM: assume 50/50 split
                print(f"[warn] centre of mass x={com_x:.2f} looks wrong; using lf = lr = wheelbase/2")
                lf = lr = wheelbase / 2.0
    except Exception as exc:  # keep defaults if the API differs
        print(f"[warn] could not read axle positions ({exc}); using defaults")
    scale = m / d["mass"]
    return {"mass": m, "iz": m * lf * lr, "lf": lf, "lr": lr,
            "cf": d["cf"] * scale, "cr": d["cr"] * scale, "max_steer_rad": max_steer}


def flatten_steering_curve(vehicle, carla_module):
    """CARLA scales the steering range down with speed by default; make it 1.0
    at all speeds so steer = delta / max_steer holds (the controllers assume it)."""
    pc = vehicle.get_physics_control()
    pc.steering_curve = [carla_module.Vector2D(0.0, 1.0), carla_module.Vector2D(300.0, 1.0)]
    vehicle.apply_physics_control(pc)


def scale_tire_friction(vehicle, scale: float) -> dict:
    """Multiply every wheel's WheelPhysicsControl.tire_friction by `scale` (integration3 --tire-friction-scale).

    CARLA API: vehicle.get_physics_control() -> VehiclePhysicsControl.wheels[i].tire_friction (float), then
    vehicle.apply_physics_control(pc). Returns {scale, applied, before, after[, error]} for agent_config.json /
    the 09 monitor (P-02 limit follows the applied scale). Never raises: a failure is recorded with applied=False
    so the run is still labelled with the tyres it really had."""
    info = {"scale": float(scale), "applied": False, "before": None, "after": None}
    try:
        pc = vehicle.get_physics_control()
        wheels = list(pc.wheels)
        info["before"] = [float(w.tire_friction) for w in wheels]
        for w in wheels:
            w.tire_friction = float(w.tire_friction) * float(scale)
        pc.wheels = wheels
        vehicle.apply_physics_control(pc)
        info["after"] = [float(w.tire_friction) for w in vehicle.get_physics_control().wheels]
        info["applied"] = all(abs(a - b * float(scale)) <= 1e-3 * max(1.0, abs(b))
                              for a, b in zip(info["after"], info["before"]))
        if not info["applied"]:
            info["error"] = "tire_friction read back differs from the requested scale"
    except Exception as exc:  # keep the run; the monitor treats an unapplied scale as 1.0
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def follow_camera(spectator, vehicle, carla_module, back=8.0, up=4.0):
    tf = vehicle.get_transform()
    f = tf.get_forward_vector()
    loc = carla_module.Location(tf.location.x - back * f.x, tf.location.y - back * f.y,
                                tf.location.z + up)
    spectator.set_transform(carla_module.Transform(loc, carla_module.Rotation(pitch=-15, yaw=tf.rotation.yaw)))
