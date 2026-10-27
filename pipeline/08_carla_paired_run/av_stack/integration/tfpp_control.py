from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Tuple

import numpy as np

from integration.behavior_control import BehaviorSpeedControl, Pedals, update_reference
from integration.carla_io import to_carla_steer
from integration.episode import build_controller, step_controller
from integration.fault_injection import FaultConfig
from integration.supervised_lateral import SupervisedLateral
from integration.unprotected_lateral import UnprotectedMPC
from integration.speed import speed_reference
from mpc_controller import MPCConfig
from planner import Trajectory


@dataclass(frozen=True)
class SensorState:
    x: float
    y: float
    yaw: float
    vx: float
    yaw_rate: float


@dataclass(frozen=True)
class OwnControl:
    throttle: float
    brake: float
    steer: float
    source: str
    target_speed: float
    tracking_error: float


# (target speed, measured speed, stop) -> (throttle, brake); e.g. the planner's own longitudinal law.
ThrottleLaw = Callable[[float, float, bool], Tuple[float, float]]


class PredictionController:
    def __init__(self, mode: str, geometry: dict[str, float], dt: float,
                 trajectory: Trajectory, fault: FaultConfig, throttle_law: ThrottleLaw | None = None,
                 curvature_speed_cap: bool = True) -> None:
        self.mode = mode
        self.curvature_speed_cap = curvature_speed_cap
        self.geometry = geometry
        self.speed = BehaviorSpeedControl(dt)
        self.throttle_law = throttle_law
        mpc = MPCConfig(horizon=20, prediction_dt=.1, model_speed='measured')
        self.supervised = SupervisedLateral(geometry, dt, trajectory, fault, mpc) if mode == 'supervised' else None
        self.unprotected = UnprotectedMPC(geometry, dt, trajectory, fault, mpc) if mode == 'mpc_unprotected' else None
        self.managed = self.supervised if self.supervised is not None else self.unprotected
        self.controller = (self.managed if self.managed is not None
                           else build_controller(mode, geometry, dt, trajectory, mpc_cfg=mpc))

    def command(self, reference: Trajectory, desired_speed: float, state: SensorState, t: float,
                lqr_reference: Trajectory | None = None) -> OwnControl:
        """reference feeds the MPC (and speed lookup); lqr_reference, if given, feeds the LQR."""
        tracking_speed = max(state.vx, .5)

        def tracking(path: Trajectory) -> Trajectory:
            return replace(path, v=np.full_like(path.v, tracking_speed), t=(path.s-path.s[0])/tracking_speed)
        tracking_reference = tracking(reference)
        tracking_lqr = tracking(lqr_reference) if lqr_reference is not None else tracking_reference
        if self.managed is not None:
            self.managed.set_reference(tracking_reference, tracking_speed, tracking_lqr)
            result = self.managed.step(t, state.x, state.y, state.yaw, state.vx, 0, state.yaw_rate)
        else:
            update_reference(self.controller, tracking_lqr if self.mode == 'lqr' else tracking_reference,
                             tracking_speed)
            result = step_controller(self.mode, self.controller, t, state.x, state.y, state.yaw,
                                     state.vx, 0, state.yaw_rate)
        stop = desired_speed < .01 or (self.supervised is not None and self.supervised.stop_required)
        if result.steer is None:
            stop = True
        target = desired_speed
        if self.curvature_speed_cap:
            target = min(target, speed_reference(reference, result.s, state.vx))
        target = 0.0 if stop else target
        if self.throttle_law is not None:
            pedals = Pedals(*self.throttle_law(target, state.vx, bool(stop)))
        else:
            pedals = self.speed.command(state.vx, target, bool(stop))
        steering = to_carla_steer(result.steer or 0, self.geometry['max_steer_rad'])
        source = self.managed.active if self.managed is not None else self.mode
        return OwnControl(float(pedals.throttle), float(pedals.brake), steering, source, float(target), float(result.e_y))
