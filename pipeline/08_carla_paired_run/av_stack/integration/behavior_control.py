"""Actuator boundary: lateral references go to MPC/LQR, stop intent overrides PI."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from integration.speed import SpeedPI
from lqr_controller import LQRLateralController
from mpc_controller import MPCLateralController
from planner import Trajectory


@dataclass(frozen=True)
class Pedals:
    throttle: float
    brake: float


class BehaviorSpeedControl:
    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.pi = SpeedPI()

    def command(self, speed: float, target: float, stop: bool) -> Pedals:
        if stop or target <= 0.0:
            self.pi.i = 0.0
            return Pedals(0.0, 1.0)
        if speed > target and self.pi.i > 0:
            self.pi.i = 0.0
        throttle, brake = self.pi(speed, target, self.dt)
        return Pedals(throttle, brake)


def update_reference(controller: MPCLateralController | LQRLateralController,
                     trajectory: Trajectory, speed_cap: float) -> None:
    if isinstance(controller, MPCLateralController):
        controller.set_reference(trajectory.x, trajectory.y, trajectory.yaw,
                                 trajectory.kappa, trajectory.s,
                                 np.minimum(trajectory.v, speed_cap))
    else:
        controller.set_reference(trajectory.x, trajectory.y, trajectory.yaw,
                                 trajectory.kappa, trajectory.s)
