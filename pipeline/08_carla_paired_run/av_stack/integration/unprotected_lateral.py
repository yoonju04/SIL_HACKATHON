"""Simulator baseline: unusable MPC commands become neutral steering, without LQR."""
from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from integration.behavior_control import update_reference
from integration.episode import LateralResult, build_controller
from integration.fault_injection import FaultConfig, MPCFaultInjector
from planner import Trajectory
from mpc_controller import MPCConfig


class UnprotectedRecord(TypedDict):
    supervisor_enabled: bool
    lqr_fallback_enabled: bool
    lateral_enabled: bool
    fault_injected: bool
    primary_status: str


class UnprotectedMPC:
    def __init__(self, geometry: dict[str, float], dt: float, trajectory: Trajectory,
                 fault: FaultConfig = FaultConfig(), mpc_config: MPCConfig | None = None) -> None:
        self.mpc = build_controller('mpc', geometry, dt, trajectory, mpc_cfg=mpc_config)
        self.injector = MPCFaultInjector(fault, geometry['max_steer_rad'])
        self.active = 'mpc'
        self.primary_status = 'not_run'
        self.t = 0.0
        self.fault_effective = False

    def set_reference(self, trajectory: Trajectory, speed_cap: float,
                      lqr_trajectory: Trajectory | None = None) -> None:
        update_reference(self.mpc, trajectory, speed_cap)  # no LQR in this baseline

    def step(self, t: float, x: float, y: float, yaw: float,
             vx: float, vy: float, yaw_rate: float) -> LateralResult:
        self.t = t
        previous = self.active
        try:
            self.fault_effective = False
            output = self.injector.apply(self.mpc.compute(x, y, yaw, vx, vy, yaw_rate, t), t)
            self.fault_effective = self.injector.effective
            usable = bool(output.ok and output.steer is not None and math.isfinite(output.steer))
            self.active = 'mpc' if usable else 'none'
            self.primary_status = output.status
            result = LateralResult(float(output.steer) if usable else 0.0, output.s,
                                   float(output.state[0]), float(output.state[2]), output.kappa,
                                   usable, output.status if usable else 'unprotected_neutral:' + output.status,
                                   output.solve_time_s * 1000, output.qp_iterations)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            self.active = 'none'
            self.primary_status = type(error).__name__
            state, station, curvature = self.mpc.ref.error_state(x, y, yaw, vx, vy, yaw_rate)
            result = LateralResult(0.0, station, float(state[0]), float(state[2]), curvature,
                                   False, 'unprotected_neutral:' + self.primary_status, None, None)
        if self.active != previous:
            print('[unprotected] t=%.2fs steering=%s; supervisor=OFF LQR=OFF' % (t, self.active), flush=True)
        return result

    def record(self) -> UnprotectedRecord:
        return UnprotectedRecord(supervisor_enabled=False, lqr_fallback_enabled=False,
                                 lateral_enabled=self.active == 'mpc',
                                 fault_injected=bool(self.injector.config.active(self.t)),
                                 primary_status=self.primary_status)
