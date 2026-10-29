"""Observe MPC health and latch steering ownership to an independent LQR."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TypedDict

import numpy as np

from integration.behavior_control import update_reference
from integration.episode import LateralResult, build_controller, step_controller
from integration.fault_injection import FaultConfig, MPCFaultInjector
from lqr_controller import LQRConfig
from mpc_controller import MPCOutput, MPCConfig
from planner import Trajectory


@dataclass(frozen=True)
class SupervisorConfig:
    """Thresholds for faults whose output values stay in range (bias, drift, saturate, intermittent bias).

    'tight' (default, integration4) is set from 61 fault-free CARLA baselines (result_2..5, 48k ticks):
    |e_y| max 0.46 m, |e_psi| max 0.23 rad, commanded steer rate max 61 deg/s (the MPC itself limits it to
    60 deg/s, mpc.py steer_rate_lim_dps). 'legacy' (1.0 m / 0.6 rad / 0.25 s, no rate check) reacted only after
    the car had left its lane: the in-lane margin is 0.83-1.08 m, and all 15 switched risk-fault runs crossed
    the lane edge before the switch."""
    max_lateral_error_m: float = 0.6
    max_heading_error_rad: float = 0.35
    divergence_hold_s: float = 0.15
    max_steer_rate_dps: float | None = 90.0   # None = no rate check; 1.5 x the MPC's own 60 deg/s constraint


SUPERVISOR_PROFILES = {'tight': SupervisorConfig(),
                       'legacy': SupervisorConfig(1.0, 0.6, 0.25, None)}


class MPCSupervisor:
    """Accumulate health observations; this monitor has no fault-injector input."""

    def __init__(self, max_steer_rad: float, deadline_ms: float = 50.0,
                 config: SupervisorConfig = SupervisorConfig()) -> None:
        self.max_steer = max_steer_rad
        self.deadline_ms = deadline_ms
        self.config = config
        self.sequence = -1
        self.tracking_bad_since: float | None = None
        self.last_steer: float | None = None
        self.last_t: float | None = None

    def inspect(self, output: MPCOutput, t: float, elapsed_ms: float) -> str | None:
        if not output.ok:
            return 'solver_failure'
        if output.steer is None:
            return 'missing_steer'
        values = (output.steer, output.s, output.kappa, output.stamp_s, *output.state)
        if not all(math.isfinite(value) for value in values):
            return 'nonfinite_output'
        if abs(output.steer) > self.max_steer + 1e-6:
            return 'steer_out_of_range'
        if elapsed_ms > self.deadline_ms:
            return 'deadline_miss'
        if abs(t - output.stamp_s) > 0.1:
            return 'stale_output'
        if output.seq <= self.sequence:
            return 'stale_sequence'
        self.sequence = output.seq
        cfg = self.config
        last_steer, last_t = self.last_steer, self.last_t
        self.last_steer, self.last_t = float(output.steer), t
        if cfg.max_steer_rate_dps is not None and last_steer is not None and t > last_t:
            # a command jump the MPC's own rate constraint cannot produce (saturate step, intermittent bias edge)
            if abs(output.steer - last_steer) / (t - last_t) > math.radians(cfg.max_steer_rate_dps) + 1e-9:
                return 'steer_rate_exceeded'
        bad_tracking = (abs(float(output.state[0])) > cfg.max_lateral_error_m
                        or abs(float(output.state[2])) > cfg.max_heading_error_rad)
        if not bad_tracking:
            self.tracking_bad_since = None
        elif self.tracking_bad_since is None:
            self.tracking_bad_since = t
        if self.tracking_bad_since is not None and t - self.tracking_bad_since >= cfg.divergence_hold_s - 1e-9:
            return 'tracking_divergence'
        return None


class SupervisionRecord(TypedDict):
    active_controller: str
    supervisor_reason: str | None
    switched_at_s: float | None
    switch_count: int
    primary_status: str
    primary_elapsed_ms: float | None
    fault_injected: bool
    safe_stop: bool


class SupervisedLateral:
    """Own the switch latch and last applied steering, keeping both references current."""

    def __init__(self, geometry: dict[str, float], dt: float, trajectory: Trajectory,
                 fault: FaultConfig = FaultConfig(), mpc_config: MPCConfig | None = None,
                 lqr_config: LQRConfig | None = None, supervisor: SupervisorConfig = SupervisorConfig()) -> None:
        self.mpc = build_controller('mpc', geometry, dt, trajectory, mpc_cfg=mpc_config)
        self.lqr = build_controller('lqr', geometry, dt, trajectory, lqr_cfg=lqr_config)
        self.monitor = MPCSupervisor(geometry['max_steer_rad'], deadline_ms=dt * 1000, config=supervisor)
        self.injector = MPCFaultInjector(fault, geometry['max_steer_rad'])
        self.dt = dt
        self.active = 'mpc'
        self.reason: str | None = None
        self.switched_at: float | None = None
        self.switch_count = 0
        self.stop_required = False
        self.rate_dps = 60.0  # LQR steering-rate limit after takeover [deg/s, road wheel]
        self.primary_status = 'not_run'
        self.elapsed_ms: float | None = None
        self.last = LateralResult(0.0, 0.0, 0.0, 0.0, 0.0, True, 'initial', None, None)
        self.t = 0.0
        self.fault_effective = False

    def set_reference(self, trajectory: Trajectory, speed_cap: float,
                      lqr_trajectory: Trajectory | None = None) -> None:
        update_reference(self.mpc, trajectory, speed_cap)
        update_reference(self.lqr, trajectory if lqr_trajectory is None else lqr_trajectory, speed_cap)

    def step(self, t: float, x: float, y: float, yaw: float,
             vx: float, vy: float, yaw_rate: float) -> LateralResult:
        self.t = t
        result = None
        self.fault_effective = False  # only an MPC call in this tick can carry the fault
        if self.active == 'mpc':
            started = time.perf_counter()
            try:
                output = self.mpc.compute(x, y, yaw, vx, vy, yaw_rate, t)
                output = self.injector.apply(output, t)
                self.fault_effective = self.injector.effective
                self.elapsed_ms = (time.perf_counter() - started) * 1000
                self.primary_status = output.status
                self.reason = self.monitor.inspect(output, t, self.elapsed_ms)
                if self.reason is None:
                    result = LateralResult(output.steer, output.s, float(output.state[0]),
                                           float(output.state[2]), output.kappa, output.ok,
                                           output.status, output.solve_time_s * 1000, output.qp_iterations)
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                self.elapsed_ms = (time.perf_counter() - started) * 1000
                self.reason = 'controller_exception'
                self.primary_status = type(error).__name__
            if self.reason is not None:
                self.active = 'lqr'
                self.switched_at = t
                self.switch_count += 1
                print('[supervisor] t=%.2fs MPC -> LQR reason=%s' % (t, self.reason), flush=True)
        if self.active == 'lqr' and not self.stop_required:
            try:
                result = step_controller('lqr', self.lqr, t, x, y, yaw, vx, vy, yaw_rate)
                values = (result.steer, result.s, result.e_y, result.e_psi, result.kappa)
                self.stop_required = not all(value is not None and math.isfinite(value) for value in values)
                self.stop_required = bool(self.stop_required or abs(result.steer or 0.0) > self.monitor.max_steer + 1e-6)
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                self.stop_required = True
                self.primary_status = 'lqr_exception:' + type(error).__name__
            if self.stop_required:
                self.active = 'safe_stop'
                print('[supervisor] t=%.2fs LQR invalid -> safe stop' % t, flush=True)
            else:
                limit = math.radians(self.rate_dps) * self.dt
                steer = float(np.clip(result.steer, self.last.steer - limit, self.last.steer + limit))
                result = replace(result, steer=steer)
        if self.stop_required:
            result = replace(self.last, ok=False, status='safe_stop')
        assert result is not None
        self.last = result
        return result

    def record(self) -> SupervisionRecord:
        return SupervisionRecord(active_controller=self.active, supervisor_reason=self.reason,
                                 switched_at_s=self.switched_at, switch_count=self.switch_count,
                                 primary_status=self.primary_status, primary_elapsed_ms=self.elapsed_ms,
                                 fault_injected=self.injector.config.active(self.t), safe_stop=self.stop_required)
