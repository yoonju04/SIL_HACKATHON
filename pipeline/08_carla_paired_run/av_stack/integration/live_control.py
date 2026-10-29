"""Live demo helpers: manual fault injection from a second terminal, supervisor console, comfort pedals.

Manual injection: `tools/inject.py bias 0.05` writes LIVE_FAULT_FILE; the running agent polls it every
tick and starts that fault at the current simulation time (commands issued before the run are ignored).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

from integration.fault_injection import FAULT_MODES, FaultConfig

LIVE_FAULT_FILE = Path(__file__).resolve().parents[1] / 'runs' / 'live_fault.json'
OPEN_ENDED_S = 1e6


class LiveFaultCommands:
    """Poll the command file; return a new FaultConfig when a fresh command arrives."""

    def __init__(self, path: Path = LIVE_FAULT_FILE) -> None:
        self.path = path
        self.started = time.time()
        self.last_issued = None

    def poll(self, now_s: float) -> FaultConfig | None:
        try:
            command = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        issued = command.get('issued')
        if issued is None or issued < self.started or issued == self.last_issued:
            return None
        self.last_issued = issued
        mode = 'none' if command.get('mode') == 'clear' else command.get('mode')
        if mode not in FAULT_MODES:
            print(f'[inject] ignored unknown mode {mode!r}', flush=True)
            return None
        return FaultConfig(mode, now_s, OPEN_ENDED_S, float(command.get('magnitude', .05)),
                           float(command.get('on_s', .2)), float(command.get('off_s', .2)))


def describe(fault: FaultConfig) -> str:
    if fault.mode == 'bias':
        return f'bias {fault.magnitude:+.3f} rad ({math.degrees(fault.magnitude):+.1f} deg) on MPC steering'
    if fault.mode == 'drift':
        return f'drift {fault.magnitude:+.3f} rad/s ({math.degrees(fault.magnitude):+.1f} deg/s) on MPC steering'
    if fault.mode == 'intermittent_bias':
        return (f'intermittent bias {fault.magnitude:+.3f} rad, on {fault.on_s:.2f} s / off {fault.off_s:.2f} s'
                ' on MPC steering')
    if fault.mode == 'saturate':
        return 'MPC steering held at the %s actuator limit' % ('positive' if fault.magnitude > 0 else 'negative')
    if fault.mode == 'output_delay':
        return f'MPC command delayed by {round(fault.magnitude)} ticks (fresh stamp)'
    return {'solver_fail': 'MPC solver failure', 'stale': 'MPC repeats a stale sample',
            'none': 'fault cleared'}[fault.mode]


REASONS = {
    'solver_failure': 'MPC solver reported failure',
    'missing_steer': 'MPC returned no steering command',
    'nonfinite_output': 'MPC output is NaN / infinite',
    'steer_out_of_range': 'MPC steering beyond the actuator limit',
    'deadline_miss': 'MPC took longer than the 50 ms control period',
    'stale_output': 'MPC output timestamp is old',
    'stale_sequence': 'MPC output was not refreshed (repeated sample)',
    'controller_exception': 'MPC raised an exception',
    'tracking_divergence': 'vehicle diverging from the path (tight: |e_y| > 0.6 m or |e_psi| > 0.35 rad for 0.15 s; '
                           'legacy: 1.0 m / 0.6 rad / 0.25 s)',
    'steer_rate_exceeded': 'MPC steering command changing faster than 90 deg/s (the MPC itself limits it to 60)',
}


class SupervisorConsole:
    """Terminal output: a line when a fault is injected, a banner when the supervisor takes over.

    period_s > 0 additionally prints one status line per period (off by default).
    """

    def __init__(self, period_s: float = 0.) -> None:
        self.period = period_s
        self.next_line = 0.
        self.previous_active = None
        self.injected_at = None

    def injected(self, t: float, fault: FaultConfig) -> None:
        self.injected_at = None if fault.mode == 'none' else t
        print(f'\n>>> [INJECT] t={t:6.2f}s  {describe(fault)}\n', flush=True)

    def update(self, t: float, active: str, record: dict, e_y: float | None, e_psi: float | None,
               speed: float) -> None:
        if self.previous_active == 'mpc' and active != 'mpc':
            reason = record.get('supervisor_reason')
            delay = ('no fault injected' if self.injected_at is None
                     else f'{t - self.injected_at:.2f} s after the fault was injected')
            line = '=' * 76
            print(f'\n{line}\n  [SUPERVISOR] fault detected -> steering switched MPC -> {active.upper()}   (t={t:.2f} s)\n'
                  f'    detected : {REASONS.get(reason, reason)}\n'
                  f'    when     : {delay}\n'
                  f'    state    : e_y={0. if e_y is None else e_y:+.2f} m, e_psi={math.degrees(e_psi or 0.):+.1f} deg, '
                  f'v={speed * 3.6:.1f} km/h\n{line}\n', flush=True)
        elif self.previous_active == 'lqr' and active == 'safe_stop':
            print(f'\n[SUPERVISOR] t={t:.2f} s  LQR output invalid -> SAFE STOP (full brake)\n', flush=True)
        self.previous_active = active
        if self.period <= 0 or t + 1e-9 < self.next_line:
            return
        self.next_line = t + self.period
        solve = record.get('primary_elapsed_ms')
        attack = 'ATTACK ON' if record.get('fault_injected') else 'no attack'
        print(f'[MONITOR t={t:6.1f}s] steering={active.upper():<9} v={speed * 3.6:5.1f} km/h  '
              f'e_y={0. if e_y is None else e_y:+.2f} m  e_psi={math.degrees(e_psi or 0.):+5.1f} deg  '
              f'MPC {"-" if solve is None else f"{solve:4.1f} ms"}  {attack}', flush=True)


@dataclass
class ComfortPedals:
    """Smooth PDM-Lite's bang-bang pedals; PDM-Lite still decides *when* to slow down or stop.

    - target speed capped at max_speed
    - hazard / stop request: brake grows with speed (0.35 at rest .. 1.0 at 6.5 m/s)
    - small overspeed: proportional brake instead of PDM-Lite's full brake (ratio > 1.03)
    - throttle capped and rate-limited (no 0 -> 1 jumps)
    """
    max_speed: float = 30 / 3.6
    throttle_max: float = .75
    throttle_rise_per_s: float = 2.0
    dt: float = .05
    throttle: float = 0.

    def __call__(self, hazard: bool, target: float, speed: float, pdm_throttle: float) -> tuple[float, float]:
        target = min(target, self.max_speed)
        if hazard or target < .05:
            self.throttle = 0.
            return 0., float(min(1., .35 + .1 * speed))
        if speed > target * 1.03:
            self.throttle = 0.
            return 0., float(min(.6, .25 * (speed - target)))
        wanted = min(pdm_throttle, self.throttle_max)
        self.throttle = min(wanted, self.throttle + self.throttle_rise_per_s * self.dt)
        return float(self.throttle), 0.


class CapturingLongitudinal:
    """Wrap PDM-Lite's longitudinal controller: cap the target speed and remember its inputs."""

    def __init__(self, inner, max_speed: float) -> None:
        self.inner = inner
        self.max_speed = max_speed
        self.last = (False, 0., 0.)

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        target_speed = min(target_speed, self.max_speed)
        self.last = (bool(hazard_brake), float(target_speed), float(current_speed))
        return self.inner.get_throttle_and_brake(hazard_brake, target_speed, current_speed)

    def __getattr__(self, name):
        return getattr(self.inner, name)
