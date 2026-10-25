"""Opt-in simulator faults at the MPC output boundary; never changes LQR.

Availability faults (solver_fail, nan, stale, exception, delay) make the MPC output visibly
unusable. Integrity attacks (bias, drift) keep a healthy-looking output but shift its steering:
bias adds `magnitude` rad, drift adds `magnitude` rad per second since the fault start.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from mpc_controller import MPCOutput


AVAILABILITY_FAULTS = ('solver_fail', 'nan', 'stale', 'exception', 'delay')
INTEGRITY_FAULTS = ('bias', 'drift')
FAULT_MODES = ('none',) + AVAILABILITY_FAULTS + INTEGRITY_FAULTS


@dataclass(frozen=True)
class FaultConfig:
    mode: str = 'none'
    at_s: float = 25.0
    duration_s: float = 2.0
    magnitude: float = 0.05  # bias [rad] or drift rate [rad/s]; integrity attacks only

    def __post_init__(self) -> None:
        if self.mode not in FAULT_MODES:
            raise ValueError('Unknown MPC fault mode.')
        if not math.isfinite(self.magnitude):
            raise ValueError('Fault magnitude must be finite.')
        if not math.isfinite(self.at_s) or self.at_s < 0:
            raise ValueError('Fault start must be finite and nonnegative.')
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError('Fault duration must be finite and positive.')

    def active(self, t: float) -> bool:
        return self.mode != 'none' and self.at_s <= t < self.at_s + self.duration_s

    def config_active_since(self, t: float, seconds: float) -> bool:
        """True once a configured fault started at least `seconds` ago (demo stop option)."""
        return self.mode != 'none' and t >= self.at_s + seconds


class InjectedMPCFailure(RuntimeError):
    """Temporary controller failure used only by the explicit simulator injector."""


def _raise_failure(output: MPCOutput) -> MPCOutput:
    raise InjectedMPCFailure('Temporary MPC computation failure')


def _delay(output: MPCOutput) -> MPCOutput:
    time.sleep(0.075)
    return output


class MPCFaultInjector:
    """Retain the last clean sample to reproduce a stale-output fault."""

    def __init__(self, config: FaultConfig) -> None:
        self.config = config
        self.previous: MPCOutput | None = None

    @staticmethod
    def _shift(output: MPCOutput, offset: float) -> MPCOutput:
        if output.steer is None:
            return output
        return replace(output, steer=output.steer + offset)

    def apply(self, output: MPCOutput, t: float) -> MPCOutput:
        if not self.config.active(t):
            self.previous = output
            return output
        faults = {
            'solver_fail': lambda value: replace(value, ok=False, steer=None, status='injected_solver_failure'),
            'nan': lambda value: replace(value, steer=math.nan),
            'stale': lambda value: self.previous if self.previous is not None else replace(value, stamp_s=t-1),
            'exception': _raise_failure,
            'delay': _delay,
            'bias': lambda value: self._shift(value, self.config.magnitude),
            'drift': lambda value: self._shift(value, self.config.magnitude * (t - self.config.at_s)),
        }
        return faults[self.config.mode](output)
