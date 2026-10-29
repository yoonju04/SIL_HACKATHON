"""Opt-in simulator faults at the MPC output boundary; never changes LQR.

Availability faults (solver_fail, stale) make the MPC output visibly unusable. Integrity attacks keep a healthy-looking output but change its steering:
  bias               adds `magnitude` rad
  drift              adds `magnitude` rad per second since the fault start
  intermittent_bias  adds `magnitude` rad for `on_s`, then nothing for `off_s`, repeating
  saturate           holds steering at `level` x the actuator limit (level 1.0 = exactly the limit), side = sign of
                     `magnitude`; with ramp_s > 0 the command blends linearly from the MPC's own command to that
                     target over ramp_s seconds (integration3, DriveFI-style continuous magnitude/timing).
                     level 1.0 and ramp_s 0 reproduce the original step to the limit.
  output_delay       sends the command computed `magnitude` ticks ago under a fresh stamp and sequence
                     (processing latency; an old stamp would be caught by the sequence check at onset)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

from mpc_controller import MPCOutput


AVAILABILITY_FAULTS = ('solver_fail', 'stale')
INTEGRITY_FAULTS = ('bias', 'drift', 'intermittent_bias', 'saturate', 'output_delay')
FAULT_MODES = ('none',) + AVAILABILITY_FAULTS + INTEGRITY_FAULTS
MAX_DELAY_TICKS = 15


@dataclass(frozen=True)
class FaultConfig:
    mode: str = 'none'
    at_s: float = 25.0
    duration_s: float = 2.0
    magnitude: float = 0.05  # bias [rad], drift rate [rad/s], saturate side (sign) or output_delay ticks
    on_s: float = 0.2  # intermittent_bias: bias on for on_s, then off for off_s
    off_s: float = 0.2
    level: float = 1.0  # saturate: fraction of the actuator limit, (0, 1]
    ramp_s: float = 0.0  # saturate: seconds to blend from the MPC command to the saturated target (0 = step)

    def __post_init__(self) -> None:
        if self.mode not in FAULT_MODES:
            raise ValueError('Unknown MPC fault mode.')
        if not math.isfinite(self.magnitude):
            raise ValueError('Fault magnitude must be finite.')
        if not math.isfinite(self.at_s) or self.at_s < 0:
            raise ValueError('Fault start must be finite and nonnegative.')
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError('Fault duration must be finite and positive.')
        if not (math.isfinite(self.on_s) and math.isfinite(self.off_s)) or self.on_s <= 0 or self.off_s <= 0:
            raise ValueError('Intermittent on/off times must be finite and positive.')
        if not math.isfinite(self.level) or not 0 < self.level <= 1:
            raise ValueError('Saturate level must be in (0, 1] (fraction of the steering limit).')
        if not math.isfinite(self.ramp_s) or self.ramp_s < 0:
            raise ValueError('Saturate ramp must be finite and nonnegative.')
        if self.mode == 'saturate' and self.magnitude == 0:
            raise ValueError('Saturate needs a nonzero magnitude; its sign picks the side.')
        if self.mode == 'output_delay' and not 1 <= round(self.magnitude) <= MAX_DELAY_TICKS:
            raise ValueError('Output delay must be 1..%d ticks.' % MAX_DELAY_TICKS)

    def active(self, t: float) -> bool:
        return self.mode != 'none' and self.at_s <= t < self.at_s + self.duration_s

    def config_active_since(self, t: float, seconds: float) -> bool:
        """True once a configured fault started at least `seconds` ago (demo stop option)."""
        return self.mode != 'none' and t >= self.at_s + seconds


class MPCFaultInjector:
    """Retain the last clean sample (stale) and recent MPC answers (output_delay)."""

    def __init__(self, config: FaultConfig, max_steer: float | None = None) -> None:
        self.config = config
        self.max_steer = max_steer
        self.previous: MPCOutput | None = None
        self.history: deque[MPCOutput] = deque(maxlen=MAX_DELAY_TICKS + 1)
        self.effective = False  # did the last apply() change what the MPC handed on (fault_effective)

    @staticmethod
    def _shift(output: MPCOutput, offset: float) -> MPCOutput:
        if output.steer is None:
            return output
        return replace(output, steer=output.steer + offset)

    def _intermittent(self, output: MPCOutput, t: float) -> MPCOutput:
        # rounded to microseconds so tick times like 1.2 s land in the intended on/off phase
        period = round(self.config.on_s + self.config.off_s, 6)
        phase = round(round(t - self.config.at_s, 6) % period, 6)
        return self._shift(output, self.config.magnitude) if phase < self.config.on_s - 1e-9 else output

    def _saturate(self, output: MPCOutput, t: float) -> MPCOutput:
        if self.max_steer is None:
            raise ValueError('Saturate fault needs the actuator limit (max_steer).')
        target = math.copysign(self.config.level * self.max_steer, self.config.magnitude)
        if self.config.ramp_s > 0 and output.steer is not None:
            frac = min(1.0, max(0.0, round(t - self.config.at_s, 6) / self.config.ramp_s))
            target = output.steer + frac * (target - output.steer)
        return replace(output, steer=target)

    def _delayed(self, output: MPCOutput) -> MPCOutput:
        old = self.history[max(0, len(self.history) - 1 - round(self.config.magnitude))]
        return replace(output, steer=old.steer, raw_steer=old.raw_steer, state=old.state, s=old.s,
                       kappa=old.kappa, predicted_e_y=old.predicted_e_y, u_sequence=old.u_sequence)

    def apply(self, output: MPCOutput, t: float) -> MPCOutput:
        self.history.append(output)
        self.effective = False
        if not self.config.active(t):
            self.previous = output
            return output
        faults = {
            'solver_fail': lambda value: replace(value, ok=False, steer=None, status='injected_solver_failure'),
            'stale': lambda value: self.previous if self.previous is not None else replace(value, stamp_s=t-1),
            'bias': lambda value: self._shift(value, self.config.magnitude),
            'drift': lambda value: self._shift(value, self.config.magnitude * (t - self.config.at_s)),
            'intermittent_bias': lambda value: self._intermittent(value, t),
            'saturate': lambda value: self._saturate(value, t),
            'output_delay': self._delayed,
        }
        result = faults[self.config.mode](output)
        self.effective = result is not output and (result.ok != output.ok or result.steer != output.steer
                                                   or result.seq != output.seq)
        return result
