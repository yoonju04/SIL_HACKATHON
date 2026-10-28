"""Inject an MPC fault into the RUNNING drive, from a second terminal (takes effect next tick).

  python tools\\inject.py bias 0.05      steering offset [rad]       (integrity attack)
  python tools\\inject.py drift 0.03     growing offset [rad/s]      (integrity attack)
  python tools\\inject.py intermittent_bias 0.08 0.2 0.1   offset [rad], on [s], off [s]
  python tools\\inject.py saturate 1      steering held at the limit, side = sign
  python tools\\inject.py output_delay 2  command computed 2 ticks ago, fresh stamp
  python tools\\inject.py solver_fail    / stale                  (availability faults)
  python tools\\inject.py clear          stop injecting (an LQR takeover stays latched)

Needs --controller supervised (or mpc_unprotected for the no-supervisor comparison).
Standard library only, so any Python works.
"""
import json
from pathlib import Path
import sys
import time

MODES = ('solver_fail', 'stale', 'bias', 'drift', 'intermittent_bias', 'saturate', 'output_delay', 'clear')
NO_MAGNITUDE = ('solver_fail', 'stale', 'clear')
COMMAND_FILE = Path(__file__).resolve().parents[1] / 'runs' / 'live_fault.json'

if len(sys.argv) < 2 or sys.argv[1] not in MODES:
    sys.exit(__doc__)
mode = sys.argv[1]
magnitude = float(sys.argv[2]) if len(sys.argv) > 2 else .05
on_s = float(sys.argv[3]) if len(sys.argv) > 3 else .2
off_s = float(sys.argv[4]) if len(sys.argv) > 4 else .2
COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
temporary = COMMAND_FILE.with_suffix('.tmp')
temporary.write_text(json.dumps(dict(mode=mode, magnitude=magnitude, on_s=on_s, off_s=off_s, issued=time.time())),
                     encoding='utf-8')
temporary.replace(COMMAND_FILE)  # atomic: the agent never reads a half-written file
print(f'sent: {mode}' + ('' if mode in NO_MAGNITUDE else f' {magnitude}'))
