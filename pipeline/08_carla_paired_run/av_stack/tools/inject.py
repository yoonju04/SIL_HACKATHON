"""Inject an MPC fault into the RUNNING drive, from a second terminal (takes effect next tick).

  python tools\\inject.py bias 0.05      steering offset [rad]       (integrity attack)
  python tools\\inject.py drift 0.03     growing offset [rad/s]      (integrity attack)
  python tools\\inject.py solver_fail    / nan / stale / exception / delay   (availability faults)
  python tools\\inject.py clear          stop injecting (an LQR takeover stays latched)

Needs --controller supervised (or mpc_unprotected for the no-supervisor comparison).
Standard library only, so any Python works.
"""
import json
from pathlib import Path
import sys
import time

MODES = ('solver_fail', 'nan', 'stale', 'exception', 'delay', 'bias', 'drift', 'clear')
COMMAND_FILE = Path(__file__).resolve().parents[1] / 'runs' / 'live_fault.json'

if len(sys.argv) < 2 or sys.argv[1] not in MODES:
    sys.exit(__doc__)
mode = sys.argv[1]
magnitude = float(sys.argv[2]) if len(sys.argv) > 2 else .05
COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
temporary = COMMAND_FILE.with_suffix('.tmp')
temporary.write_text(json.dumps(dict(mode=mode, magnitude=magnitude, issued=time.time())), encoding='utf-8')
temporary.replace(COMMAND_FILE)  # atomic: the agent never reads a half-written file
print(f'sent: {mode}' + (f' {magnitude}' if mode in ('bias', 'drift') else ''))
