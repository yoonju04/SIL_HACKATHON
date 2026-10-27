#!/usr/bin/env python3
"""08 paired run: turn a planned fault into av_stack/run_benchmark.py arguments.

The fault catalog (what can be injected, and in which ranges) lives in stage 04:
pipeline/04_feasibility_expansion/scenario_space.yaml. Running CARLA belongs here in 08,
so the conversion to run_benchmark.py flags lives here too.

  python fault_args.py --self-test

Input `scenario` is one chosen plan in scenario_space.py's sample format:
  {"fault_id": "F-04", "params": {...}, "timing": {...}, "environment": {...}}
onset_s is the absolute fault start the caller found on its route for
timing.onset_event + offset_s (event detection is not done here).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

SPACE_PATH = Path(__file__).resolve().parents[1] / '04_feasibility_expansion' / 'scenario_space.yaml'
WHOLE_RUN_S = 500.0  # fault window for faults without a duration: stays on until the route ends


def load_space(path: Path = SPACE_PATH) -> Dict[str, Any]:
    with Path(path).open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _find_fault(space: Dict[str, Any], fault_id: str) -> Dict[str, Any]:
    for fault in space['faults']:
        if fault['id'] == fault_id:
            return fault
    raise KeyError('unknown fault id %s' % fault_id)


def to_cli(space: Dict[str, Any], scenario: Dict[str, Any], onset_s: float) -> List[str]:
    """run_benchmark.py arguments for one planned fault."""
    fault = _find_fault(space, scenario['fault_id'])
    if not fault['implemented']:
        raise NotImplementedError('%s (%s) is not in the injector yet' % (fault['id'], fault['name']))
    params = scenario['params']
    # --fault-magnitude carries the one number each mode needs (see av_stack/integration/fault_injection.py)
    if 'delay_ticks' in params:
        magnitude = params['delay_ticks']
    elif 'side' in params:
        magnitude = 1.0 if params['side'] == 'positive' else -1.0
    else:
        magnitude = params.get('magnitude_rad', params.get('rate_rad_s'))
    duration = params.get('duration_s', WHOLE_RUN_S)
    args = ['--mpc-fault', fault['injector_mode'], '--fault-start', '%.2f' % onset_s,
            '--fault-duration', '%.2f' % duration]
    if magnitude is not None:
        args += ['--fault-magnitude', '%.4f' % magnitude]
    if 'on_s' in params:
        args += ['--fault-on', '%.3f' % params['on_s'], '--fault-off', '%.3f' % params['off_s']]
    for name, value in (scenario.get('environment') or {}).items():
        flag = (space.get('environment', {}).get(name) or {}).get('cli')
        if flag:
            args += [flag, '%.1f' % value]
    return args


def _self_test() -> None:
    sys.path.insert(0, str(SPACE_PATH.parent))
    import random
    import scenario_space  # stage 04 helper, used here only to draw test plans

    space = load_space()
    rng = random.Random(0)
    for fault in space['faults']:
        scenario = scenario_space.sample(space, rng, fault_id=fault['id'])
        if fault['implemented']:
            args = to_cli(space, scenario, onset_s=12.0)
            assert args[1] == fault['injector_mode'] and args[3] == '12.00', args
        else:
            try:
                to_cli(space, scenario, onset_s=12.0)
            except NotImplementedError:
                pass
            else:
                raise AssertionError('%s should not produce CLI args' % fault['id'])
    print('self-test passed: to_cli for %d faults' % len(space['faults']))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
