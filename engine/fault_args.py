#!/usr/bin/env python3
"""08 paired run: turn a planned fault into av_stack/run_benchmark.py arguments (teammate's to_cli, sim-eval 13c2dbf;
moved from pipeline/08_carla_paired_run/fault_args.py — functional code lives in engine/).

The av_stack fault catalog (what can be injected, and in which ranges) lives in stage 04:
pipeline/04_feasibility_expansion/scenario_space_av_stack.yaml (the teammate's scenario_space.yaml; our
scenario_space.yaml keeps the team template). engine.carla_runner calls to_cli; engine.scenarios.fault_cli_local
is the same mapping kept as fallback.

  python -m engine.fault_args --self-test

Input `scenario` is one chosen plan in engine.fault_catalog.sample's format:
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

SPACE_PATH = Path(__file__).resolve().parents[1] / 'pipeline' / '04_feasibility_expansion' / 'scenario_space_av_stack.yaml'
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
    args += param_flags(fault, params)
    args += environment_flags(space, scenario.get('environment') or {})
    return args


def param_flags(fault: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    """Extra per-parameter flags declared with `cli:` in the catalog (integration3: saturate --saturate-level /
    --fault-ramp). A parameter that is absent from the plan is not passed, so run_benchmark.py keeps its default
    (level 1.0, ramp 0 = the original saturate)."""
    args: List[str] = []
    for name, spec in (fault.get('params') or {}).items():
        flag = (spec or {}).get('cli')
        if flag and params.get(name) is not None:
            args += [flag, '%.4f' % float(params[name])]
    return args


def environment_flags(space: Dict[str, Any], environment: Dict[str, Any]) -> List[str]:
    """Environment axes with a `cli:` flag (max_speed_kmh -> --max-speed '%.1f'; tire_friction_scale ->
    --tire-friction-scale '%.2f', integration3). Format from the axis' `cli_format`, default '%.1f'."""
    args: List[str] = []
    for name, value in environment.items():
        spec = space.get('environment', {}).get(name) or {}
        flag = spec.get('cli')
        if flag and value is not None:
            args += [flag, spec.get('cli_format', '%.1f') % float(value)]
    return args


def _self_test() -> None:
    import random

    try:
        from engine import fault_catalog as scenario_space  # stage 04 helper, used here only to draw test plans
    except ImportError:  # run as a plain script
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fault_catalog as scenario_space

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
