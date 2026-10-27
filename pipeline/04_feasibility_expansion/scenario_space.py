#!/usr/bin/env python3
"""Load, check and sample scenario_space_av_stack.yaml, and turn a scenario into run_benchmark.py arguments.

  python scenario_space.py --check
  python scenario_space.py --sample 5 --seed 0 [--fault F-03] [--implemented-only]
  python scenario_space.py --self-test

Needs PyYAML (present in av_stack's .venv-tfpp).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_PATH = Path(__file__).with_name('scenario_space_av_stack.yaml')
GROUPS = ('control', 'risk')
WHOLE_RUN_S = 500.0  # fault window for faults without a duration: stays on until the route ends


def load(path: Path = DEFAULT_PATH) -> Dict[str, Any]:
    with Path(path).open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _check_param(name: str, spec: Dict[str, Any]) -> List[str]:
    if 'range' in spec:
        low, high = spec['range']
        return [] if low < high else ['%s: range low >= high' % name]
    if 'ranges' in spec:
        return [] if all(low < high for low, high in spec['ranges']) else ['%s: a range has low >= high' % name]
    if 'values' in spec:
        return [] if spec['values'] else ['%s: empty values' % name]
    return ['%s: needs range, ranges or values' % name]


def check(space: Dict[str, Any]) -> List[str]:
    """Return problems in the space; an empty list means it is usable."""
    problems: List[str] = []
    ids = [item.get('id') for item in space.get('faults', []) + space.get('extensions', [])]
    if len(ids) != len(set(ids)):
        problems.append('duplicate fault ids')
    sources = space.get('meta', {}).get('observation_sources', {})
    for fault in space.get('faults', []):
        fid = fault.get('id', '?')
        if fault.get('group') not in GROUPS:
            problems.append('%s: group must be one of %s' % (fid, GROUPS))
        if not fault.get('next_question'):
            problems.append('%s: next_question is empty' % fid)
        for observation in fault.get('observations', []):
            if observation.get('source') not in sources:
                problems.append('%s: observation source %r not in meta.observation_sources' % (fid, observation.get('source')))
            if not isinstance(observation.get('runs'), int) or observation['runs'] < 1:
                problems.append('%s: observation runs must be a positive integer' % fid)
        for name, spec in fault.get('params', {}).items():
            problems += _check_param('%s.%s' % (fid, name), spec)
    for name, spec in space.get('environment', {}).items():
        problems += _check_param('environment.%s' % name, spec)
    for name, spec in space.get('fault_timing', {}).items():
        problems += _check_param('fault_timing.%s' % name, spec)
    return problems


def _draw(spec: Dict[str, Any], rng: random.Random) -> Any:
    if 'values' in spec:
        return rng.choice(spec['values'])
    if 'range' in spec:
        return rng.uniform(*spec['range'])
    # several intervals: pick one in proportion to its width, then a point inside it
    widths = [high - low for low, high in spec['ranges']]
    low, high = rng.choices(spec['ranges'], weights=widths)[0]
    return rng.uniform(low, high)


def find_fault(space: Dict[str, Any], fault_id: str) -> Dict[str, Any]:
    for fault in space['faults']:
        if fault['id'] == fault_id:
            return fault
    raise KeyError('unknown fault id %s' % fault_id)


def sample(space: Dict[str, Any], rng: random.Random, fault_id: Optional[str] = None,
           implemented_only: bool = False) -> Dict[str, Any]:
    """Draw one scenario uniformly from the space (one fault, its params, timing, environment)."""
    if fault_id is not None:
        fault = find_fault(space, fault_id)
    else:
        pool = [f for f in space['faults'] if f['implemented'] or not implemented_only]
        fault = rng.choice(pool)
    return {
        'fault_id': fault['id'],
        'fault': fault['name'],
        'params': {name: _draw(spec, rng) for name, spec in fault.get('params', {}).items()},
        'timing': {name: _draw(spec, rng) for name, spec in space['fault_timing'].items()},
        'environment': {name: _draw(spec, rng) for name, spec in space['environment'].items()},
    }


def to_cli(space: Dict[str, Any], scenario: Dict[str, Any], onset_s: float) -> List[str]:
    """run_benchmark.py arguments for a scenario. onset_s is the absolute fault start the caller
    found for timing.onset_event + offset_s on its route (event detection is not done here)."""
    fault = find_fault(space, scenario['fault_id'])
    if not fault['implemented']:
        raise NotImplementedError('%s (%s) is not in the injector yet' % (fault['id'], fault['name']))
    params = scenario['params']
    # --fault-magnitude carries the one number each mode needs (see integration/fault_injection.py)
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
    for name, value in scenario['environment'].items():
        flag = space['environment'][name].get('cli')
        if flag:
            args += [flag, '%.1f' % value]
    return args


def _self_test() -> None:
    space = load()
    assert check(space) == [], check(space)
    rng = random.Random(0)
    for fault in space['faults']:
        scenario = sample(space, rng, fault_id=fault['id'])
        for name, spec in fault.get('params', {}).items():
            value = scenario['params'][name]
            if 'range' in spec:
                assert spec['range'][0] <= value <= spec['range'][1], (fault['id'], name, value)
            elif 'ranges' in spec:
                assert any(low <= value <= high for low, high in spec['ranges']), (fault['id'], name, value)
        if fault['implemented']:
            args = to_cli(space, scenario, onset_s=12.0)
            assert args[1] == fault['injector_mode'] and '--fault-start' in args
        else:
            try:
                to_cli(space, scenario, onset_s=12.0)
            except NotImplementedError:
                pass
            else:
                raise AssertionError('%s should not produce CLI args' % fault['id'])
    pool = {sample(space, rng, implemented_only=True)['fault_id'] for _ in range(200)}
    assert pool == {f['id'] for f in space['faults'] if f['implemented']}, pool
    print('self-test passed: %d faults, check, sample, to_cli' % len(space['faults']))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--space', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--check', action='store_true', help='validate the YAML and list the faults')
    parser.add_argument('--sample', type=int, metavar='N', help='print N sampled scenarios as JSON lines')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fault', help='sample only this fault id, e.g. F-03')
    parser.add_argument('--implemented-only', action='store_true', help='skip faults the injector does not have yet')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    space = load(args.space)
    problems = check(space)
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        return 1
    if args.check or not args.sample:
        for fault in space['faults']:
            print('%s %-18s %-7s %-16s observations=%d' % (
                fault['id'], fault['name'], fault['group'],
                'ready' if fault['implemented'] else 'not in injector', len(fault.get('observations', []))))
        for extension in space.get('extensions', []):
            print('%s %-18s (extension, not sampled)' % (extension['id'], extension['name']))
    if args.sample:
        rng = random.Random(args.seed)
        for _ in range(args.sample):
            print(json.dumps(sample(space, rng, args.fault, args.implemented_only), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
