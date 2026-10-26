"""Add verified-domain-prior context to Nemotron without trusting LLM safety judgments."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected mapping: {path}')
    return data


def make_nemotron_payload(profile: dict, hypotheses: dict, prior: dict, scenario_space: dict) -> dict:
    """Create provider-agnostic chat payload (no API requests and no invented values)."""
    observations = profile.get('code_evidence', {}).get('observations', [])
    verified = [o for o in observations if o.get('verified') is True]
    return {
        'messages': [
            {'role': 'system', 'content': (
                'Generate MPC-to-LQR fault scenario hypotheses. Cite existing PK-/O-/H- identifiers. '
                'Never invent code parameters; distinguish physically justified statements from '
                'testable hypotheses. Return structured JSON; do not declare any scenario safe. '
                'Physics checks and final safety decisions belong to deterministic software.')},
            {'role': 'user', 'content': json.dumps({
                'verified_code_observations': verified,
                'domain_prior_knowledge': prior,
                'hypotheses': hypotheses.get('hypotheses', []),
                'scenario_space': scenario_space,
                'task': 'Propose candidate scenarios and cite grounding IDs with limitations.'
            }, ensure_ascii=False)}
        ],
        'mode': 'offline_payload_only',
        'unverified_observations_omitted': len(observations) - len(verified)
    }


def screen_lateral_demand(candidate: dict[str, Any], max_lateral_acc_mps2: float | None = None) -> dict:
    """Simple steady-state stress tag; not a physical feasibility proof."""
    speed = candidate.get('speed_mps')
    curvature = candidate.get('road_curvature_per_m')
    if speed is None or curvature is None:
        return {'tag': 'insufficient_data', 'reason': 'speed or route curvature unavailable'}
    for value in (speed, curvature):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return {'tag': 'invalid_input', 'reason': 'finite numeric speed and curvature required'}
    if speed < 0:
        return {'tag': 'invalid_input', 'reason': 'speed must be nonnegative'}
    demand = float(speed)**2 * abs(float(curvature))
    output = {'nominal_lateral_acc_mps2': demand, 'formula': 'v^2*abs(curvature)',
              'limitation': 'steady-state approximation; not a feasibility proof'}
    if max_lateral_acc_mps2 is None:
        output['tag'] = 'not_evaluated'
    elif not isinstance(max_lateral_acc_mps2, (int, float)) or isinstance(max_lateral_acc_mps2, bool) or not math.isfinite(max_lateral_acc_mps2) or max_lateral_acc_mps2 <= 0:
        output['tag'] = 'invalid_threshold'
    else:
        output['tag'] = 'stress_test' if demand > max_lateral_acc_mps2 else 'within_nominal_bound'
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default=str(root/'01_safety_profile/safety_profile.yaml'))
    parser.add_argument('--hypotheses', default=str(root/'02_hypotheses/hypotheses.yaml'))
    parser.add_argument('--prior', default=str(root/'00_prior_knowledge/prior_knowledge.yaml'))
    parser.add_argument('--scenario-space', default=str(root/'04_feasibility_expansion/scenario_space.yaml'))
    parser.add_argument('--output', default='nemotron_prior_payload.json')
    args = parser.parse_args()
    payload = make_nemotron_payload(load_yaml(args.profile), load_yaml(args.hypotheses),
                                    load_yaml(args.prior), load_yaml(args.scenario_space))
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote offline Nemotron payload: {args.output}')


if __name__ == '__main__':
    main()
