"""Flatten pipeline/04_feasibility_expansion/scenario_space.yaml into named sampling dimensions."""
from __future__ import annotations

from typing import Any


def _dimension(spec: dict[str, Any]) -> dict[str, Any] | None:
    if 'range' in spec:
        low, high = spec['range']
        return {'min': low, 'max': high, **({'unit': spec['unit']} if 'unit' in spec else {})}
    if 'values' in spec:
        return {'choices': list(spec['values'])}
    if spec.get('type') == 'bool':
        return {'choices': [True, False]}
    return None


def normalize_space(space: dict[str, Any]) -> dict[str, Any]:
    """Return {'dimensions': {name: {min,max}|{choices}}, 'fault_types': [...]}.

    Names: environment keys as-is, 'fault_type', '<fault id>.<param>',
    'onset_event' (fault_timing.reference_events), 'onset_offset_s'.
    Faults marked co_injected_only are valid fault ids but are not drawn as fault_type.
    An already-flat space (with 'dimensions') is returned unchanged.
    """
    if 'dimensions' in space:
        return space
    dims: dict[str, Any] = {}
    for name, spec in (space.get('environment') or {}).items():
        dim = _dimension(spec)
        if dim is not None:
            dims[name] = dim
    faults = space.get('faults') or []
    fault_ids = [f['id'] for f in faults]
    drawable = [f['id'] for f in faults if not f.get('co_injected_only')]
    if drawable:
        dims['fault_type'] = {'choices': drawable}
    for fault in faults:
        for param, spec in (fault.get('params') or {}).items():
            dim = _dimension(spec)
            if dim is not None:
                dims[f"{fault['id']}.{param}"] = dim
    timing = space.get('fault_timing') or {}
    if timing.get('reference_events'):
        dims['onset_event'] = {'choices': list(timing['reference_events'])}
    if isinstance(timing.get('offset_s'), dict):
        dim = _dimension(timing['offset_s'])
        if dim is not None:
            dims['onset_offset_s'] = dim
    return {'meta': space.get('meta'), 'dimensions': dims, 'fault_types': fault_ids}
