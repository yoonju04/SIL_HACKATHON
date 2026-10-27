"""Minimal JSON-schema subset: validate model output and sample stub output (no extra dependency).

Supported keywords: type, enum, minimum, maximum, required, properties, additionalProperties(false),
items, minItems, maxItems, anyOf.
"""
from __future__ import annotations

import random
from typing import Any

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def validate(schema: dict, obj: Any, path: str = "$") -> list[str]:
    errs: list[str] = []
    if "anyOf" in schema:
        options = [validate(s, obj, path) for s in schema["anyOf"]]
        if all(options):
            best = min(options, key=len)
            return [f"{path}: matches no anyOf branch ({best[0] if best else ''})"]
        return []
    if "enum" in schema and obj not in schema["enum"]:
        return [f"{path}: {obj!r} not in enum"]
    t = schema.get("type")
    if isinstance(t, list):
        if not any(validate(dict(schema, type=tt), obj, path) == [] for tt in t):
            return [f"{path}: type not in {t}"]
        return []
    if t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)):
            return [f"{path}: not a number"]
    elif t == "integer":
        if isinstance(obj, bool) or not isinstance(obj, int):
            return [f"{path}: not an integer"]
    elif t in _TYPES and not isinstance(obj, _TYPES[t]):
        return [f"{path}: not {t}"]
    if t in ("number", "integer"):
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: {obj} < {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} > {schema['maximum']}")
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}.{k}: required")
        if schema.get("additionalProperties") is False:
            for k in obj:
                if k not in props:
                    errs.append(f"{path}.{k}: not allowed")
        for k, sub in props.items():
            if k in obj:
                errs += validate(sub, obj[k], f"{path}.{k}")
    if t == "array":
        if "minItems" in schema and len(obj) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(obj) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(obj):
            errs += validate(schema.get("items", {}), item, f"{path}[{i}]")
    return errs


def sample(schema: dict, rng: random.Random, n_items: int | None = None) -> Any:
    """Random instance of a schema (used by the --mock stub LLM; output is SYNTHETIC)."""
    if "anyOf" in schema:
        return sample(rng.choice(schema["anyOf"]), rng)
    if "enum" in schema:
        return rng.choice(schema["enum"])
    t = schema.get("type")
    if isinstance(t, list):
        t = [x for x in t if x != "null"][0]
    if t == "object":
        return {k: sample(s, rng) for k, s in schema.get("properties", {}).items()}
    if t == "array":
        lo = schema.get("minItems", 1)
        hi = schema.get("maxItems", max(lo, 3))
        k = n_items if n_items is not None else rng.randint(lo, min(hi, lo + 2))
        return [sample(schema.get("items", {}), rng) for _ in range(max(lo, min(hi, k)))]
    if t == "number":
        return round(rng.uniform(schema.get("minimum", 0.0), schema.get("maximum", 1.0)), 3)
    if t == "integer":
        return rng.randint(int(schema.get("minimum", 0)), int(schema.get("maximum", 10)))
    if t == "boolean":
        return rng.random() < 0.5
    if t == "string":
        return "SYNTHETIC stub text"
    return None
