"""Locate and load pipeline YAMLs, the sim-eval av_stack and its scenario space.

Path rules (so the same code works before and after new_pipeline + junwoo/sim-eval are merged):
  * av_stack:        <repo>/pipeline/08_carla_paired_run/av_stack  (in-repo, after the merge)
                     else $AV_STACK_ROOT, else $SIM_EVAL_ROOT/pipeline/08_carla_paired_run/av_stack
  * scenario space:  <repo>/pipeline/04_feasibility_expansion/scenario_space_av_stack.yaml (sim-eval file)
                     else the same path under $SIM_EVAL_ROOT
                     else (mock only) engine/fixtures/scenario_space_mock.yaml  (SYNTHETIC)
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
AV_STACK_REL = Path("pipeline/08_carla_paired_run/av_stack")
SPACE_REL = Path("pipeline/04_feasibility_expansion/scenario_space_av_stack.yaml")
SPACE_TOOL_REL = Path("pipeline/04_feasibility_expansion/scenario_space.py")
FIXTURE_SPACE = Path(__file__).resolve().parent / "fixtures" / "scenario_space_mock.yaml"

STAGE_FILES = {
    "prior_knowledge": "00_prior_knowledge/prior_knowledge.yaml",
    "safety_profile": "01_safety_profile/safety_profile.yaml",
    "vulnerability_patterns": "02_hypotheses/vulnerability_patterns.yaml",
    "objectives_common": "02_hypotheses/search_objectives/_common.yaml",
    "S1": "02_hypotheses/search_objectives/S1_operating_condition_recovery_boundary.yaml",
    "nemotron_a": "03_nemotron_a/nemotron_a.yaml",
    "feasibility_expansion": "04_feasibility_expansion/feasibility_expansion.yaml",
    "deterministic_features": "05a_deterministic_features/deterministic_features.yaml",
    "nemotron_b": "05b_nemotron_b/nemotron_b.yaml",
    "surrogate": "06_gp_surrogate/surrogate.yaml",
    "orchestrator": "07_orchestrator/orchestrator.yaml",
    "paired_runs": "08_carla_paired_run/paired_runs.yaml",
    "error_table": "09_monitor/error_table.yaml",
    "safety_spec": "09_monitor/safety_spec.yaml",
    "video_explainer": "10_video_explainer/video_explainer.yaml",
    "llm": "llm_models.yaml",
    "ablations": "evaluation/ablations.yaml",
}


def load_yaml(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_stage_configs(pipeline_dir: Path = PIPELINE_DIR) -> dict[str, Any]:
    return {name: load_yaml(pipeline_dir / rel) for name, rel in STAGE_FILES.items()}


def _candidates(rel: Path) -> list[Path]:
    out = [REPO_ROOT / rel]
    for env in ("SIM_EVAL_ROOT",):
        if os.environ.get(env):
            out.append(Path(os.environ[env]) / rel)
    return out


def resolve_av_stack() -> Path | None:
    paths = [REPO_ROOT / AV_STACK_REL]
    if os.environ.get("AV_STACK_ROOT"):
        paths.append(Path(os.environ["AV_STACK_ROOT"]))
    paths += _candidates(AV_STACK_REL)[1:]
    for p in paths:
        if (p / "run_benchmark.py").is_file():
            return p.resolve()
    return None


def resolve_scenario_space(allow_fixture: bool) -> tuple[Path, bool]:
    """Return (path, is_synthetic_fixture)."""
    for p in _candidates(SPACE_REL):
        if p.is_file():
            return p.resolve(), False
    if allow_fixture:
        return FIXTURE_SPACE, True
    raise FileNotFoundError(
        f"{SPACE_REL} not found. Merge junwoo/sim-eval into this branch or set SIM_EVAL_ROOT.")


def import_scenario_space_tool():
    """sim-eval pipeline/04_feasibility_expansion/scenario_space.py (to_cli etc.) or None."""
    for p in _candidates(SPACE_TOOL_REL):
        if p.is_file():
            spec = importlib.util.spec_from_file_location("sim_eval_scenario_space", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    return None
