"""Config check: load the 15 stage YAMLs under pipeline/ and report null items that must be filled before a real run.

  python run.py --check            (exit 2 while required nulls remain)

Example values in hypotheses.yaml (e.g. design_speed 15.0) are example_only and are NOT used as defaults here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import PIPELINE_DIR
from .judgment import _get

# pipeline/ 의 15개 YAML. 파일이 하나라도 빠지면 load_pipeline()이 실패한다.
# hypotheses.yaml은 python -m engine.hypothesis_tools.generate_hypotheses 로 생성한다.
PIPELINE_FILES = {
    "prior_knowledge": "00_prior_knowledge/prior_knowledge.yaml",
    "safety_profile": "01_safety_profile/safety_profile.yaml",
    "hypotheses": "02_hypotheses/hypotheses.yaml",
    "vulnerability_patterns": "02_hypotheses/vulnerability_patterns.yaml",
    "nemotron_a": "03_nemotron_a/nemotron_a.yaml",
    "feasibility_expansion": "04_feasibility_expansion/feasibility_expansion.yaml",
    "scenario_space": "04_feasibility_expansion/scenario_space.yaml",
    "deterministic_features": "05a_deterministic_features/deterministic_features.yaml",
    "nemotron_b": "05b_nemotron_b/nemotron_b.yaml",
    "surrogate": "06_gp_surrogate/surrogate.yaml",
    "orchestrator": "07_orchestrator/orchestrator.yaml",
    "paired_runs": "08_carla_paired_run/paired_runs.yaml",
    "error_table": "09_monitor/error_table.yaml",
    "safety_spec": "09_monitor/safety_spec.yaml",
    "ablations": "evaluation/ablations.yaml",
}


def load_pipeline(pipeline_dir: str | Path) -> dict[str, Any]:
    """15개 YAML을 모두 읽는다. 빠진 파일이 있으면 FileNotFoundError."""
    root = Path(pipeline_dir)
    loaded: dict[str, Any] = {}
    for name, rel in PIPELINE_FILES.items():
        path = root / rel
        with path.open(encoding="utf-8") as fh:
            loaded[name] = yaml.safe_load(fh)
    return loaded


def collect_missing(config: dict[str, Any]) -> list[str]:
    """실행 전에 반드시 채워야 하는 null 항목을 경로 목록으로 반환한다.

    hypotheses.yaml의 예시값(예: design_speed 15.0)은 example_only라서
    여기서 기본값으로 쓰지 않는다. safety_profile.yaml이 비어 있으면 비어
    있는 그대로 보고한다.
    """
    profile = config["safety_profile"]
    missing: list[str] = []

    def check(value: Any, path: str) -> None:
        if value is None or value == [] or value == {}:
            missing.append(path)

    check(_get(profile, "controllers", "lqr", "feedback_design", "design_speed_mps"),
          "safety_profile.controllers.lqr.feedback_design.design_speed_mps")
    check(_get(profile, "controllers", "lqr", "feedback_design", "design_curvature_per_m"),
          "safety_profile.controllers.lqr.feedback_design.design_curvature_per_m")
    check(_get(profile, "controllers", "mpc", "solver_runtime", "configured_time_limit_ms"),
          "safety_profile.controllers.mpc.solver_runtime.configured_time_limit_ms")
    check(_get(profile, "switching", "failure_detection_period_ms"),
          "safety_profile.switching.failure_detection_period_ms")
    check(_get(profile, "switching", "triggers"),
          "safety_profile.switching.triggers")
    # lane_recovery_robustness(GP 타깃)는 런타임 차선 폭·차체 반폭(monitor_probe)으로 계산한다.
    # 아래 프로필 값은 런마다 달라 null이 정상이지만, 옛 검사와의 호환으로 남겨 둔다.
    for path in (("vehicle", "lane_half_width_m"), ("vehicle", "vehicle_half_width_m"),
                 ("controllers", "lqr", "constraints_and_behavior", "steering_saturation"),
                 ("controllers", "lqr", "constraints_and_behavior", "steering_rate_limit")):
        check(_get(profile, *path), "safety_profile." + ".".join(path))
    check(_get(config["nemotron_a"], "nemotron", "model"),
          "nemotron_a.nemotron.model (.env NEMOTRON_MODEL_ID)")
    check(_get(config["nemotron_b"], "nemotron", "model"),
          "nemotron_b.nemotron.model (.env NEMOTRON_MODEL_ID)")
    spec_props = _get(config["safety_spec"], "safety_specification", "properties") or []
    p02 = next((p for p in spec_props if p.get("id") == "P-02"), {})
    check(p02.get("lateral_acc_threshold_mps2"),
          "safety_spec.P-02.lateral_acc_threshold_mps2")
    carla = config["paired_runs"].get("carla", {})
    check(carla.get("version"), "paired_runs.carla.version")
    check(carla.get("map_name"), "paired_runs.carla.map_name")
    check(carla.get("routes"), "paired_runs.carla.routes")
    check(carla.get("fixed_delta_seconds"), "paired_runs.carla.fixed_delta_seconds")
    budget = config["orchestrator"].get("budget", {})
    check(budget.get("max_simulations"), "orchestrator.budget.max_simulations")
    check(budget.get("wall_time_budget_s"), "orchestrator.budget.wall_time_budget_s")
    check(budget.get("sim_seconds_per_run_measured"),
          "orchestrator.budget.sim_seconds_per_run_measured")
    check(_get(config["paired_runs"], "execution", "parallel_workers"),
          "paired_runs.execution.parallel_workers")
    return missing


def catalog_report(pipeline_dir: Path = PIPELINE_DIR) -> str:
    """One line on the av_stack fault catalog (scenario_space_av_stack.yaml, engine/fault_catalog.py check)."""
    from . import fault_catalog
    path = Path(pipeline_dir) / "04_feasibility_expansion" / "scenario_space_av_stack.yaml"
    if not path.is_file():
        return f"av_stack 고장 목록 없음: {path} (real 모드는 SIM_EVAL_ROOT 필요)"
    space = fault_catalog.load(path)
    problems = fault_catalog.check(space)
    ready = sum(1 for f in space.get("faults", []) if f.get("implemented"))
    head = f"av_stack 고장 목록: {len(space.get('faults', []))}개 (주입기 구현 {ready}개) {path.name}"
    return head + ("" if not problems else " — 문제: " + "; ".join(problems))


def cmd_check(pipeline_dir: Path = PIPELINE_DIR) -> int:
    pipeline_dir = Path(pipeline_dir)
    config = load_pipeline(pipeline_dir)
    missing = collect_missing(config)
    print(f"YAML {len(PIPELINE_FILES)}개 로드 완료: {pipeline_dir}")
    hyps = (config["hypotheses"] or {}).get("hypotheses") or []
    counts: dict[str, int] = {}
    for h in hyps:
        counts[h["status"]] = counts.get(h["status"], 0) + 1
    print("가설 상태:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "없음")
    metric_of = {h["id"]: h["prediction"]["metric"] for h in hyps
                 if h["status"] in ("proposed", "testing")}
    print("활성 GP 지표:", ", ".join(sorted(set(metric_of.values()))) or "없음 (전부 blocked)")
    print(catalog_report(pipeline_dir))
    if not missing:
        print("실행에 필요한 null 항목이 없다.")
        return 0
    print("실행 전에 채워야 하는 null 항목:")
    for item in missing:
        print(f"  - {item}")
    return 2
