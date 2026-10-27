#!/usr/bin/env python3
"""SIL_HACKATHON 파이프라인 러너 스켈레톤 (DRAFT — 저장소 루트에서 실행: python3 scripts/runner_skeleton.py --check).

이 파일은 pipeline/ 의 15개 YAML을 순서대로 읽어 실행하는 진입점의 뼈대다.
2026-10-26에 확인한 pipeline/ YAML 기준으로 만들었고, YAML에 null로 비어 있는
값은 절대 0이나 예시값으로 대체하지 않는다. 비어 있으면 --check 가 목록으로
보고하고, 실행은 그 지점에서 멈춘다.

구현된 것 (순수 함수, --self-test 로 검증 가능):
  * DF-01~05 계산 (05a deterministic_features.yaml)
  * 타당성 게이트 G-01~G-03 (04 feasibility_expansion.yaml)
  * 지표별 GP 타깃 + 방향·스케일 정규화 z (06 surrogate.yaml, 02 metric_rules[*].gp)
  * 가설 지표별 LCB 선택 + 배치 구성 + 다양성/시드 제한 (07 orchestrator.yaml)
  * 모니터 P-01~03 robustness + 결과 5분류 + margin_delta_m (09 monitor)
  * 짝 실행 baseline 캐시 키 (08 paired_runs.yaml)

스텁으로 남은 것 (외부 시스템이 필요해서 여기서 구현하지 않음):
  * Nemotron A 시나리오 생성, Nemotron B 코드 질문 — model이 null이다.
  * GP 서러게이트 학습 (지표마다 하나) — 스펙은 Matern 5/2 with ARD + WhiteKernel 이지만,
    이 환경에 scikit-learn이 없어 어댑터가 명시적으로 실패한다.
    scikit-learn을 설치하거나 GP 백엔드를 교체하면 연결 지점은 GPModel 이다.
  * CARLA 짝 실행 + 고장 주입 — version/map/route/fixed_delta_seconds가 null이다.
  * G-04 baseline 완료 판정 — 실제 시뮬레이션 결과 없이는 판정할 수 없다.

사용법:
  python3 runner_skeleton.py --pipeline-dir pipeline --check
  python3 runner_skeleton.py --self-test
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

# pipeline/ 의 15개 YAML. 파일이 하나라도 빠지면 load_pipeline()이 실패한다.
# hypotheses.yaml은 python -m pipeline.hypothesis_tools.generate_hypotheses 로 생성한다.
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

# 순수 함수·상수는 pipeline/engine/judgment.py 로 옮겼다 (전체 루프 pipeline/run.py 와 공유). 여기서는 다시 내보낸다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from pipeline.engine.judgment import *  # noqa: E402,F401,F403
from pipeline.engine.judgment import _get, _held  # noqa: E402,F401


class MissingConfigError(ValueError):
    """YAML에 null로 비어 있어 실행할 수 없는 항목이 있을 때 발생."""


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



# ---------------------------------------------------------------------------
# 외부 시스템 연결 지점 — 구현되지 않았음을 명시적으로 실패한다.
# ---------------------------------------------------------------------------

class NemotronAClient:
    """03 Nemotron A: 시나리오 후보 생성 (per_round 15, 근거 id 필수)."""

    def generate(self, observations, hypotheses, scenario_space, prior_failures_top5):
        raise NotImplementedError(
            "TODO: nemotron_a.nemotron.model이 null이다. "
            "NEMOTRON_MODEL_ID를 채우고 이 메서드를 구현할 것.")


class NemotronBClient:
    """05b Nemotron B: 코드 질문 답변 one-hot. 실패 사례·결과를 보지 않는다.

    call_once_per: seed_scenario — 섭동 후보는 시드의 답변을 재사용한다.
    unknown은 false가 아니다.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, str]] = {}

    def answer(self, seed_scenario_id: str, observations, scenario) -> dict[str, str]:
        if seed_scenario_id in self._cache:
            return self._cache[seed_scenario_id]
        raise NotImplementedError(
            "TODO: nemotron_b.nemotron.model이 null이다. "
            "NEMOTRON_MODEL_ID를 채우고 Q-B1~Q-B3 답변 로직을 구현할 것.")


class GPModel:
    """06 GP: gaussian_process_regression, Matern 5/2 with ARD + WhiteKernel.

    지표마다 인스턴스 하나 (metric). 타깃은 run_record.gp_targets[metric] (z).
    refit: per_batch, 학습 데이터는 09 run_record의 all_valid_runs 중 타깃이 정의된 런.
    """

    def __init__(self, metric: str) -> None:
        self.metric = metric

    def fit(self, features, targets):
        raise NotImplementedError(
            "TODO: GP 백엔드가 없다. 이 환경에는 scikit-learn이 없어 "
            "Matern 5/2 + ARD + WhiteKernel 어댑터를 연결할 수 없는 상태다.")

    def predict(self, features):
        raise NotImplementedError("TODO: GP 백엔드 미연결 (fit 참조).")


class CarlaRunner:
    """08 CARLA 짝 실행: no_fault_baseline vs fault_injected, Nebius 병렬.

    감지기는 주입된 고장 자체를 직접 보지 못한다. version/map/route/
    fixed_delta_seconds/parallel_workers가 null이라 실행할 수 없다.
    """

    def run_pair(self, candidate, baseline_cache: dict):
        raise NotImplementedError(
            "TODO: paired_runs.carla.version/map_name/routes/fixed_delta_seconds와 "
            "execution.parallel_workers가 null이다. 값을 채우고 구현할 것.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_check(pipeline_dir: Path) -> int:
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
    if not missing:
        print("실행에 필요한 null 항목이 없다.")
        return 0
    print("실행 전에 채워야 하는 null 항목:")
    for item in missing:
        print(f"  - {item}")
    return 2


def cmd_self_test() -> int:
    """순수 함수만 검증한다. 아래 수치는 테스트용 임의값이며, 저장소 값이 아니다."""
    profile = {
        "controllers": {
            "lqr": {"feedback_design": {"design_speed_mps": 10.0, "design_curvature_per_m": 0.01}},
            "mpc": {"solver_runtime": {"configured_time_limit_ms": 20}},
        },
        "switching": {"failure_detection_period_ms": 20},
    }
    feats = compute_df(
        {"speed_mps": 15.0, "road_curvature_per_m": 0.02,
         "fault": {"delay_ms": 50}, "onset_time": 10.0, "curve_entry_time": 10.01},
        profile)
    assert feats.as_dict() == {
        "DF-01": 5.0, "DF-02": 0.01, "DF-03": 30, "DF-04": True, "DF-05": 4.5,
    }, feats.as_dict()
    assert beta_schedule(0, 10) == 2.0 and abs(beta_schedule(9, 10) - 0.5) < 1e-9
    assert lcb(0.0, 1.0, 2.0) == -2.0
    ok, tags = feasibility_gate(
        {"onset_event": "curve_entry", "nominal_lateral_acc_mps2": 5.0},
        ["curve_entry"], lateral_acc_threshold=4.0, carla_mapping_ok=True)
    assert ok and tags == ["stress_test"]
    assert robustness_p01([0.5, -0.1, 0.3]) == -0.1
    assert robustness_p02([1.0, -2.0], 3.0) == 1.0
    assert classify_outcome(simulation_invalid=False, triggered=True, switched=True,
                            final_safe=True) == "switched_safe"
    assert classify_outcome(simulation_invalid=False, triggered=True, switched=True,
                            final_safe=None) == "switched_unsafe"
    assert classify_outcome(simulation_invalid=True, triggered=True, switched=True,
                            final_safe=True) == "simulation_invalid"
    # recovery_outcome: 100틱 = 5 s 창, 1 s 유지 = 20틱
    ok_m, ok_e = [0.5] * 100, [0.1] * 100
    kw = dict(switched=True, window_complete=True, collision=False)
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, **kw) == "safe_recovery"
    assert classify_recovery(margins=ok_m, e_y=[0.5] * 100, e_psi=[0.0] * 100, **kw) == "unsettled"
    dep = [-0.1] * 30 + [0.5] * 70
    assert classify_recovery(margins=dep, e_y=ok_e, e_psi=[0.0] * 100, **kw) == "recovered_after_departure"
    assert classify_recovery(margins=[0.5] * 70 + [-0.1] * 30, e_y=ok_e, e_psi=[0.0] * 100,
                             **kw) == "unrecovered"
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, switched=True,
                             window_complete=True, collision=True) == "unrecovered"
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, switched=True,
                             window_complete=False, collision=False) == "simulation_invalid"
    assert classify_recovery(margins=ok_m, e_y=ok_e, e_psi=[0.0] * 100, switched=False,
                             window_complete=True, collision=False) == "no_switch"
    labels = recovery_labels(acc_excess=1.0, acc_total_max=4.0, jerk_excess=2.0,
                             jerk_total_max=6.0, margin_at_switch=-0.2, safe_stop=False)
    assert labels["lat_acc_excess"] == {"primary": False, "secondary": True}
    assert labels["departed_at_switch"] is True
    assert final_safe_recovery("safe_recovery", labels) is True
    assert final_safe_recovery("unsettled", labels) is False
    assert lane_margin_m(3.5, 0.5, 0.9) == 0.35
    assert abs(lane_recovery_robustness([0.85, 0.0], [3.5, 3.5], 0.9, False) - 0.0) < 1e-9
    assert lane_recovery_robustness([0.85], [3.5], 0.9, True) == -1.0
    s = travelled_distance([10.0] * 5)
    assert s == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert excess_over_baseline([0.25, 1.0], [2.0, 5.0], [0.0, 0.5, 1.0], [1.0, 1.0, 1.0]) == 4.0
    assert abs(lateral_jerk([0.0, 0.5, 1.0], window_s=0.05)[2] - 10.0) < 1e-9
    assert margin_delta_m(-0.1, 0.4) == -0.5
    assert cold_start_split(24) == (12, 12)
    rules = {
        "recovery_success": {"support_direction": "negative",
                             "gp": {"target": "lane_recovery_robustness",
                                    "form": "absolute_post_switch", "scale": 0.1}},
        "switch_count": {"support_direction": "positive",
                         "gp": {"target": "switch_count",
                                "form": "paired_delta_if_fault_else_absolute", "scale": 1.0}},
    }
    z = gp_targets(rules, {"lane_recovery_robustness": -0.05, "switch_count": 4},
                   {"switch_count": 1}, has_fault=True)
    assert abs(z["recovery_success"] + 0.5) < 1e-9 and z["switch_count"] == -3.0, z
    assert gp_targets(rules, {"switch_count": 2}, None, has_fault=False) == {"switch_count": -2.0}
    assert gp_targets(rules, {"switch_count": 2}, None, has_fault=True) == {}  # baseline 없음
    metric_of = {"H-01": "recovery_success", "H-03": "switch_count"}
    cands = [
        {"id": "a", "hypothesis_ids": ["H-01"], "gp_mean": {"recovery_success": 0.0},
         "gp_std": {"recovery_success": 1.0}, "normalized_features": [0.0], "seed_scenario_id": "s1"},
        {"id": "b", "hypothesis_ids": ["H-03"], "gp_mean": {"switch_count": -1.0},
         "gp_std": {"switch_count": 0.1}, "normalized_features": [0.5], "seed_scenario_id": "s2"},
        {"id": "c", "hypothesis_ids": ["H-03"], "gp_mean": {}, "gp_std": {},
         "normalized_features": [1.0], "seed_scenario_id": "s3"},
    ]
    assert candidate_lcb(cands[2], metric_of, 2.0) is None
    batch = select_batch(cands, metric_of, beta=2.0, batch_size=4)
    assert [c["id"] for c in batch[:2]] == ["a", "b"] and len(batch) == 3
    print("self-test 통과: DF-01~05, LCB, G-03, P-01/P-02, 결과 분류, recovery_outcome·라벨·"
          "final_safe_recovery, lane_recovery_robustness, baseline 위치 정렬, margin_delta_m, "
          "지표별 GP 타깃, 가설 지표 LCB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", default="pipeline")
    parser.add_argument("--check", action="store_true", help="15개 YAML 로드 + null 항목 보고")
    parser.add_argument("--self-test", action="store_true", help="순수 함수 자가 검증")
    args = parser.parse_args(argv)
    if args.self_test:
        return cmd_self_test()
    if args.check:
        return cmd_check(Path(args.pipeline_dir))
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
