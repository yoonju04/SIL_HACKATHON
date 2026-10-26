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

# 05b Nemotron B 질문 은행. B는 실패 사례·CARLA 결과·GP 예측을 보지 않는다.
QUESTION_BANK = {
    "Q-B1": ["hold_last", "zero", "stale_mpc", "lqr_early", "unknown"],
    "Q-B2": ["yes", "no", "unknown"],
    "Q-B3": ["yes", "no", "unknown"],
}

# 08 짝 실행 baseline 캐시 매칭 키 (paired_runs.yaml의 match 목록 그대로).
BASELINE_MATCH_FIELDS = (
    "map",
    "route_id",
    "initial_state",
    "seed",
    "weather",
    "traffic",
    "speed_mps",
    "tire_friction_scale",
)

# 09 run_record 스키마 (error_table.yaml의 results_schema.run_record 그대로).
RUN_RECORD_FIELDS = (
    "run_id", "scenario_id", "seed_scenario_id", "round", "selector",
    "profile_id", "source_commit", "feature_schema_version", "hypothesis_ids",
    "parameters", "planned_fault_onset", "actual_fault_onset", "trigger_time",
    "switch_time", "switch_count", "outcome_category", "P01", "P02", "P03",
    "margin_delta_m", "switch_state_label", "metrics", "baseline_metrics", "gp_targets",
    "gp_mean", "gp_std", "sim_status", "invalid_reason",
)


class MissingConfigError(ValueError):
    """YAML에 null로 비어 있어 실행할 수 없는 항목이 있을 때 발생."""


def _get(data: dict, *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


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
    # recovery_robustness(GP 타깃)를 계산하려면 admissible 제약의 한계값이 필요하다.
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
# 05a: DF-01~05 — LLM을 쓰지 않고 프로필 + 시나리오 값으로 계산한다.
# ---------------------------------------------------------------------------

@dataclass
class DeterministicFeatures:
    df01_speed_offset_mps: float | None = None
    df02_curvature_offset: float | None = None
    df03_fault_delay_over_deadline_ms: float | None = None
    df04_expected_switch_after_curve_entry: bool | None = None
    df05_nominal_lateral_acc_mps2: float | None = None
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """_missing 지시자 규칙: 비어 있는 값은 0으로 대체하지 않는다."""
        out: dict[str, Any] = {
            "DF-01": self.df01_speed_offset_mps,
            "DF-02": self.df02_curvature_offset,
            "DF-03": self.df03_fault_delay_over_deadline_ms,
            "DF-04": self.df04_expected_switch_after_curve_entry,
            "DF-05": self.df05_nominal_lateral_acc_mps2,
        }
        for name in self.missing:
            out[f"{name}_missing"] = True
        return out


def compute_df(scenario: dict[str, Any], profile: dict[str, Any]) -> DeterministicFeatures:
    """scenario: speed_mps, road_curvature_per_m, fault{delay_ms},
    onset_time, curve_entry_time. profile: safety_profile.yaml 내용."""
    design_speed = _get(profile, "controllers", "lqr", "feedback_design", "design_speed_mps")
    design_curv = _get(profile, "controllers", "lqr", "feedback_design", "design_curvature_per_m")
    time_limit = _get(profile, "controllers", "mpc", "solver_runtime", "configured_time_limit_ms")
    detection_ms = _get(profile, "switching", "failure_detection_period_ms")
    speed = scenario.get("speed_mps")
    curv = scenario.get("road_curvature_per_m")
    delay = _get(scenario, "fault", "delay_ms")
    feats = DeterministicFeatures()
    if speed is None or design_speed is None:
        feats.missing.append("DF-01")
    else:
        feats.df01_speed_offset_mps = speed - design_speed
    if curv is None or design_curv is None:
        feats.missing.append("DF-02")
    else:
        feats.df02_curvature_offset = curv - design_curv
    if delay is None or time_limit is None:
        feats.missing.append("DF-03")
    else:
        feats.df03_fault_delay_over_deadline_ms = delay - time_limit
    onset, entry = scenario.get("onset_time"), scenario.get("curve_entry_time")
    if onset is None or detection_ms is None or entry is None:
        feats.missing.append("DF-04")
    else:
        feats.df04_expected_switch_after_curve_entry = onset + detection_ms / 1000.0 > entry
    if speed is None or curv is None:
        feats.missing.append("DF-05")
    else:
        feats.df05_nominal_lateral_acc_mps2 = speed ** 2 * curv
    return feats


# ---------------------------------------------------------------------------
# 04: 타당성 게이트 G-01~G-03. G-04는 시뮬레이션 결과 후에 별도로 판정한다.
# ---------------------------------------------------------------------------

def feasibility_gate(candidate: dict[str, Any], route_events: Iterable[str],
                     lateral_acc_threshold: float | None,
                     carla_mapping_ok: bool) -> tuple[bool, list[str]]:
    """반환: (통과 여부, 태그). G-03 위반은 탈락이 아니라 stress_test 태그다."""
    tags: list[str] = []
    if candidate.get("onset_event") not in set(route_events):
        return False, ["G-01_reject"]  # route가 onset_event를 실제로 포함하지 않음
    if not carla_mapping_ok:
        return False, ["G-02_reject"]  # 파라미터가 CARLA API로 매핑되지 않음
    nominal = candidate.get("nominal_lateral_acc_mps2")
    if nominal is not None and lateral_acc_threshold is not None \
            and nominal > lateral_acc_threshold:
        tags.append("stress_test")  # G-03: 버리지 않고 별도 태깅
    return True, tags


# ---------------------------------------------------------------------------
# 06: 지표별 GP 타깃. 규칙 원본은 02 vulnerability_patterns.yaml metric_rules[*].
# ---------------------------------------------------------------------------

def gp_target(rule: dict[str, Any], run_metrics: dict[str, Any],
              baseline_metrics: dict[str, Any] | None, has_fault: bool) -> float | None:
    """런 1개의 원래 단위 타깃 y. 계산할 수 없으면 None (0으로 대체하지 않는다)."""
    gp = rule["gp"]
    value = run_metrics.get(gp["target"])
    if value is None:
        return None
    if gp["form"] == "absolute_post_switch" or not has_fault:
        return float(value)
    if gp["form"] != "paired_delta_if_fault_else_absolute":
        raise ValueError(f"알 수 없는 GP 타깃 형태: {gp['form']}")
    base = (baseline_metrics or {}).get(gp["target"])
    return None if base is None else float(value) - float(base)


def orient(y: float, rule: dict[str, Any]) -> float:
    """z = sign * y / scale. z가 낮을수록 가설 방향(더 위험). 07이 지표끼리 비교한다."""
    sign = 1.0 if rule["support_direction"] == "negative" else -1.0
    return sign * y / rule["gp"]["scale"]


def gp_targets(metric_rules: dict[str, Any], run_metrics: dict[str, Any],
               baseline_metrics: dict[str, Any] | None, has_fault: bool) -> dict[str, float]:
    """run_record.gp_targets: 지표별 z. 정의되지 않은 지표는 빠진다."""
    out: dict[str, float] = {}
    for name, rule in metric_rules.items():
        y = gp_target(rule, run_metrics, baseline_metrics, has_fault)
        if y is not None:
            out[name] = orient(y, rule)
    return out


# ---------------------------------------------------------------------------
# 07: LCB 오케스트레이터 — 지표별 GP 평균/표준편차(z)가 주어졌을 때의 선택 로직.
# ---------------------------------------------------------------------------

def beta_schedule(round_index: int, total_rounds: int) -> float:
    """2.0에서 0.5로 라운드에 따라 선형 감소 (orchestrator.yaml, 예시)."""
    if total_rounds <= 1:
        return 0.5
    frac = min(max(round_index / (total_rounds - 1), 0.0), 1.0)
    return 2.0 + (0.5 - 2.0) * frac


def lcb(mean_z: float, std_z: float, beta: float) -> float:
    """낮을수록 가설 방향 효과가 예상되는, 즉 더 위험한 후보. 최소화 방향."""
    return mean_z - beta * std_z


def candidate_lcb(candidate: dict[str, Any], metric_of: dict[str, str],
                  beta: float) -> float | None:
    """candidate: hypothesis_ids, gp_mean/gp_std = {metric: z}. metric_of: 가설 id → 주 지표.

    인용한 가설 지표 중 가장 낮은 LCB. 해당 지표 GP 예측이 없으면 None.
    """
    scores = []
    for hid in candidate.get("hypothesis_ids") or []:
        metric = metric_of.get(hid)
        mean = (candidate.get("gp_mean") or {}).get(metric)
        std = (candidate.get("gp_std") or {}).get(metric)
        if mean is not None and std is not None:
            scores.append(lcb(mean, std, beta))
    return min(scores) if scores else None


def select_batch(candidates: list[dict[str, Any]], metric_of: dict[str, str], beta: float,
                 seed: int = 0, batch_size: int = 12, min_distance: float = 0.1,
                 max_per_seed: int = 2) -> list[dict[str, Any]]:
    """배치 12개 = LCB 9개 + 무작위 space filling 3개 (07, 예시 비율).

    candidates 각 항목은 hypothesis_ids, gp_mean/gp_std({metric: z}),
    normalized_features(list[float]), seed_scenario_id를 가져야 한다.
    LCB를 계산할 수 없는 후보는 무작위 몫으로만 뽑힌다.
    다양성: 정규화 feature 공간 최소 거리 0.1, 시드당 최대 2개.
    """
    rng = random.Random(seed)
    scored = [(candidate_lcb(c, metric_of, beta), c) for c in candidates]
    ranked = [c for s, c in sorted((x for x in scored if x[0] is not None), key=lambda x: x[0])]
    picked: list[dict[str, Any]] = []
    per_seed: dict[str, int] = {}
    for cand in ranked:
        if len(picked) >= math.ceil(batch_size * 0.75):
            break
        sid = cand["seed_scenario_id"]
        if per_seed.get(sid, 0) >= max_per_seed:
            continue
        feats = cand["normalized_features"]
        if any(math.dist(feats, p["normalized_features"]) < min_distance for p in picked):
            continue
        picked.append(cand)
        per_seed[sid] = per_seed.get(sid, 0) + 1
    rest = [c for c in candidates if c not in picked]
    rng.shuffle(rest)
    picked.extend(rest[: batch_size - len(picked)])
    return picked


def cold_start_split(total_runs: int = 24) -> tuple[int, int]:
    """GP off 상태: Nemotron A 50% + space filling 50% (07, 예시)."""
    return total_runs // 2, total_runs - total_runs // 2


# ---------------------------------------------------------------------------
# 08: 짝 실행 — baseline 캐시 키. 실제 CARLA 실행은 CarlaRunner 스텁 참조.
# ---------------------------------------------------------------------------

def baseline_cache_key(candidate: dict[str, Any]) -> tuple:
    return tuple(candidate.get(f) for f in BASELINE_MATCH_FIELDS)


# ---------------------------------------------------------------------------
# 09: 모니터 P-01~03 + 결과 5분류 + margin_delta_m.
# ---------------------------------------------------------------------------

def robustness_p01(signed_lane_margins_m: list[float]) -> float:
    return min(signed_lane_margins_m)  # 0보다 작으면 실패


def robustness_p02(lateral_acc_values: list[float], threshold_mps2: float) -> float:
    return threshold_mps2 - max(abs(a) for a in lateral_acc_values)  # 0보다 작으면 실패


def classify_outcome(*, simulation_invalid: bool, triggered: bool,
                     switched: bool, p01: float, p02: float,
                     collision: bool) -> str:
    if simulation_invalid:
        return "simulation_invalid"  # 안전으로 치지 않음
    if not triggered:
        return "no_trigger"
    if not switched:
        return "triggered_no_switch"
    if p01 >= 0 and p02 >= 0 and not collision:
        return "switched_safe"
    return "switched_unsafe"


def margin_delta_m(p01_fault: float, p01_baseline: float) -> float:
    """GP 타깃: P01_robustness(fault) - P01_robustness(baseline)."""
    return p01_fault - p01_baseline


def build_run_record(**kwargs: Any) -> dict[str, Any]:
    record = {name: kwargs.get(name) for name in RUN_RECORD_FIELDS}
    unknown = sorted(set(kwargs) - set(RUN_RECORD_FIELDS))
    if unknown:
        raise ValueError(f"run_record 스키마에 없는 필드: {unknown}")
    return record


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
                            p01=0.2, p02=0.5, collision=False) == "switched_safe"
    assert classify_outcome(simulation_invalid=True, triggered=True, switched=True,
                            p01=0.2, p02=0.5, collision=False) == "simulation_invalid"
    assert margin_delta_m(-0.1, 0.4) == -0.5
    assert cold_start_split(24) == (12, 12)
    rules = {
        "recovery_success": {"support_direction": "negative",
                             "gp": {"target": "recovery_robustness",
                                    "form": "absolute_post_switch", "scale": 0.1}},
        "switch_count": {"support_direction": "positive",
                         "gp": {"target": "switch_count",
                                "form": "paired_delta_if_fault_else_absolute", "scale": 1.0}},
    }
    z = gp_targets(rules, {"recovery_robustness": -0.05, "switch_count": 4},
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
    print("self-test 통과: DF-01~05, LCB, G-03, P-01/P-02, 결과 분류, margin_delta_m, "
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
