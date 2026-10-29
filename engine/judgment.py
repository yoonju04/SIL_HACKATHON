"""Pure judgment / selection functions shared by run.py, engine/monitor.py and the tests.

Everything here is deterministic and side-effect free (self-test: tests/test_engine_judgment.py). Numeric defaults marked PROVISIONAL
are not validated (02 outcome_definitions.recovery_judgment).
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable


def _get(data: dict, *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur

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
    "vehicle_mass_scale",
    "maneuver",
    "velocity_disturbance",
    "steering_disturbance",
    "max_speed_kmh",
    "planner",
)

# 09 run_record 스키마 (error_table.yaml의 results_schema.run_record 그대로).
RUN_RECORD_FIELDS = (
    "run_id", "scenario_id", "seed_scenario_id", "round", "selector",
    "profile_id", "source_commit", "feature_schema_version", "hypothesis_ids",
    "parameters", "planned_fault_onset", "actual_fault_onset", "trigger_time",
    "trigger_source", "trigger_attribution", "switch_time", "switch_count",
    "switch_state", "handover_saturation", "outcome_category",
    "recovery_outcome", "recovery_labels", "final_safe_recovery", "lane_recovery_robustness",
    "lat_acc_excess_mps2", "P01", "P02", "P03",
    "margin_delta_m", "switch_state_label", "metrics", "baseline_metrics", "gp_targets",
    "gp_mean", "gp_std", "sim_status", "invalid_reason",
    # 2026-10-27 integration (engine/sr_checks.py, teammate Monitor v2): extra fields, never used for our labels
    "sr_outcome", "sr_failed_requirements", "sr_labels", "sr_scores", "detection_latency_s", "ftti_s", "sr_checks",
)

# ---------------------------------------------------------------------------
# 05a: DF-01~06 — LLM을 쓰지 않고 프로필 + 시나리오 값으로 계산한다.
# DF-06 (integration3 #6): 타이어 마찰 이용률 v²|κ| / (μ·g), μ = 0.9 × tire_friction_scale (P-02 와 같은 μ).
#   1.0 = 명목 타이어 한계 전부, 0.5 = P-02 한계 (0.5·μ·g). CARLA tire_friction 은 물리 μ 가 아니라 상대값.
# ---------------------------------------------------------------------------

NOMINAL_MU = 0.9      # safety_spec P-02 lateral_acc_formula (0.5 * mu * g, mu = 0.9 at tire_friction_scale 1.0)
G_MPS2 = 9.81


@dataclass
class DeterministicFeatures:
    df01_speed_offset_mps: float | None = None
    df02_curvature_offset: float | None = None
    df03_fault_delay_over_deadline_ms: float | None = None
    df04_expected_switch_after_curve_entry: bool | None = None
    df05_nominal_lateral_acc_mps2: float | None = None
    df06_friction_utilization: float | None = None
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """_missing 지시자 규칙: 비어 있는 값은 0으로 대체하지 않는다."""
        out: dict[str, Any] = {
            "DF-01": self.df01_speed_offset_mps,
            "DF-02": self.df02_curvature_offset,
            "DF-03": self.df03_fault_delay_over_deadline_ms,
            "DF-04": self.df04_expected_switch_after_curve_entry,
            "DF-05": self.df05_nominal_lateral_acc_mps2,
            "DF-06": self.df06_friction_utilization,
        }
        for name in self.missing:
            out[f"{name}_missing"] = True
        return out


def compute_df(scenario: dict[str, Any], profile: dict[str, Any]) -> DeterministicFeatures:
    """scenario: speed_mps, road_curvature_per_m, fault{delay_ms},
    onset_time, curve_entry_time, tire_friction_scale (default 1.0). profile: safety_profile.yaml 내용."""
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
    scale = scenario.get("tire_friction_scale", 1.0)
    if feats.df05_nominal_lateral_acc_mps2 is None or not isinstance(scale, (int, float)) or not math.isfinite(scale) \
            or scale <= 0:
        feats.missing.append("DF-06")
    else:
        feats.df06_friction_utilization = feats.df05_nominal_lateral_acc_mps2 / (NOMINAL_MU * float(scale) * G_MPS2)
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


# 2026-10-27 recovery_judgment (02 outcome_definitions). 아래 기본 수치는 PROVISIONAL (검증 안 됨).
# 실제 값은 09 safety_spec.yaml (safety_specification.recovery_judgment / P-02 / P-04) 에서 읽는다 (engine.monitor).
# 팀원 기준 (junwoo/sim-eval a5242b2): window 3 s (SR5 within_s), 횡가속도 0.5*0.9*9.81 = 4.41, 충돌 -2, clip [-2, 1].
PROVISIONAL_RECOVERY = {
    "e_settle_m": 0.3, "psi_settle_rad": 0.1, "settle_hold_s": 1.0, "window_s": 3.0,
    "lat_acc_threshold_mps2": 4.41, "lat_jerk_threshold_mps3": 5.0, "filter_s": 0.5,
    "collision_penalty": -2.0, "robustness_clip_min": -2.0, "robustness_clip_max": 1.0,
}


def lane_margin_m(lane_width_m: float, lane_offset_m: float, bbox_half_width_m: float) -> float:
    """lane_width/2 - |offset| - 차체 반폭. 방향각 무시 근사 (monitor_probe.jsonl 필드)."""
    return lane_width_m / 2.0 - abs(lane_offset_m) - bbox_half_width_m


def _held(flags: list[bool], need: int) -> bool:
    run = 0
    for ok in flags:
        run = run + 1 if ok else 0
        if run >= need:
            return True
    return False


def classify_recovery(*, switched: bool, window_complete: bool, margins: list[float],
                      e_y: list[float], e_psi: list[float], collision: bool,
                      dt: float = 0.05, th: dict[str, float] | None = None) -> str:
    """전환 후 창 [t_sw, t_sw+window] 시계열 → recovery_outcome (상호 배타, 위에서부터)."""
    th = th or PROVISIONAL_RECOVERY
    if not window_complete or not margins or len(margins) != len(e_y) or len(e_y) != len(e_psi):
        return "simulation_invalid"
    if not switched:
        return "no_switch"
    need = max(1, round(th["settle_hold_s"] / dt))
    settled = [abs(a) <= th["e_settle_m"] and abs(b) <= th["psi_settle_rad"] and m >= 0
               for a, b, m in zip(e_y, e_psi, margins)]
    departed = [m < 0 for m in margins]
    if collision:
        return "unrecovered"
    if not any(departed):
        return "safe_recovery" if _held(settled, need) else "unsettled"
    last = max(i for i, d in enumerate(departed) if d)
    if last == len(margins) - 1:
        return "unrecovered"  # 창 끝까지 복귀 못 함
    return "recovered_after_departure" if _held(settled[last + 1:], need) else "unrecovered"  # returned_not_settled


def moving_average(values: list[float], window_s: float, dt: float = 0.05) -> list[float]:
    n = max(1, round(window_s / dt))
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= n:
            acc -= values[i - n]
        out.append(acc / min(i + 1, n))
    return out


def lateral_jerk(a_y: list[float], dt: float = 0.05, window_s: float = 0.5) -> list[float]:
    """d(a_y)/dt 차분 후 이동평균 (E-01 사용자 제공: 0.5 s MA)."""
    diff = [0.0] + [(a_y[i] - a_y[i - 1]) / dt for i in range(1, len(a_y))]
    return moving_average(diff, window_s, dt)


def travelled_distance(vx: list[float], dt: float = 0.05) -> list[float]:
    """경로 진행 근사 s = cumsum(vx*dt). 경로 인덱스 로깅(L-5) 전 임시 정렬 좌표."""
    s, out = 0.0, []
    for v in vx:
        out.append(s)
        s += max(v, 0.0) * dt
    return out


def interp_at(s_query: float, s_ref: list[float], values: list[float]) -> float | None:
    """Linear interpolation of values at s_query on a non-decreasing s_ref; None outside the range."""
    if not s_ref or s_query < s_ref[0] or s_query > s_ref[-1]:
        return None
    i = bisect.bisect_left(s_ref, s_query)
    if i == 0:
        return values[0]
    s0, s1 = s_ref[i - 1], s_ref[i]
    w = 0.0 if s1 == s0 else (s_query - s0) / (s1 - s0)
    return values[i - 1] + w * (values[i] - values[i - 1])


def excess_over_baseline(s_fault: list[float], v_fault: list[float],
                         s_base: list[float], v_base: list[float]) -> float | None:
    """max |v_fault(t) - v_base(s(t))|. 같은 위치 baseline이 없는 점은 건너뛴다. 전부 없으면 None."""
    diffs = [abs(v - b) for s, v in zip(s_fault, v_fault)
             if (b := interp_at(s, s_base, v_base)) is not None]
    return max(diffs) if diffs else None


def lane_recovery_robustness(margins: list[float], lane_widths: list[float],
                             bbox_half_width_m: float, collision: bool,
                             penalty: float = PROVISIONAL_RECOVERY["collision_penalty"],
                             clip: tuple[float, float] | None = (PROVISIONAL_RECOVERY["robustness_clip_min"],
                                                                 PROVISIONAL_RECOVERY["robustness_clip_max"])) -> float:
    """min margin/(lane_width/2 - 반폭). 1 = 중앙, 0 = 경계, <0 이탈. 충돌이면 penalty와 min. clip=[lo, hi] (팀원 SR4 눈금)."""
    value = min(m / (w / 2.0 - bbox_half_width_m) for m, w in zip(margins, lane_widths))
    value = min(value, penalty) if collision else value
    return value if clip is None else max(clip[0], min(clip[1], value))


def recovery_labels(*, acc_excess: float | None, acc_total_max: float | None,
                    jerk_excess: float | None, jerk_total_max: float | None,
                    margin_at_switch: float | None, safe_stop: bool,
                    th: dict[str, float] | None = None) -> dict[str, Any]:
    th = th or PROVISIONAL_RECOVERY
    gt = lambda v, lim: None if v is None else v > lim
    return {
        "lat_acc_excess": {"primary": gt(acc_excess, th["lat_acc_threshold_mps2"]),
                           "secondary": gt(acc_total_max, th["lat_acc_threshold_mps2"])},
        "lat_jerk_excess": {"primary": gt(jerk_excess, th["lat_jerk_threshold_mps3"]),
                            "secondary": gt(jerk_total_max, th["lat_jerk_threshold_mps3"])},
        "departed_at_switch": None if margin_at_switch is None else margin_at_switch < 0,
        "safe_stop": bool(safe_stop),
    }


def final_safe_recovery(outcome: str, labels: dict[str, Any]) -> bool | None:
    """safe_recovery AND NOT 가속도/저크 primary. 라벨 계산 불가(None)면 None (안전으로 치지 않음)."""
    if outcome == "simulation_invalid":
        return None
    if outcome != "safe_recovery":
        return False
    acc, jerk = labels["lat_acc_excess"]["primary"], labels["lat_jerk_excess"]["primary"]
    if acc is None or jerk is None:
        return None
    return not acc and not jerk


def classify_outcome(*, simulation_invalid: bool, triggered: bool,
                     switched: bool, final_safe: bool | None) -> str:
    """09 결과 5분류 (역추적용 유지). switched_* 는 final_safe_recovery로 정한다."""
    if simulation_invalid:
        return "simulation_invalid"  # 안전으로 치지 않음
    if not triggered:
        return "no_trigger"
    if not switched:
        return "triggered_no_switch"
    return "switched_safe" if final_safe is True else "switched_unsafe"


def margin_delta_m(p01_fault: float, p01_baseline: float) -> float:
    """GP 타깃: P01_robustness(fault) - P01_robustness(baseline)."""
    return p01_fault - p01_baseline


def build_run_record(**kwargs: Any) -> dict[str, Any]:
    record = {name: kwargs.get(name) for name in RUN_RECORD_FIELDS}
    unknown = sorted(set(kwargs) - set(RUN_RECORD_FIELDS))
    if unknown:
        raise ValueError(f"run_record 스키마에 없는 필드: {unknown}")
    return record
