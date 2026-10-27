# 09 Monitor (integration: recovery v2 + 팀원 Monitor v2 SR 검사)

09 Monitor는 08 짝 실행이 끝난 뒤 런 폴더를 읽어 채점합니다. 읽기 전용·사후 실행이고, 판정을 내리는 곳은 여기 하나입니다
(10 영상 해설은 판정을 바꾸지 않습니다).

2026-10-27 `integration` 브랜치에서 `new_pipeline`(우리)과 `junwoo/sim-eval`(팀원 a5242b2 Monitor v2)을 합쳤습니다.
겹치는 부분은 우리 구현을 쓰고, 팀원만 가진 검사(SR1/SR2/SR4/SR6)는 `engine/sr_checks.py`로 옮겨 run_record에 필드로 덧붙였습니다.

| 파일 | 내용 |
|---|---|
| `safety_spec.yaml` | P-01~04 (우리), `sr_checks` 절 (SR1/SR2/SR4/SR6 수치, 팀원 값). 모니터가 읽는 유일한 수치 출처 |
| `error_table.yaml` | run_record / failure_table 스키마 (`sr_*` 필드 추가) |
| [`engine/monitor.py`](../../engine/monitor.py) | `evaluate(run_dir, spec, baseline_dir=, no_reaction_dir=)` → run_record 하나. CLI `python -m engine.monitor` |
| [`engine/judgment.py`](../../engine/judgment.py) | recovery v2 판정 순수 함수 (recovery_outcome, 라벨, final_safe_recovery, GP 타깃) |
| [`engine/sr_checks.py`](../../engine/sr_checks.py) | 팀원 Monitor v2의 SR1(FTTI)·SR2·SR4(차체 사각형)·SR6 + 유효성 V-01~11·G-04 |
| [`engine/dashboard.py`](../../engine/dashboard.py) | 사람이 보는 화면 `outputs/<run>/screen/campaign.html` (run.py가 라운드마다 갱신) |
| [`engine/av_stack_probe.py`](../../engine/av_stack_probe.py) | CARLA 실측 probe (의도된 차선 기준 오프셋·차선 폭·차체 크기). CARLA 안에서 도는 사본은 `08_carla_paired_run/av_stack/integration/monitor_logs.py` |

## 입력 (런 폴더, av_stack run_benchmark.py 출력 = 우리 engine/carla_runner 가 만드는 형식)

`steps.jsonl`, `behavior.jsonl`, `monitor_probe.jsonl`, `agent_controls.jsonl`, `collisions.json`, `summary.json`,
`agent_config.json`, (`front.mp4` + `front_frames.jsonl`). `engine.monitor.load_av_stack_run`이 t로 묶고,
`engine.sr_checks.trace_from_run`이 같은 데이터를 SR 검사용 Trace로 바꿉니다
(조향 명령: `steer_rad`가 없으면 `agent_controls.steer × geometry.max_steer_rad`, 부호는 우수계로).

런 세 종류:
- fault: 고장 있음, 감독기 있음 (`--controller supervised`)
- baseline: 같은 경로·속도, 고장 없음 (08 match key로 캐시)
- no-reaction (선택): 같은 고장·시점, 감독기 없음 (`--controller mpc_unprotected`). SR1 FTTI 측정용.
  `08_carla_paired_run/paired_runs.yaml runner.no_reaction_run` (mock 기본 on, real 기본 off), `run.py --no-reaction-runs on|off`.

## 출력 (run_record, `engine.judgment.RUN_RECORD_FIELDS`)

우리 필드가 판정입니다: `recovery_outcome`, `recovery_labels`, `final_safe_recovery`, `outcome_category`,
`lane_recovery_robustness`, `lat_acc_excess_mps2`, `P01~P03`, `metrics`, `gp_targets` …

팀원 검사 결과는 참고 필드로 덧붙습니다 (우리 라벨·GP 타깃·outcome_category를 바꾸지 않음):

| 필드 | 내용 |
|---|---|
| `sr_outcome` | `invalid` · `no_trigger` · `F1_no_or_late_switch` · `F2_switch_transient` · `F3_not_recovered` · `pass` |
| `sr_failed_requirements` | 위반한 SR 목록 (SR5는 recovery v2에서 파생) |
| `sr_labels` | `late_or_no_switch`, `switch_transient`, `footprint_departure`, `actuator_limit_exceeded` (bool / null) |
| `sr_scores` | SR별 여유 점수 (1 여유, 0 한계, <0 위반, 충돌 −2) + `overall` |
| `detection_latency_s`, `ftti_s` | SR1: 실제 고장 → 전환 지연, no-reaction 런에서 잰 FTTI |
| `sr_checks` | 전체 상세 (sr.SR1..SR6, sr_measures, invalid_reasons, evaluation_window, no_reaction_run …) |

## SR 검사 (engine/sr_checks.py)

| SR | 질문 | 기준 (safety_spec.yaml `sr_checks`) |
|---|---|---|
| SR1 전환 시간 | 실제 고장 시작 → 첫 전환 | ≤ 0.5 × FTTI (no-reaction 런). 없으면 0.5 s 고정. FTTI = ∞ 이면 적용 안 함 (`undetected_harmless`) |
| SR2 전환 순간 | 전환 −dt ~ +0.5 s 최대 조향 속도 | ≤ 60 °/s |
| SR4 차선·충돌 | 차체 네 모서리 vs 의도된 차선 경계, 충돌 | ≥ 0 m, 충돌 없음. baseline이 위반하면 쌍 거절 (G-04) |
| SR5 회복 | **우리 recovery v2에서 파생** | `recovery_outcome == safe_recovery` |
| SR6 액추에이터 | 최대 \|조향\|, 최대 조향 속도 | 차량 max_steer, 90 °/s |

`sr_outcome` 우선순위: invalid → no_trigger → F1 (SR1) → F2 (SR2) → F3 → pass.
F3는 전환한 런이면 `final_safe_recovery == false` (우리 판정), 전환 안 한 런이면 SR4 위반입니다.

## 팀원 버전과 달라진 점 (겹치는 부분은 우리 것)

| 항목 | 팀원 (a5242b2) | 지금 |
|---|---|---|
| 회복 판정 | SR5: ±0.3 m 띠에 1 s, 3 s 안 (참고용) | recovery v2 6분류 + 라벨. 수치는 팀원 기준: 0.3 m·1 s 유지·**3 s** 창 (+ 우리 0.1 rad 방향각 조건) |
| 횡가속도 | P-02 0.5·μ·g = **4.41 m/s²** (전체 값, 보고용) | P-02 **4.41 m/s²** (팀원 값), baseline 대비 primary 라벨로 final_safe_recovery 에 반영 |
| SR 평가 창 | onset + 10 s (최소 5 s), baseline 같은 길이 (V-08) | 같음 (팀원 원본, `sr_checks.evaluation_window`) |
| 충돌 GP 값 | SR4 충돌 −2, clip [−2, 1] | lane_recovery_robustness 충돌 −2, [−2, 1]로 자름 (이전 −1, 자르지 않음) |
| run_record | runner_skeleton 28필드 + metrics.monitor | 우리 RUN_RECORD_FIELDS + `sr_*` |
| 팀 지표·GP 타깃 | run_metrics / runner_gp_targets | engine.monitor + engine.judgment.gp_targets |
| 피드백 (2·3·6단계) | feedback.py (가설 추적 bootstrap, failure_table, gp_training) | engine/feedback.py + run.py |
| 영상·해설 | video_clip.py, explain.py (블라인드 판단 + 해설) | engine/video.py, engine/explainer.py (10 단계, 모니터 시각으로 검증) |
| CLI | `python monitor.py --fault-run … --baseline-run … --onset …` | `python -m engine.monitor RUN_DIR --baseline DIR [--no-reaction DIR]` (onset은 런 설정에서 읽음) |

삭제: `pipeline/09_monitor/monitor.py`, `feedback.py`, `explain.py`, `video_clip.py`, `test_monitor.py`, `test_explain.py`.
옮김: `dashboard.py` → `engine/dashboard.py` (가설 패널 제거, 우리 판정·해설 표시), `av_stack_probe.py` → `engine/`,
`test_probe.py` → `tests/test_engine_av_stack_probe.py`, SR 테스트 → `tests/test_engine_sr_checks.py`.

## 실행

```
python -m engine.monitor outputs/<run>/runs/R00-0001-drift --baseline outputs/<run>/runs/baselines/<hash> [--no-reaction …] --out rr.jsonl
python -m engine.dashboard --records outputs/<run>/run_records.jsonl --out outputs/<run>/screen --open
```

모든 수치는 PROVISIONAL (검증 안 됨). 인증 기준이 아닙니다.
