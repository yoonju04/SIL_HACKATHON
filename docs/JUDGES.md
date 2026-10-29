# 05b 판정자 (laya-judge v2) — J1~J9 × 25 판정 + 질문별 회귀

[English below](#english)

## 한 줄 요약

05b 는 System One 형 판정 모델에 **판정 25개를 한 요청으로** 묻고(`POST /v1/systemone`), 질문별 L2 로지스틱 회귀가
그 확률을 하나의 답으로 모읍니다 (Quantitative LLM Judges 방식). v2 에서는 **판정자에게 숫자로 계산되는 질문을 묻지
않습니다.** 파라미터로 계산되는 고장 역학은 결정론 feature DF-07~09 로 GP 에 바로 들어가고, 판정자에게는 코드를
읽어야 아는 질문과 실행 결과로 채점되는 질문만 남깁니다.

## 질문 은행 (`pipeline/05b_judges/questions.yaml`)

| id | 종류 | 질문 | 보기 | 라벨 | 판정 수 |
|---|---|---|---|---|---|
| J1 | 코드 | 이 고장은 어떤 감지 경로로 잡히는가 | immediate_check / deviation_accumulation / no_dedicated_check / unknown | 프로필별 코드 정답 키 | 4 |
| J3 | 코드 | 간헐·짧은 고장이 유지(hold) 조건을 피해 가는가 (intermittent_bias, saturate 에만) | caught_by_rate_check / can_evade / cannot_evade / unknown | 코드 정답 키 | 3 |
| J4 | 코드 | 전환 전 차량에 어떤 조향이 나가는가 (옛 Q-B1, 프로필별) | lqr_early / corrupted_mpc / stale_mpc / hold_last / unknown | 코드 정답 키 | 3 |
| J7 | 코드 | LQR 인계 순간 조향이 이어지는가 | continuous_from_clean / continuous_from_faulted / discontinuous / unknown | 코드 정답 키 | 3 |
| J8 | 코드 | 고장이 LQR 경로에도 닿는가 | mpc_output_only / also_reaches_lqr / unknown | 코드 정답 키 (한 클래스 → 보고만) | 2 |
| J9 | 코드 | 전환 뒤 MPC 로 돌아가는가 (옛 Q-B2) | latched_no_return / returns_to_mpc / unknown | 코드 정답 키 (한 클래스 → 보고만) | 2 |
| J2 | 결과 | 고장 시작부터 감지까지 지연 구간 | under_0_5s / 0_5_to_2s / 2_to_5s / over_5s_or_never | 실행 기록 `switch_time − actual_fault_onset` | 3 |
| J5 | 결과 | 감지 전 차선 이탈 위험 | low / moderate / high / near_or_departed | `pre_switch_max_lateral_error_m` 0.15 / 0.4 / 0.7 m, 이탈이면 마지막 | 3 |
| J6 | 결과 | LQR 첫 명령이 속도 제한에 걸리는가 | rate_limited / within_limit | `handover_saturation.rate_saturation_ratio > 0` (전환된 런만) | 2 |

- 판정 25개 = 질문 × 변형 (en_notes, en_notes_rev = 보기 순서 반대, ko, focus_supervisor / focus_fault / focus_switch_rev = 코드 구역 지시). 같은 말 바꾸기보다 서로 다른 질문을 늘렸습니다.
- J3 은 intermittent_bias·saturate 에만 묻습니다 (요청당 25개, 다른 모드는 22개).
- **독립성 규칙:** state 에는 정답 키도 실행 결과도 들어가지 않습니다 (`FORBIDDEN_STATE_KEYS`). 결과는 회귀 라벨로만 씁니다. 루프 안에서는 같은 판정 키로 묶인 끝난 실행들의 다수 라벨, 오프라인 평가에서는 실행마다의 라벨.
- 판정 캐시 키 = 고장 모드 | 시작 이벤트 | 질문 파라미터 구간 | 속도 구간 (20 / 35 km/h) | 경로. state 에는 속도·경로·마찰도 들어갑니다 (결과 질문에 필요).
- J8·J9 는 코드상 답이 하나뿐이라 회귀하지 않고 (`regress: false`), 판정자가 코드를 맞게 읽는지 보는 점검용입니다.
- GP 입력: 코드 질문은 보기별 회귀 확률 (`unknown` 제외), 결과 질문은 기대 구간 번호 하나 (`gp_encoding: ordinal`), 캠페인 동안 안 변하는 열은 뺍니다.

## 계산되는 값 → 결정론 feature (DF-07~09, `engine/judgment.py fault_dfs`)

| id | 이름 | 식 | 정의되는 모드 |
|---|---|---|---|
| DF-07 | first_tick_jump_over_rate_check | 고장이 한 틱에 만드는 조향 점프 / (90 °/s × 0.05 s = 0.0785 rad). > 1 이면 tight 속도 검사가 그 틱에 걸림 | bias·intermittent = magnitude_rad, saturate = level × 1.2217 rad (ramp 0) 또는 × tick / ramp_s, drift = rate × tick, solver_fail·stale = 0. output_delay 는 MPC 동역학에 달려 없음 |
| DF-08 | fault_on_time_over_hold | intermittent on_s 또는 saturate duration_s / 유지 시간 (tight 0.15 s, legacy 0.25 s) | intermittent_bias, saturate |
| DF-09 | applied_offset_over_lqr_step | 감지 전 차량에 나가는 오프셋 / LQR 인계 한 틱 (60 °/s × 0.05 s). tight 에서 DF-07 > 1 이면 0 | bias, intermittent_bias, saturate |

속도·곡률·마찰은 이미 DF-05 (v²κ), DF-06 (마찰 사용률) 가 담당합니다. 없는 값은 0 + `_missing` 지시 열로 GP 에 들어가고,
`deterministic_features.yaml fault_mechanics_features.enabled` 로 끌 수 있습니다.

## 백엔드 (`--judge-b`, `pipeline/05b_judges/judges.yaml`)

| 값 | 무엇 | 키 | state |
|---|---|---|---|
| `laya` (기본) | 오픈소스 Laya 0.4.17 (Apache-2.0, HF convaiinnovations/laya), 로컬 `laya-serve` 또는 프로세스 안 | 없음 | compact (감독기·해당 고장 줄·전환 줄) |
| `jev` | 호스팅 TypeSafe Jev, 같은 프로토콜 | `TYPESAFE_API_KEY` | full |
| `nemotron` | Token Factory Nemotron (역할 B) 을 System One 판정자로: json_schema 로 보기별 확률만 받음 | `NEBIUS_API_KEY` | full |
| `nemotron_b` | 예전 생성형 Nemotron B + Q-B1~5 은행 (`05b_nemotron_b/`), 비교용 | `NEBIUS_API_KEY` | – |

서버·키가 없거나 `--mock` 이면 결정론 stub (SYNTHETIC) 으로 돌고 run_config.json `judges.client` 에 남습니다.

## result_8 오프라인 평가 (`tools/judges_offline_eval.py`)

result_8 의 sim_status ok 실행 82개 (모두 tight). 실행마다 state 1개, 질문은 위 은행. 숫자는 `/workspace` 평가 산출물에서.

**Nemotron (System One 에뮬레이션), 82 상태** — 요청 82회, 입력 중앙값 9.1k tok (합 0.76M), 6 병렬, 요청당 중앙값 43 s.

| 질문 | 라벨 분포 (한 답으로 정해진 것) | 판정자 정확도 | 다수결 | 회귀 CV | 고장 모드 조회 (판정자 없음) | 모드 하나 빼기: 회귀 / 다수 |
|---|---|---|---|---|---|---|
| J1 | dev 35 · imm 28 · none 11 (+8 둘 다) | 0.82 | 0.82 | 0.81 | 0.74 | 0.78 / 0.30 |
| J3 | caught 15 (+18 목록) | 0.55 | 0.55 | 평균 (한 클래스) | – | – |
| J4 | corrupted 35 · lqr_early 28 · stale 11 (+8) | 0.79 | 0.79 | 0.80 | 0.74 | 0.74 / 0.30 |
| J7 | faulted 46 · clean 28 (+8) | 0.75 | 0.77 | 0.70 | 0.74 | 0.62 / 0.62 |
| J8 | mpc_output_only 82 | 1.00 | 1.00 | 보고만 | – | – |
| J9 | latched 82 | 0.98 | 0.99 | 보고만 | – | – |
| J2 | <0.5 s 34 · >5 s/없음 33 · 2–5 s 6 · 0.5–2 s 6 | 0.48 | 0.48 | 0.56 | 0.75 | 0.46 / 0.15 |
| J5 | low 37 · moderate 24 · high 20 | 0.56 | 0.56 | 0.53 | 0.56 | 0.51 / 0.46 |
| J6 | within 38 · rate_limited 12 | 0.54 | 0.54 | 0.72 | 0.74 | 0.68 / 0.76 |

**Laya (multilingual, zero-shot), 15 상태 (고장 모드 고르게)** — 요청당 중앙값 73 s (CPU 8코어, 다른 작업과 경합), 입력 30k tok.

| 질문 | 판정자 정확도 | 판정자 답 |
|---|---|---|
| J1 | 0.43 | 대부분 immediate_check |
| J4 | 0.42 | lqr_early / stale_mpc |
| J7 | 0.16 | 대부분 discontinuous |
| J8 | 0.00 | 항상 also_reaches_lqr |
| J9 | 0.00 | 거의 항상 returns_to_mpc |
| J2 | 0.18 | 2_to_5s / over_5s |
| J5 | 0.22 | 거의 항상 moderate |
| J6 | 0.50 | 거의 항상 rate_limited |

**해석**

- Nemotron 은 코드 질문을 대체로 맞게 읽습니다 (J1 0.82, J4 0.79, J7 0.75, J8·J9 ≈ 1.0). 고장 모드만 보고 맞히는 기준선(0.74) 보다 조금 높고, 고장 모드 하나를 통째로 빼도 J1·J4 회귀는 0.78·0.74 로 유지됩니다 (다수 라벨 0.30). 즉 고장 모드 지문이 아니라 판정 내용이 쓰입니다.
- Nemotron 의 변형 판정들은 거의 같은 답을 냅니다 (한 번의 호출에서 25개를 같이 답하므로). 변형으로 얻는 다양성은 작습니다.
- 결과 질문(J2·J5·J6) 은 판정자 단독으로는 다수 라벨 수준 이하이고, 회귀도 고장 모드 조회 기준선을 넘지 못합니다. 이 값들은 시뮬레이션해야 알 수 있고 그 부분은 GP 가 실행 결과로 배웁니다.
- Laya zero-shot 은 이 질문들에서 쓸 수 없는 수준입니다 (J8·J9 처럼 코드에 답이 그대로 있는 질문도 0.0). README 대로 기본 체크포인트는 미세조정 없이는 거의 무작위입니다. 기본값은 키 없이 도는 laya 로 두되, 실제 캠페인에서는 `--judge-b nemotron` 을 권합니다. Laya 를 쓰려면 이 은행의 코드 정답 키와 결과 라벨로 미세조정이 필요합니다.

## DF-07~09 가 오프라인 GP 를 낫게 하는가 (result_8, safety_margin, LOO)

같은 GP 코드·같은 result_8 실행(전환되고 라벨 있는 50개, 안전 25개), Nemotron B b_cache, 타깃 safety_margin.
`/workspace/gpt/offline_gp_df.py df_off | df_on`.

| 설정 | GP 입력 열 | LOO 균형 정확도 | LOO AUC | 라운드 순서 (앞 라운드로 학습, n 42) 균형 정확도 / AUC |
|---|---|---|---|---|
| c03dd95 기준 (참고) | – | 0.76 | 0.859 | – |
| DF-07~09 끔 (이 브랜치) | 52 | 0.74 | 0.848 | 0.741 / 0.807 |
| DF-07~09 켬 | 58 | **0.80** | **0.893** | 0.716 / 0.793 |

- LOO 에서는 DF-07~09 가 0.74 → 0.80, AUC 0.848 → 0.893 으로 좋아집니다 (c03dd95 의 0.76 / 0.859 보다도 높음).
- 하지만 앞 라운드만으로 다음 라운드를 맞히는 순서 평가에서는 0.741 → 0.716 로 약간 나빠집니다. 표본이 50개라 한 두 런 차이이고, 개선이 확정적이라고 말할 수는 없습니다.
- DF 끔이 c03dd95 (0.76 / 0.859) 보다 조금 낮은 건, 이 브랜치에서 Nemotron B 의 판정 키가 바뀌어 (magnitude_rad 구간) result_8 의 예전 캐시 일부가 맞지 않기 때문으로 보입니다.

## 재현

```bash
python tools/judges_offline_eval.py --campaign result/result_8 --backend nemotron --out /tmp/judges_eval   # NEBIUS_API_KEY
laya-serve --checkpoint multilingual &                                                                     # 로컬 Laya
python tools/judges_offline_eval.py --campaign result/result_8 --backend laya --limit 15 --out /tmp/judges_eval
python run.py --mock --judge-b laya          # stub 판정자, 키 없음
```

---

<a id="english"></a>
# English

**What changed (v2).** The 05b judges no longer get numeric / computable questions. Fault mechanics that follow from
the scenario parameters are deterministic features DF-07..09 that go straight into the GP. The judges keep
code-reading questions (labelled by the profile-aware code answer key) and outcome questions (labelled by finished run
records; outcomes never enter the judge state).

**Bank (`pipeline/05b_judges/questions.yaml`).** Code: J1 detection path, J3 whether an intermittent / short fault
evades the hold condition (intermittent_bias and saturate only), J4 what steering reaches the car before the switch
(old Q-B1), J7 continuity of the LQR takeover, J8 whether the fault reaches the LQR path, J9 switch-back (old Q-B2).
Outcome: J2 detection-delay bucket (<0.5 / 0.5–2 / 2–5 / >5 s or never), J5 pre-detection lane-departure risk
(0.15 / 0.4 / 0.7 m), J6 whether the first LQR command is rate-limited. 25 judges = questions × variants (wording,
reversed options, Korean, code-section focus). J8 / J9 are single-class and report-only. GP inputs: per-option
regressed probabilities for code questions (no `unknown`), one expected-bucket column for outcome questions.
Judge cache key = fault mode | onset | parameter bins | speed bin (20 / 35 km/h) | route.

**DFs.** DF-07 first-tick steering jump / (90 °/s × 0.05 s); DF-08 fault on-time / supervisor hold time; DF-09
offset applied before detection / one LQR takeover step (60 °/s × 0.05 s), 0 when the tight rate check fires on the
first tick. Missing values get a `_missing` indicator.

**Backends.** `laya` (default, open-source, local, no key), `jev` (hosted, TYPESAFE_API_KEY), `nemotron` (Token Factory
Nemotron emulating System One via json_schema per-option probabilities, NEBIUS_API_KEY), `nemotron_b` (the earlier
generative Nemotron B and its Q-B bank, comparison only).

**result_8 offline evaluation.** Nemotron on 82 states: judge accuracy J1 0.82, J4 0.79, J7 0.75, J8 1.00, J9 0.98,
J3 0.55; outcome questions J2 0.48, J5 0.56, J6 0.54. Regression CV is close to the raw judge mean for code questions
(0.81 / 0.80 / 0.70) and holds under leave-one-fault-mode-out for J1 / J4 (0.78 / 0.74 vs a 0.30 majority), so the
judges add information beyond the fault mode. For outcome questions neither the judges nor the regression beat the
fault-mode lookup baseline (≈0.75 / 0.56 / 0.74). Laya zero-shot on 15 states is unusable (J8 / J9 0.0, J7 0.16) and
takes ~73 s per request on this CPU; fine-tuning on this bank would be needed. Recommended for real campaigns:
`--judge-b nemotron`.

**DF effect on the offline GP (result_8, safety_margin, same GP, 50 switched labelled runs).** LOO balanced accuracy /
AUC: c03dd95 0.76 / 0.859; this branch without DF-07..09 0.74 / 0.848; with DF-07..09 0.80 / 0.893. Round-ordered
(train on earlier rounds, n 42): 0.741 / 0.807 without vs 0.716 / 0.793 with. LOO improves, the forward-in-time check
does not; with 50 runs the difference is a few runs, so the gain is not established.
