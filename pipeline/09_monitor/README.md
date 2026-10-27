# 09 Monitor 설계 (v2)

09 Monitor는 08 짝 실행이 끝난 뒤 로그를 읽어 채점합니다.

**채점하는 런**
- fault: 고장 있음, Supervisor 있음
- baseline: 고장 없음
- no-reaction (선택): 같은 고장, Supervisor 없음

**확인하는 것**
1. 고장이 났는데 LQR로 **전환을 못 했거나 늦었다** → SR1 (F1)
2. **전환은 했는데** 그 순간이나 이후 **주행이 불안했다** → SR2 (F2), SR4·충돌 (F3)

| 파일 | 내용 |
|---|---|
| `safety_spec.yaml` | 안전 명세 SR1~SR6. 숫자와 근거 논문이 들어 있고, monitor가 직접 읽음 |
| `monitor.py` | 짝 하나를 채점해서 `run_record`를 만듦 |
| `feedback.py` | 여러 `run_record`로 2·3·6단계에 넘길 파일을 만듦 |
| `explain.py`, `video_clip.py` | 영상 자르기 + Cosmos Reason 영상 판단(모니터와 독립) + 비교 + 해설 |
| `dashboard.py` | 사람이 보는 화면 (한 페이지: 왼쪽 방금 돌린 런·영상·해설, 오른쪽 전체) |
| `av_stack_probe.py` | CARLA 실측값(의도된 차선 기준 오프셋, 차선 폭, 차체 크기, fault_effective)을 기록 |
| `error_table.yaml` | 출력 형식 정의 |
| `test_monitor.py`, `test_explain.py`, `test_probe.py` | 테스트 47개 통과 |

## 1. 원칙

- **읽기 전용, 사후 실행.** 제어 루프 밖에서 동작하고 런에 영향을 주지 않습니다.
- **정보 경계.**
  - 모니터는 시뮬레이터 실측값과 주입기 기록을 봅니다.
  - Supervisor는 둘 다 못 봅니다.
- **모르면 무효.** 차선 폭이나 차체 크기를 모르면 추정하지 않고 invalid로 처리합니다.
- **명세 동결.** 판정 런 전에 `safety_spec.yaml`을 커밋하고 `frozen_at_commit`을 기록합니다.

## 2. 안전 명세 (safety_spec.yaml)

| SR | 질문 | 재는 법 | 기준 | 근거 |
|---|---|---|---|---|
| SR1 전환 시간 | 늦지 않게 넘어갔나 | 첫 전환 − 실제 고장 시작 | ≤ 0.5 × FTTI | ISO 26262 FTTI, Jäger 2025, DriveFI |
| SR2 전환 매끄러움 | 전환 순간 조향이 튀었나 | 전환 −dt ~ +0.5 s 구간의 최대 조향 속도 (실제 송신값) | ≤ 60°/s | Zaccarian & Teel 2005, Autoware MPC |
| SR4 차선 유지 | 차체가 의도된 차선 안에 있었나 | 차체 네 모서리와 차선 경계 사이 최소 여유. 충돌도 포함 | ≥ 0 m | Kayotee, DriveFI |
| SR5 회복 | 차선 중심으로 돌아왔나 | ±0.3 m 안에 1 s 머물기까지 걸린 시간 | ≤ 3 s, 참고용 | 설계값 (출처 없음) |
| SR6 액추에이터 한계 | 조향각과 조향 속도가 한계 안이었나 | 최대 \|조향\|, 최대 속도 | 차량 한계, 90°/s | 안전 필터 값 |

### SR1의 FTTI

- FTTI는 no-reaction 런에서 잽니다. "아무도 대응하지 않으면 고장 후 몇 초 만에 차선을 벗어나는가"입니다.
- 그 시간의 절반 안에 전환해야 합니다. 나머지 절반은 LQR이 회복할 시간입니다.
- no-reaction 런이 끝까지 차선 안에 있으면 FTTI = ∞이고 SR1은 적용하지 않습니다.
  - 이 경우 전환을 안 했어도 `pass`이고, note에 `undetected_harmless`를 붙입니다.
  - 리뷰에서 나온 "미감지 bias는 차선 안에 머문다" 사례를 실패로 잘못 세지 않기 위한 처리입니다.
- no-reaction 런이 없으면 고정 한계 0.5 s를 씁니다 (`limit_source: fixed_fallback`).

### 결과 분류 (위에서부터 먼저 걸리는 것)

| 분류 | 조건 |
|---|---|
| `invalid` | 유효성 검사 실패 (V-01~09, G-04) |
| `no_trigger` | 고장이 MPC 출력을 한 번도 바꾸지 않음 |
| `F1_no_or_late_switch` | SR1 위반. `unsafe` 플래그로 실제 차선 이탈 여부를 따로 남김 |
| `F2_switch_transient` | 전환했는데 SR2 위반 |
| `F3_not_recovered` | SR4 위반 또는 충돌 |
| `pass` | 위 어디에도 해당 없음 |

## 3. 모니터가 넘기는 것

```
08 짝 실행 → monitor.py (짝마다 run_record 1줄) → run_records.jsonl
                                                    │ feedback.py
            ┌───────────────────────────────────────┼──────────────────────────┐
            ▼                                       ▼                          ▼
 02 hypothesis_tracking.json             03 failure_table.json         06 gp_training.jsonl
 가설이 맞았나                            어디서 실패했나                  얼마나 안전했나 (숫자 하나)
```

### 러너와 맞춘 형식 (`scripts/runner_skeleton.py`, `new_pipeline` 브랜치)

`monitor.evaluate_pair()`는 러너가 받는 `run_record`의 **28개 필드만** 그대로 내보냅니다. 러너의 `build_run_record()`는 모르는 필드가 있으면 에러를 내기 때문입니다.

| 필드 | 내용 |
|---|---|
| `outcome_category` | 러너 5분류: no_trigger / triggered_no_switch / switched_safe / switched_unsafe / simulation_invalid |
| `P01` `P02` `P03` | 차선 여유(차체 기준) / 횡가속도 여유 / 충돌 없음(true = 충돌 없음) |
| `metrics`, `baseline_metrics` | 팀원 02 정의 지표 11개 (아래 표). 계산할 수 없으면 null |
| `metrics.monitor` | **우리 SR1~SR6 판정 전체** (F1~F3 실패 유형, SR별 점수와 원래 값, 실행 폴더 경로) |
| `gp_targets` | 지표별 z (`z = 부호 × y / scale`). 러너의 `gp_targets()`와 같은 계산 (`--patterns`로 규칙을 줄 때) |
| `parameters` | 실행 정보 + `environment`(A) + `fault_plan`(B) + `arm`(조건군/대조군) |
| `switch_state_label` | null. 회복 가능 영역 판정 자료(P 행렬, reach table)가 아직 없음. 전환 시점 상태는 `metrics.monitor.switch_state`에 남김 |

| 팀 지표 | 계산 |
|---|---|
| `recovery_robustness` | 전환 후 5 s 동안 4개 제약 대비 최소 비율: 횡오차, 조향각, 조향 속도(90°/s), 횡가속도(4.41 m/s²). 전환이 없으면 null |
| `recovery_success` | 위 값 ≥ 0. 전환을 안 했으면 false |
| `pre_switch_max_lateral_error_m` | 고장 시작부터 전환까지의 최대 \|e_y\| |
| `lateral_error_rmse_m` | 평가 창 안의 e_y RMS |
| `switch_count` | 평가 창 안의 전환 횟수 |
| `command_jump_rad` | \|u(전환) − u(전환 직전)\| |
| `post_switch_saturation_ratio` | 전환 후 1 s 동안 조향각·조향 속도가 한계에 닿은 비율 |
| `max_steering_rate_dps`, `margin_delta_m` | 최대 조향 속도, 고장 런 − baseline 차선 여유 |
| `deadline_miss_rate` | `steps.jsonl`의 `solve_ms` > `--deadline-ms`인 비율. 한계를 안 주면 null |
| `mpc_constraint_violation_rate` | MPC가 제약 위반 플래그를 기록해야 계산 가능. 지금은 null |

**두 판정의 차이**
- 러너 5분류는 차선, 횡가속도, 충돌만 봅니다.
- 그래서 전환 순간 조향이 180°/s로 튀어도 차선 안에 있으면 `switched_safe`로 나옵니다.
- 우리 판정(`metrics.monitor.failure_type`)은 같은 런을 F2로 잡습니다.
- 화면에는 두 판정이 함께 보입니다.

**검증:** `new_pipeline` 브랜치의 실제 러너에서 다음을 확인했습니다.
- `build_run_record()`가 우리 레코드를 그대로 받음
- `gp_targets()` 결과가 우리 값과 같음
- `--check`에서 P-02 빈칸이 사라짐
- 러너 self-test 통과

### 02 가설 → `hypothesis_tracking.json`

- **판정 규칙:** 팀원 `decision_protocol`을 따릅니다.
  - 가설마다 `prediction.metric`(예: H-01 `recovery_success`)을 `metrics`에서 읽습니다.
  - `metric_rules`의 `minimum_effect`, `support_direction`을 씁니다. `non_inferior`(H-07)도 지원합니다.
- **조건군/대조군:** `parameters.arm`과 `hypothesis_ids`로 구분합니다.
- **신뢰구간:** bootstrap 99%로 구합니다.
- **판정 시점:** 각 군이 10, 20, 30회가 될 때마다 판정합니다.
- **제외:** `simulation_invalid` 런과 지표가 null인 런은 뺍니다.

### 03 Nemotron A (+ B) → `failure_table.json`

- **후보:** F1, F2, F3로 실패한 시나리오입니다.
- **재현 확인:** 같은 `scenario_id`가 2번 이상 실패해야 `verified`로 들어갑니다.
- **정렬과 선택:** gp_target이 낮은(나쁜) 순서로 정렬하고, (고장 종류, 시점) 조합당 1개씩, 최대 5개를 고릅니다.
- **각 행의 내용:**
  - 환경 (A가 정한 경로와 ODD)
  - 고장 계획 (B가 정한 종류, 세기, 지속 시간, 시점)
  - 어긴 SR과 그 정도 (`how_much`, 예: 지연 1.2 s / 한계 0.4 s)
  - 실제 전환 행동
  - 재현 횟수
  - 한 줄 요약 (`summary`, few-shot에 그대로 쓰기 좋음)
- **`candidates`:** 1번만 실패한 것입니다. 다시 돌려서 확인할 후보입니다.

### 06 GP → `gp_training.jsonl`

러너 `surrogate.yaml`의 `feedback_in` 형식을 따릅니다.
- **x:** `env.*`(A) + `fault.*`(B) + `param.*` + `reason.*`
- **y:** `gp_targets`. 팀 지표마다 z가 하나씩 있고, 지표마다 GP를 하나씩 둡니다.
- **함께 보내는 것:** `metrics`와 `baseline_metrics`
- **null 처리:** 값이 없는 지표(예: 전환 없음 → `recovery_robustness`)는 아예 빠집니다. 0으로 채우지 않습니다.
- **우리 SR 점수:** `sr_scores`로 같이 보내지만 GP 타깃이 아닙니다. 판정, 실패 목록, 화면에서 씁니다.

### 판정 → 영상 판단 → 해설 → 화면

**역할 나누기**
- **모니터 (숫자):** 통과/실패, SR1~SR6, 지연, 차선 여유, 충돌, 팀 지표를 시뮬레이터 실측값으로 판정합니다.
- **Cosmos Reason (영상):** 숫자로 못 하는 판단을 맡고, 해설도 씁니다.

```
monitor.py  (숫자 판정)        → run_records.jsonl
explain.py  1. 영상만 보고 판단  (모델, 모니터 판정은 안 알려줌)
            2. 비교              (코드) 모니터 판정 vs 영상 위험도 → 일치 / 불일치 → 사람 확인 목록
            3. 해설              (모델) 모니터 수치 + 자기 판단을 보고 한국어 설명
                                → screen/clips/<run>.mp4, screen/commentary.jsonl
dashboard.py (화면)             → screen/campaign.html
feedback.py  (3단계)            → failure_table.json에 영상 판단 반영 (--judgments screen/commentary.jsonl)
```

**1. 영상만 보고 판단 (blind)**

모델에게 모니터의 판정과 수치는 주지 않습니다. 고장 시작 시각과 프레임 8장만 줍니다. 그래서 모니터와 독립적인 두 번째 판정이 됩니다. 정해진 단어로만 답하게 하고, 목록에 없는 단어는 "불분명"으로 바꿉니다.

| 항목 | 값 | 모니터가 못 하는 이유 |
|---|---|---|
| 차선 행동 | 중앙 유지 / 차선 안에서 밀림 / 차선에 닿음 / 차선 넘음 / 도로 이탈 | (모니터도 잼 → 교차 확인용) |
| 움직임 (실패 양상) | 부드러움 / 서서히 밀림 / 좌우로 흔들림 / 급조향 / 전환 후 과잉 보정 / 정지 | 같은 F1이라도 어떻게 실패했는지는 숫자 하나로 안 나옴 |
| 주변 | 맞은편 차, 차량, 보행자, 자전거, 도로 끝, 방호벽, 주차 차량 + 방향 + 거리(멀리/가까이/아주 가까이) | 모니터는 차선 기하와 충돌만 봄. 넘어간 쪽에 뭐가 있었는지 모름 |
| 아슬아슬한 근접 | 있음/없음 + 설명 | 충돌이 아니면 모니터는 안 잡음 |
| 위험도 0~3 | 0 위험 없음 / 1 경미 / 2 위험(실제 도로면 운전자 개입 필요) / 3 충돌·직전 | 같은 0.5 m 이탈이라도 옆이 빈 차선인지 맞은편 차인지에 따라 다름 |
| 전환 표시 | 화면의 STEERING: MPC/LQR 글자가 바뀐 프레임 (`view.mp4`일 때만) | 로그와 영상의 교차 확인 |

**2. 비교 (코드가 함)**

| 모니터 | 영상 위험도 | 결과 | 쓰임 |
|---|---|---|---|
| 실패 | 2~3 | 일치 · 둘 다 위험 | |
| 실패 | 1 | 대체로 일치 · 경미 | |
| 실패 | 0 | **불일치** · 영상상 위험 안 보임 | 기준이 너무 엄하거나 측정 문제 → 사람 확인 |
| 통과 | 2~3 | **불일치** · 모니터가 놓쳤을 수 있음 | 모니터 사각지대(근접 등) → 사람 확인 |
| 통과 | 0~1 | 일치 · 둘 다 안전 | |

전환 표시 프레임이 로그의 전환 시각과 다르면 "로그 확인"을 띄웁니다.

**3. 해설:** 두 번째 호출에서 모니터 수치와 영상 판단을 함께 주고 해설을 받습니다. 받는 내용은 요약, 시간순 사건, 왜 이 결과인지, 그리고 같은 결과라도 이 장면이 더 또는 덜 위험한 이유입니다.

**어디에 쓰이나**
- **실패 Top 5 (3단계 → Nemotron A·B):**
  - 순위: 영상 위험도가 높은 순서로 먼저 정렬하고, 같으면 모니터 점수가 나쁜 순서로 정렬합니다.
  - 각 줄에 `vision`(위험도, 차선, 움직임, 주변, 근접)이 붙고, 요약 문장에 "영상: 차선 넘음, 서서히 밀림, 왼쪽 맞은편 차량 아주 가까이, 위험도 3/3"이 들어갑니다.
- **`needs_review`:** 불일치 런은 이 목록으로 따로 모입니다.
- **화면:**
  - 왼쪽: 영상, AI 판단(위험도·차선·움직임·주변·근접·비교 결과), AI 해설
  - 오른쪽: AI 위험도 열과 불일치 ⚠, 위쪽 개수에 "AI 불일치"
- **GP 학습:** 영상 판단은 GP 학습(6단계)에는 넣지 않습니다. GP는 숫자 지표만 씁니다.

**대상과 비용**
- 기본으로 통과 런까지 모든 런을 봅니다. 모니터가 놓친 것은 통과 런 안에 있기 때문입니다.
- 런마다 모델 호출이 2번 생깁니다 (프레임 8장 × 2).
- `--failures-only`로 실패만 볼 수 있지만, 그러면 "모니터가 놓쳤을 수 있음"은 못 찾습니다.

| 환경 변수 | 뜻 |
|---|---|
| `NEBIUS_TOKEN_FACTORY_KEY` | Token Factory API 키 (`v1.`로 시작) |
| `NEBIUS_TOKEN_FACTORY_BASE_URL` | 기본 `https://api.tokenfactory.nebius.com/v1/`. 직접 띄운 vLLM이면 그 주소 |
| `COSMOS_MODEL` | 모델 id. 공개 카탈로그의 Cosmos3 추론 모델이 2026년 8월에 내려갔기 때문에 코드에 고정하지 않음. `python explain.py --check`로 확인 |

**영상 (`video_clip.py`)**
- **원본:** av_stack `run_benchmark.py --record-front`로 녹화한 영상입니다.
  - `view.mp4`: 추격 카메라, 왼쪽 위에 MPC/LQR 표시
  - `front.mp4`: 지붕 카메라
- **시각 맞추기:** `*_frames.jsonl`의 시뮬레이션 시각으로 영상과 모니터 시각을 맞춥니다.
  - `front.mp4`에는 `front_frames.jsonl`이 이미 있습니다.
  - `view.mp4`는 아직 프레임 시각 파일이 없어서 시작 시각을 0으로 가정합니다 (화면에 "시각은 근사"로 표시).
  - av_stack `ChaseView.show()`에서 프레임마다 `{"t": 시각}` 한 줄을 `view_frames.jsonl`에 쓰면 정확해집니다.
- **형식:** 클립은 브라우저에서 바로 재생되는 H.264 mp4입니다.
- **ffmpeg:** PATH에 있는 것을 쓰고, 없으면 `pip install imageio-ffmpeg`에 포함된 것을 씁니다.

### 사람이 보는 화면 → `dashboard.py`

브라우저로 보는 HTML 한 페이지(`campaign.html`)입니다. 서버나 인터넷 없이 동작하고, 5초마다 스스로 새로고침합니다. 영상을 재생하는 동안에는 새로고침하지 않습니다.

| 위치 | 내용 |
|---|---|
| 왼쪽 | **방금 돌린 시나리오**: 결과 한 줄 설명, (실패면) 실패 구간 영상과 AI 해설, 고장 종류·세기·시점, 전환 성공 여부(지연 / 한계), 전환 후 운행 안정성 목록 |
| 왼쪽 아래 | **가설 판정**: 가설마다 내용, 조건군/대조군 런 수, 효과와 99% 구간, 판정 |
| 오른쪽 | **지금까지 돌린 시나리오 전부**: 결과별 개수, 런마다 한 줄(고장, 시점, 전환, 운행 안정성, 가설, 결과, 영상 여부). 줄을 누르면 왼쪽에 그 런이 뜸 |

```bash
python pipeline/09_monitor/explain.py --check                      # 키와 모델 확인
# 런마다: 판정 → 해설·화면
python pipeline/09_monitor/monitor.py ... --out results/run_records.jsonl
python pipeline/09_monitor/explain.py --records results/run_records.jsonl --screen results/screen \
       --hypotheses pipeline/02_hypotheses/hypotheses.yaml --patterns pipeline/02_hypotheses/vulnerability_patterns.yaml
# 또는 캠페인 내내 켜두기 (새 판정이 생길 때마다 영상·해설·화면 갱신)
python pipeline/09_monitor/explain.py --records results/run_records.jsonl --screen results/screen --open --watch
```

## 4. 유효성 검사

| id | 무효 조건 |
|---|---|
| V-01 | 로그가 비었거나, 길이가 다르거나, NaN이 있음 |
| V-02 | 틱 간격 이상 (프레임 누락) |
| V-03 | 평가 창이 5 s 미만 (SUT가 스스로 안전 정지한 경우는 예외) |
| V-04 | 평가 창 안에서 기준 경로가 바뀜 |
| V-05 | 차선 폭이나 차체 크기를 모름 |
| V-06 | 실제 고장 시작이 계획보다 1틱 넘게 다름 |
| V-07 | baseline에 고장이 켜져 있음 |
| V-08 | fault와 baseline의 평가 창 길이가 다름 |
| V-09 | fault 런에 주입기 로그가 없음 |
| V-10 | fault 런이 고장 시작 전에 이미 전환됨. 전환을 채점할 수 없음 |
| V-11 | 충돌이 시각 없이 기록됨. 평가 창 안인지 알 수 없으므로 버리지 않고 무효 처리 |

**충돌이 나면**
- SR4 실패로 처리합니다. SR4 점수는 −2(최저)이고 `unsafe` 플래그가 붙습니다.
- 결과 분류는 전환이 없거나 늦었으면 F1, 제때 전환했으면 F3입니다.
- 충돌로 런이 일찍 끝나도 무효(V-03, V-08)가 아닙니다. 충돌 시점까지로 채점합니다.
- `measures`에 충돌 시각과 상대 물체를 남깁니다.
| G-04 | baseline이 이미 SR4를 위반함. 고장 탓이 아니므로 짝 전체를 제외 |

## 5. 실행

```bash
python pipeline/09_monitor/test_monitor.py      # 40개
python pipeline/09_monitor/test_explain.py      # 6개 (ffmpeg 필요, 가짜 API 서버로 시험)
python pipeline/09_monitor/test_probe.py        # 1개

# 짝 하나 채점 → run_records.jsonl에 1줄 추가
python pipeline/09_monitor/monitor.py --fault-run runs/f1 --baseline-run runs/b1 \
       --no-reaction-run runs/n1 --onset 25 --run-info runs/f1/run_info.json --out results/run_records.jsonl \
       --patterns pipeline/02_hypotheses/vulnerability_patterns.yaml [--deadline-ms 50]

# 실패 영상 + AI 해설 + 화면 (위 "판정 → 해설 → 화면" 참고)
python pipeline/09_monitor/explain.py --records results/run_records.jsonl --screen results/screen

# 2·3·6단계 파일 생성
python pipeline/09_monitor/feedback.py --records results/run_records.jsonl --out results/feedback \
       --hypotheses pipeline/02_hypotheses/hypotheses.yaml --patterns pipeline/02_hypotheses/vulnerability_patterns.yaml \
       --judgments results/screen/commentary.jsonl
```

`run_info.json` 예시:

```json
{"run_id": "r-0012", "scenario_id": "S-07", "round": 3, "selector": "hypothesis",
 "environment": {"route": "Town04_highway", "speed_mps": 20},
 "fault_plan": {"fault_type": "bias", "magnitude": 0.08, "duration_s": 2.0, "onset_event": "curve_entry"},
 "hypothesis_ids": ["H-02"], "arm": "condition"}
```

## 6. av_stack에 연결할 때 필요한 것 (PR #1 기준)

1. **프로브 호출:** 루프 안의 `world.tick()` 직전에 `MonitorProbe.record(t, fault_effective)`를 호출합니다.
2. **`fault_effective`:** 주입기가 "이번 틱에 MPC 출력을 실제로 바꿨는지"를 돌려주게 합니다.
3. **충돌 시각:** `collisions.json`의 각 이벤트에 `t`와 `other_actor`를 기록합니다.
   - 지금은 frame 번호만 남기 때문에 이대로면 V-11로 무효가 됩니다.
4. **조향 기록:** `behavior.jsonl`의 `steer_rad`에는 안전 필터를 거쳐 실제로 보낸 값을 남깁니다 (SR2·SR6용).
5. **고정 경로:** 판정 런에서는 재계획을 끕니다. 그렇지 않으면 V-04로 무효가 됩니다.
6. **no-reaction 런:** 같은 고장으로 `--controller mpc_unprotected`를 한 번 더 돌려 FTTI를 잽니다.
   - 시나리오마다 1번이면 충분합니다.

## 7. 남은 결정

- **설계값 숫자:** 출처가 없는 값들입니다. 동결 전에 팀과 확정합니다.
  - SR1의 FTTI 비율 0.5
  - SR5 전체 (현재 참고용)
  - `switch_count`의 delta_min 1.0
- **GP 구성:** 기준마다 GP를 따로 둘지(SR1, SR2, SR4), `overall` 하나만 쓸지 06 담당과 정해야 합니다.
  - 데이터는 두 방식 모두에 맞게 보냅니다.
- **failure_table 받는 곳:** A만 받을지, B도 받을지 정해야 합니다. 현재 행에 두 쪽 정보가 다 들어 있습니다.
- **B의 이유 항목:** B가 `reason_items`를 낼지 정해야 합니다. 내면 GP 입력 `reason.*`으로 자동으로 들어갑니다.
- **`runner_skeleton.py`:** `robustness_p01`과 `classify_outcome`을 `monitor.evaluate_pair` 호출로 바꿉니다.
