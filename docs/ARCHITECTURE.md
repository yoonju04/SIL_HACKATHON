# Architecture & Interfaces (제안)

## 계층 구분

1. **Intelligence plane:** Nemotron 모델을 사용해 테스트 후보를 추천한다. LLM 출력은 신뢰할 수 없는 제안으로 취급한다. 05b 코드 판정은 System One 형 판정 모델(기본 오픈소스 Laya, 호환 백엔드 TypeSafe Jev, Nemotron 에뮬레이션)에 J1~J9 판정 25개를 한 요청으로 묻고, 질문별 회귀로 확률을 모은다 ([JUDGES.md](JUDGES.md)). 계산되는 값은 묻지 않고 DF-07~09 로 계산하며, 결과 질문(J2·J5·J6)은 끝난 실행의 결과를 회귀 라벨로만 쓴다. 판정자는 글을 만들지 않고 정해진 보기의 확률만 내므로 인용·환각 검사 대신 정답 키 일치와 보정(Brier, ECE)으로 검사한다.
2. **Control plane:** 시나리오 스키마 검증, 실험 ID, queue, 워커 할당, 재시도, 상태 추적, 비용 예산 처리.
3. **Simulation plane:** SIL 우선, CARLA 추후 도입. `world state → MPC (or LQR) → steering command → steering actuator model → plant → state` 구조. 조향 액추에이터는 응답 지연(1차 지연) · 조향 속도 제한 · 조향각 포화를 가진 단순 모델로 두며, 고장 주입 대상이 아니라 plant의 일부다.
4. **Evaluation plane:** 로그로부터 기준 위반 판정 및 시각화. 동일 로그·동일 기준이면 동일 결과가 나와야 한다.

## Fault injection 위치

```mermaid
flowchart LR
  Plant[Vehicle Dynamics] --> Obs[State Estimate]
  Obs --> MPC[MPC Primary]
  Obs --> LQR[LQR Backup]
  MPC --> FI[Fault Injector]
  FI --> Supervisor[Health Monitor + Supervisor]
  LQR --> Supervisor
  Supervisor --> Actuator[Steering Actuator Model]
  Actuator --> Plant
  FI -.-> Events[Timestamped Fault Events]
  Supervisor -.-> Events
```

Fault injector는 제어기 실행 중단, 출력값 왜곡, output stale, 계산 deadline miss를 모델링한다. 고장은 **MPC에만** 주입하며 센서 · 액추에이터 고장은 다루지 않는다. Supervisor는 LQR이 제어권을 넘겨받는 순간의 차량 상태(`state_at_switch`)를 기록한다. `Supervisor`는 정해진 fault-detection policy에 따라 실제 applied command source를 전환한다. **오류를 감지하는 조건이 테스트에서 변경되면 전환 지연과 실패 라벨도 함께 기록**한다.

## 최소 데이터 계약

`Scenario`: `scenario_id`, `seed`, `initial_speed_mps`, `road_curvature_inv_m`, `fault.type`, `fault.start_s`, `fault.duration_s`, `fault.magnitude`, `detection_delay_s`, `actuator.time_constant_s`, `actuator.rate_limit_rad_s`, `actuator.max_angle_rad`, `supervisor.detection_threshold`, `supervisor.debounce_s`, `controller_config_hash`, `simulator_version`.

`StepLog`: `time_s`, `x_m`, `y_m`, `yaw_rad`, `speed_mps`, `reference_path`, `lateral_error_m`, `primary_command_rad`, `backup_command_rad`, `applied_steering_rad`, `active_controller`, `fault_active`, `fault_detected`, `switch_event`.

`Evaluation`: `scenario_id`, `run_id`, `failure_labels`, `max_lateral_error_m`, `lane_boundary_violation`, `max_steering_jump_rad`, `detection_latency_s`, `state_at_switch` {`yaw_rate_rad_s`, `lat_accel_mps2`, `sideslip_rad`, `steering_rad`, `steering_rate_rad_s`, `lateral_error_m`, `heading_error_rad`, `mpc_lqr_command_gap_rad`}, `handover_latency_s`, `recovery_time_s`, `seed`, `git_commit`, `criteria_version`.

## 권장 프로세스 격리

- 실제 시스템과 달리 초기 MVP에서는 두 제어기를 동일 프로세스에서 실행해도 된다. 이 경우 **process crash / ECU power loss에 대한 독립 복구를 검증할 수 없다**.
- 물리적/계산적 격리를 테스트하려면 분리 프로세스·ECU·전원·통신 모델을 명시적으로 추가해야 한다.
- Timeout 판정은 simulation time vs wall clock time을 구분한다. 고정 시뮬레이션 주기에서 인위적인 latency fault를 모델링하는 경우 hardware deadline guarantee라고 주장하면 안 된다.
- CARLA를 사용할 때 synchronous mode, fixed delta, seed, map/weather/vehicle versions을 기록하고 각 CARLA world는 단일 tick owner를 둔다.

## 05b 판정자 인터페이스 (laya-judge)

```mermaid
flowchart LR
  BANK["05b_judges/questions.yaml<br/>질문 은행 J1~J9 + 프로필별 정답 키 / 결과 라벨 규칙"] --> GEN["judges.judge_questions<br/>질문 25개 (변형)"]
  CODE["av_stack 코드 발췌<br/>(Laya: 감독기·해당 고장 줄만)"] --> ST["state<br/>scenario + supervisor_profile + code"]
  CTX["시나리오 입력<br/>고장 모드·파라미터·시작 이벤트·속도·도로·마찰"] --> ST
  GEN --> API["POST /v1/systemone<br/>laya-serve (로컬) | api.typesafe.ai (Jev) | Token Factory (Nemotron 에뮬레이션)"]
  ST --> API
  API --> PARSE["parse_answers<br/>질문별 판정자 확률"]
  PARSE --> REG["JudgeRegression<br/>L2 로지스틱, k-fold CV, 라벨 적으면 평균"]
  BANK -. "코드 정답 키 (J1 J3 J4 J7)" .-> REG
  RUNS["끝난 실행 기록"] -. "결과 라벨 (J2 J5 J6)" .-> REG
  REG --> GPF["GP feature<br/>코드 질문 보기별 확률 + 결과 질문 기대 구간, 상수 열 제외<br/>(J8·J9 는 보고만)"]
  DF["DF-07~09 (계산, 판정자 아님)"] --> GPF2["GP 입력"]
  GPF --> GPF2
  REG --> CHK["agreement · Brier · ECE · low_confidence"]
```

- 요청 1회 = state 1개 + 판정 25개 (J3 은 intermittent_bias·saturate 에만, 다른 모드는 22개). 응답 `answers[name]` 은 `{type: choice, choice, probabilities, confidence}` 또는 `{type: noul, noul}`.
- state 에는 정답 키·런 결과·GP 예측이 들어가지 않는다 (`judges.FORBIDDEN_STATE_KEYS` 로 검사). 결과는 회귀 라벨로만 쓴다.
- 판정 캐시 키 = 고장 모드 | 시작 이벤트 | 질문 파라미터 구간 | 속도 구간 (20 / 35 km/h) | 경로.
- 서버가 없거나 키가 없으면 결정론적 stub (SYNTHETIC) 으로 돌고 run_config.json `judges.client` 에 그 사실이 남는다.

## 클라우드 적용

- **Token Factory:** `GET /v1/models`로 계정에서 사용 가능한 Nemotron 모델을 확인한 뒤 API를 호출한다. 인증키는 환경 변수/secret manager로만 제공한다.
- **Nebius AI Cloud:** 설치된 이미지와 리소스가 실험에 적합한지 확인 후 worker 컨테이너를 실행한다. 특정 서비스의 GPU 수·시간·지원 기능을 미리 보장하지 않는다.
- **선택적 AWS:** 과거 플랫폼 구상과 호환되도록 S3/object storage, SQS/queue, DB/metadata store로 대체 가능하다. 초기 구현의 필수 의존성은 아니다.
- **Reproducibility:** scenario hash, run ID, random seed, container digest, controller commit, result checksum 기록; job retries는 idempotent writes 사용.

## 예시 API (확정 인터페이스 아님)

```text
POST /experiments        # budget, baselines, safety criteria
POST /scenarios/validate # schema + range checks
POST /runs                # validated scenario, controller builds
GET  /runs/{id}          # logs, events, metrics, replay
GET  /experiments/{id}   # cumulative failure discovery summary
```

실제 배포 전 OpenAPI 스키마, 인증, resource quotas, input sandboxing을 설계해야 한다.
