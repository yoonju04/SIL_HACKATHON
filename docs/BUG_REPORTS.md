# Bug reports

실제 실행(`run.py --real`)에서 발견한 버그와 수정 기록.

## BUG-001: CARLA 실행 결과가 엉뚱한 폴더에 저장되어 루프가 멈춤

- **발견:** 2026-10-27 23:42, `run.py --real --cold-start 2 --rounds 0 --out results/real_smoke_1028` (integration2 59786c6)
- **증상:** 08 단계의 첫 baseline 실행 직후 루프 종료.
  ```
  FileNotFoundError: [Errno 2] No such file or directory: 'results\real_smoke_1028\runs\baselines\f6c6c7b44a67\.complete'
  ```
- **원인:**
  - `engine/carla_runner.py` `RealRunner._run`이 `run_benchmark.py`에 결과 폴더를 **상대경로**(`--out results\...`)로 넘깁니다.
  - 그런데 하위 프로세스는 `cwd=av_stack`으로 실행됩니다.
  - 그래서 결과가 `pipeline/08_carla_paired_run/av_stack/results/...`에 저장됐고, 엔진은 저장소 루트의 `results/...`에서 찾다가 실패했습니다.
  - CARLA 주행 자체는 정상 종료(returncode 0, 영상·모니터 로그 생성).
- **영향:** `--out`을 상대경로로 준 모든 `--real` 실행. `--mock`은 하위 프로세스를 쓰지 않아 영향 없음.
- **수정:** `_run`에서 `out = Path(out).resolve()`로 절대경로를 넘깁니다.
- **확인:** 2026-10-27 23:49 같은 명령 재실행(bbb808f). 기준 실행 2개, 고장 실행 2개, Cosmos 해설 1개까지 끝까지 완료
  (`results/real_smoke_1028`, 실행 1회 약 35~45 s).

## BUG-002: 감독기가 못 잡은 고장이 `no_trigger`(고장 효과 없음)로 기록됨

- **발견:** 2026-10-27 23:49, `results/real_smoke_1028` 런 `R00-0001-output_delay` (integration2 bbb808f)
- **증상:** 출력 지연 4틱 고장이 13.8 s부터 끝까지(약 3 s, 60틱) 주입됐는데 결과가 `outcome_category = no_trigger`,
  `trigger_time = None`. 감독기가 전환하지 않은 경우인데 "고장이 아무 영향도 주지 않음"으로 분류됨.
- **원인:**
  - 러너 계약(`scripts/runner_skeleton.py` `classify_outcome`)에서 `triggered`는 "고장이 MPC 출력을 실제로 바꿨나"이고,
    전환 여부는 `switched`로 따로 본다. `no_trigger` = 고장이 효과 없음, `triggered_no_switch` = 효과는 있었는데 전환 못 함.
  - `engine/monitor.py compute_events`는 `trigger_time`을 **감독기가 이상을 감지한 시각**(`supervisor_reason`이 처음 찍힌 틱)으로
    잡는다. 그래서 감지 실패 = 전환 없음 = trigger 없음 = `no_trigger`가 된다.
  - av_stack은 "이번 틱에 고장이 출력을 바꿨나"(`fault_effective`)를 기록하지 않는다 (`monitor_probe.jsonl`에 항상 null).
- **영향:** 감지를 피하는 고장(bias, drift, 간헐 bias, 출력 지연)이 전환에 실패해도 `triggered_no_switch`가 아니라 `no_trigger`로
  기록된다. 이 프로젝트가 찾으려는 "전환 실패"가 결과표에서 사라지고, GP·가설 판정에도 잘못 들어간다.
- **수정:**
  1. av_stack: `MPCFaultInjector`가 틱마다 출력을 실제로 바꿨는지(`effective`)를 남기고, `SupervisedLateral` /
     `UnprotectedMPC`가 그 틱의 값을 `fault_effective`로 내보내며, `monitor_logs`가 `behavior.jsonl`·`monitor_probe.jsonl`에 기록한다.
  2. engine: `trigger_time` = 첫 `fault_effective` 틱. `fault_effective`가 없는 옛 로그는 실제 주입 시각(`fault_injected`)으로 대신한다.
     감독기 감지 시각은 `detection_time`, 이유는 지금처럼 `trigger_source`에 남긴다.
- **확인:**
  - 단위 테스트: `test_supervisor.py`(주입기 `effective`: bias, drift 시작 틱, 간헐 bias 꺼짐 구간), `test_monitor_logs.py`(두 파일 기록),
    `tests/test_engine_monitor.py::test_bug002_undetected_fault_is_triggered_not_no_trigger`(기록 있음 / 효과 없음 / 옛 로그).
  - `real_smoke_1028` 재채점: `R00-0001-output_delay` `no_trigger` → **`triggered_no_switch`** (trigger 13.8 s, 전환 없음),
    `R00-0000-saturate`는 `switched_unsafe` 그대로 (trigger 11.8 s, 감지·전환 12.45 s).
  - 실제 재실행(2026-10-28 00:05, `results/real_smoke_1028_b`, 2b03cd3): av_stack이 `fault_effective`를 기록하고
    (최대각 고정 0.5 s = 10틱 true), 출력 지연 런이 **`triggered_no_switch`**(trigger 13.8 s)로 나옴.
  - 남은 표시 문제: `run.py` 진행 줄의 `trigger=`는 `trigger_source`(감독기 감지 이유)라서 감지 못 한 런은 `trigger=None`으로 찍힘.

## BUG-003: 경로 끝 직전에 고장을 넣어 평가 구간이 모자람 (V-03 무효)

- **발견:** 2026-10-28 00:05, `results/real_smoke_1028_b` 런 `R00-0001-output_delay`
- **증상:** 경로 26956(끝 16.75 s)에서 주입 시각 13.76 s(`curve_exit` - 0.19 s). 평가 구간이 3 s뿐이라 SR 모니터가 V-03(최소 5 s 미만)으로 무효 처리.
- **원인:** 04 타당성 게이트가 "주입 뒤 평가 구간(최소 `evaluation_window.min_length_s`)이 경로 안에 남는가"를 확인하지 않음.
  `onset_event`가 경로에 있는지만 본다(G-01).
- **수정:** 08 단계에서 기준 실행이 끝나 주입 시각이 정해지면, 고장 실행 전에 SR 모니터와 같은 `check_run`으로 기준 실행을 검사
  (`run.py Loop.baseline_gate`). 구간이 모자라면 고장 실행을 하지 않고 `V-03_reject`로 기록한 뒤, 후보 풀의 다음 후보로 바꿔 돌린다.

## BUG-004: 고장 없는 기준 실행부터 충돌하는 조건에서 고장 실행까지 돌림 (G-04 무효)

- **발견:** 2026-10-27 23:49 / 2026-10-28 00:05, 경로 26956의 기준 실행 두 개 (최고 속도 30, 35 km/h)
- **증상:** 고장 실행 결과가 SR 모니터에서 G-04("baseline already breaks SR4")로 무효.
- **원인:** 처음에는 차체 모서리 차선 여유 계산 문제로 봤지만 아니었다. 기준 실행의 차선 여유는 끝까지 양수(최소 0.11~0.19 m)였다.
  실제 원인은 **기준 실행 자체가 13.75 s에 교차로에서 다른 차(vehicle.mini.cooper_s_2021)와 충돌**한 것. 최고 속도를 30·35 km/h로
  낮추면 교차로 도착 시각이 바뀌어 시나리오 차량과 부딪친다(같은 경로·시드에서 40 km/h는 충돌 0, 점수 100).
  G-04 판정 자체는 맞지만, 루프가 이걸 고장 실행 **뒤에** 알아서 CARLA 시간을 버리고 그 후보는 결과가 없다.
- **수정:** BUG-003과 같은 `baseline_gate`가 기준 실행이 평가 구간 안에서 차선을 벗어나거나 충돌하면 고장 실행 전에
  `G-04_reject`로 걸러내고 다음 후보로 바꾼다.
- **확인:** 위 두 기준 실행에 게이트 적용 → `G-04 baseline collides at 13.75 s`, `baseline: V-03 window shorter than minimum`.
  `run.py --mock` 36런 정상, 엔진 테스트 11/11.
- **남은 것:** 어떤 (경로, 속도) 조합이 기준 실행부터 위험한지 GP·후보 생성에 알려주지는 않는다(같은 조합이 다시 뽑힐 수 있음).

## 참고: Cosmos 해설 거절 (버그 아님)

- `R00-0000-saturate` 해설이 전환 시각을 12.0 s(실제 12.65 s, 허용 ±0.5 s)로, 충돌을 "없음"(실제 12.75 s)으로 말해 검증에서 거절.
  모델 답이 틀린 경우로, 검증 단계가 의도대로 걸러낸 것. 진행 로그의 `trigger=`는 `trigger=<시각>s, detected=<감지 이유>`로 바꿈.

## BUG-005: 09 모니터 `explain.py`(Cosmos 캠페인 인사이트)가 루프에서 한 번도 안 돌았음

- **발견:** 2026-10-28, `results/real_12runs_1028` 결과를 보다가 "Cosmos 인사이트가 어디 있냐"는 질문에서 확인
- **증상:** `run.py`가 끝나도 `insights/`가 없음. 09 모니터의 Cosmos 인사이트 코드(`pipeline/09_monitor/explain.py`)는 따로 손으로 돌려야만 실행됨.
- **원인:** 브랜치를 합칠 때 `run.py`에는 SR 모니터 채점(`engine/sr_checks.py`)만 연결되고 `explain.py` 호출은 빠짐.
- **수정 (6ca357e):** `engine/insights.py`가 루프가 끝난 뒤 `explain.py`를 불러 `OUT/insights/`에 씀. `--insights on|off`(기본 on),
  mock은 네트워크 없음, 실패해도 런 전체는 멈추지 않음. 이미 끝난 결과에는 `python -m engine.insights RESULTS_DIR`.
- **확인:** `real_12runs_1028`에 실제 Cosmos 호출 → 인사이트 5개 (`result/result_2/insights/`, 9c3d3b0).

## BUG-006: 엔진 기록과 `explain.py` 입력 형식이 안 맞음

- **발견:** 2026-10-28, BUG-005 연결 작업 중
- **증상:** 엔진 `run_records.jsonl`을 그대로 넣으면 `explain.py`가 채점된 런을 0개로 셈 / 키를 못 찾음.
- **원인:**
  - 판정 키: `explain.py`는 `outcome_category`에 SR 결과(F1/F2/F3/pass)를 기대하지만, 엔진은 SR 결과를 `sr_outcome`에 둠
    (`outcome_category`는 러너 분류).
  - 상태 값: `explain.py`는 `sim_status == "valid"`, 엔진은 `"ok"`.
  - 접속 정보: `explain.py`는 `NEBIUS_TOKEN_FACTORY_KEY / NEBIUS_TOKEN_FACTORY_BASE_URL / COSMOS_MODEL`, `.env`는
    `NEBIUS_API_KEY / COSMOS_API_KEY / COSMOS_BASE_URL / COSMOS_MODEL_ID`.
  - 근거 값 비교: `explain.py`는 모델이 인용한 근거 숫자를 기록 값과 `==`로 비교함. 반올림 안 된 float(예: 0.78935328…)은
    모델이 그대로 옮겨 적지 못해 답 전체가 거절됨.
- **수정 (6ca357e):** `engine/insights.py adapt_record`가 `sr_outcome → outcome_category`, `ok → valid`, fault_plan·environment·sr_checks를
  펼쳐 넣고, float를 소수 3자리로 반올림. 접속 정보는 엔진의 C_cosmos 설정으로 `explain.ModelClient`를 만들어 넘김.
  `explain.py`(09 모니터 원본)는 이 단계에서 고치지 않음.
- **확인:** `tests/test_engine_insights.py`(형식 변환, 채점 수 세기).

## BUG-007: Cosmos 인사이트 품질·입력 크기·화면 표시 문제

- **발견:** 2026-10-28, `result_2` 첫 인사이트(6ca357e로 생성)와 그 수정 작업 중, 사용자 확인
  ("인사이트도 html에 들어가야 한다", "html이 열려서 실시간으로 보는 것")
- **증상:**
  1. 답 내용
     - 답에 중국어(한자) 문장이 섞임.
     - "모든 실험이 같은 시나리오"라는 한계 문장 (실제로는 경로가 여러 개).
     - 근거·통계에 없는 소수 숫자가 제목·본문에 나옴.
     - SR 기준이나 임계값을 느슨하게 하자는 "다음 실험" 제안.
  2. 나눠 묻기: 런을 여러 묶음으로 나눠 보내자 모델이 그 묶음에 없는 런 ID를 근거로 인용 → `explain.py`가 답을 거절 (`result_2`, 12런).
  3. 화면
     - 인사이트가 `insights/campaign.html`에만 있고 메인 `screen/campaign.html`에는 없음.
     - `run.py`가 메인 화면을 **라운드가 끝날 때만** 다시 그림. 브라우저도 자동으로 안 열림.
       (페이지 자체의 5초 자동 새로고침은 원래 있었음)
- **원인:**
  1. `explain.py`의 검사는 런 ID와 근거 값만 보고 문장 내용은 안 봄. 프롬프트에 언어·데이터 사실(런/시나리오/경로 수)이 없었음.
  2. 보고서용 진단 필드(`sr_checks.diagnostics`, 런당 약 3 KB)까지 넣어서 12런이 한 번에 안 들어가 묶음이 나뉨.
  3. 화면 갱신·인사이트 표시는 `run.py`/`engine/dashboard.py`에 연결돼 있지 않았음.
- **수정:**
  1. 96a9ce6: 프롬프트에 한국어 강제, 데이터 사실, 고장 종류별 지표 평균, 안전 기준 완화 금지를 넣음.
     `check_quality`가 한자·가나, 근거·통계에 없는 소수, 데이터와 다른 "같은 시나리오" 주장, SR/임계값 완화 제안을 거절하고,
     거절 이유를 붙여 최대 3번 다시 물음. 거절된 답은 `insights.json rejected_answers`에 남김.
  2. 96a9ce6: `engine/insights.py`에서 보고서 전용 진단 필드를 빼서 12런이 한 묶음에 들어가게 함. 전체 재시도 2번.
  3. 21f6018: 메인 화면에 인사이트 표시. b2d2b10·52857fe: 오른쪽 열 시나리오 목록 바로 아래의 작은 칸으로, 핵심만 크게.
     493d75a: 시작할 때 `screen/campaign.html`을 기본 브라우저로 한 번 열고(`--open-screen on|off`), 런이 하나 끝날 때마다와
     단계가 바뀔 때마다 다시 그림. 맨 위에 "지금 하는 일" 표시.
- **확인:** `pipeline/09_monitor/test_explain.py` 추가 테스트, `result_2` 재생성에서 답 내용 문제 네 가지 모두 사라짐.
  `tests/test_live_screen.py`. PC에서 `run.py --mock --out results/mock_live_test` 72런 동안 89번 다시 그려지고 탭은 1개만 열림.
- **남은 것 (미해결):** 정수(개수)는 검사하지 않음. 예: "SR4 실패 9개"(실제 8개)가 그대로 통과. `real_gen_1028` 인사이트의
  "한계" 문장은 같은 말이 반복됨.

## BUG-011: 기준 실행부터 충돌하는 경로 26956이 계속 뽑힘

- **발견:** 2026-10-28, `real_gen_1028` (BUG-004의 "남은 것")
- **증상:** 걸러진 후보 11개 중 5개가 26956의 기준 실행 충돌(13.25~13.75 s, G-04). 처음 Nemotron A가 만든 4개 중 2개도 여기서 빠짐.
  28035는 경로가 짧아 평가 구간 부족(V-03)이 4번.
- **원인:** 26956은 교차로에서 시나리오 차량과 부딪치는 시각이 속도에 따라 겹침(BUG-004). 후보 생성은 이를 모름.
- **수정 (5126a9c):** `paired_runs.yaml`에서 26956을 `enabled: false`로 끄고(항목은 남김), 새 경로 3개 추가.
  Nemotron A 형식, 무작위 탐색, GP 특징은 모두 `enabled` 경로 목록을 그대로 씀 (코드에 경로 목록 하드코딩 없음).
- **새 경로 검증 (2026-10-28, PC CARLA 0.9.15, 30 km/h, 고장 없음 기준 실행 1회 + `baseline_gate`):**
  | 경로 | 맵 | 결과 | 통과한 고장 시점 |
  |---|---|---|---|
  | bench2drive:24206 | Town03 | 36.4 s 완주 | 직선, 커브 진입/정점/출구 모두 |
  | bench2drive:28048 | Town02 | 38.5 s 완주 | 직선, 커브 진입 (정점/출구는 V-03) |
  | bench2drive:28154 | Town10HD | 31.3 s 완주 | 직선만 (커브가 끝에 있어 V-03) |
  - 떨어진 후보: 26393(충돌), 25300(실패), 24258(MinSpeed 기준 FAILURE → 종료코드≠0), 25854(실패),
    25896(모든 시점 G-04 차선 이탈), town04_highway(직선만 통과, 기준 실행 6분이라 제외).
  - 30 km/h 한 속도만 확인함. 다른 속도에서는 아직 미확인.
- **확인:** pytest 통과, `run.py --mock --rounds 0 --cold-start 4` 실행에서 26956 경로가 한 번도 안 뽑힘.
  mock용 곡률 모양(`engine/fixtures/synthetic_sim.py`)도 새 경로 3개 추가 (대략적인 모양, SYNTHETIC).

## BUG-012: "직선" 고장 시점이 차가 멈춘 구간에 잡힘

- **발견:** 2026-10-28, BUG-011 경로 검증 중
- **증상:** 기준 실행이 끝난 5개 경로 중 4개(24206, 28048, 28154, 25896)에서 `straight` 시점의 속도(`speed_at.straight`)가
  거의 0 m/s (town04_highway만 8.0 m/s). 28035(4.0 m/s), town10_long(2.1 m/s)도 출발 가속 중에 잡힘.
  즉 "직선 주행 중 고장"이 실제로는 차가 서 있거나 막 출발할 때 들어감.
- **원인:** `engine/scenarios.py detect_route_events` 는 곡률 = yaw_rate / max(vx, 1)로 계산. 멈춰 있으면 yaw_rate가 0이라
  곡률도 0 → 정지 구간이 "직선"으로 잡힘. 그 구간의 가운데를 고르므로 정지 중인 시각이 뽑힘.
- **수정 (14e6a21):** 실제로 움직이는 틱만 씀.
  - v_ref = 이동 속도(>0.5 m/s)의 90번째 백분위 (그 기준 실행의 순항 속도)
  - 직선: vx ≥ max(2.0, 0.6·v_ref), 곡선: vx ≥ max(2.0, 0.5·v_ref) (저속 곡률은 흔들려서)
  - 처음 직선 속도에 닿기 전(출발 구간)과 t < 5 s는 제외. 멈추면 구간이 끊김
  - 값은 `paired_runs.yaml event_detection`에 (`min_speed_mps`, `straight_speed_frac`, `curve_speed_frac`)
- **검증:** 오늘 PC CARLA 기준 실행(30 km/h) 기록으로 다시 계산, `baseline_gate` 통과 여부 포함.
  | 경로 | 직선 전 | 직선 후 | 곡선 (진입/정점/출구) |
  |---|---|---|---|
  | 24206 | 23.00 s, 0.0 m/s | 17.50 s, 8.4 m/s (통과) | 진입 6.20→6.90 s (1.7→4.3 m/s), 정점 6.40→8.60 s; 출구 같음 |
  | 28048 | 17.75 s, 0.0 m/s | 36.75 s, 8.4 m/s (V-03: 경로 끝) | 같음 (진입만 통과) |
  | 28154 | 10.75 s, 0.0 m/s | 6.70 s, 7.9 m/s (통과) | 같음 (모두 V-03) |
  | 28035 | 8.95 s, 4.0 m/s | 없음 (5 s 이후 3 s 이상 순항하는 직선 없음) | 같음 |
  | town10_long | 8.65 s, 2.1 m/s | 17.25 s, 8.5 m/s (통과) | 같음 |
  | town04_highway | 46.70 s, 8.0 m/s | 같음 | 없음 |
  - 결과: 28035, 28048은 이제 "직선" 시점을 못 씀 (gate가 V-03/없음으로 거름). 대신 남은 직선은 모두 순항 중.
  - 테스트 2개 추가 (정지·출발 구간 무시, 저속 곡선 무시). pytest 통과, mock 18개 실행 정상.

## 참고: 같은 시나리오만 나오는 것처럼 보임 (사용법)

- `--cold-start 2`면 처음 탐색에서 Nemotron A에게 1개만 요청함(AI 50% + 무작위 50%). `--seed` 기본값 0이라 무작위 쪽도 매번 같음.
  `--out`을 새로 만들면 이전 실행의 실패 사례가 A에게 전달되지 않음.
- A가 바꿀 수 있는 것은 켜진 경로, 최고 속도, 고장 종류·세기·길이, 고장 시점(직선/커브 진입/정점/출구)과 앞뒤 몇 초뿐.
  날씨·다른 차·마찰·새 맵은 형식에 없고 `run_benchmark.py`에도 옵션이 없음.
- 다양하게 돌리려면 `--cold-start` 6 이상, `--seed` 변경, `--rounds` 1 이상. 경로는 `paired_runs.yaml`에 추가.

## 참고: 영상 해설이 자주 거절됨 (모델 한계, 미해결)

- `real_gen_1028` Super-VL 해설 5개 모두 거절: 차선 이탈 시각(`departure_time_s`)을 놓치거나 0.85~1.65 s 틀림 (허용 ±0.5 s).
- 같은 5개를 Cosmos로 다시 해설 → 1개 통과(R01-0006). 거절 이유: 모니터는 차선 이탈이 없는데 이탈했다고 함,
  전환이 없는데 전환했다고 함, 이탈 시각 0.59·0.6 s 틀림, 충돌을 못 봄, 목록에 없는 태그.
- 모니터는 차체 외곽(bbox)이 차선 밖으로 나간 순간을 이탈로 봄(`engine/monitor.py lane_margin_m < 0`). 프롬프트의 정의
  ("바퀴/차체가 차선 밖")와 같음. 4 fps 영상(프레임 간격 0.25 s)에서 모델이 순간을 정확히 못 짚는 것이 주 원인으로 보임.
  검증 단계가 의도대로 걸러낸 것이라 코드 수정은 하지 않음.

## BUG-013: 예전에 실패한 기준 실행을 같은 폴더에 다시 돌리려다 캠페인 전체가 멈춤

- **발견:** 2026-10-28 13:45 (PT), `run.py --real --cold-start 8 --rounds 3 --batch-size 6 --seed 11 --out results/real_gen2_1028`
  (integration2 5a0abd1). 20런을 마치고 라운드 3 시작 직후 종료.
  ```
  FileExistsError: ...\runs\baselines\aa67077909ad already has results.json (run_benchmark.py needs a fresh --out)
  ```
- **원인:**
  - 기준 실행 폴더 이름은 (경로, 속도, 시드)로 정해짐. 같은 조합이 다시 뽑히면 같은 폴더를 다시 씀.
  - `aa67077909ad`(town10_long 10 km/h)는 라운드 0 (12:54)에 이미 돌았는데 MinSpeedTest FAILURE로 종료코드≠0
    → `.complete`가 안 생기고 "baseline failed"로 걸러짐.
  - `ensure_baseline`은 `.complete`만 보고 "아직 안 돌았다"고 판단해 다시 실행 → `_run`의 "새 `--out`만 허용" 검사에 걸려 예외.
    같은 시드라 다시 돌려도 결과는 같음.
- **영향:** 실패한 기준 실행 조합이 다시 뽑히는 모든 `--real` 캠페인. 라운드 3의 런 6개가 안 돌았음 (20런에서 멈춤).
- **수정 (a686079):** `engine/carla_runner.py ensure_baseline` / `ensure_no_reaction`이 `results.json`은 있고 `.complete`가 없으면
  다시 돌리지 않고 그 폴더를 돌려줌 → 호출하는 쪽(`run.py stage_08_09`)이 지금처럼 `G-04_reject "baseline failed"`로 거르고 다음 후보로 바꿈.
  `results.json`도 없는 반쯤 남은 폴더(중간에 끊긴 실행)는 지금처럼 다시 돌림.
- **확인:** `tests/test_engine_carla_runner.py::test_failed_baseline_is_not_rerun`, 전체 pytest 통과.

## BUG-014: Cosmos가 "준비됨"이 된 직후 호출이 전부 HTTP 404 → 영상 해설이 모두 Super-VL로 넘어감

- **발견:** 2026-10-28, `results/real_gen2_1028` 정리 중 (`llm_calls.jsonl`, `real_gen2_1028_finalize.log`)
- **증상:**
  - `--cosmos-power auto`가 엔드포인트를 켜고 `readiness=ready`(status=updating)를 확인한 바로 뒤 Cosmos 호출이 모두 HTTP 404.
    `real_gen2_1028` 라운드 0~2 영상 해설 8개 모두 `C:video`·`C:contact_sheet` 404 → **8개 전부 대체 모델(Nemotron-3.5-Super-VL)이 해설**.
  - 13:59 (PT) 캠페인 인사이트도 404로 `status: error`, 인사이트 0개. 켜져 있던 시간(라운드마다 약 3~4분)은 과금됐는데 Cosmos는 한 번도 답하지 않음.
- **원인:** 관리 API(`/v0/dedicated_endpoints`)가 `ready`라고 한 뒤에도, 실제 호출 주소(`/v1/chat/completions`)에 라우팅되기까지
  약 20초가 더 걸림. `CosmosPower.ensure_on`은 관리 API 상태만 보고 바로 "사용 가능"을 돌려줬고, 영상 해설·`explain.py`는 404를 재시도하지 않음.
- **수정 (c317c7c):** `engine/cosmos_power.py ensure_on`이 `ready` 뒤에 아주 작은 호출(`max_tokens 1`)로 실제 주소를 확인하고,
  404/429/5xx면 10초마다 다시 확인(전체 15분 제한). 끝내 안 되면 False → 영상 해설은 Super-VL, 인사이트는 건너뜀 (그리고 끔).
  주소는 `COSMOS_BASE_URL`(없으면 Token Factory `/v1`).
- **확인:** 가짜 HTTP 단위 테스트 3개 추가. 실제 켜기 2번 (2026-10-28 PT):
  - 14:55: 관리 API 준비 166초, 그 순간 호출은 404, 187초에 200.
  - 15:01: 관리 API 준비 155초, 그 순간 404, 166초에 200.
  - 두 번 모두 그 뒤 Cosmos가 실제로 답함(인사이트 호출 4번, 10번). 다만 답이 `explain.py` 근거 검사에서 모두 거절됨
    (인사이트 문장에 그 인사이트 근거에 없는 숫자 0.96, 46.45를 씀). 이건 BUG-007의 검사가 의도대로 거른 것.
- **남은 것:** `real_gen2_1028`의 영상 해설 8개는 Super-VL 해설 그대로 (Cosmos로 다시 해설하지 않음).

## BUG-015 (표시): 백업 전환이 없었는데 "운행 안정성 ✓"로 보임

- **발견:** 2026-10-28, 사용자 질문 ("전환을 안 하는데 운행 안정성이 있는 건 모순 아닌가")
- **증상:** `recovery_outcome = no_switch`(고장을 넣었는데 감독기가 백업으로 전환 안 함)인 런도 화면의 운행 안정성 칸이 초록 ✓.
- **원인:** `engine/dashboard.py _stability_short`는 SR2·SR4·SR5·SR6 실패가 없으면 ✓. 전환 여부는 안 봄.
  전환이 없으면 SR2(전환 순간 조향)는 아예 적용되지 않아 더 쉽게 ✓가 됨.
- **수정 (5cdc964, 판정 로직은 그대로, 화면만):**
  - 시나리오 목록의 운행 안정성 칸: `no_switch`면 회색 "전환 없음 · 판정 보류". 실제 차선 이탈(SR4)이나 충돌은 그대로 빨간 ✗.
  - 런 상세의 "전환 후 운행 안정성" 표: 제목 옆에 "전환 없음 · 판정 보류 (아래는 런 전체 관측값)", 행에는 초록 ✓ 대신 회색 –
    (측정값은 그대로 보임). 차선·충돌 행의 실패만 ✗.
- **확인:** `tests/test_engine_dashboard_insights.py` 테스트 2개. `results/real_gen2_1028/screen/campaign.html` 다시 그림.

## BUG-016: 경로가 끝나는 시각이 0.05~0.1초 달라서 SR 판정이 "무효"가 됨 (V-08)

- **발견:** 2026-10-28, `results/real_gen2_1028` SR 무효 4개 (R00-0000-bias, R01-0008-drift, R01-0010-intermittent_bias, R01-0012-bias)
- **증상:** recovery v2는 채점했는데 SR 모니터는 4개 모두 `V-08 fault and baseline windows differ`로 무효.
- **원인:**
  - SR 평가 구간 = 고장 시작부터 10초, 단 기록이 먼저 끝나면 기록 끝까지. 이 4개는 고장이 경로 끝 근처에 들어가서
    고장 런과 기준 런 둘 다 10초가 되기 전에 **경로를 다 달려서** 기록이 끝남.
  - 고장 때문에 차가 조금 늦거나 빨라져서 끝나는 시각이 1~2틱 다름: 51.65 vs 51.55 s, 41.65 vs 41.75 s, 54.90 vs 54.80 s, 52.95 vs 52.90 s.
  - `pipeline/09_monitor/monitor.py`는 두 구간 길이가 1틱(0.05 s)만 달라도 무효 처리. 사양(`evaluation_window: end simulation_end,
    align_baseline_to_fault_run: true`)은 기준 런을 고장 런에 맞추라고 돼 있는데 맞추지 않고 버림. 시뮬레이션 문제가 아니라 비교 방법 버그.
- **수정 (fcfcd00):** `paired_checks`가 두 구간이 다르면 **더 일찍 끝난 쪽에 맞춰 둘 다 자르고** 다시 검사함. 최소 구간 5초(V-03)는 잘린 구간에서
  그대로 검사(짧으면 여전히 무효). 충돌·안전 정지 런은 예전 그대로. 맞춘 경우 `notes`에 "V-08 aligned ..."를 남김.
  화면의 무효 표시는 "판정 불가 (무효, 실패 아님)"으로 바꾸고 이유를 같이 보여줌.
- **확인:** `test_monitor.py::test_route_end_a_few_ticks_apart_is_aligned_not_invalid`. 기존 20런을 CARLA 없이 다시 채점:
  무효 4 → 0. 4개 모두 `F1_no_or_late_switch` (bias 2, drift 1, intermittent_bias 1). 나머지 16런 결과는 하나도 안 바뀜.
  원본은 `run_records.jsonl.before_rescore`로 남김.

## BUG-017: 영상 위 시각 글씨가 너무 작아 모델이 시각을 못 읽고 찍음 → 영상 해설 7/8 거절

- **발견:** 2026-10-28, `real_gen2_1028` 영상 해설 8개 중 7개 거절 (모두 Super-VL, BUG-014 때문)
- **증상:** 거절 이유 대부분 차선 이탈 시각(`departure_time_s`). 모델이 쓴 시각이 클립 끝 시각처럼 딱 떨어지는 값이고
  실제와 0.65~6.65 s 차이. 근거 프레임에 클립 밖 시각(고장 시작 35.45 s 등, 문맥으로 준 값)을 씀.
- **원인:**
  1. 프레임 아래 `t=.. s` 글씨를 PIL 기본 글꼴(약 10 px)로 씀. 글씨가 작아서 모델이 영상을 줄여 보면 읽을 수 없음
     → 모델이 시각을 추측. (영상 프레임 수와 `front_frames.jsonl` 시각 개수는 일치 확인)
  2. 프롬프트에 프레임별 시각 목록이 없어서, 문맥으로 준 고장 시작·감지 시각(클립 밖)을 근거 프레임으로 씀.
  3. 검증기: 4 fps 프레임 사이에만 있었던 짧은 이탈(R01-0008: 37.15~37.30 s, 프레임 37.10·37.35)은 영상에 안 보이는데도
     "이탈 시각"을 요구함. 기준도 구간 시작(프레임이 아닌 시각)이었음.
- **수정 (60ec745):**
  - 시각 글씨를 프레임 높이의 1/12(최소 18 px) 굵은 글꼴, 노란색으로.
  - 프롬프트에 "프레임 i의 시각 = [...]" 목록을 넣고, 모든 시각은 이 목록에서 고르라고 함. 근거 프레임도 목록에서만.
  - 검증 기준 = 이탈 구간 안에 있는 **첫 프레임** 시각. 프레임에 전혀 안 보인 이탈은 null이 정답이고, 시각을 쓰면 그 구간 ±0.5 s 안이어야 함.
  - 허용 오차(0.5 s), 태그 목록 검사, 충돌 검사는 그대로 (느슨하게 하지 않음).
- **확인:** 저장된 답 8개를 새 검증기로 다시 검사 → **전과 같이 1/8** (모두 실제로 틀린 답이라 검증기 수정만으로는 안 바뀜.
  예: R00-0006은 클립 끝 시각 46.75 s를 이탈 시각이라고 함, R02-0014는 35.35 s 이탈을 39.0 s라 하고 35.55 s 충돌을 놓침, R01-0011은 목록에 없는 태그를 만들었고 내용도 틀림(전환 52.1 s 전인 51.85 s에 이탈), R02-0019는 없던 이탈을 있다고 함).
  거절된 7개 런의 클립을 새 글씨로 다시 만듦 (예전 클립은 `clips_before_fix/`). 테스트 3개 추가.
- **남은 것:** 새 클립으로 Cosmos 다시 해설하는 것은 아직 못 함 (아래 "남은 일").

## BUG-018: 인사이트 검사가 실제 데이터에 있는 숫자도 거절하고, 고치는 방법을 안 알려줌

- **발견:** 2026-10-28, `real_gen2_1028` 인사이트 2번 실행 모두 거절 (Cosmos 응답 4번, 10번)
- **증상:** "수치 0.96이 evidence에 없다", "수치 46.45가 evidence에 없다"로 거절. 두 번째는 "모든 실패 런은 curve_entry" 같은 틀린 주장도 있었음.
- **원인:**
  - 0.96은 실제로 보낸 데이터에 있음: R00-0004-drift `metrics.pre_switch_max_lateral_error_m` = 0.96 (그 밖에 R01-0010 `fault_plan.onset_offset_s`).
    46.45도 있음: R02-0018-solver_fail `switch_time` / `planned_fault_onset` = 46.45 (통과 런).
  - 검사는 숫자를 "그 인사이트의 evidence 값 + 전체 통계"에서만 찾음. 문장에 쓴 런의 다른 값은 인정 안 함.
    다시 물을 때 "근거를 추가하라"만 말하고 어느 런의 어떤 값인지 안 알려줘서 모델이 같은 실수를 반복.
  - 고장 시점별 실패 개수가 통계에 없어서 모델이 "모든 실패는 커브 진입"처럼 틀린 전체 주장을 하기 쉬웠음.
    실제 실패 시점: curve_entry 5, curve_exit 5, straight 1 (재채점 전 기준).
  - 거절된 답 원문이 저장 안 돼서 나중에 다시 검사할 수 없었음 (이유 문자열만 남음).
- **수정 (ab7bb6b):**
  - 숫자 근거: evidence 값, 통계 값, 그리고 **그 인사이트가 evidence로 인용한 런의 기록 값**. 지어낸 숫자는 여전히 거절.
  - 거절 메시지에 그 숫자가 실제로 있는 곳을 적음 (예: "0.96은 R00-0004-drift metrics.pre_switch_max_lateral_error_m에 있습니다. 그 런을 evidence에 넣으세요").
  - 통계에 `by_onset_event`(시점별 런 수·결과·실패 수)와 `facts.failure_onset_events` 추가. 프롬프트에 인용 규칙 추가.
  - "모든/전부/모두 + 실패 + 시점" 주장을 실제 실패 시점 분포와 대조해 틀리면 거절.
  - 거절된 답 원문을 `insights.json rejected_raw`에 저장.
- **확인:** 저장된 거절 답은 원문이 없어서 그대로 재검사는 못 함. 대신 실제 데이터셋으로 같은 주장을 만들어 확인:
  0.96을 인용 런 없이 쓰면 거절 + 위치 안내, R00-0004를 evidence에 넣으면 통과, "모든 실패 런은 curve_entry"는 거절.
  `test_explain.py` 테스트 3개 추가.

## 남은 일 (2026-10-28 PT 기준)

- Cosmos 한 번 켜서 (1) 새 클립으로 거절된 영상 해설 7개 다시 해설, (2) 인사이트 재생성 하는 스크립트는 준비했지만
  실행 명령이 자동 안전 검사에서 계속 막혀서 못 돌림. Cosmos는 켜지지 않았고 꺼진 상태 확인.
