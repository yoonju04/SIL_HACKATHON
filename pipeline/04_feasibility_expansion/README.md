# 04 타당성 검사 + 후보 확장

| 파일 | 내용 |
|---|---|
| [`feasibility_expansion.yaml`](feasibility_expansion.yaml) | G-01~04 게이트, 섭동·중복 제거 (팀 템플릿) |
| [`scenario_space.yaml`](scenario_space.yaml) | 팀 템플릿 탐색 공간 (지연 계열 F-01~03) |
| [`scenario_space_av_stack.yaml`](scenario_space_av_stack.yaml) | av_stack(PDM-Lite + MPC/LQR 감독기)용 탐색 공간: 아래 고장 7종 + 확장 5종 |
| [`scenario_space.py`](scenario_space.py) | 위 YAML 읽기, 검증, 샘플링, `run_benchmark.py` 인자 변환 |

`scenario_space_av_stack.yaml`의 F-01~07은 팀 템플릿 `scenario_space.yaml`의 F-01~03과 다른 고장입니다.

## 우리 감독기는 어떻게 판단하나

감독기는 0.05 s마다 MPC 출력을 검사합니다 (`08_carla_paired_run/av_stack/integration/supervised_lateral.py`).

1. **출력이 쓸 만한가**: 솔버 실패, NaN, 조향 범위 초과(최대각 *초과*만), 계산 시간 초과, 오래된 출력(시각 차이 0.1 s 초과, 번호가 늘지 않음). 하나라도 걸리면 그 주기에 바로 LQR로 넘어갑니다.
2. **경로를 잘 따라가나**: MPC가 보고한 횡오차가 1 m(방향 오차 0.6 rad)를 넘은 상태가 0.25 s 이어지면 넘어갑니다. 중간에 정상 샘플이 하나라도 나오면 타이머가 처음부터 다시 셉니다.

LQR로 넘어가면 MPC로 돌아오지 않습니다.

## 고장 7종

서로 다른 실패 원인(명시적 실패, 오래된 메시지, 시간 지연, 은밀한 값 오염, 점진적 오염, 급격한 이상, 감독기 타이머)을 하나씩 구분해서 시험합니다.
F-04·F-05·F-07은 같은 조향 오염에 시간 패턴(상수, 기울기, 켜고 끔)만 다르게 적용합니다.

| ID | 고장 | 뭐 하는 고장인지 | 구분 | 확인할 것 | 범위 | 주입기 |
|---|---|---|---|---|---|---|
| F-01 | 명시적 계산 실패 | MPC가 "답을 못 구했다"고 보고 | 대조군 | 실패 상태를 받으면 정상 전환하는가 | 지속 0.1–3 s | 있음 |
| F-02 | 오래된 메시지 | 직전 메시지를 값·시각·번호까지 통째로 반복 | 대조군 | 출력 중단을 제때 감지하는가 | 지속 0.1–3 s | 있음 |
| F-03 | 실제 출력 지연 | MPC 명령을 k틱 늦게 적용 (원래 시각 유지) | 위험: 시간 문제 | 늦은 명령 처리와 전환 지연 | 1, 2, 3, 4, 6틱, 지속 0.5–5 s | 없음 |
| F-04 | 지속 bias | 조향값에 일정한 각도를 계속 더함 | 위험: 은밀한 값 오염 | 감지 누락과 전환 후 회복 | 0.02–0.10 rad | 있음 |
| F-05 | drift | 조향 오차가 시간이 갈수록 커짐 | 위험: 점진적 값 오염 | 악화 속도에 따른 감지·전환 지연 | 0.005–0.05 rad/s | 있음 |
| F-06 | 최대 조향 고정 | 조향을 정확히 최대각에 고정 (범위 검사 통과) | 위험: 급격한 이상 | 급격히 악화된 상태에서의 인계·회복 | 지속 0.1–1.0 s, 좌/우 | 없음 |
| F-07 | 간헐·주기적 bias | bias를 간헐·주기적으로 켜고 끔 | 위험: 감독기 취약점 | 타이머 리셋으로 인한 전환 누락 | 0.05–0.15 rad, 켜짐 0.1–1.0 s, 꺼짐 0.05–1.0 s | 없음 |

F-02는 "값만 고정하고 시각·번호는 갱신"하는 경우와 다른 시험입니다. 그 경우는 정상 메시지처럼 보여 확장 E-01로 둡니다.
F-03은 기존 주입기의 `delay`(75 ms sleep)와 다릅니다. 기존 `delay`는 계산 시간 초과로 즉시 전환되고, CARLA 동기 모드라 차량에는 영향이 없습니다.

## 관측 결과와 다음 가설

관측은 실제 실행 결과이며 조건과 횟수를 함께 적습니다. 한 번 나온 결과를 모든 상황의 특성으로 일반화하지 않고, 다음 시나리오를 만드는 근거로 씁니다.

- route 26956: `av_stack/runs/campaign_26956_1025_0202/summary.md` (PDM, supervised, 주입 11.5 s, 조건당 1회)
- Town10 라이브: `av_stack/runs/town10_live_check.log` (town10_long, 30 km/h, `tools/inject.py`로 주입)

`runs/`는 git에 올라가지 않는 로컬 폴더입니다.

| ID | 관측 | 다음에 검증할 가설 |
|---|---|---|
| F-01 | route 26956: 0.0 s에 전환, 최대 e_y 0.625 m, 차선 이탈 0%, 점수 100 (1회). Town10 라이브: 0.0 s에 전환 (30·60 km/h 각 1회) | 커브 정점·고속에서 전환 직후 조향 점프와 회복 거리는 얼마인가 |
| F-02 | route 26956: 0.0 s에 전환, 이유 stale_sequence (1회) | 값만 고정하고 시각·번호를 갱신하면 감지되는가 (E-01) |
| F-03 | 없음 | 1–2틱(0.1 s 이하) 지연은 시각 검사를 통과해 감지 없이 주행에 영향을 주는가 |
| F-04 | **전환 못 함**: Town10 라이브 0.05 rad (1회), route 26956 0.02 rad, 최대 e_y 0.707 m (1회). route 26956 0.05 rad는 4.7 s 뒤 전환 (최대 e_y 1.129 m), 0.10 rad는 2.3 s 뒤 전환 (최대 e_y 1.566 m) (각 1회) | 속도·곡률·주입 방향을 바꿔도 감지하지 못하는가. 감지 경계 크기는 어디인가 |
| F-05 | route 26956: 0.05 rad/s는 2.6 s 뒤 전환 (최대 e_y 1.457 m). 0.02 rad/s는 2.0 s 뒤 전환했지만 이유가 계산 시간 초과(deadline_miss)라 drift 감지가 아님 (각 1회) | 악화 속도와 전환 지연은 어떤 관계인가. 우연한 deadline_miss는 기준 실행에서도 생기는가 |
| F-06 | 없음 | 오차 1 m + 0.25 s 뒤 전환될 때 LQR이 차선 안에서 회복하는가 |
| F-07 | 없음 | 이상 조건이 반복해서 해제돼 감지 타이머가 초기화되는가 (경계 근처 진동 유도는 후속) |

## 확장 목록 (샘플링하지 않음)

| ID | 고장 | 내용 |
|---|---|---|
| E-01 | 헤더 갱신 replay | 값은 고정, 시각·번호는 갱신 (정상 메시지처럼 보임) |
| E-02 | 위치 추정 오염 | 제어기가 받는 차량 위치를 옆으로 틀리게 줌. LQR도 같이 오염되므로 MPC 단독 고장과 분리 |
| E-03 | covert | 조향 오염 + 감독기에 보고하는 오차 위조. 감독기가 쓰는 정보의 출처에 따라 시험 의미가 달라짐 |
| E-04 | scale | 조향 × k. 첫 버전은 bias·drift에 탐색 집중 |
| E-05 | 경계선 진동 | 횡오차를 1 m 경계 근처에서 오르내리게 유도 (F-07 후속) |

## 함께 바꿀 조건

| 조건 | 범위 | 비고 |
|---|---|---|
| 주입 시점 | 직선, 커브 진입, 커브 정점, 커브 탈출 (± 1 s) | 저속 급커브는 고장 후 차선 이탈까지 시간이 가장 짧음 |
| 최고 속도 | 10–40 km/h | `run_benchmark.py --max-speed` |
| 노면 마찰 | 0.3–1.0 | `run_benchmark.py`에 아직 옵션 없음 (1.0으로 실행) |

## 사용법

`.venv-tfpp`(PyYAML 포함)로 실행합니다. 이 폴더에서:

```
..\08_carla_paired_run\av_stack\.venv-tfpp\Scripts\python.exe scenario_space.py --check
..\08_carla_paired_run\av_stack\.venv-tfpp\Scripts\python.exe scenario_space.py --sample 5 --seed 0 --implemented-only
..\08_carla_paired_run\av_stack\.venv-tfpp\Scripts\python.exe scenario_space.py --self-test
```

`--sample`은 시나리오를 JSON 한 줄씩 출력합니다:

```json
{"fault_id": "F-05", "fault": "drift", "params": {"rate_rad_s": 0.040}, "timing": {"onset_event": "straight", "offset_s": -0.02}, "environment": {"max_speed_kmh": 36.8, "tire_friction_scale": 0.57}}
```

`to_cli(space, scenario, onset_s)`는 주입기에 있는 고장(F-01, F-02, F-04, F-05)을 `run_benchmark.py` 인자로 바꿉니다.
`onset_s`(주입 시각)는 호출하는 쪽이 경로에서 `onset_event` + `offset_s`를 찾아 넘깁니다.
주입기에 없는 고장(F-03, F-06, F-07)은 `NotImplementedError`를 냅니다.

```
['--mpc-fault', 'bias', '--fault-start', '11.50', '--fault-duration', '500.00', '--fault-magnitude', '0.0965', '--max-speed', '20.8']
```
