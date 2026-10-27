# 04 타당성 검사 + 후보 확장

| 파일 | 내용 |
|---|---|
| [`feasibility_expansion.yaml`](feasibility_expansion.yaml) | G-01~04 게이트, 섭동·중복 제거 (팀 템플릿) |
| [`scenario_space.yaml`](scenario_space.yaml) | 팀 템플릿 탐색 공간 (지연 계열 F-01~03) |
| [`scenario_space_av_stack.yaml`](scenario_space_av_stack.yaml) | av_stack(PDM-Lite + MPC/LQR 감독기)용 탐색 공간: 아래 고장 9개 |
| [`scenario_space.py`](scenario_space.py) | 위 YAML 읽기, 검증, 샘플링, `run_benchmark.py` 인자 변환 |

`scenario_space_av_stack.yaml`의 F-01~09는 팀 템플릿 `scenario_space.yaml`의 F-01~03과 다른 고장입니다.

## 우리 감독기는 어떻게 판단하나

감독기는 0.05 s마다 MPC 출력을 검사합니다 (`08_carla_paired_run/av_stack/integration/supervised_lateral.py`).

1. **출력이 쓸 만한가**: 솔버 실패, NaN, 조향 범위 초과, 계산 시간 초과, 오래된 출력. 하나라도 걸리면 그 주기에 바로 LQR로 넘어갑니다.
2. **경로를 잘 따라가나**: MPC가 보고한 횡오차가 1 m(방향 오차 0.6 rad)를 넘은 상태가 0.25 s 이어지면 넘어갑니다. 중간에 정상 샘플이 하나라도 나오면 타이머가 처음부터 다시 셉니다.

LQR로 넘어가면 MPC로 돌아오지 않습니다.

## 고장 9개

- 전환 실패: 감독기가 MPC→LQR 전환을 하지 않거나 늦게 함
- 전환 후 안정성 저하: 전환은 했지만 이후 차선 이탈, 큰 오차, 급조향 등이 생김
- (실측) = CARLA에서 돌려서 확인함. 나머지는 감독기 코드 구조와 논문을 근거로 한 추정

| ID | 고장 | 뭐 하는 고장인지 | 구분 | 전환 실패 가능성 | 전환 후 안정성 저하 가능성 | 범위 | 주입기 |
|---|---|---|---|---|---|---|---|
| F-01 | solver_fail | MPC가 "답을 못 구했다"고 보고 | 대조군 | 낮음 (실측: 즉시 전환) | 낮음 (실측: 100점, 충돌 0) | 지속 0.1–3 s | 있음 |
| F-02 | stale | MPC가 직전에 낸 답을 그대로 반복 | 대조군 | 낮음 | 낮음 | 지속 0.1–3 s | 있음 |
| F-03 | bias | 조향값에 작은 각도를 계속 더함 | 위험 | 높음 (실측: 0.05 rad 감지 못 함) | 중간 | 0.02–0.10 rad | 있음 |
| F-04 | drift | 조향 오차가 시간이 갈수록 커짐 | 위험 | 높음 (실측: 2–5 s 늦게 전환) | 중간~높음 | 0.005–0.05 rad/s | 있음 |
| F-05 | scale | 조향값을 일정 비율로 줄이거나 키움 (커브에서만 티 남) | 위험 | 중간 | 중간 | × 0.5–0.9, 1.1–1.5 | 없음 |
| F-06 | saturate | 조향을 최대각에 고정 | 위험 | 낮음~중간 (감지는 되지만 늦음) | 높음 | 지속 0.1–1.0 s, 좌/우 | 없음 |
| F-07 | 위치 추정 오염 | 제어기가 받는 차량 위치를 옆으로 틀리게 줌 (LQR도 같이 틀림) | 위험 | 높음 | 높음 | 0.1–0.6 m, 지연 0–200 ms | 없음 |
| F-08 | covert | 조향을 틀리게 하면서 감독기에 보고하는 오차는 정상값으로 거짓 보고 | 위험 | 매우 높음 | 높음 (크기에 따라) | bias 0.05–0.2 rad | 없음 |
| F-09 | 경계선 진동 | 횡오차가 1 m 감지선 근처를 오르내리게 해서 감지 타이머를 계속 리셋 | 위험 | 높음 (진동을 만들 수 있다면) | 중간~높음 | 기본 bias 0.06–0.12 rad, 흔들림 0.01–0.05 rad, 주기 0.2–1.0 s | 없음 |

대조군(F-01, F-02)은 감독기가 잘 잡는 고장이며, 서로 다른 검사(상태, 순번)로 감지됩니다.
F-07은 MPC 고장이 아니라 입력 고장이라 보조 사례로 둡니다.
F-08은 결과가 구조상 정해져 있어(항상 감지 실패) 탐색보다 설계 결함 시연용입니다.

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
{"fault_id": "F-04", "fault": "drift", "params": {"rate_rad_s": 0.040}, "timing": {"onset_event": "straight", "offset_s": -0.02}, "environment": {"max_speed_kmh": 36.8, "tire_friction_scale": 0.57}}
```

`to_cli(space, scenario, onset_s)`는 주입기에 있는 고장(F-01~04)을 `run_benchmark.py` 인자로 바꿉니다.
`onset_s`(주입 시각)는 호출하는 쪽이 경로에서 `onset_event` + `offset_s`를 찾아 넘깁니다.
주입기에 없는 고장(F-05~09)은 `NotImplementedError`를 냅니다.

```
['--mpc-fault', 'bias', '--fault-start', '11.50', '--fault-duration', '500.00', '--fault-magnitude', '0.0965', '--max-speed', '20.8']
```
