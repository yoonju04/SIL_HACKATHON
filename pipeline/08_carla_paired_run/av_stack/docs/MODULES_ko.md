# av_stack: 궤적 생성 / MPC / LQR 분리 구현 (CARLA 실행)

CARLA 실행은 `run_benchmark.py --planner pdm --controller supervised` 하나입니다 (README의 Quick start).
경로·속도·위험 판단은 PDM-Lite가, 조향은 아래 MPC/LQR이 맡습니다.

공개 오픈소스(Autoware, Apollo, PythonRobotics)와 논문을 근거로 만든 세 개의 독립 모듈입니다. CARLA에서 바로 돌아가도록 연결해 두었습니다.

| 모듈 | 하는 일 | 근거 |
|---|---|---|
| `planner/` | **자율주행 = 궤적 생성만.** 차선 중심선 → 경로·곡률·속도 프로파일. 조향/페달 명령은 내지 않음 | PythonRobotics CubicSpline, Quintic (MIT), Werling et al. ICRA 2010 |
| `mpc_controller/` | MPC 횡방향 제어: 궤적 → 조향각 | Autoware `autoware_mpc_lateral_controller` (Apache-2.0), Falcone et al. TCST 2007, OSQP (Stellato et al. 2020) |
| `lqr_controller/` | LQR 횡방향 제어: 궤적 → 조향각 | Apollo `lat_based_lqr_controller` (Apache-2.0), Rajamani *Vehicle Dynamics and Control* 2장 |

```
planner ──Trajectory(s, x, y, yaw, kappa, v, t)──▶ mpc_controller  ─┐
                                              └─▶ lqr_controller  ─┴─▶ 조향각 ─▶ CARLA
integration/  (CARLA 좌표 변환, 속도 PI, 로그)  ← 세 모듈을 묶는 유일한 곳
```

- **분리 규칙:** 세 모듈은 서로를 import하지 않습니다. CARLA도 import하지 않습니다. 의존성은 `numpy`/`scipy`뿐입니다. `tests/test_separation.py`가 import를 검사해서 이 규칙을 강제합니다.
- **폴더 하나만 떼어 써도 됩니다.** 그래서 MPC와 LQR은 경로 매칭 코드(`reference.py`)를 각자 갖고 있습니다. 중복은 의도한 것입니다.

## 1. planner: 궤적 생성만

```python
from planner import ReferencePlanner, PlannerConfig, LaneChange, route_from_carla_map
traj = ReferencePlanner(PlannerConfig(v_max=15, a_lat_max=3.0)).plan(route_x, route_y,
                                                                     (LaneChange(120, 50, -3.5),))
traj.s, traj.x, traj.y, traj.yaw, traj.kappa, traj.v, traj.t, traj.lane_x, traj.lane_y
```

**처리 순서**

1. 차선 중심선 점에서 3 m 간격으로 매듭점을 뽑습니다.
2. 자연 3차 스플라인을 맞춥니다. C2 연속이라 곡률도 연속입니다.
3. 0.5 m 간격으로 다시 샘플링합니다.
4. 속도 프로파일을 계산합니다.
   - `v²·|κ| ≤ min(a_lat_max, 0.5·μ·g)`를 지킵니다. 물리적으로 불가능한 시나리오를 미리 막기 위해서입니다.
   - 가속·감속 제한은 전진·후진 패스로 적용합니다.

**아키텍처 리뷰에서 정한 규칙**

- 궤적은 차선 중심선 기준으로 **시나리오당 한 번만** 만듭니다. 현재 차량 위치에서 다시 계획하지 않습니다.
- 고장을 모르고 결정론적입니다. 같은 경로와 같은 설정이면 같은 궤적이 나옵니다. 그래서 "고장 없는 같은 시나리오" 재실행이 성립합니다.
- 차선 변경은 선택입니다. 횡오프셋 d(s)를 5차 다항식으로 만듭니다(Werling).
- 평가용 기준으로 `lane_x/lane_y`(원래 차선 중심선)를 따로 보관합니다.

`route_from_carla_map()`은 CARLA 맵 waypoint를 차선을 따라 1 m씩 이어 붙입니다. 교차로에서는 첫 갈래를 탑니다. `carla`를 import하지 않습니다.

## 2. mpc_controller: Autoware MPC 이식

```python
import mpc_controller as M
mpc = M.MPCLateralController(M.VehicleParams(...), dt=0.05)        # MPCConfig로 조정
mpc.set_reference(traj.x, traj.y, traj.yaw, traj.kappa, traj.s, traj.v, traj.t)
out = mpc.compute(x, y, yaw, vx, vy, yaw_rate, t)   # out.steer, out.ok, out.status, out.solve_time_s, out.qp_iterations, out.seq
```

**Autoware와 같은 부분**

- 동역학 자전거 오차 모델 `[e_y, ė_y, e_ψ, ė_ψ]`, bilinear 이산화
- N=50, 예측 간격 0.1 s
- 비용(Autoware 기본 가중치)
  - 횡오차
  - 헤딩오차·v²
  - (조향 − 피드포워드)²
  - 횡저크
  - 종단 가중치
- 제약
  - 조향각 한계
  - 조향속도 한계
- 출력에 3 Hz 2차 Butterworth 저역통과 필터
- 선택: 입력 지연 보상
- **조향속도 제약의 기준은 MPC 자신의 직전 raw 출력입니다**(`m_raw_steer_cmd_prev`). 차량에 실제로 적용된 명령이 아닙니다.

**Autoware와 다른 부분**

- 동역학 모델만 구현했습니다. Autoware 기본값은 운동학 모델입니다.
- 조향속도 한계는 상수 60°/s 하나입니다. Autoware는 곡률·속도별 맵을 씁니다.
- 조향가속도 가중치(1e-6)는 넣지 않았습니다.
- ė_y와 ė_ψ는 측정값을 씁니다. Autoware는 수치미분 후 LPF를 씁니다.
- 피드포워드의 Kv는 축(axle) 강성으로 일관되게 계산했습니다. Autoware와 Apollo는 A 행렬에서는 축 강성을 쓰고 Kv에서는 1/2을 곱해서 서로 맞지 않습니다.

**QP 솔버**

- OSQP 알고리즘(ADMM)을 numpy로 다시 구현했습니다(`qp.py`).
  - 스케일링, 적응형 ρ, polishing을 포함합니다.
  - 무작위 문제 20개에서 KKT 조건으로 검증했습니다.
- `osqp` 패키지가 설치되어 있으면 `MPCConfig(qp_solver="osqp")`로 바꿔 쓸 수 있습니다.
- 계산 시간
  - 평균 약 4–5 ms, p99 약 8 ms(이 머신 기준)
  - 반복 횟수는 평균 약 41회, 최대 70회입니다. 반복 횟수는 기계와 무관한 결정론적 지표입니다.

**이전 `steering_sil` 재현 옵션**

- 이전 코드는 조향속도 제약을 *적용된* 명령에 걸었습니다.
- 아키텍처 리뷰에서 찾은 "감지됐는데 2.19 m 이탈"이 이 구조에서 나온 결과였습니다.
- 비교 실험용으로 `MPCConfig(rate_reference="applied")`와 `compute(..., applied_prev=...)`를 남겨 두었습니다.

## 3. lqr_controller: Apollo LQR 이식

```python
import lqr_controller as L
lqr = L.LQRLateralController(L.VehicleParams(...), dt=0.05)        # LQRConfig로 조정
lqr.set_reference(traj.x, traj.y, traj.yaw, traj.kappa, traj.s)
out = lqr.compute(x, y, yaw, vx, vy, yaw_rate)      # out.steer, out.steer_feedback, out.steer_feedforward, out.state
```

**Apollo와 같은 부분**

- 같은 오차 모델을 씁니다.
- 이산화도 Apollo 방식입니다(A는 bilinear, B·T).
- 반복 Riccati로 풀고, 속도별 게인을 캐시합니다.
- 가중치도 Apollo 기본값 그대로입니다(Q = [0.05, 0, 1, 0], R = 1).
- 조향 명령은 `δ = −Kx + δ_ff`입니다. 피드포워드는 Rajamani 식이고, 축 강성으로 일관되게 계산합니다. 선형 모델에서 정상상태 횡오차가 0이 되는 것을 테스트로 확인했습니다.

**Apollo와 다른 부분**

- lead/lag 보상, MRAC, look-ahead 오차 보정, 후진은 넣지 않았습니다.
- ė_y에 횡속도 vy를 포함합니다. Apollo는 v·sin(e_ψ)로 계산합니다.

## 실행

```bash
# CARLA 서버를 먼저 켭니다 (start_carla_stable.ps1). 명령 전체는 README의 Quick start / Command reference.
.\.venv-tfpp\Scripts\python.exe run_benchmark.py --route-id 26956 --planner pdm --controller supervised --out runs\<new>

# 테스트 (pytest 없이)
python tests/run_all.py
```

## CARLA에서 주의할 점

- **좌표계**
  - CARLA는 왼손 좌표계입니다(y 오른쪽, yaw 시계방향, steer + = 오른쪽).
  - 변환은 `integration/carla_io.py` 한 곳에서만 합니다.
- **동기 모드:** `fixed_delta_seconds = dt`(기본 0.05 s)이고, 매 주기 `world.tick()`을 호출합니다. 실행 중에는 BLAS 스레드를 1로 고정합니다(결정성).
- **조향 곡선:** CARLA는 기본적으로 속도가 높을수록 최대 조향각을 줄입니다. 실행기는 `steering_curve`를 평평하게 바꿔서 `steer = δ / max_steer` 관계를 유지합니다.
- **차량 파라미터:** 질량, 축 위치, 최대 조향각은 CARLA physics control에서 읽습니다.
  - **타이어 코너링 강성은 CARLA가 제공하지 않습니다.** 그래서 기본값을 질량 비율로 맞춘 값을 씁니다.
  - 추종이 나쁘면 `--cf/--cr`로 조정하세요. 그 차량으로 강성을 식별하는 것이 가장 좋습니다.

## 검증 결과 (2026-10-22, 당시 오프라인·가짜 CARLA 실행기 기준. 그 실행기는 삭제됨)

| 시나리오 | MPC 최대 \|e_y\| | LQR 최대 \|e_y\| |
|---|---|---|
| 오프라인 demo 경로 (R60 좌, R45 우, 최고 15 m/s) | 0.097 m | 0.050 m |
| 오프라인 R45 원호, a_lat 4.4 m/s² (마찰 한계 근처) | 0.094 m | 0.067 m |
| 오프라인 차선 변경 3.5 m / 50 m | 0.047 m | 0.023 m |
| 가짜 CARLA demo 경로 (왼손 좌표계) | 0.094 m | 0.047 m |

- 오프라인 차량은 제어기의 선형 모델과 일부러 다르게 만들었습니다(타이어 포화, 조향 액추에이터 지연 0.08 s, Iz 차이).

## 남은 것

- **MPC→LQR 전환(Supervisor)과 고장 주입기**는 `integration/supervised_lateral.py`, `integration/fault_injection.py`에 있습니다 (`SUPERVISOR.md`).
- **라이선스:** 저장소 LICENSE는 팀이 정해야 합니다. 이식한 코드의 고지는 `THIRD_PARTY_NOTICES.md`에 있습니다.
