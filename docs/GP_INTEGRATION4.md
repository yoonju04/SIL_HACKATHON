# GP 학습 수정 (integration4)

integration2에서 갈라진 브랜치. CARLA를 더 돌려 학습 데이터를 쌓기 전에 GP가 데이터를 제대로 쓰게 고친다.

## 왜 고쳤나 (result_2~5 점검)

- 위험 고장에서 전환된 15런 모두 **전환 전에 차선을 벗어났다**. 감독기의 추종 오차 기준(1 m, 0.25 s 유지)이 차선 안 여유(0.83~1.08 m)보다 크다.
  그래서 예전 목표(전환 후 복구 점수, 전환된 런만 정의)에는 찾을 경계가 없고, 학습점도 캠페인마다 2~5개뿐이었다.
- result_2 1라운드를 같은 데이터로 다시 학습해 보니, 거리 척도가 거의 모두 한계(0.01/100)에 붙고 노이즈가 0이었다.
  고장 세기 칸(`param_u0`)은 척도 100(무시)이었다. 그래서 drift 0.035와 0.044가 소수점 8자리까지 같은 예측을 받았다.
- `param_u0` 한 칸에 saturate 지속 시간, drift 속도, bias 크기가 섞여 있었다.
- GP가 특징을 학습 데이터의 최소/최대로 다시 맞췄다. 학습에 없는 고장의 칸은 상수가 되어 아무것도 배우지 못했다.
- 전환 여부를 ±1 회귀로 학습해서 예측이 1.00014, −0.93처럼 확률 범위를 벗어났다.

## 바뀐 것

| # | 내용 | 위치 |
|---|---|---|
| 0 | S1 목표를 **SR4 차선 여유 점수**(`sr4_margin_score`, 1 지킴, 0 차선 경계, <0 이탈, −2 충돌)로 바꿈. 모든 런에 정의되고, 0 경계는 유해/무해 경계다. 전환 벌점은 끔(`needs_switch: false`) | `orchestrator.yaml objective_acquisition.S1`, `run.py gp_row_targets` |
| 1 | `--prior-runs result/result_2 ...`: 이전 캠페인 런을 GP 학습 데이터에 더함(재판정 기록 우선). 그 캠페인의 기준 실행에서 경로 이벤트를, 05b 답에서 Nemotron B 캐시를 채움. 같은 시나리오는 다시 돌리지 않음 | `run.py Loop.load_prior`, `gp_rows` |
| 2 | 고장 파라미터마다 칸을 따로 둠(`F-05.rate_rad_s`, `F-06.duration_s`, …, 다른 고장은 0) | `engine/scenarios.py numeric_features` (feature schema 1.2) |
| 3 | GP가 특징을 다시 맞추지 않음. 특징은 이미 시나리오 공간 범위로 [0, 1] | `engine/surrogate.py GPModel` |
| 4 | 커널 한계: 거리 척도 0.1~10, 노이즈 1e-3~1 | `surrogate.yaml kernel_bounds` |
| 5 | 전환 여부는 GP 분류기(평균 = 전환 확률, std = √(p(1−p))). 한 클래스만 있으면 상수 | `surrogate.py GPClassModel`, `surrogate.yaml classify` |
| 6 | 오프라인 검증: `python -m engine.gp_validate result/result_2 result/result_3 result/result_4 result/result_5` (CARLA·LLM 없음, 1~2분) | `engine/gp_validate.py` |
| 7 | 라운드마다 학습 점 수(`prior_rows` 포함)와 커널 진단(거리 척도, 노이즈, 한계에 붙은 칸, 학습에서 상수인 칸)을 `loop_stats.json`에 기록 | `run.py fit_gp`, `gp_kernels` |

## 오프라인 검증 결과 (result_2~5, 유효 48런, leave-one-out)

| 목표 | 새 GP | 예전 GP | 다수 클래스로 찍기 |
|---|---|---|---|
| SR4 부호 (유해/무해) | **79 %** | 67 % | 54 % |
| 전환 여부 | 60 % | 54 % (예측 6개가 ±1 밖) | 60 % |

- 유해/무해 경계는 이제 데이터로 배운다. 전체 48런으로 학습하면 drift 속도(척도 0.69), saturate 지속 시간(0.16), bias 크기 같은 칸이 실제로 학습된다.
- 전환 분류기는 확률 범위는 맞지만 아직 다수 클래스보다 낫지 않다. 41차원에 48점이라 부족하고, 전환은 거의 고장 종류로 정해진다.
  데이터를 쌓은 뒤 다시 본다.
- Nemotron B 답 칸 4개는 모든 런에서 같아 정보가 없다.

## 실행

```
.\pipeline\08_carla_paired_run\av_stack\.venv-tfpp\Scripts\python.exe run.py --real --prior-runs result/result_2 result/result_3 result/result_4 result/result_5 --no-reaction-runs on --video-explainer off --cosmos-power manual --out outputs\<new>
```
