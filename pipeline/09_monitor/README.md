# 09 Monitor — safety spec v3

검증 대상: **MPC → LQR 전환 및 전환 직후 안정성**.
LLM은 판정하지 않습니다. `monitor.py`가 실측 로그와 고정 명세로 판정하고,
Cosmos Reason은 판정 수치 모음에서 해석·실험 제안을 생성합니다. 실패 영상은 사람이 확인하는 리플레이입니다.

## 판정 기준

- **SR1 전환 시간:** 무대응 런에서 관측한 FTTI의 0.5배 이내. 0.5배와 무대응 런이 없을 때의 0.5 s는 실험 설계값입니다.
- **SR2 인계 매끄러움:** 전환 -dt~+0.5 s의 실제 송신 road-wheel 조향 속도. 60°/s는 현재 실험 설정이며 보편적 규격값이 아닙니다.
- **SR3 요·횡슬립/회복 가능 영역:** 차량·차속·노면별 검증된 안정 영역이 없어 미판정입니다. 원시 상태는 참고 지표로 기록합니다.
- **SR4 차선·충돌:** 차체 모서리의 차선 여유가 음수이거나 충돌하면 실패. 고의 차선 변경·회피 기동은 별도 허용 영역 명세가 필요합니다.
- **SR5 회복:** 0.3 m·3 s·1 s는 실험 설계값이며 기본 참고 지표입니다. hold 전체가 평가 창 안에 있어야 합니다.
- **SR6 명령 한계:** 차량/안전 필터에서 설정한 각도·속도 한계. 90°/s를 실제 액추에이터 물리 한계나 법규값으로 주장하지 않습니다. v3부터 enforce=true이면 판정과 overall에 반영합니다.

`pass`는 구성된 관측 기준 통과이며 실차 안전·SR3 안정 영역 보장·법규 인증이 아닙니다.
필수 송신 조향/한계나 전환 구간 관측이 부족하면 `assessment_complete=false`로 표시합니다.
누락 값을 0으로 대체하지 않습니다. 무대응 런에서 위험이 발견되지 않은 결과도 유한 관측 창에 의해
검열된 결과이지 영원히 무해하다는 뜻이 아닙니다.

결과 우선순위는 invalid → no_trigger → F1(인계 실패/지연) → F2(인계 조향 급변)
→ F3(차선·충돌·enforced SR6/SR5 위반) → pass입니다. 모든 위반은 failed_requirements에도 남깁니다.

## 전환 전후 주행 성능 지표

`metrics.monitor.diagnostics`에 전체 창, 전환 전, 전환 순간, 전환 이후 값을 기록합니다.
전환 이후는 기본 전환+0.5~+5 s이며 관측 부족은 truncated/unavailable로 표시합니다.

- 횡오차 RMSE/MAE/최대값, 최대 헤딩 오차
- 최소 차체 차선 여유, 요레이트/요가속도, 저속을 제외한 횡슬립
- 횡가속도, 횡jerk
- 국소 TLC: 차체 경계 여유 / 양의 경계 접근 속도. 실제 FTTI나 곡선도로 비선형 TLC와 다릅니다.
  미접근·저속·기하 미측정은 null과 사유를 기록하며 임의 통과 기준을 두지 않습니다.
- 같은 전환 시각으로 정렬한 baseline 구간 지표와 fault−baseline 차이

TreeIRL Appendix B.1의 |ay|=4.89 m/s², |yaw rate|=0.95 rad/s, |yaw acceleration|=1.93 rad/s²는
**쾌적성 벤치마크 참고값 일부**로만 비교합니다. 안전 판정·전체 nuPlan comfort score·ISO 2631 충족을
의미하지 않습니다. 횡jerk 하나를 전체 jerk vector norm 한계와 비교하지 않습니다.

기본 수집은 20 Hz입니다. Euro NCAP LSS v4.3 §4.1.1의 최소 100 Hz 시험과 같지 않고,
샘플 사이 peak를 놓칠 수 있습니다. 필터와 미분 방식은 결과에 기록합니다.

제공 문헌 24개의 확인 범위·적용/제외 사유는 `safety_spec.yaml.references`,
설계 결정은 [SAFETY_SPEC_REVIEW.md](SAFETY_SPEC_REVIEW.md)에 정리했습니다. 기준은 캠페인 결과를 보기 전에 고정합니다.
`load_spec()`는 실제 명세의 SHA-256을 기록합니다. 직접 MonitorConfig를 만들면 inline_unversioned로 표시됩니다.

## 데이터 계약과 피드백

기존 `junwoo/sim-eval`용 `monitor.evaluate_pair()`는 최상위 28개 필드를 유지합니다.
최신 `new_pipeline` 러너는 **37개 필드**이며, `runner_adapter.py`로 연결합니다.
확인한 러너 커밋: `d743fac6ec11b5f8c1ba53033dc2523483db59d1`.
`runner_skeleton.py`가 다시 내보내는 `pipeline/engine/judgment.py`와 실제
`pipeline/engine/monitor.py`를 읽어 스키마, 복구 판정, GP 출력의 일치를 검사했습니다.

어댑터는 해당 러너가 계산한 모든 필드·지표·복구 라벨·GP 타깃을 그대로 보존하고
`metrics.monitor`에 새 SR 판정을 추가합니다. 러너의 `switched_safe`와 SR의 `pass`는
정의가 다릅니다. 화면과 수치 인사이트는 `monitor.view()`를 통해 SR 판정을 사용합니다.
현재 `pipeline/run.py`는 engine 모니터를 직접 호출하므로, **09 파일만 수정한 이 변경으로
러너 전체 실행 경로가 자동 교체되지는 않습니다.** 아래 어댑터 진입점을 사용해야 합니다.

```powershell
python pipeline/09_monitor/runner_adapter.py --runner-root C:/path/to/new_pipeline --fault-run results/fault --baseline-run results/baseline --patterns pipeline/02_hypotheses/vulnerability_patterns.yaml --out results/run_records.jsonl
```

`--runner-root`는 실제 engine 소스가 있는 체크아웃입니다. new_pipeline에서 실행하면 생략합니다.
실행한 engine 파일의 SHA-256도 결과에 기록합니다. 기록된 고장 시작 시각과 `--onset`이
다르면 중단합니다. API/시뮬레이션을 호출하지 않고 이미 저장된 로그만 읽습니다.
`agent_controls.jsonl`의 정규화된 조향은 `agent_config.json.geometry.max_steer_rad`를
곱해 실제 road-wheel radians로 변환합니다. 명령이나 변환 한계가 없으면 미판정입니다.
새 SR 결과, 진단 지표, 명세 provenance는 `metrics.monitor` 아래에 저장합니다.
`monitor.view()`는 러너 결과와 상세 결과를 함께 읽는 화면용 뷰입니다.

- 기존 28필드 모니터의 5분류는 P01/P02/P03 기준입니다. 최신 37필드 러너는 recovery_outcome·baseline 상대 가속도/jerk 라벨 기준이며 어댑터가 이를 보존합니다. 상세 F1/F2/F3 판정과 같다고 가정하지 않습니다.
- `metrics`, `baseline_metrics`, `gp_targets`의 팀 지표 구조는 유지합니다.
- `switch_state_label`은 검증된 회복 가능 영역이 없어 null입니다.
- SR1/2/4와 enforced SR6(+설정 시 SR5)의 적용 가능한 최소 정규화 여유가 상세 overall입니다.
- 미적용 지표는 null이며, comfort/TLC/SR3 미판정 값은 안전 GP 타깃에 넣지 않습니다.

`feedback.py`는 원본 판정 모음으로 2단계 hypothesis_tracking.json, 3단계 failure_table.json,
6단계 gp_training.jsonl을 만듭니다. 가설의 조건군/대조군 기술통계는 체크포인트 전에도 집계하지만,
신뢰구간/최종 판정은 사전 정의된 10·20·30회 체크포인트를 따릅니다. 무효·결측·비유한 값과
불완전 안전 평가는 완전 데이터 분석에서 제외합니다. 화면의 관측 평균 차이와 정식 통계 판정은 구분합니다.

## Cosmos Reason 수치 인사이트

`explain.py`는 판정 모음 전체(통과·실패·무효·미발동)를 읽고, 결과 개수·유효 시험 실패율·지표 통계를
코드로 계산합니다. Cosmos에는 실험 조건, 판정, baseline/진단 지표와 계산된 통계를 텍스트로 전달합니다.
모델은 기존 판정을 바꾸지 않고 패턴·해석·다음 실험을 제안합니다. 인용 run_id, 지표 경로, 값은 코드가 검증합니다.
이 검증은 인용 일치만 확인하며 해석의 인과성을 보장하지 않습니다.

- 기본 40개 런/JSON 60,000자 한도 배치 분석 후 계층적 종합. 조용히 레코드를 자르지 않습니다.
- `--failures-only`: 개별 입력만 실패로 제한하고 전체 통계/선택 편향은 명시합니다.
- 같은 데이터·모델·프롬프트·옵션은 캐시합니다. `--force` 재생성, `--redo-errors` 오류 재시도.
- 깨진 JSON, 잘린 응답, 없는 ID·지표·다른 값은 오류로 저장하며 통과 해설로 취급하지 않습니다.
- `insight_dataset.json`은 정리된 입력, `insights.json`은 생성 결과입니다. API 키는 입력/캐시에 넣지 않습니다.

저장소 루트 `.env`를 실행 위치에 관계없이 자동으로 읽으며, 기존 환경변수가 우선합니다.
`.env.example`을 복사해 설정합니다. `.env`는 Git에 포함하지 않습니다.

```dotenv
NEBIUS_TOKEN_FACTORY_KEY=
NEBIUS_TOKEN_FACTORY_BASE_URL=https://api.tokenfactory.nebius.com/v1/
COSMOS_MODEL=
```

`COSMOS_MODEL`에는 해당 계정 모델 ID 또는 배포한 Dedicated 모델의 routing key를 입력합니다.
`--check`는 목록 조회로 설정을 확인하며 추론/과금 가능 여부나 배포 readiness까지 보장하지 않습니다.

```powershell
python pipeline/09_monitor/explain.py --check
python pipeline/09_monitor/explain.py --records results/run_records.jsonl --screen results/screen --open
python pipeline/09_monitor/explain.py --records results/run_records.jsonl --screen results/screen --watch
```

## 화면과 실패 리플레이

- 왼쪽 위: 검증 대상과 최근 시나리오의 전환 결과·실패 리플레이
- 왼쪽 아래: Nemotron 취약 가설, 비교군 정의, 표본 수·평균·차이·신뢰구간·판정
- 오른쪽 위: 전환 판정 기록
- 오른쪽 아래: Cosmos Reason 인사이트. 해석을 먼저 보여주고 근거·실험·한계는 접습니다.

`replay.py`는 `metrics.monitor.run_dirs.fault`의 view.mp4/front.mp4에서 실패 구간을 자릅니다.
고장 1초 전부터 전환/위반 3초 후까지이며 평가 창을 넘지 않습니다. 프레임 시각 로그가 없으면
근사 시각임을 표시합니다. 모델에는 영상을 보내지 않습니다. ffmpeg/imageio-ffmpeg가 없어도 수치 분석은 됩니다.
기존 commentary.jsonl은 읽거나 생성하지 않습니다. feedback.py의 --judgments는 과거 영상 결과 호환용입니다.

## 실행

기본 판정 의존성은 numpy/PyYAML입니다. 원본 녹화는 av_stack의 `run_benchmark.py --record-front`로 생성합니다.

```powershell
python pipeline/09_monitor/monitor.py --fault-run runs/f1 --baseline-run runs/b1 --no-reaction-run runs/n1 --onset 25 --run-info runs/f1/run_info.json --out results/run_records.jsonl --patterns pipeline/02_hypotheses/vulnerability_patterns.yaml
python pipeline/09_monitor/feedback.py --records results/run_records.jsonl --out results/feedback --hypotheses pipeline/02_hypotheses/hypotheses.yaml --patterns pipeline/02_hypotheses/vulnerability_patterns.yaml
python pipeline/09_monitor/dashboard.py --records results/run_records.jsonl --out results/screen --open --watch
```

원본 기록이 바뀌면 이전 데이터의 인사이트는 숨기고 재분석 안내를 표시합니다.
예시 캠페인 32건은 합성 추적이며 조건군/대조군 각 10건의 평균·99% 신뢰구간·가설 지지는
계산과 화면의 예시일 뿐 실제 차량/Nemotron/Cosmos 실험 결과가 아닙니다.

## 검증

```powershell
python pipeline/09_monitor/test_monitor.py
python pipeline/09_monitor/test_spec_v3.py
python pipeline/09_monitor/test_explain.py
python pipeline/09_monitor/test_replay.py
python pipeline/09_monitor/test_probe.py
```

새 v3 회귀 테스트는 설정 한계 일관성, SR6 반영, 평가 창 밖 참조 방지, 측정 누락,
선형 TLC(`tlc_linear_min_s`), 저속 null, 실제 시각 jerk, baseline 짝 비교와 비정상 채널 처리를 검증합니다.

## 러너 호환성 검사 재현

```powershell
$env:SIL_RUNNER_ROOT="C:/path/to/new_pipeline"
python -m unittest discover -s pipeline/09_monitor -p test_runner_compat.py -v
```

실제 engine으로 만든 원본 결과와 어댑터 결과에서 `metrics.monitor`만 제거한 결과가
37개 필드 모두 정확히 같은지 비교합니다. 정상 복구, SR6 위반, 조향 로그 누락,
복구 창 부족, 충돌을 포함합니다. `SIL_RUNNER_ROOT`가 없으면 실제 러너 검사는 skip되므로
skip을 호환성 통과로 해석하지 않습니다. 검증 입력은 합성 로그이며 CARLA 실주행 검증은 아닙니다.
