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

`monitor.evaluate_pair()`는 최상위 28개 필드를 돌려줍니다. 루프(`run.py`)에서는 `engine/sr_checks.py`가
`grade_pair()`를 직접 불러 run_record의 `sr_*` 필드를 채웁니다 (예전 러너 어댑터와 호환성 검사는 삭제).

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
```

새 v3 회귀 테스트는 설정 한계 일관성, SR6 반영, 평가 창 밖 참조 방지, 측정 누락,
선형 TLC(`tlc_linear_min_s`), 저속 null, 실제 시각 jerk, baseline 짝 비교와 비정상 채널 처리를 검증합니다.

