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
