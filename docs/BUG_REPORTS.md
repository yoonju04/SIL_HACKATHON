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
- **확인:** 같은 명령으로 재실행 (아래 결과 갱신).
