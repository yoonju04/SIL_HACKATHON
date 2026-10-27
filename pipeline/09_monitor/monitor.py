#!/usr/bin/env python3
"""09 Monitor CLI: one sim-eval av_stack run folder (+ optional no-fault baseline) -> run_record JSON.

  python pipeline/09_monitor/monitor.py RUN_DIR [--baseline BASELINE_DIR] [--out record.json]

sim-eval's integration/monitor_logs.py refers to this file and its loader ``load_av_stack_run``.
Implementation: pipeline/engine/monitor.py. Thresholds are PROVISIONAL (02 outcome_definitions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.engine.config import load_stage_configs  # noqa: E402
from pipeline.engine.monitor import (compute_run_record, load_av_stack_run,  # noqa: E402,F401
                                     monitor_timeline, thresholds_from_config)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    cfg = load_stage_configs()
    th = thresholds_from_config(cfg)
    run = load_av_stack_run(args.run_dir)
    base = load_av_stack_run(args.baseline) if args.baseline else None
    rec = compute_run_record(run, base, metric_rules=cfg["vulnerability_patterns"]["metric_rules"], th=th)
    rec["_timeline"] = monitor_timeline(run, rec, th)
    text = json.dumps(rec, indent=2, ensure_ascii=False, default=str)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
