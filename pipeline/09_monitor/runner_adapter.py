"""Current new_pipeline runner -> v3 SR monitor bridge; modifies only metrics.monitor.

The runner owns its 37-field recovery contract. Its labels/GP targets are preserved;
the v3 switching verdict is separate and is read through monitor.view().
No CARLA execution, LLM call, or network access occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import monitor as sr_monitor

TESTED_RUNNER_COMMIT = "d743fac6ec11b5f8c1ba53033dc2523483db59d1"


def load_runner(root):
    """Load actual engine sources without replacing local monitor or pipeline modules."""
    engine = Path(root).resolve() / "pipeline" / "engine"
    name = "_sil_runner_" + hashlib.sha256(str(engine).encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name + ".monitor"]
    for filename in ("judgment.py", "monitor.py"):
        if not (engine / filename).is_file():
            raise FileNotFoundError(f"{engine / filename}: current new_pipeline runner required; use --runner-root")
    package = types.ModuleType(name)
    package.__path__ = [str(engine)]
    sys.modules[name] = package
    try:
        for module in ("judgment", "monitor"):
            spec = importlib.util.spec_from_file_location(name + "." + module, engine / (module + ".py"))
            obj = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = obj
            spec.loader.exec_module(obj)
        return sys.modules[name + ".monitor"]
    except Exception:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        raise


def evaluate_folder(fault_dir, baseline_dir, *, runner_root=None, cfg=None,
                    onset=None, no_reaction_dir=None, run_info=None, metric_rules=None,
                    stage_config=None):
    root = Path(runner_root) if runner_root else Path(__file__).resolve().parents[2]
    runner = load_runner(root)
    fault = runner.load_av_stack_run(fault_dir)
    baseline = runner.load_av_stack_run(baseline_dir)
    info = dict(run_info or {})
    th = runner.thresholds_from_config(stage_config)
    record = runner.compute_run_record(
        fault, baseline, candidate=info, metric_rules=metric_rules,
        th=th, run_id=info.get("run_id"), round_index=info.get("round"), selector=info.get("selector"))
    if tuple(record) != runner.J.RUN_RECORD_FIELDS:
        raise ValueError("runner output no longer matches RUN_RECORD_FIELDS; review the adapter")
    planned = record.get("planned_fault_onset")
    if onset is None:
        onset = planned
    if onset is None:
        raise ValueError("fault onset is not logged; pass --onset explicitly")
    if planned is not None and abs(onset - planned) > 1e-6:
        raise ValueError("explicit onset differs from runner planned_fault_onset")
    detail = sr_monitor.grade_pair(
        sr_monitor.load_av_stack_run(fault_dir), sr_monitor.load_av_stack_run(baseline_dir),
        onset, cfg or sr_monitor.load_spec(Path(__file__).with_name("safety_spec.yaml")),
        sr_monitor.load_av_stack_run(no_reaction_dir) if no_reaction_dir else None, info)
    detail["run_id"] = record["run_id"]
    detail["run_dirs"] = {"fault": str(Path(fault_dir).resolve()),
                          "baseline": str(Path(baseline_dir).resolve()),
                          "no_reaction": str(Path(no_reaction_dir).resolve()) if no_reaction_dir else None}
    detail["runner_contract"] = {
        "tested_commit": TESTED_RUNNER_COMMIT,
        "engine_sha256": {name: hashlib.sha256((root / "pipeline" / "engine" / name).read_bytes()).hexdigest()
                          for name in ("judgment.py", "monitor.py")},
        "field_count": len(record), "labels_preserved": True,
        "note": "runner recovery labels and SR switching verdict are separate definitions"}
    record["metrics"]["monitor"] = detail
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--no-reaction-run")
    parser.add_argument("--runner-root", type=Path)
    parser.add_argument("--onset", type=float)
    parser.add_argument("--spec", default=str(Path(__file__).with_name("safety_spec.yaml")))
    parser.add_argument("--patterns", help="02 vulnerability_patterns.yaml: GP rules and recovery thresholds")
    parser.add_argument("--run-info")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    stage, rules = None, None
    if args.patterns:
        import yaml
        patterns = yaml.safe_load(Path(args.patterns).read_text(encoding="utf-8"))
        stage, rules = {"vulnerability_patterns": patterns}, patterns.get("metric_rules")
    info = json.loads(Path(args.run_info).read_text(encoding="utf-8")) if args.run_info else None
    record = evaluate_folder(args.fault_run, args.baseline_run, runner_root=args.runner_root,
                             cfg=sr_monitor.load_spec(args.spec), onset=args.onset,
                             no_reaction_dir=args.no_reaction_run, run_info=info,
                             metric_rules=rules, stage_config=stage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"saved {record['run_id']}: runner={record['outcome_category']}, "
          f"SR={record['metrics']['monitor']['failure_type']}")


if __name__ == "__main__":
    main()
