"""Campaign insights: the 09 monitor's text-only Cosmos analysis (pipeline/09_monitor/explain.py) on engine runs.

explain.py reads SR-monitor style records (outcome_category = F1/F2/F3/pass/invalid, sim_status == "valid",
sr/sr_scores/... at the top level). Engine run_records keep the engine verdict in outcome_category and the SR
monitor result under sr_outcome / sr_checks. This module maps one to the other WITHOUT touching the source
run_records.jsonl, and calls explain.process() with a client built from the engine's C_cosmos credentials
(pipeline/llm_models.yaml: COSMOS_MODEL_ID / COSMOS_BASE_URL / COSMOS_API_KEY, else NEBIUS_API_KEY).

Outputs go to <out>/insights/ so the engine dashboard (<out>/screen/campaign.html) is never overwritten:
  run_records_sr.jsonl   adapted records (input of explain.py)
  insight_dataset.json   exactly what the model sees (numbers only, no video)
  insights.json          validated insights (status complete / error / empty)
  campaign.html          09 monitor dashboard with the insights panel

Standalone on an existing results folder:
  python -m engine.insights results/real_12runs_1028 [--force] [--max-tokens 12000]
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from .config import PIPELINE_DIR, REPO_ROOT

MONITOR_DIR = PIPELINE_DIR / "09_monitor"
# SR-monitor fields that the engine nests under sr_checks; explain.project_record expects them at the top level.
SR_KEYS = ("sr", "sr_scores", "sr_measures", "failed_requirements", "sr_labels", "detection_latency_s", "ftti_s",
           "assessment_complete", "unassessed_requirements", "evaluation_window", "spec_provenance")
# sr_checks.diagnostics (report_only, ~3 KB/run) is left out: it pushed 12 runs past explain.py's input budget, and split
# batches made the model cite run ids it could not see.
METRIC_MIRRORS = ("detection_latency_s", "ftti_s")
DEFAULT_MAX_TOKENS = 12000      # Cosmos reasoner thinks before answering; explain.py raises on finish_reason=length


def load_explain():
    """pipeline/09_monitor/explain.py as the top-level module `explain` (it imports feedback/monitor/dashboard)."""
    if str(MONITOR_DIR) not in sys.path:
        sys.path.insert(0, str(MONITOR_DIR))
    return importlib.import_module("explain")


def _round(value, ndigits: int = 3):
    """Round floats so the model can copy evidence values exactly (explain.py compares them with ==)."""
    if isinstance(value, float):
        return round(value, ndigits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _round(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, ndigits) for v in value]
    return value


def adapt_record(rec: dict, out: Path | None = None) -> dict:
    """Engine run_record -> explain.py / 09 dashboard schema. The engine verdict is kept as runner_outcome."""
    r = dict(rec)
    sc = rec.get("sr_checks") if isinstance(rec.get("sr_checks"), dict) else {}
    for k in SR_KEYS:
        if r.get(k) is None and sc.get(k) is not None:
            r[k] = sc[k]
    if r.get("failed_requirements") is None and rec.get("sr_failed_requirements") is not None:
        r["failed_requirements"] = rec["sr_failed_requirements"]
    # The model tends to cite every number as metrics.<name>; mirror the SR timing scalars there (values unchanged).
    metrics = dict(r.get("metrics") or {})
    for k in METRIC_MIRRORS:
        if k not in metrics and isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool):
            metrics[k] = r[k]
    r["metrics"] = metrics
    r["runner_outcome"] = rec.get("outcome_category")          # presence also makes monitor.view() a no-op
    sr_outcome = rec.get("sr_outcome") or sc.get("sr_outcome")
    if sc.get("valid") is False:
        sr_outcome = "invalid"
        r["invalid_reason"] = r.get("invalid_reason") or "; ".join(map(str, sc.get("invalid_reasons") or [])) or None
    r["outcome_category"] = sr_outcome or "invalid"
    status = rec.get("sim_status")
    r["sim_status"] = "valid" if status in ("ok", "valid") else (status or "invalid")
    p = rec.get("parameters") or {}
    f = p.get("fault") or {}
    if not rec.get("fault_plan"):
        r["fault_plan"] = {"fault_type": f.get("mode"), "fault_id": f.get("fault_id"), "params": f.get("params"),
                           "onset_event": p.get("onset_event"), "onset_offset_s": p.get("onset_offset_s"),
                           "onset_s": p.get("onset_s")}
    if not rec.get("environment"):
        r["environment"] = {"route_id": p.get("route_id"), **(p.get("environment") or {})}
    if out is not None and not r.get("run_dirs") and rec.get("run_id"):
        d = Path(out) / "runs" / str(rec["run_id"])
        if d.is_dir():
            r["run_dirs"] = {"fault": str(d)}                   # 09 replay clips for failed runs (human view only)
    return _round(r)


def cosmos_client(llm_cfg: dict, timeout_s: float = 300.0):
    """explain.ModelClient with the same Cosmos endpoint/key/model resolution as the engine's C_cosmos role."""
    from .llm import TokenFactoryClient, role_settings
    role = "C_cosmos"
    s = role_settings(llm_cfg, role)
    if not s.get("model"):                                     # same fallback as run.py Loop._setup_clients
        role, s = "C_fallback", role_settings(llm_cfg, "C_fallback")
    tf = TokenFactoryClient.from_config(llm_cfg, role="C_cosmos" if role == "C_cosmos" else None)
    ex = load_explain()
    client = ex.ModelClient(key=tf._key or "", base_url=tf.base_url, model=s.get("model") or "", timeout=timeout_s)
    return client, role


def run_insights(out: Path, *, mode: str = "real", llm_cfg: dict | None = None, force: bool = False,
                 max_tokens: int = DEFAULT_MAX_TOKENS, client=None, attempts: int = 2) -> dict:
    """Write <out>/insights/. mode == "mock": dataset + dashboard only, no network."""
    out = Path(out)
    src = out / "run_records.jsonl"
    dst = out / "insights"
    dst.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if src.is_file() else []
    rec_path = dst / "run_records_sr.jsonl"
    tmp = rec_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(adapt_record(rec, out), ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, rec_path)
    ex = load_explain()
    result: dict[str, Any] = {"path": str(dst / "insights.json"), "html": str(dst / "campaign.html"),
                              "n_records": len(records)}
    if mode == "mock":
        dataset = ex.build_dataset(ex.read_records(rec_path))
        ex._write_json(dst / "insight_dataset.json", dataset)
        from dashboard import refresh
        refresh(rec_path, dst)
        result.update(status="skipped_mock", path=str(dst / "insight_dataset.json"), model_calls=0, n_insights=0,
                      error=None, role=None, model=None)
        return result
    if client is None:
        if llm_cfg is None:
            from .config import load_stage_configs
            llm_cfg = load_stage_configs()["llm"]
        client, role = cosmos_client(llm_cfg)
    else:
        role = "custom"
    calls, errors = 0, []
    for attempt in range(max(1, attempts)):
        # explain.py rejects the whole answer if one evidence value/path is wrong; sampling again usually fixes it.
        calls += ex.process(rec_path, dst, client, force=force or attempt > 0, max_tokens=max_tokens) or 0
        row = json.loads((dst / "insights.json").read_text(encoding="utf-8"))
        errors.extend(row.get("rejected_answers") or [])     # answers explain.py rejected and re-asked
        if row.get("status") != "error" or not client.ready:
            break
    row["model_calls"] = calls if calls else row.get("model_calls")
    result.update(status=row.get("status"), error=row.get("error"), model_calls=row.get("model_calls"),
                  n_insights=len(row.get("insights") or []), model=row.get("model"), role=role,
                  rejected_attempts=errors)
    return result


def _load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Same rules as run.load_dotenv: existing env wins, values never printed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not v.startswith("replace-with") and k not in os.environ:
            os.environ[k] = v


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="09 monitor Cosmos campaign insights on an engine results folder")
    ap.add_argument("out", type=Path, help="results folder containing run_records.jsonl")
    ap.add_argument("--force", action="store_true", help="re-run the model even if the dataset is unchanged")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--attempts", type=int, default=2, help="whole-analysis re-runs (explain.py re-asks each rejected answer twice with the reason)")
    ap.add_argument("--mock", action="store_true", help="no network: dataset + dashboard only")
    a = ap.parse_args(argv)
    if not a.mock:
        _load_dotenv()
    res = run_insights(a.out, mode="mock" if a.mock else "real", force=a.force, max_tokens=a.max_tokens,
                       attempts=a.attempts)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("status") in ("complete", "skipped_mock", "empty") else 1


if __name__ == "__main__":
    sys.exit(main())
