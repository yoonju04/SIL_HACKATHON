"""engine/insights.py: engine run_records -> 09 explain.py schema, and the offline (mock) path."""
import json
from pathlib import Path

from engine import insights as INS


def _engine_record(run_id="R00-0000-saturate", sr_outcome="F1_no_or_late_switch", valid=True, status="ok"):
    return {"run_id": run_id, "scenario_id": "SC-1", "round": 0, "outcome_category": "switched_unsafe",
            "sim_status": status, "sr_outcome": sr_outcome, "sr_failed_requirements": ["SR1"],
            "parameters": {"route_id": "bench2drive:1", "environment": {"max_speed_kmh": 35.0},
                           "fault": {"fault_id": "F-06", "mode": "saturate", "params": {"duration_s": 0.5}},
                           "onset_event": "curve_entry", "onset_offset_s": -0.3, "onset_s": 46.4},
            "metrics": {"lane_recovery_robustness": -2.0},
            "sr_checks": {"valid": valid, "invalid_reasons": [] if valid else ["no baseline"],
                          "sr_outcome": sr_outcome, "sr": {"SR1": {"latency_s": 0.75}},
                          "sr_scores": {"SR1": -0.5}, "failed_requirements": ["SR1"], "assessment_complete": True}}


def test_adapt_record_maps_sr_schema(tmp_path):
    (tmp_path / "runs" / "R00-0000-saturate").mkdir(parents=True)
    r = INS.adapt_record(_engine_record(), tmp_path)
    assert r["outcome_category"] == "F1_no_or_late_switch"
    assert r["runner_outcome"] == "switched_unsafe"
    assert r["sim_status"] == "valid"
    assert r["fault_plan"]["fault_type"] == "saturate" and r["fault_plan"]["onset_event"] == "curve_entry"
    assert r["environment"] == {"route_id": "bench2drive:1", "max_speed_kmh": 35.0}
    assert r["sr"]["SR1"]["latency_s"] == 0.75 and r["assessment_complete"] is True
    assert r["run_dirs"]["fault"].endswith("R00-0000-saturate")
    bad = INS.adapt_record(_engine_record(valid=False))
    assert bad["outcome_category"] == "invalid" and bad["invalid_reason"] == "no baseline"


def test_explain_dataset_counts_engine_records():
    ex = INS.load_explain()
    recs = [INS.adapt_record(_engine_record()), INS.adapt_record(_engine_record("R01", "pass"))]
    ds = ex.build_dataset(recs)
    s = ds["summary"]
    assert s["graded_triggered_runs"] == 2 and s["failure_count"] == 1
    assert s["groups"][0]["fault_type"] == "saturate"
    assert ds["records"][0]["metrics"]["lane_recovery_robustness"] == -2.0


def test_run_insights_mock_offline(tmp_path):
    with (tmp_path / "run_records.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_engine_record()) + "\n")
    res = INS.run_insights(tmp_path, mode="mock")
    assert res["status"] == "skipped_mock" and res["model_calls"] == 0
    assert (tmp_path / "insights" / "insight_dataset.json").is_file()
    assert (tmp_path / "insights" / "campaign.html").is_file()
    assert not (tmp_path / "insights" / "insights.json").exists()


class _FakeClient:
    model, base, calls, ready = "fake/cosmos", "http://fake/", 0, True

    def chat(self, messages, max_tokens=4000, temperature=0.2):
        self.calls += 1
        return json.dumps({"summary": "s", "insights": [{"title": "t", "finding": "f", "interpretation": "i",
                           "confidence": "low", "evidence": [{"run_id": "R00-0000-saturate",
                           "metric": "metrics.lane_recovery_robustness", "value": -2.0}]}],
                           "next_experiments": [], "limitations": []})


def test_run_insights_real_path_with_fake_client(tmp_path):
    with (tmp_path / "run_records.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_engine_record()) + "\n")
    res = INS.run_insights(tmp_path, mode="real", client=_FakeClient())
    assert res["status"] == "complete" and res["n_insights"] == 1 and res["model_calls"] == 1
    row = json.loads((tmp_path / "insights" / "insights.json").read_text(encoding="utf-8"))
    assert row["statistics"]["failure_count"] == 1
