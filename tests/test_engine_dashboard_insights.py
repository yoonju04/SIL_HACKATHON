"""engine/dashboard.py shows <results>/insights/insights.json on the main screen."""
import json

from engine import dashboard as D


def _rec():
    return {"run_id": "R00-0000-saturate", "scenario_id": "SC-1", "sr_outcome": "F1_no_or_late_switch",
            "parameters": {"fault": {"mode": "saturate"}, "onset_event": "curve_entry"}}


def test_main_screen_renders_insights(tmp_path):
    (tmp_path / "run_records.jsonl").write_text(json.dumps(_rec()) + "\n", encoding="utf-8")
    (tmp_path / "insights").mkdir()
    (tmp_path / "insights" / "campaign.html").write_text("x", encoding="utf-8")
    (tmp_path / "insights" / "insights.json").write_text(json.dumps({
        "status": "complete", "summary": "요약 <b>", "model": "m", "statistics": {"failure_rate": 0.5},
        "insights": [{"title": "포화 고장", "finding": "f", "interpretation": "i", "confidence": "high",
                      "evidence": [{"run_id": "R00-0000-saturate", "metric": "metrics.margin_delta_m", "value": -7.384}]}],
        "next_experiments": [{"proposal": "직선 비교", "rationale": "r", "run_ids": ["R00-0000-saturate"]}],
        "limitations": ["표본 부족"]}, ensure_ascii=False), encoding="utf-8")
    page = D.refresh(tmp_path / "run_records.jsonl", tmp_path / "screen", spec={}).read_text(encoding="utf-8")
    assert "Cosmos 캠페인 인사이트" in page and "포화 고장" in page and "-7.384" in page
    assert "직선 비교" in page and "표본 부족" in page and "요약 &lt;b&gt;" in page
    assert "../insights/campaign.html" in page
    assert 'id="run-' in page                       # run panels still there


def test_main_screen_without_insights(tmp_path):
    (tmp_path / "run_records.jsonl").write_text(json.dumps(_rec()) + "\n", encoding="utf-8")
    page = D.refresh(tmp_path / "run_records.jsonl", tmp_path / "screen", spec={}).read_text(encoding="utf-8")
    assert "아직 인사이트가 없습니다" in page
