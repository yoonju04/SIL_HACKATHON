"""engine/dashboard.py shows <results>/insights/insights.json as a bottom-right panel on the main screen."""
import json

from engine import dashboard as D


def _rec():
    return {"run_id": "R00-0000-saturate", "scenario_id": "SC-1", "sr_outcome": "F1_no_or_late_switch",
            "parameters": {"fault": {"mode": "saturate"}, "onset_event": "curve_entry"}}


def _insight(title, conf):
    return {"title": title, "finding": f"{title} 관측", "interpretation": "i", "confidence": conf,
            "evidence": [{"run_id": "R00-0000-saturate", "metric": "metrics.margin_delta_m", "value": -7.384}]}


def test_main_screen_renders_compact_insight_panel(tmp_path):
    (tmp_path / "run_records.jsonl").write_text(json.dumps(_rec()) + "\n", encoding="utf-8")
    (tmp_path / "insights").mkdir()
    (tmp_path / "insights" / "campaign.html").write_text("x", encoding="utf-8")
    (tmp_path / "insights" / "insights.json").write_text(json.dumps({
        "status": "complete", "summary": "요약 <b>", "model": "m",
        "statistics": {"failure_rate": 0.5, "graded_triggered_runs": 2, "failure_count": 1},
        "insights": [_insight("낮은 것", "low"), _insight("중간 것", "medium"), _insight("포화 고장", "high"),
                     _insight("두번째 높음", "high")],
        "next_experiments": [{"proposal": "직선 비교", "rationale": "r", "run_ids": ["R00-0000-saturate"]}],
        "limitations": ["표본 부족"]}, ensure_ascii=False), encoding="utf-8")
    page = D.refresh(tmp_path / "run_records.jsonl", tmp_path / "screen", spec={}).read_text(encoding="utf-8")
    panel = page[page.index('id="ins"'):]
    big = panel[:panel.index("<details>")]
    assert "실패율 50.0%" in big
    assert big.index("포화 고장") < big.index("두번째 높음") < big.index("중간 것")   # high confidence first
    assert "낮은 것" not in big                                                      # only top 3 are big
    details = panel[panel.index("<details>"):panel.index("</details>")]
    assert "직선 비교" in details and "표본 부족" in details and "요약 &lt;b&gt;" in details
    assert "../insights/campaign.html" in details
    assert 'id="insfold"' in panel and "position:fixed" not in page
    # a normal card in the right column, directly after the scenario list card
    right = page[page.index("지금까지 돌린 시나리오"):]
    assert right.index("</section>") < right.index('id="ins"') and right.index('id="ins"') < right.index("</div>\n</div>")


def test_main_screen_without_insights_shows_small_badge(tmp_path):
    (tmp_path / "run_records.jsonl").write_text(json.dumps(_rec()) + "\n", encoding="utf-8")
    page = D.refresh(tmp_path / "run_records.jsonl", tmp_path / "screen", spec={}).read_text(encoding="utf-8")
    assert 'class="insbadge"' in page and "아직 없음" in page
    assert 'id="ins"' not in page and page.index("지금까지 돌린 시나리오") < page.index('id="insbadge"')
