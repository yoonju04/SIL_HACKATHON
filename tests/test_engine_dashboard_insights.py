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
    assert "인사이트" in panel and "AI 분석 요약" not in panel
    assert "실패 비율" in panel and "50.0%" in panel
    assert panel.index("포화 고장") < panel.index("두번째 높음") < panel.index("중간 것")   # high confidence first
    assert "낮은 것" not in panel                                                      # only top 3
    assert "<details" not in panel and "원문" not in panel
    assert "position:fixed" not in page
    right = page[page.index("지금까지 돌린 시나리오"):]
    assert right.index("</section>") < right.index('id="ins"') and right.index('id="ins"') < right.index("</div>\n</div>")


def test_main_screen_without_insights_shows_small_badge(tmp_path):
    (tmp_path / "run_records.jsonl").write_text(json.dumps(_rec()) + "\n", encoding="utf-8")
    page = D.refresh(tmp_path / "run_records.jsonl", tmp_path / "screen", spec={}).read_text(encoding="utf-8")
    assert 'class="insbadge"' in page and "아직 없음" in page
    assert 'id="ins"' not in page and page.index("지금까지 돌린 시나리오") < page.index('id="insbadge"')


def test_live_status_shown_in_header(tmp_path):
    from engine import dashboard as D
    page = D.refresh(tmp_path / "none.jsonl", tmp_path / "screen", status="라운드 0: 시뮬레이션 중")
    html = page.read_text(encoding="utf-8")
    assert '<span class="live">라운드 0: 시뮬레이션 중</span>' in html
    assert "location.reload" in html          # self-reload (works on file:// pages)


def test_method_card_counts_a_hypotheses(tmp_path):
    import json
    from engine import dashboard as D
    camp = tmp_path / "camp"
    (camp / "round_00").mkdir(parents=True)
    (camp / "01_profile_summary.json").write_text(json.dumps({"observations": 25, "verified": 0}), encoding="utf-8")
    pool = [{"scenario_id": "SA", "source": "nemotron_a", "expected_side": "fail", "expected_mechanism": "m"},
            {"scenario_id": "SB", "source": "nemotron_a", "expected_side": "recover"},
            {"scenario_id": "SR", "source": "space_filling"}]
    (camp / "round_00" / "03_04_pool.json").write_text(json.dumps(pool), encoding="utf-8")
    (camp / "round_00" / "05b_answers.json").write_text(json.dumps(
        {"k": {"answers": {"Q-B1": {"evidence": [{"file": "lqr_controller/lqr.py", "line": 3}]}}}}), encoding="utf-8")
    recs = [{"run_id": "R1", "scenario_id": "SA", "seed_scenario_id": "SA", "recovery_outcome": "unrecovered",
             "final_safe_recovery": False, "lane_recovery_robustness": -2.0},
            {"run_id": "R2", "scenario_id": "SB", "seed_scenario_id": "SB", "recovery_outcome": "no_switch",
             "final_safe_recovery": None, "lane_recovery_robustness": None},
            {"run_id": "R3", "scenario_id": "SX", "seed_scenario_id": "SR", "recovery_outcome": "safe_recovery",
             "final_safe_recovery": True, "lane_recovery_robustness": 0.9}]
    (camp / "run_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    m = D.load_method(camp / "screen")
    assert m["n_obs"] == 25 and m["n_props"] == 2 and m["b_files"] == ["lqr_controller/lqr.py"]
    assert [(x["run_id"], x["hit"]) for x in m["tests"]] == [("R1", True), ("R2", None)]
    html = D.render_method(camp / "screen")
    assert "예상 적중 1/1" in html and "전환 없음 1" in html and "lqr.py" in html


def test_stability_short_no_switch_is_held_not_green():
    from engine import dashboard as D
    base = {"display_outcome": "fail", "recovery_outcome": "no_switch", "failed_requirements": ["SR5"],
            "sr": {"SR4": {"pass": True, "collision": False}}}
    s = D._stability_short(base)
    assert "판정 보류" in s and "✓" not in s and 'class="mut"' in s
    lane = dict(base, failed_requirements=["SR4", "SR5"], sr={"SR4": {"pass": False, "collision": False}})
    assert "✗" in D._stability_short(lane) and "차선" in D._stability_short(lane)
    hit = dict(base, failed_requirements=["SR4"], sr={"SR4": {"pass": False, "collision": True}})
    assert "충돌" in D._stability_short(hit)
    ok = {"display_outcome": "pass", "recovery_outcome": "safe_recovery", "failed_requirements": [],
          "sr": {"SR4": {"pass": True}}}
    assert "✓" in D._stability_short(ok)


def test_checks_no_switch_rows_have_no_green_but_keep_lane_and_collision():
    from engine import dashboard as D
    lim = {"sr2": 60, "sr6": 90, "win": 5, "acc": 3.0, "rj": {}}
    r = {"recovery_outcome": "no_switch", "switch_time": None,
         "sr": {"SR4": {"pass": False, "collision": True, "margin_m": -0.4}, "SR6": {"applies": True, "pass": True}},
         "recovery_labels": {"lat_acc_excess": {"primary": False}}}
    marks = [ok for _, ok, _, _ in D._checks(r, lim)]
    assert True not in marks
    assert marks[1] is False and marks[2] is False
    clean = dict(r, sr={"SR4": {"pass": True, "collision": False, "margin_m": 0.5}})
    assert all(ok is None for _, ok, _, _ in D._checks(clean, lim))
    sw = dict(clean, recovery_outcome="safe_recovery", switch_time=10.0)
    assert True in [ok for _, ok, _, _ in D._checks(sw, lim)]
