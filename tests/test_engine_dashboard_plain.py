"""engine/dashboard.py one-screen view: topic header, method band, hypotheses + insights, scenario TABLE,
terse run lines; technical block always visible (no fold)."""
import json
import re

from engine import dashboard as D


def _rec(run_id="R00-0000-saturate", mode="saturate", params=None, outcome="F1_no_or_late_switch", note=None,
         onset=46.5, switch=47.25, latency=0.75, violation=46.8, collision_t=51.45, margin=-7.27, recovery="unrecovered"):
    sr4 = {"applies": True, "pass": violation is None and collision_t is None, "margin_m": margin,
           "first_violation_t": violation, "collision": collision_t is not None}
    return {"run_id": run_id, "scenario_id": "SC-1", "round": 0, "sim_status": "ok", "sr_outcome": outcome,
            "actual_fault_onset": onset, "planned_fault_onset": onset, "switch_time": switch,
            "detection_latency_s": latency, "recovery_outcome": recovery, "sr_failed_requirements": ["SR1", "SR4"],
            "selector": "acquisition", "lane_recovery_robustness": -2.0,
            "sr_checks": {"sr": {"SR4": sr4, "SR1": {"applies": True, "pass": False}}, "note": note,
                          "sr_measures": {"SR4_collision_t": collision_t}},
            "parameters": {"route_id": "bench2drive:28035", "environment": {"max_speed_kmh": 30.0, "tire_friction_scale": 0.6},
                           "fault": {"fault_id": "F-06", "mode": mode,
                                     "params": params if params is not None else {"duration_s": 0.5, "magnitude": 0.5}},
                           "onset_event": "curve_entry"}}


def _harmless():
    return _rec("R01-0004-bias", "bias", {"magnitude_rad": 0.0399}, "pass", "undetected_harmless", switch=None,
                latency=None, violation=None, collision_t=None, margin=0.37, recovery="no_switch")


def _invalid():
    r = _rec("R01-0009-stale", "stale", {}, "invalid", switch=None, latency=None, violation=None, collision_t=None,
             margin=None, recovery="simulation_invalid")
    r["sim_status"], r["invalid_reason"] = "invalid", "baseline evaluation window too short"
    return r


def _main_view(page):
    body = page.split("<body", 1)[1]
    body = re.sub(r"<script>.*?</script>", "", body, flags=re.S)
    body = body.split("</header>", 1)[1]               # the title may name MPC and LQR
    body = re.sub(r'<div class="tech">.*?</div>', " ", body, flags=re.S)  # tech block may use short codes
    while "<details" in body:
        body, n = re.subn(r"<details[^>]*>(?:(?!<details).)*?</details>", "", body, flags=re.S)
        if not n:
            break
    return re.sub(r"<[^>]+>", " ", body)


def _camp(tmp_path, recs, planned=(4, 1, 4)):
    camp = tmp_path / "camp"
    (camp / "round_00").mkdir(parents=True)
    (camp / "run_config.json").write_text(json.dumps(dict(zip(("cold_start", "rounds", "batch_size"), planned))), encoding="utf-8")
    (camp / "run_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return camp


def test_terse_line_uses_recorded_seconds_only():
    r = D.view(_rec())
    t = D.times(r)
    assert abs(t["notice"] - 0.75) < 1e-9 and abs(t["left"] - 0.3) < 1e-9 and abs(t["crash"] - 4.95) < 1e-9
    head, line = D.story(r)
    assert head == "Controller handover failed/late"
    assert line == "Lane departure 0.3 s after the fault · detected and switched late at 0.75 s · collision at 4.95 s"
    name, _, strength = D.fault_plain(r)
    assert name == "Steering Saturation" and strength == "50% of max angle · for 0.5 s"
    assert not re.search("[\uAC00-\uD7A3]", head + line)   # English only


def test_harmless_and_invalid_are_not_failures():
    h, i = D.view(_harmless()), D.view(_invalid())
    assert D._bucket(h) == "harmless" and D._bucket(i) == "na"
    assert D.story(h) == ("Undetected · harmless", "Not detected · Lane kept (harmless)")
    assert "not a failure" in D.story(i)[0]
    assert D.fault_plain(h)[0] == "Steering Bias" and D.fault_plain(h)[2] == "2.3°" and D.fault_plain(i)[2] == D.UNKNOWN


def test_one_screen_page(tmp_path):
    camp = _camp(tmp_path, [_rec(), _harmless(), _invalid()])
    page = D.refresh(camp / "run_records.jsonl", camp / "screen", spec={}, status="Round 0").read_text(encoding="utf-8")
    head = page[:page.index("</header>")]
    assert "MPC → LQR Handover Stability Evaluation" in head and "<h1>" in head and '<html lang="en">' in page
    assert "render_scorecards" not in page and 'class="cards"' not in page and "3/8" not in head
    assert "overflow:hidden" in page and "height:100vh" in page and "position:fixed" not in page
    assert 'font-family:"Inter"' in page or "font-family:\"Inter\"" in page or "Inter" in page
    flow = page[page.index('id="flow"'):page.index('class="grid"')]
    for step in ("Nemotron code analysis", "Hypothesis", "Fault scenario generation", "CARLA simulation",
                 "Statistical verification"):
        assert step in flow
    assert page.count("<tr class=") == 3 and "Controller handover failed/late" in page and "Undetected · harmless" in page
    for col in ("Scenario", "Fault", "Handover", "Driving Safety", "Video Commentary"):
        assert f">{col}</th>" in page, col
    assert ">Verdict</th>" not in page and "Recovery" not in page.split("<script")[0]
    assert 'table class="list"' in page
    strip = page[page.index('class="counts"'):page.index('class="listbox"')]
    for label in ("Total", "Pass", "Controller handover failed/late", "Unstable at handover", "Unstable after handover",
                  "Not tested · invalid"):
        assert label in strip, label
    assert "Verification failed" not in strip
    assert "Latest Scenario" in page and "Hypotheses" in page and "Insights" in page
    assert 'class="col left"' in page and 'class="col mid"' in page and 'class="col hypo"' in page and 'class="col ins"' in page
    assert 'class="tech"' in page and "<details" not in page.split("<script")[0]
    assert "Driving stability after handover" in page and "<th>Item</th>" in page and "<th>Measured</th>" in page \
        and "<th>Threshold</th>" in page                                      # full check table, always shown
    assert not re.search("[\uAC00-\uD7A3]", page)                          # all UI text is English
    main = _main_view(page)
    for word in ("F1", "F2", "F3", "lrr", "acquisition", "recovery v2", "FTTI"):
        assert word not in main, word
        assert "location.reload" in page
    assert "Pretendard Variable" in page and (camp / "screen" / "fonts" / "PretendardVariable.woff2").is_file()


def test_run_list_caps_at_max_rows(tmp_path):
    recs = [_rec(run_id=f"R00-{i:04d}-saturate") for i in range(D.MAX_ROWS + 3)]
    camp = _camp(tmp_path, recs, planned=(30, 0, 1))
    page = D.refresh(camp / "run_records.jsonl", camp / "screen", spec={}).read_text(encoding="utf-8")
    assert page.count("<tr class=") == D.MAX_ROWS and "+3 more" in page
    body = page.split('class="listbox')[1]
    assert 'class="listbox dense"' in page                                  # 12+ rows: one line per row
    assert f'data-id="R00-{D.MAX_ROWS + 2:04d}-saturate"' in body and 'data-id="R00-0000-saturate"' not in body


def test_hypotheses_come_from_campaign_files(tmp_path):
    camp = _camp(tmp_path, [dict(_rec(), scenario_id="SA", seed_scenario_id="SA", final_safe_recovery=False)])
    (camp / "02_objectives.json").write_text(json.dumps({"active": ["S1"], "S1_objective": {"question_ko": "전환 순간 오차가 크면 차선을 못 지키는가?",
                                                                                                  "question_en": "Does a large error at the switch break lane keeping?"}},
                                                        ensure_ascii=False), encoding="utf-8")
    prior = {"domain_prior_knowledge": {"principles": [
        {"id": "PK-001", "kind": "physical_relation", "equation": "a_y = v^2 k"},
        {"id": "PK-002", "kind": "testable_hypothesis", "statement": "Fixed-gain LQR degrades."},
        {"id": "PK-009", "kind": "testable_hypothesis", "statement": "Unmapped statement."}]}}
    (camp / "00_prior_payload.json").write_text(json.dumps({"messages": [{"role": "system", "content": "x"},
                                                                       {"role": "user", "content": json.dumps(prior)}]}), encoding="utf-8")
    (camp / "round_00" / "03_04_pool.json").write_text(json.dumps([{"scenario_id": "SA", "source": "nemotron_a",
                                                                    "expected_side": "recover", "evidence_ids": ["PK-002"]}]), encoding="utf-8")
    h = D.load_hypotheses(camp / "screen")
    assert h["questions"] == [("S1", "Does a large error at the switch break lane keeping?")]   # English first
    assert [p["id"] for p in h["prior"]] == ["PK-002", "PK-009"]          # only testable hypotheses, as filed
    assert h["prior"][0]["text"] == D.PK_KO["PK-002"] and h["prior"][1]["text"] == "Unmapped statement."
    assert (h["prior"][0]["n"], h["prior"][0]["miss"]) == (1, 1)          # expected lane kept, run was unsafe
    html = D.render_hypotheses(camp / "screen")
    assert "Hypotheses" in html and "Evaluation question" in html and "Hit 0" in html and "Miss 1" in html
    assert "Prior knowledge · predictions by Nemotron A" in html and "<details" not in html


def test_plain_text_keeps_numbers_and_ids():
    s = D.plain_text("saturate run (R00-0002-stale): metrics.lateral_error_rmse_m 5.413m, detection_latency_s 0.75s.")
    assert s == "Steering Saturation run (R00-0002-stale): lateral error RMS 5.413m, detection latency 0.75s."


def test_insight_panel_shows_key_points_unfolded(tmp_path):
    camp = tmp_path / "camp"
    (camp / "insights").mkdir(parents=True)
    (camp / "insights" / "insights.json").write_text(json.dumps({
        "status": "complete", "summary": "F1_no_or_late_switch 5 runs", "statistics": {"failure_rate": 0.625,
        "graded_triggered_runs": 8, "failure_count": 5},
        "insights": [{"title": "saturate faults did not recover", "finding": "margin_delta_m -7.644m", "confidence": "high"}]},
        ensure_ascii=False), encoding="utf-8")
    page = D.refresh(camp / "none.jsonl", camp / "screen", spec={}).read_text(encoding="utf-8")
    panel = page[page.index('id="ins"'):]
    assert "Insights" in panel
    assert "Failure rate" in panel and "62.5%" in panel and "(5 of 8 graded runs failed)" in panel
    assert "by Cosmos" in panel
    assert "Steering Saturation faults did not recover" in panel and "-7.644m" in panel and "saturate" not in panel
    assert "<details" not in panel


def test_insight_headline_and_evidence_from_structured_fields():
    it = {"title": "saturate faults cause high lateral_error_rmse_m", "confidence": "high", "finding": "x",
          "evidence": [{"run_id": "R00-0000-saturate", "metric": "metrics.lateral_error_rmse_m", "value": 5.285},
                       {"run_id": "R00-0000-saturate", "metric": "metrics.margin_delta_m", "value": -7.791},
                       {"run_id": "R01-0006-saturate", "metric": "metrics.lateral_error_rmse_m", "value": 5.541},
                       {"run_id": "R01-0006-saturate", "metric": "metrics.margin_delta_m", "value": -7.496}]}
    out = {"R00-0000-saturate": "F1_no_or_late_switch", "R01-0006-saturate": "F1_no_or_late_switch"}
    head, ev = D.insight_view(it, out)
    assert head == "Steering Saturation: Controller handover failed/late in 2/2 runs"
    assert ev == "Lateral error RMS avg 5.41 m · Lane margin change avg −7.64 m · R00-0000, R01-0006"
    apex = {"title": "faults at curve_apex mostly pass", "evidence": [
        {"run_id": "R00-0002-stale", "metric": "monitor_outcome", "value": "pass"},
        {"run_id": "R00-0003-bias", "metric": "monitor_outcome", "value": "pass"}]}
    assert D.insight_view(apex, {}) == ("Curve apex faults: Pass in 2/2 runs", "R00-0002, R00-0003")


def test_missing_clip_file_says_so_instead_of_empty_player(tmp_path):
    out = tmp_path / "screen"
    out.mkdir()
    gone = D._clip_html({"clip": {"video": str(tmp_path / "clips" / "R00-0000" / "clip.mp4")}}, out)
    assert "<video" not in gone and "No video file" in gone
    real = tmp_path / "clips" / "R00-0000" / "clip.mp4"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"\0")
    shown = D._clip_html({"clip": {"video": str(real), "start_s": 1.0, "end_s": 4.0}}, out)
    assert '<video class="clip" src="../clips/R00-0000/clip.mp4"' in shown
    assert D._clip_html({}, out) == ""


def test_question_falls_back_to_built_in_english(tmp_path):
    camp = _camp(tmp_path, [])
    (camp / "02_objectives.json").write_text(json.dumps({"active": ["S1"], "S1_objective": {"question_ko": "한국어 질문"}},
                                                        ensure_ascii=False), encoding="utf-8")
    assert D.load_hypotheses(camp / "screen")["questions"] == [("S1", D.QUESTION_EN["S1"])]
    lines = D._question_lines(D.QUESTION_EN["S1"])
    assert len(lines) == 2 and lines[1].startswith("and how hard")
