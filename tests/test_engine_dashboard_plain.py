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
    body = body.split("</header>", 1)[1]               # the title/subtitle may name MPC, LQR and 감독기
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
    assert head == "제어기 전환 실패/늦음"
    assert line == "고장 0.3초 후 차선 이탈 · 0.75초 후 감지·전환 (늦음) · 4.95초 후 충돌"
    name, _, strength = D.fault_plain(r)
    assert name == "Steering Saturation" and strength == "최대각의 50% · 0.5초간"
    assert "어요" not in head + line                     # concise, no chatty endings


def test_harmless_and_invalid_are_not_failures():
    h, i = D.view(_harmless()), D.view(_invalid())
    assert D._bucket(h) == "harmless" and D._bucket(i) == "na"
    assert D.story(h) == ("미감지·무해", "미감지 · 차선 유지 (무해)")
    assert "실패 아님" in D.story(i)[0]
    assert D.fault_plain(h)[0] == "Steering Bias" and D.fault_plain(h)[2] == "2.3°" and D.fault_plain(i)[2] == D.UNKNOWN


def test_one_screen_page(tmp_path):
    camp = _camp(tmp_path, [_rec(), _harmless(), _invalid()])
    page = D.refresh(camp / "run_records.jsonl", camp / "screen", spec={}, status="라운드 0").read_text(encoding="utf-8")
    head = page[:page.index("</header>")]
    assert "MPC → LQR 제어기 전환 안정성 평가" in head and "<h1>" in head
    assert "render_scorecards" not in page and 'class="cards"' not in page and "3/8" not in head
    assert "overflow:hidden" in page and "height:100vh" in page and "position:fixed" not in page
    assert 'font-family:"Inter"' in page or "font-family:\"Inter\"" in page or "Inter" in page
    flow = page[page.index('id="flow"'):page.index('class="grid"')]
    for step in ("코드 분석", "가설 수립", "고장 시나리오 생성", "CARLA 시뮬레이션", "통계 검증"):
        assert step in flow
    assert page.count("<tr class=") == 3 and "제어기 전환 실패/늦음" in page and "미감지·무해" in page
    assert "시나리오" in page and "전환 여부" in page and "운행 안전성" in page and "영상 해설" in page
    assert ">판정</th>" not in page and "대응 결과" not in page.split("<script")[0]
    assert 'table class="list"' in page
    strip = page[page.index('class="counts"'):page.index('class="listbox"')]
    for label in ("전체", "통과", "제어기 전환 실패/늦음", "전환하는 순간 불안정", "전환 후 불안정", "시험 안 됨·무효"):
        assert label in strip, label
    assert "해설 검증 실패" not in strip
    assert "최근 시나리오" in page and "가설" in page and "인사이트" in page
    assert "AI 분석 요약" not in page
    assert 'class="col left"' in page and 'class="col mid"' in page and 'class="col hypo"' in page and 'class="col ins"' in page
    assert 'class="tech"' in page and page.split("<script")[0].count("자세히 보기") == 0
    assert "운행 안전성" in page and "대응 결과" not in page.split("<script")[0]
    assert "전환 후 운행 안정성" in page and "<th>측정값</th>" in page and "<th>기준</th>" in page   # full check table, always shown
    main = _main_view(page)
    for word in ("F1", "F2", "F3", "lrr", "acquisition", "recovery v2", "회복", "FTTI", "어요"):
        assert word not in main, word
    assert "무슨 일이 언제 일어났나" not in page and "전문가용 기록" not in page
    assert "location.reload" in page
    assert "Pretendard Variable" in page and (camp / "screen" / "fonts" / "PretendardVariable.woff2").is_file()


def test_run_list_caps_at_max_rows(tmp_path):
    recs = [_rec(run_id=f"R00-{i:04d}-saturate") for i in range(D.MAX_ROWS + 3)]
    camp = _camp(tmp_path, recs, planned=(30, 0, 1))
    page = D.refresh(camp / "run_records.jsonl", camp / "screen", spec={}).read_text(encoding="utf-8")
    assert page.count("<tr class=") == D.MAX_ROWS and "외 3개" in page
    body = page.split('class="listbox')[1]
    assert 'class="listbox dense"' in page                                  # 12+ rows: one line per row
    assert f'data-id="R00-{D.MAX_ROWS + 2:04d}-saturate"' in body and 'data-id="R00-0000-saturate"' not in body


def test_hypotheses_come_from_campaign_files(tmp_path):
    camp = _camp(tmp_path, [dict(_rec(), scenario_id="SA", seed_scenario_id="SA", final_safe_recovery=False)])
    (camp / "02_objectives.json").write_text(json.dumps({"active": ["S1"], "S1_objective": {"question_ko": "전환 순간 오차가 크면 차선을 못 지키는가?"}},
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
    assert h["questions"] == [("S1", "전환 순간 오차가 크면 차선을 못 지키는가?")]
    assert [p["id"] for p in h["prior"]] == ["PK-002", "PK-009"]          # only testable hypotheses, as filed
    assert h["prior"][0]["text"] == D.PK_KO["PK-002"] and h["prior"][1]["text"] == "Unmapped statement."
    assert (h["prior"][0]["n"], h["prior"][0]["miss"]) == (1, 1)          # expected 차선 유지, run was unsafe
    html = D.render_hypotheses(camp / "screen")
    assert "가설" in html and "평가 질문" in html and "적중 0" in html and "빗나감 1" in html
    assert "Nemotron A" in html and "사전 지식" in html
    assert "AI 예상" not in html and "<details" not in html


def test_plain_text_keeps_numbers_and_ids():
    s = D.plain_text("saturate 고장 런(R00-0002-stale)의 lateral_error_rmse_m은 5.413m, detection_latency_s는 0.75s이다.")
    assert s == "Steering Saturation 고장 시험(R00-0002-stale)의 길 가운데에서 벗어난 평균 거리는 5.413m, 고장을 알아채는 데 걸린 시간은 0.75초이다."


def test_insight_panel_shows_key_points_unfolded(tmp_path):
    camp = tmp_path / "camp"
    (camp / "insights").mkdir(parents=True)
    (camp / "insights" / "insights.json").write_text(json.dumps({
        "status": "complete", "summary": "F1_no_or_late_switch 5개", "statistics": {"failure_rate": 0.625,
        "graded_triggered_runs": 8, "failure_count": 5},
        "insights": [{"title": "saturate 고장은 회복 실패", "finding": "margin_delta_m -7.644m", "confidence": "high"}]},
        ensure_ascii=False), encoding="utf-8")
    page = D.refresh(camp / "none.jsonl", camp / "screen", spec={}).read_text(encoding="utf-8")
    panel = page[page.index('id="ins"'):]
    assert "인사이트" in panel and "AI 분석 요약" not in panel
    assert "실패 비율" in panel and "62.5%" in panel and "(채점 8건 중 실패 5건)" in panel and "위험 비율" not in panel
    assert "by Cosmos" in panel
    assert "Steering Saturation 고장은 복귀 실패" in panel and "-7.644m" in panel and "saturate" not in panel
    assert "원문" not in panel and "<details" not in panel


def test_insight_headline_and_evidence_from_structured_fields():
    it = {"title": "saturate 고장은 높은 lateral_error_rmse_m를 유발", "confidence": "high", "finding": "x",
          "evidence": [{"run_id": "R00-0000-saturate", "metric": "metrics.lateral_error_rmse_m", "value": 5.285},
                       {"run_id": "R00-0000-saturate", "metric": "metrics.margin_delta_m", "value": -7.791},
                       {"run_id": "R01-0006-saturate", "metric": "metrics.lateral_error_rmse_m", "value": 5.541},
                       {"run_id": "R01-0006-saturate", "metric": "metrics.margin_delta_m", "value": -7.496}]}
    out = {"R00-0000-saturate": "F1_no_or_late_switch", "R01-0006-saturate": "F1_no_or_late_switch"}
    head, ev = D.insight_view(it, out)
    assert head == "Steering Saturation: 2건 모두 제어기 전환 실패/늦음"
    assert ev == "중심 오차 RMS 평균 5.41 m · 차선 여유 변화 평균 −7.64 m · R00-0000, R01-0006"
    apex = {"title": "curve_apex에서의 고장은 통과율이 높음", "evidence": [
        {"run_id": "R00-0002-stale", "metric": "monitor_outcome", "value": "pass"},
        {"run_id": "R00-0003-bias", "metric": "monitor_outcome", "value": "pass"}]}
    assert D.insight_view(apex, {}) == ("커브 정점 고장: 2건 모두 통과", "R00-0002, R00-0003")


def test_missing_clip_file_says_so_instead_of_empty_player(tmp_path):
    out = tmp_path / "screen"
    out.mkdir()
    gone = D._clip_html({"clip": {"video": str(tmp_path / "clips" / "R00-0000" / "clip.mp4")}}, out)
    assert "<video" not in gone and "영상 파일 없음" in gone
    real = tmp_path / "clips" / "R00-0000" / "clip.mp4"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"\0")
    shown = D._clip_html({"clip": {"video": str(real), "start_s": 1.0, "end_s": 4.0}}, out)
    assert '<video class="clip" src="../clips/R00-0000/clip.mp4"' in shown
    assert D._clip_html({}, out) == ""
