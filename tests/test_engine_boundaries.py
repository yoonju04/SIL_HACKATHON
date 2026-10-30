"""integration4: critical faults and boundary table (engine.boundaries)."""
from engine import boundaries as B


def test_bracket_statuses():
    assert B.bracket([(0.02, False), (0.03, False), (0.05, True)]) == {
        "n": 3, "n_bad": 1, "values": [0.02, 0.03, 0.05], "status": "bracketed", "range": [0.03, 0.05]}
    assert B.bracket([(0.02, False), (0.05, True), (0.07, False)])["status"] == "overlap"
    assert B.bracket([(0.02, False), (0.05, False)])["status"] == "all_ok"
    assert B.bracket([(0.02, True)])["bad_from"] == 0.02


def table():
    def run(i, v, bad):
        return {"run_id": f"r{i}", "value": v, "margin": -0.1 if bad else 0.3, "harm_bad": bad, "detect_bad": bad,
                "speed_kmh": 35.0, "friction": 0.9, "onset_event": "curve_exit", "detected_by": None}
    g = {"harm": B.bracket([(0.03, False), (0.055, True)]), "detection": B.bracket([(0.03, False), (0.055, True)]),
         "runs": [run(1, 0.03, False), run(2, 0.055, True)]}
    return {"modes": {"bias": {"runs": 2, "harmful_runs": 1, "late_or_missed_runs": 1, "switched_runs": 1, "harmful_rate": 0.5,
                               "late_rate": 0.5, "worst_margin": -0.1, "worst_run": "r2", "detected_by": {},
                               "parameters": {"magnitude_rad": {"space_range": [0.02, 0.1], "unit": "rad",
                                                                "groups": {"all conditions": g}}}}},
            "critical_ranking": ["bias"]}


def test_check_answer_keeps_what_agrees_with_the_table():
    good = {"critical_faults": [{"rank": 1, "fault": "bias", "reason": "위험 비율이 가장 높음", "run_ids": ["r2"]}],
            "boundaries": [{"fault": "bias", "parameter": "magnitude_rad", "kind": "harm", "condition": "all conditions",
                            "range": [0.03, 0.055], "status": "bracketed", "statement": "0.03~0.055 rad 사이"},
                           {"fault": "bias", "parameter": "magnitude_rad", "kind": "detection", "condition": "all conditions",
                            "range": [0.04, 0.05], "status": "estimated", "statement": "대략 0.04~0.05 rad로 추정"}]}
    kept, problems = B.check_answer(good, table())
    assert problems == [] and len(kept["boundaries"]) == 2
    bad = {"critical_faults": [{"rank": 1, "fault": "nope", "run_ids": []}],
           "boundaries": [{"fault": "bias", "parameter": "magnitude_rad", "kind": "harm", "condition": "all conditions",
                           "range": [0.01, 0.055], "status": "bracketed"},           # outside the space range
                          {"fault": "bias", "parameter": "magnitude_rad", "kind": "harm", "condition": "all conditions",
                           "range": [0.04, 0.06], "status": "bracketed"}]}          # not the table's bracket
    kept, problems = B.check_answer(bad, table())
    assert kept["boundaries"] == [] and kept["critical_faults"] == [] and len(problems) >= 3


def test_check_answer_rejects_han_characters():
    a = {"critical_faults": [{"rank": 1, "fault": "bias", "reason": "偏差", "run_ids": []}], "boundaries": []}
    assert any("한자" in p for p in B.check_answer(a, table())[1])


def test_deterministic_summary_marks_full_range():
    t = table()
    t["modes"]["bias"]["parameters"]["magnitude_rad"]["groups"]["all conditions"]["detection"] = \
        B.bracket([(0.02, False), (0.1, False)])
    text = "\n".join(B.deterministic_summary(t))
    assert "시험 범위 전체" in text and "0.03~0.055" in text
