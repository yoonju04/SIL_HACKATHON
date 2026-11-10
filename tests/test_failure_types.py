"""engine/failure_types.py: labels from run records, judge uncertainty, cost thresholds, models, acquisition term."""
import math

import numpy as np
import pytest

from engine import failure_types as FT


def rec(**kw):
    r = {"sim_status": "ok", "switch_time": 10.0, "recovery_outcome": "safe_recovery", "lane_recovery_robustness": 0.5,
         "parameters": {"environment": {"tire_friction_scale": 0.9}},
         "metrics": {"lat_jerk_excess_mps3": 1.0, "lat_acc_excess_mps2": 0.1},
         "recovery_labels": {"departed_at_switch": False}}
    for k, v in kw.items():
        if k in ("lat_jerk_excess_mps3", "lat_acc_excess_mps2"):
            r["metrics"][k] = v
        else:
            r[k] = v
    return r


def test_labels_multi_label_and_limits():
    cfg = FT.load_config()
    assert FT.type_labels(rec(), cfg) == {"jerk": False, "lat_acc": False, "lane_departure": False, "unsettled": False}
    lab = FT.type_labels(rec(lat_jerk_excess_mps3=16.0, lat_acc_excess_mps2=4.0), cfg)
    assert lab["jerk"] and lab["lat_acc"]                                # 4.0 > 4.41 * 0.9 = 3.969, both on
    assert not FT.type_labels(rec(lat_jerk_excess_mps3=14.9), cfg)["jerk"]   # P-04 = 15 (integration4), not 5
    assert FT.type_labels(rec(recovery_outcome="recovered_after_departure"), cfg)["lane_departure"]
    assert FT.type_labels(rec(lane_recovery_robustness=-0.1), cfg)["lane_departure"]
    assert FT.type_labels(rec(recovery_outcome="unsettled"), cfg)["unsettled"]
    assert FT.type_labels(rec(switch_time=None), cfg) is None            # never switched: no failure-type label
    assert FT.type_labels(rec(sim_status="invalid"), cfg) is None


def test_sr4_option_and_severity():
    cfg = FT.load_config()
    r = rec(sr_checks={"sr": {"SR4": {"pass": False}}})
    assert not FT.type_labels(r, cfg)["lane_departure"]
    cfg["types"]["lane_departure"]["include_sr4_footprint"] = True
    assert FT.type_labels(r, cfg)["lane_departure"]
    order = cfg["ordinal"]["order"]
    lab = FT.type_labels(rec(lat_jerk_excess_mps3=20.0), FT.load_config())
    assert order[FT.severity_level(rec(), lab, order)] == "jerk_only"
    lab = FT.type_labels(rec(recovery_outcome="unrecovered", lat_jerk_excess_mps3=20.0), FT.load_config())
    assert order[FT.severity_level(rec(recovery_outcome="unrecovered"), lab, order)] == "unrecovered"


def test_judge_uncertainty_features():
    specs = {"a": {"order": "forward"}, "b": {"order": "reverse"}}
    same = {"J1": [{"judge": "a", "probs": {"x": 0.9, "y": 0.1}}, {"judge": "b", "probs": {"x": 0.9, "y": 0.1}}]}
    flip = {"J1": [{"judge": "a", "probs": {"x": 0.9, "y": 0.1}}, {"judge": "b", "probs": {"x": 0.1, "y": 0.9}}]}
    assert FT.judge_uncertainty(same, specs)[0] == [0.0, 0.0]
    (d, g), names = FT.judge_uncertainty(flip, specs)
    assert names == ["judge_disagreement", "judge_order_gap"]
    assert d == pytest.approx(0.8) and g == pytest.approx(0.8)
    assert FT.judge_uncertainty({}, specs)[0] == [0.0, 0.0]


def test_cost_thresholds_and_columns():
    th = FT.cost_thresholds({"costs": {"false_alarm": 1.0, "miss": {"lane_departure": 20.0, "jerk": 1.0}}})
    assert th["lane_departure"] == pytest.approx(1 / 21) and th["jerk"] == pytest.approx(0.5)
    names = ["route=a", "DF-05", "DF-05_missing", "B_blind_spot", "J4=p(lqr_early)", "fault=F-01"]
    assert FT.select_columns(names, ["df"]) == [1, 2]
    assert FT.select_columns(names, ["judges"]) == [3, 4]
    assert FT.select_columns(names, ["scenario"]) == [0, 5]


@pytest.mark.parametrize("method", ["logistic_balanced", "hier_bayes"])
def test_models_fit_predict_decide(method):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    Y = np.stack([(X[:, 0] > 0.5), (X[:, 0] > 1.2), (X[:, 1] > 1.5), np.zeros(60, bool)], axis=1).astype(float)
    Y[0, 2] = np.nan
    cfg = FT.load_config()
    m = FT.FailureTypeModel(cfg, method=method).fit(X, Y)
    pr = m.predict(X)
    assert pr["probs"].shape == (60, 4) and np.all((pr["probs"] >= 0) & (pr["probs"] <= 1))
    hi, lo = m.predict(np.array([[2.0, 0, 0], [-2.0, 0, 0]]))["probs"][:, 0]
    assert hi > lo
    d = m.decide(pr["probs"])
    assert d.shape == (60, 4) and d.dtype == bool


def test_rare_type_score_uses_probability_times_epistemic():
    pred = {"probs": np.array([[0.9, 0.1, 0.5, 0.0], [0.9, 0.1, 0.5, 0.0]]),
            "epistemic": np.array([[9.0, 0.0, 0.1, 0.0], [0.0, 0.0, 2.0, 0.0]])}
    s = FT.rare_type_score(pred, {"acquisition": {"rare_types": ["lane_departure"], "weight": 1.0}})
    assert s[1] > s[0] and math.isclose(s[1], 1.0)


def test_ordinal_severity_model():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))
    lv = np.clip(np.digitize(X[:, 0], [0.0, 0.8, 1.5]), 0, 5)
    m = FT.OrdinalSeverityModel(FT.load_config()).fit(X, lv)
    p = m.predict(np.array([[3.0, 0.0], [-3.0, 0.0]]))["probs"]
    assert p.shape == (2, 6) and p[1, 0] > p[0, 0]
