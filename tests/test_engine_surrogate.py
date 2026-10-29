import random

import pytest

from engine.surrogate import MetricGPs, acquisition_score, select_batch_scored, straddle
from run import SWITCH_METRIC, switch_aware_score


def test_straddle_prefers_uncertain_points_near_zero():
    assert straddle(0.0, 1.0) > straddle(2.0, 1.0)
    assert straddle(0.0, 1.0) > straddle(0.0, 0.1)
    assert straddle(0.5, 0.2, beta=2.0) == pytest.approx(-0.1)


def test_gp_learns_a_sign_boundary():
    rng = random.Random(0)
    rows = []
    for _ in range(30):
        x = [rng.random(), rng.random()]
        rows.append((x, {"recovery_success": (0.5 - x[0]) * 4}))
    g = MetricGPs(["recovery_success", "other"], seed=0)
    counts = g.fit(rows)
    assert counts == {"recovery_success": 30, "other": 0} and "other" not in g.models
    mu, sd = g.predict([[0.1, 0.5], [0.9, 0.5]])
    assert mu[0]["recovery_success"] > 0 > mu[1]["recovery_success"]
    assert sd[0]["recovery_success"] >= 0


def test_acquisition_and_batch():
    obj = {"metric": "m", "acquisition": "straddle", "beta": 1.96}
    cands = []
    for i in range(20):
        c = {"gp_mean": {"m": (i - 10) / 5}, "gp_std": {"m": 0.5}, "seed_scenario_id": f"s{i}",
             "normalized_features": [i / 20.0]}
        cands.append(c)
    batch = select_batch_scored(cands, lambda c: acquisition_score(c, obj, 1.0), batch_size=8, acquisition_fraction=0.75,
                                min_distance=0.01, max_per_seed=2, rng=random.Random(0), maximize=True)
    assert len(batch) == 8
    acq = [c for c in batch if c["selected_by"] == "acquisition"]
    assert len(acq) == 6 and all(abs(c["gp_mean"]["m"]) <= 0.6 for c in acq)
    lcb = {"metric": "m", "acquisition": "lower_confidence_bound"}
    assert acquisition_score(cands[0], lcb, 1.0) > acquisition_score(cands[-1], lcb, 1.0)


def test_switch_aware_penalty():
    obj = {"metric": "m", "acquisition": "straddle", "beta": 1.96}
    yes = {"gp_mean": {"m": 0.0, SWITCH_METRIC: 0.9}, "gp_std": {"m": 0.5}}     # P(switch)
    no = {"gp_mean": {"m": 0.0, SWITCH_METRIC: 0.1}, "gp_std": {"m": 0.5}}
    assert switch_aware_score(yes, obj, 1.0) > switch_aware_score(no, obj, 1.0)
    # a target defined for every run (SR4 margin) needs no switch: no penalty
    free = dict(obj, needs_switch=False)
    assert switch_aware_score(yes, free, 1.0) == switch_aware_score(no, free, 1.0)


def test_switch_classifier_returns_probabilities():
    rng = random.Random(1)
    rows = [([x := rng.random(), rng.random()], {SWITCH_METRIC: 1.0 if x > 0.5 else 0.0}) for _ in range(30)]
    g = MetricGPs([SWITCH_METRIC], seed=0, classify=[SWITCH_METRIC])
    g.fit(rows)
    mu, sd = g.predict([[0.05, 0.5], [0.95, 0.5]])
    p_lo, p_hi = mu[0][SWITCH_METRIC], mu[1][SWITCH_METRIC]
    assert 0.0 <= p_lo < 0.5 < p_hi <= 1.0
    assert sd[1][SWITCH_METRIC] == pytest.approx((p_hi * (1 - p_hi)) ** 0.5)
    one = MetricGPs([SWITCH_METRIC], seed=0, classify=[SWITCH_METRIC])
    one.fit([([rng.random()], {SWITCH_METRIC: 1.0}) for _ in range(5)])   # one class only
    assert one.predict([[0.3]])[0][0][SWITCH_METRIC] == 1.0
    assert one.diagnostics()[SWITCH_METRIC]["constant"] == 1.0


def test_gp_uses_unit_features_as_given_and_reports_bounds():
    # integration4: no re-scaling on the training data. Before, a column that was constant in training (a fault
    # never run) became (x - lo) / 1 and a column spanning 0.4..0.6 was stretched to 0..1.
    rng = random.Random(2)
    rows = [([0.4 + 0.2 * rng.random(), 0.0], {"m": rng.random()}) for _ in range(12)]
    g = MetricGPs(["m"], seed=0)
    g.fit(rows)
    model = g.models["m"]
    assert not hasattr(model, "lo") and not hasattr(model, "span")
    d = g.diagnostics(["a", "b"])["m"]
    assert d["kind"] == "regression" and set(d["length_scales"]) == {"a", "b"} and d["noise"] >= 1e-3
    assert all(0.1 * 0.999 <= v <= 10.0 * 1.001 for v in d["length_scales"].values())
    assert d["constant_in_training"] == ["b"]             # a fault never run: the GP knows nothing about that column
