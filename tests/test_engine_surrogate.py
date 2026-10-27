import random

import pytest

from pipeline.engine.surrogate import MetricGPs, acquisition_score, select_batch_scored, straddle
from pipeline.run import SWITCH_METRIC, switch_aware_score


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
    yes = {"gp_mean": {"m": 0.0, SWITCH_METRIC: 0.8}, "gp_std": {"m": 0.5}}
    no = {"gp_mean": {"m": 0.0, SWITCH_METRIC: -0.8}, "gp_std": {"m": 0.5}}
    assert switch_aware_score(yes, obj, 1.0) > switch_aware_score(no, obj, 1.0)
