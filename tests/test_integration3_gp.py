"""integration3 #5: GP features / classifier / level-set acquisition / --gp-history / min training count / metrics."""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from engine import scenarios as SC
from engine.config import REPO_ROOT, SPACE_REL, load_yaml
from engine.surrogate import GPModel, MetricGPs, SwitchClassifier, boundary_score, level_set_score
from run import SWITCH_METRIC, Loop, boundary_metrics, switch_aware_score

ROUTES = ["town10_long", "bench2drive:26956"]
POOL = ["F-01", "F-02", "F-04", "F-06"]


@pytest.fixture(scope="module")
def space():
    return load_yaml(REPO_ROOT / SPACE_REL)


def _sat(space, mag=0.6, ramp=0.1, fr=1.0, speed=30, dur=0.5):
    return SC.make_candidate(space, route_id="town10_long", speed=speed, fault_id="F-06",
                             params={"duration_s": dur, "side": "positive", "magnitude": mag, "ramp_s": ramp},
                             onset_event="curve_entry", offset=0.0, source="space_filling", friction=fr)


def test_features_are_unit_and_carry_magnitude_ramp_friction(space):
    x, names = SC.numeric_features(_sat(space, mag=1.0, ramp=0.0, fr=0.5), space, ROUTES, POOL)
    assert all(0.0 <= v <= 1.0 for v in x)
    f = dict(zip(names, x))
    assert f["param:magnitude"] == pytest.approx(1.0) and f["param:ramp_s"] == pytest.approx(0.0)
    assert {"tire_friction_u", "max_speed_u", "param:duration_s"} <= set(names)
    assert "param_u0" not in names                                     # old shared slots are gone
    lo, hi = SC.friction_axis(space)["range"]
    assert f["tire_friction_u"] == pytest.approx((0.5 - lo) / (hi - lo))
    # a fault without the parameter gets 0.5 (neutral), not a value from another fault's parameter
    c1 = SC.make_candidate(space, route_id="town10_long", speed=30, fault_id="F-04",
                           params={k: v.get("default") for k, v in SC.faults_by_id(space)["F-04"]["params"].items()},
                           onset_event="straight", offset=0.0, source="space_filling")
    f1 = dict(zip(*reversed(SC.numeric_features(c1, space, ROUTES, POOL))))
    assert f1["param:magnitude"] == 0.5 and f1["param:ramp_s"] == 0.5


def test_differing_continuous_param_gives_differing_prediction(space):
    """result_2 R01-0007/0008: two plans that differed only in a continuous parameter got the same gp_mean to 8 digits."""
    rng = random.Random(3)
    rows = []
    for _ in range(12):
        m = rng.uniform(0.2, 1.0)
        x, _ = SC.numeric_features(_sat(space, mag=round(m, 2)), space, ROUTES, POOL)
        rows.append((x, {"recovery_success": 6.0 * (0.6 - m)}))       # boundary at magnitude 0.6
    g = MetricGPs(["recovery_success"], seed=0, prior_means={"recovery_success": 0.0})
    g.fit(rows)
    xa, _ = SC.numeric_features(_sat(space, mag=0.3), space, ROUTES, POOL)
    xb, _ = SC.numeric_features(_sat(space, mag=0.9), space, ROUTES, POOL)
    (ma, mb), _ = g.predict([xa, xb])
    assert abs(ma["recovery_success"] - mb["recovery_success"]) > 0.5
    assert ma["recovery_success"] > 0 > mb["recovery_success"]
    ls = g.models["recovery_success"].model.kernel_.k1.k2.length_scale
    assert np.all(ls > 0.1 - 1e-9) and np.all(ls < 10 + 1e-9)        # unit-feature kernel bounds


def test_prior_mean_at_level_far_from_data():
    """Training data only from a failing corner: far away the GP must say 'boundary/unknown' (prior = level 0),
    not 'fails like the training average' (legacy normalize_y)."""
    rows = [([0.0 + 0.02 * i, 0.0], {"m": -10.0 + 0.1 * i}) for i in range(8)]
    fixed = MetricGPs(["m"], seed=0, prior_means={"m": 0.0}); fixed.fit(rows)
    legacy = MetricGPs(["m"], seed=0); legacy.fit(rows)
    far = [[1.0, 1.0]]
    f_far, l_far = fixed.predict(far)[0][0]["m"], legacy.predict(far)[0][0]["m"]
    assert l_far < -8.0                       # legacy: the unseen corner "fails like the training average"
    assert abs(f_far) < 0.6 * abs(l_far)      # fixed prior: pulled towards the level


def test_switch_classifier_probability_latent_and_one_class():
    rng = random.Random(0)
    X = [[rng.random(), rng.random()] for _ in range(30)]
    rows = [(x, {SWITCH_METRIC: 1.0 if x[0] > 0.5 else -1.0}) for x in X]
    g = MetricGPs([SWITCH_METRIC], seed=0, switch_metric=SWITCH_METRIC, switch_model="classifier")
    g.fit(rows)
    assert isinstance(g.models[SWITCH_METRIC], SwitchClassifier)
    (lo, hi), (slo, shi) = g.predict([[0.1, 0.5], [0.9, 0.5]])
    assert 0 <= lo[SWITCH_METRIC] < 0.5 < hi[SWITCH_METRIC] <= 1
    assert lo[SWITCH_METRIC + "__latent"] < 0 < hi[SWITCH_METRIC + "__latent"]
    assert slo[SWITCH_METRIC + "__latent"] > 0
    one = SwitchClassifier().fit([[0.1], [0.2], [0.3]], [1, 1, 1])
    p, mu, sd = one.predict([[0.5]])
    assert p[0] == pytest.approx(0.8) and sd[0] == 1.0


def test_level_set_default_and_boundary_score():
    orch = load_yaml(REPO_ROOT / "pipeline/07_orchestrator/orchestrator.yaml")
    assert orch["objective_acquisition"]["S1"]["acquisition"] == "level_set"
    sur = load_yaml(REPO_ROOT / "pipeline/06_gp_surrogate/surrogate.yaml")["surrogate"]
    assert sur["switch_model"] == "classifier" and sur["prior_mean"] == "level"
    assert sur["min_train_runs_for_acquisition"] >= sur["min_train_points"]
    assert level_set_score(0.0, 1.0) > level_set_score(3.0, 1.0)
    mk = lambda m, s, p, mf, sf: {"gp_mean": {"r": m, SWITCH_METRIC: p, SWITCH_METRIC + "__latent": mf},
                                  "gp_std": {"r": s, SWITCH_METRIC + "__latent": sf}}
    near = boundary_score(mk(0.1, 1.0, 0.95, 3.0, 0.1), "r", SWITCH_METRIC, beta=1.96)
    clear_fail = boundary_score(mk(-8.0, 1.0, 0.95, 3.0, 0.1), "r", SWITCH_METRIC, beta=1.96)
    no_switch = boundary_score(mk(0.1, 1.0, 0.02, -4.0, 0.1), "r", SWITCH_METRIC, beta=1.96)
    switch_unsure = boundary_score(mk(-8.0, 1.0, 0.5, 0.0, 1.0), "r", SWITCH_METRIC, beta=1.96)
    assert near > clear_fail and near > no_switch and switch_unsure > clear_fail
    assert boundary_score({"gp_mean": {}, "gp_std": {}}, "r", SWITCH_METRIC, beta=1.96) is None
    # legacy straddle with a classifier: P(switch)=0.02 is mapped to ~-1 and penalised
    obj = {"metric": "r", "acquisition": "straddle", "beta": 1.96}
    assert switch_aware_score(mk(0.1, 1.0, 0.02, -4.0, 0.1), obj, 1.96) < switch_aware_score(mk(0.1, 1.0, 0.98, 4.0, 0.1), obj, 1.96)


def test_min_training_count_falls_back_to_space_filling(tmp_path):
    loop = Loop(mode="mock", out=tmp_path / "o", ablation="X-5", video_on=False, seed=0, cold_start=4, batch_size=4,
                rounds=1, quiet=True)
    loop.min_train_for_acq = 1000
    c = {"gp_mean": {"recovery_success": 0.0, SWITCH_METRIC: 0.5, SWITCH_METRIC + "__latent": 0.0},
         "gp_std": {"recovery_success": 1.0, SWITCH_METRIC + "__latent": 1.0}}
    loop.gp_ready = False
    assert loop.acq_score(c, 1.96) is None
    loop.run()
    tc = loop.stats["gp_train_counts"]
    assert tc and all(t["gp_acquisition_active"] is False for t in tc)
    assert not any(r["selector"] == "acquisition" for r in loop.records)


def test_gp_history_pools_earlier_runs(tmp_path):
    src = tmp_path / "old"
    old = Loop(mode="mock", out=src, ablation="X-5", video_on=False, seed=0, cold_start=6, batch_size=4, rounds=1, quiet=True)
    old.run()
    n_ok = sum(r.get("sim_status") == "ok" for r in old.records)
    with (src / "run_records.jsonl").open("a") as fh:     # junk the loader must skip
        fh.write(json.dumps({"sim_status": "failed", "parameters": {}}) + "\n")
        fh.write(json.dumps({"sim_status": "ok", "parameters": {"route_id": "nowhere", "fault": {"fault_id": "F-06"}}}) + "\n")
    new = Loop(mode="mock", out=tmp_path / "new", ablation="X-5", video_on=False, seed=1, cold_start=4, batch_size=4,
               rounds=1, quiet=True, gp_history=[str(tmp_path / "old" / "run_records.jsonl")])
    st = new.stats["gp_history"]
    assert st["loaded"] == n_ok and st["skipped"].get("not sim_status ok") == 1
    assert st["skipped"].get("route not in this campaign") == 1
    new.run()
    first = new.stats["gp_train_counts"][0]
    assert first["history_runs"] == n_ok and first["gp_acquisition_active"] is True


def test_cli_has_gp_history():
    import run, sys
    src = Path(run.__file__).read_text()
    assert '"--gp-history"' in src


def test_boundary_metrics():
    rec = lambda mode, out, safe, lrr, mu: {"sim_status": "ok", "scenario_id": f"{mode}{lrr}",
                                            "parameters": {"fault": {"mode": mode}}, "recovery_outcome": out,
                                            "final_safe_recovery": safe, "lane_recovery_robustness": lrr,
                                            "gp_mean": {"recovery_success": mu}}
    rs = [rec("saturate", "unrecovered", False, -0.9, -3.0), rec("saturate", "safe_recovery", True, 0.1, 0.4),
          rec("bias", "no_switch", False, None, 0.4), rec("bias", "no_switch", False, None, 0.4)]
    m = boundary_metrics(rs, {"saturate0.1": {"source": "nemotron_a"}}, band=0.2)
    assert m["boundary_band_runs"] == 1 and m["boundary_band_runs_0.5"] == 1
    assert m["modes_with_mixed_outcome"] == ["saturate"] and m["per_fault_mode"]["bias"]["mixed"] is False
    d = m["gp_prediction_diversity"]["recovery_success"]
    assert d["n"] == 4 and d["distinct"] == 2
    assert m["nemotron_a_share"] == pytest.approx(0.25)
