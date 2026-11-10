"""engine/bayes_logistic.py: MAP agrees with sklearn, Laplace variance behaves, hierarchical sharing, ordinal order."""
import numpy as np
import pytest

from engine import bayes_logistic as BL


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    z = X @ np.array([1.5, -1.0, 0.0]) + 0.3
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y


def test_binary_map_matches_sklearn_l2():
    from sklearn.linear_model import LogisticRegression
    X, y = _data()
    b = BL.BayesLogistic(prior_var=0.5, intercept_var=1e8).fit(X, y)
    sk = LogisticRegression(C=0.5, max_iter=2000).fit(X, y)
    assert np.allclose(b.w_[:-1], sk.coef_[0], atol=2e-3)
    assert abs(b.w_[-1] - sk.intercept_[0]) < 2e-3


def test_binary_epistemic_grows_away_from_data():
    X, y = _data()
    b = BL.BayesLogistic(prior_var=1.0).fit(X, y)
    near, far = b.epistemic(np.zeros((1, 3)))[0], b.epistemic(np.full((1, 3), 8.0))[0]
    assert far > 10 * near
    p = b.predict_proba(np.full((1, 3), 8.0))[0]
    assert 0.0 < p < 1.0


def test_multinomial_map_matches_sklearn_and_mi_nonnegative():
    from sklearn.linear_model import LogisticRegression
    X, _ = _data()
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    m = BL.BayesMultinomial(prior_var=0.5, intercept_var=1e8, seed=1).fit(X, y)
    sk = LogisticRegression(C=0.5, max_iter=3000).fit(X, y)
    assert np.abs(m.predict_map(X) - sk.predict_proba(X)).max() < 1e-3
    u = m.uncertainty(X[:5])
    assert np.all(u["epistemic_mi"] >= 0) and np.allclose(u["probs"].sum(axis=1), 1.0)
    assert np.allclose(m.uncertainty(X[:5])["probs"], u["probs"])      # seeded MC: deterministic


def test_hierarchical_rare_task_borrows_shared_slope():
    rng = np.random.default_rng(3)
    n = 150
    X = rng.normal(size=(n, 2))
    z = 2.0 * X[:, 0]
    Y = np.stack([(rng.random(n) < 1 / (1 + np.exp(-z))).astype(float),
                  (rng.random(n) < 1 / (1 + np.exp(-(z - 1)))).astype(float),
                  np.full(n, np.nan)], axis=1)
    Y[:20, 2] = (X[:20, 0] > 1.2).astype(float)                       # rare task: few labels, few positives
    h = BL.HierarchicalLogistic(tau=0.25).fit(X, Y)
    p = h.predict_proba(np.array([[2.0, 0.0], [-2.0, 0.0]]))
    assert p[0, 2] > p[1, 2]                                             # same direction as the shared slope
    # independent fit of the rare task alone is much flatter / more uncertain
    v_h = h.epistemic(np.array([[2.0, 0.0]]))[0, 2]
    m = ~np.isnan(Y[:, 2])
    b = BL.BayesLogistic(prior_var=1.0).fit(X[m], Y[m, 2])
    assert v_h < b.epistemic(np.array([[2.0, 0.0]]))[0]


def test_hierarchical_tau_by_evidence_and_masking():
    X, y = _data(80)
    Y = np.stack([y, y, np.where(np.arange(80) % 3 == 0, np.nan, y)], axis=1).astype(float)
    h = BL.HierarchicalLogistic(tau=None, tau_grid=(0.1, 1.0)).fit(X, Y)
    assert h.tau_ in (0.1, 1.0) and set(h.evidence_) == {0.1, 1.0}
    assert h.evidence_[0.1] > h.evidence_[1.0]                         # identical tasks -> pooled (small tau) wins


def test_ordinal_orders_classes():
    X, _ = _data(150)
    y = np.digitize(X @ np.array([1.5, -1.0, 0.0]), [-1.0, 0.0, 1.0])
    o = BL.OrdinalLogistic(prior_var=1.0, n_classes=4).fit(X, y)
    lo, hi = o.predict_map(np.array([[-2.0, 2.0, 0.0], [2.0, -2.0, 0.0]]))
    assert lo.argmax() == 0 and hi.argmax() == 3
    u = o.uncertainty(X[:3])
    assert np.allclose(u["probs"].sum(axis=1), 1.0)


def test_metrics():
    assert BL.pr_auc([0.9, 0.8, 0.1], [1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert BL.roc_auc([0.9, 0.8, 0.1], [1, 0, 1]) == pytest.approx(0.5)
    assert BL.pr_auc([0.1, 0.2], [0, 0]) is None
    assert BL.binary_ece([0.0, 1.0], [0, 1]) == 0.0
    w = BL.balanced_weights(np.array([0, 0, 0, 1]))
    assert w[3] == pytest.approx(2.0) and w[0] == pytest.approx(4 / 6)
