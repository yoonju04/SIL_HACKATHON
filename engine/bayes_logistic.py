"""Small Bayesian / regularised logistic models in numpy + scipy (no new dependency), for tens of labelled runs.

  BayesLogistic           binary logistic, Gaussian prior, MAP + Laplace. predict -> p (probit-corrected) and the
                          epistemic logit variance x' S x ("the model does not know here").
  BayesMultinomial        softmax logistic, Gaussian prior, MAP + Laplace; predictive by Monte Carlo over the
                          Laplace posterior -> mean probabilities, and the epistemic part = mutual information
                          (BALD: H[mean p] - mean H[p]) plus the summed variance of the class probabilities.
  HierarchicalLogistic    T binary tasks (failure types) whose slopes share a common prior:
                          w_t = mu + delta_t, mu ~ N(0, s_mu^2), delta_t ~ N(0, tau^2), b_t ~ N(0, s_b^2).
                          A rare type borrows strength from the others through mu (tau small = pooled,
                          tau large = independent). MAP + Laplace over (mu, delta, b); tau optionally chosen by
                          the Laplace evidence on a grid. Missing labels (NaN) are masked per task.
  OrdinalLogistic         cumulative logit P(y <= k) = sigmoid(theta_k - x'w), ordered thresholds
                          (theta_1, log increments), L2 on w; MAP + Laplace (numerical Hessian), MC predictive.

All inputs are standardised by the caller or by `Standardizer`. Every model takes `sample_weight` (class balance).
The Laplace covariance is the inverse Hessian of the negative log posterior at the MAP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp, softmax


# ---------------------------------------------------------------- helpers
class Standardizer:
    """Column z-scoring fit on training rows; constant columns get scale 1 (they become 0 after centring)."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0) if len(X) else np.zeros(X.shape[1])
        sd = X.std(axis=0) if len(X) else np.ones(X.shape[1])
        self.scale_ = np.where(sd > 1e-9, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """sklearn class_weight='balanced': n / (n_classes * n_c) per sample."""
    y = np.asarray(y)
    cls, cnt = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(cls) * n) for c, n in zip(cls, cnt)}
    return np.array([w[v] for v in y], dtype=float)


def _inv_psd(H: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    H = 0.5 * (H + H.T)
    for k in range(6):
        try:
            L = np.linalg.cholesky(H + (jitter * 10 ** k) * np.eye(len(H)))
            Li = np.linalg.inv(L)
            return Li.T @ Li
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(H)


def _logdet_psd(H: np.ndarray) -> float:
    s, ld = np.linalg.slogdet(0.5 * (H + H.T))
    return float(ld) if s > 0 else float("inf")


# ---------------------------------------------------------------- binary
@dataclass
class BayesLogistic:
    prior_var: float = 1.0           # slopes ~ N(0, prior_var); sklearn C  <->  prior_var = C (same objective)
    intercept_var: float = 100.0
    w_: np.ndarray | None = None
    cov_: np.ndarray | None = None

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X, np.ones((len(X), 1))])

    def fit(self, X, y, sample_weight=None) -> "BayesLogistic":
        Z = self._design(X)
        y = np.asarray(y, dtype=float)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        prec = np.r_[np.full(Z.shape[1] - 1, 1.0 / self.prior_var), 1.0 / self.intercept_var]

        def f(w):
            z = Z @ w
            nll = -np.sum(sw * (y * log_expit(z) + (1 - y) * log_expit(-z)))
            return nll + 0.5 * np.sum(prec * w * w), Z.T @ (sw * (expit(z) - y)) + prec * w

        w0 = np.zeros(Z.shape[1])
        res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.w_ = res.x
        p = expit(Z @ self.w_)
        H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + np.diag(prec)
        self.cov_ = _inv_psd(H)
        return self

    def logit_moments(self, X) -> tuple[np.ndarray, np.ndarray]:
        Z = self._design(X)
        return Z @ self.w_, np.einsum("ij,jk,ik->i", Z, self.cov_, Z)

    def predict_proba(self, X) -> np.ndarray:
        """P(y=1) with the probit approximation of the posterior predictive (MacKay 1992)."""
        m, v = self.logit_moments(X)
        return expit(m / np.sqrt(1.0 + math.pi * v / 8.0))

    def epistemic(self, X) -> np.ndarray:
        return self.logit_moments(X)[1]


# ---------------------------------------------------------------- multinomial
@dataclass
class BayesMultinomial:
    prior_var: float = 0.5
    intercept_var: float = 100.0
    n_samples: int = 256
    seed: int = 0
    classes_: np.ndarray | None = None
    W_: np.ndarray | None = None      # (K, D+1)
    cov_: np.ndarray | None = None    # (K(D+1), K(D+1))

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X, np.ones((len(X), 1))])

    def fit(self, X, y, sample_weight=None) -> "BayesMultinomial":
        Z = self._design(X)
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        K, D = len(self.classes_), Z.shape[1]
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        prec = np.tile(np.r_[np.full(D - 1, 1.0 / self.prior_var), 1.0 / self.intercept_var], K)

        def f(wf):
            W = wf.reshape(K, D)
            S = Z @ W.T
            lse = logsumexp(S, axis=1)
            nll = -np.sum(sw * (np.sum(Y * S, axis=1) - lse))
            P = np.exp(S - lse[:, None])
            g = ((P - Y) * sw[:, None]).T @ Z
            return nll + 0.5 * np.sum(prec * wf * wf), g.ravel() + prec * wf

        res = minimize(f, np.zeros(K * D), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        self.W_ = res.x.reshape(K, D)
        P = softmax(Z @ self.W_.T, axis=1)
        H = np.zeros((K * D, K * D))
        for i in range(len(Z)):
            A = np.diag(P[i]) - np.outer(P[i], P[i])
            H += sw[i] * np.kron(A, np.outer(Z[i], Z[i]))
        H += np.diag(prec)
        self.cov_ = _inv_psd(H)
        return self

    def _samples(self, X) -> np.ndarray:
        """(S, n, K) class probabilities under W ~ N(W_map, cov) (seeded: same X -> same numbers)."""
        Z = self._design(X)
        K, D = self.W_.shape
        rng = np.random.default_rng(self.seed)
        L = np.linalg.cholesky(self.cov_ + 1e-10 * np.eye(len(self.cov_)))
        eps = rng.standard_normal((self.n_samples, K * D))
        Ws = (self.W_.ravel()[None, :] + eps @ L.T).reshape(self.n_samples, K, D)
        return softmax(np.einsum("nd,skd->snk", Z, Ws), axis=2)

    def predict_proba(self, X) -> np.ndarray:
        return self._samples(X).mean(axis=0)

    def predict_map(self, X) -> np.ndarray:
        return softmax(self._design(X) @ self.W_.T, axis=1)

    def uncertainty(self, X) -> dict[str, np.ndarray]:
        """mean probs, total entropy, epistemic mutual information (BALD) and summed probability variance."""
        Ps = self._samples(X)
        pm = Ps.mean(axis=0)
        ent = lambda p: -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0.0), axis=-1)
        h_tot = ent(pm)
        mi = np.maximum(0.0, h_tot - ent(Ps).mean(axis=0))
        return {"probs": pm, "entropy": h_tot, "epistemic_mi": mi, "prob_var": Ps.var(axis=0).sum(axis=1)}


# ---------------------------------------------------------------- hierarchical binary (shared slopes)
@dataclass
class HierarchicalLogistic:
    """T binary tasks on the same inputs. Y: (n, T) with NaN = no label for that task. sample_weight: (n, T)."""
    mu_var: float = 1.0
    tau: float | None = 0.5           # None -> pick by Laplace evidence over tau_grid
    tau_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
    intercept_var: float = 25.0
    theta_: np.ndarray | None = None
    cov_: np.ndarray | None = None
    evidence_: dict = field(default_factory=dict)

    def _unpack(self, th, D, T):
        mu = th[:D]
        dl = th[D:D + T * D].reshape(T, D)
        b = th[D + T * D:]
        return mu, dl, b

    def _fit_tau(self, X, Y, M, SW, tau):
        n, D = X.shape
        T = Y.shape[1]
        P = D + T * D + T
        prec = np.r_[np.full(D, 1.0 / self.mu_var), np.full(T * D, 1.0 / tau ** 2), np.full(T, 1.0 / self.intercept_var)]
        Y0 = np.nan_to_num(Y)

        def f(th):
            mu, dl, b = self._unpack(th, D, T)
            Wt = mu[None, :] + dl                      # (T, D)
            Zs = X @ Wt.T + b[None, :]                 # (n, T)
            ll = M * SW * (Y0 * log_expit(Zs) + (1 - Y0) * log_expit(-Zs))
            r = M * SW * (expit(Zs) - Y0)              # dNLL/dz
            gW = r.T @ X                               # (T, D)
            g = np.r_[gW.sum(axis=0), gW.ravel(), r.sum(axis=0)]
            return -ll.sum() + 0.5 * np.sum(prec * th * th), g + prec * th

        res = minimize(f, np.zeros(P), jac=True, method="L-BFGS-B", options={"maxiter": 5000})
        th = res.x
        mu, dl, b = self._unpack(th, D, T)
        Zs = X @ (mu[None, :] + dl).T + b[None, :]
        A = M * SW * expit(Zs) * (1 - expit(Zs))       # (n, T) curvature
        H = np.diag(prec).astype(float)
        for t in range(T):
            # d z_t / d theta = [x (mu), x at delta_t block, 1 at b_t]
            idx = np.r_[np.arange(D), D + t * D + np.arange(D), D + T * D + t]
            J = np.hstack([X, X, np.ones((n, 1))])
            Ht = (J * A[:, t][:, None]).T @ J
            H[np.ix_(idx, idx)] += Ht
        neg_post = res.fun
        # Laplace log evidence: -neg_post + 0.5 log|prior prec| - 0.5 log|H|  (constants in 2pi cancel)
        ev = -neg_post + 0.5 * np.sum(np.log(prec)) - 0.5 * _logdet_psd(H)
        return th, H, ev

    def fit(self, X, Y, sample_weight=None) -> "HierarchicalLogistic":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        M = (~np.isnan(Y)).astype(float)
        SW = np.ones_like(Y) if sample_weight is None else np.nan_to_num(np.asarray(sample_weight, dtype=float))
        self.D_, self.T_ = X.shape[1], Y.shape[1]
        taus = [self.tau] if self.tau is not None else list(self.tau_grid)
        best = None
        for tau in taus:
            th, H, ev = self._fit_tau(X, Y, M, SW, float(tau))
            self.evidence_[float(tau)] = float(ev)
            if best is None or ev > best[2]:
                best = (th, H, ev, float(tau))
        self.theta_, H, _, self.tau_ = best
        self.cov_ = _inv_psd(H)
        return self

    def logit_moments(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(n, T) posterior mean and variance of each task's logit."""
        X = np.asarray(X, dtype=float)
        D, T = self.D_, self.T_
        n = len(X)
        mean = np.zeros((n, T))
        var = np.zeros((n, T))
        for t in range(T):
            idx = np.r_[np.arange(D), D + t * D + np.arange(D), D + T * D + t]
            J = np.hstack([X, X, np.ones((n, 1))])
            mean[:, t] = J @ self.theta_[idx]
            C = self.cov_[np.ix_(idx, idx)]
            var[:, t] = np.einsum("ij,jk,ik->i", J, C, J)
        return mean, var

    def predict_proba(self, X) -> np.ndarray:
        m, v = self.logit_moments(X)
        return expit(m / np.sqrt(1.0 + math.pi * v / 8.0))

    def epistemic(self, X) -> np.ndarray:
        return self.logit_moments(X)[1]


# ---------------------------------------------------------------- ordinal (cumulative logit)
@dataclass
class OrdinalLogistic:
    """Classes are 0..K-1 in their order. P(y <= k | x) = sigmoid(theta_k - x'w), k = 0..K-2."""
    prior_var: float = 0.5
    n_classes: int | None = None
    n_samples: int = 256
    seed: int = 0
    params_: np.ndarray | None = None
    cov_: np.ndarray | None = None

    def _theta(self, a):
        return np.cumsum(np.r_[a[0], np.exp(a[1:])])

    def _nll(self, par, X, y, sw):
        K = self.n_classes
        D = X.shape[1]
        w, a = par[:D], par[D:]
        th = self._theta(a)
        eta = X @ w
        cdf = expit(th[None, :] - eta[:, None])                # (n, K-1)
        cdf = np.hstack([np.zeros((len(X), 1)), cdf, np.ones((len(X), 1))])
        p = np.clip(cdf[np.arange(len(X)), y + 1] - cdf[np.arange(len(X)), y], 1e-12, 1.0)
        return -np.sum(sw * np.log(p)) + 0.5 * np.sum(w * w) / self.prior_var + 0.5 * np.sum(a * a) / 100.0

    def fit(self, X, y, sample_weight=None) -> "OrdinalLogistic":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes = int(self.n_classes or (y.max() + 1))
        K = self.n_classes
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        D = X.shape[1]
        # init thresholds at the empirical cumulative logits
        cum = np.clip(np.array([(y <= k).mean() for k in range(K - 1)]), 0.02, 0.98)
        th0 = np.log(cum / (1 - cum))
        th0 = np.maximum.accumulate(th0 + 1e-3 * np.arange(K - 1))
        a0 = np.r_[th0[0], np.log(np.maximum(np.diff(th0), 1e-2))]
        x0 = np.r_[np.zeros(D), a0]
        res = minimize(self._nll, x0, args=(X, y, sw), method="L-BFGS-B", options={"maxiter": 3000})
        self.params_ = res.x
        self.D_ = D
        # numerical Hessian (central differences, small parameter count)
        P = len(res.x)
        h = 1e-4
        H = np.zeros((P, P))
        f0 = self._nll(res.x, X, y, sw)
        for i in range(P):
            for j in range(i, P):
                e_i = np.zeros(P); e_i[i] = h
                e_j = np.zeros(P); e_j[j] = h
                if i == j:
                    v = (self._nll(res.x + e_i, X, y, sw) - 2 * f0 + self._nll(res.x - e_i, X, y, sw)) / h ** 2
                else:
                    v = (self._nll(res.x + e_i + e_j, X, y, sw) - self._nll(res.x + e_i - e_j, X, y, sw)
                         - self._nll(res.x - e_i + e_j, X, y, sw) + self._nll(res.x - e_i - e_j, X, y, sw)) / (4 * h * h)
                H[i, j] = H[j, i] = v
        self.cov_ = _inv_psd(H)
        return self

    def _probs(self, par, X):
        D = X.shape[1]
        w, a = par[:D], par[D:]
        th = self._theta(a)
        cdf = expit(th[None, :] - (X @ w)[:, None])
        cdf = np.hstack([np.zeros((len(X), 1)), cdf, np.ones((len(X), 1))])
        return np.clip(np.diff(cdf, axis=1), 0.0, 1.0)

    def predict_map(self, X) -> np.ndarray:
        return self._probs(self.params_, np.asarray(X, dtype=float))

    def uncertainty(self, X) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        L = np.linalg.cholesky(self.cov_ + 1e-10 * np.eye(len(self.cov_)))
        Ps = np.stack([self._probs(self.params_ + L @ rng.standard_normal(len(self.params_)), X)
                       for _ in range(self.n_samples)])
        Ps = Ps / Ps.sum(axis=2, keepdims=True)
        pm = Ps.mean(axis=0)
        ent = lambda p: -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0.0), axis=-1)
        return {"probs": pm, "entropy": ent(pm), "epistemic_mi": np.maximum(0.0, ent(pm) - ent(Ps).mean(axis=0)),
                "prob_var": Ps.var(axis=0).sum(axis=1)}

    def predict_proba(self, X) -> np.ndarray:
        return self.uncertainty(X)["probs"]


# ---------------------------------------------------------------- metrics
def pr_auc(scores, labels) -> float | None:
    """Average precision (area under the precision-recall step curve); None without positives."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec[y]) / y.sum())


def roc_auc(scores, labels) -> float | None:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def binary_ece(p, y, bins: int = 10) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (p >= lo) & ((p < hi) if b < bins - 1 else (p <= hi))
        if sel.any():
            out += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(out)
