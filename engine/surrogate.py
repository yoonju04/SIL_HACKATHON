"""06 GP surrogate (one GP per metric, Matern 5/2 ARD + White, scikit-learn) and 07 acquisitions.

* Targets are run_record.gp_targets[metric] = z = sign*y/scale (judgment.gp_targets); z low = riskier.
* Features come from engine.scenarios.numeric_features, already on [0, 1] (one-hot / unit-scaled ranges), so they
  are NOT re-normalised on the training span (integration3: the old min/max scaling on a handful of runs blew tiny
  differences up and pinned ARD length scales at their 1e-2 / 1e2 bounds -> constant predictions, e.g. result_2
  R01-0007/0008). Only columns outside [0, 1] (05a DF features) are shifted/divided to fit.
* Kernel bounds are set for unit features (length scale 0.1..10, noise 1e-4..0.5 of the normalised variance) so
  that a GP on few points cannot collapse to "every point independent" (constant prior mean everywhere).
* switched_indicator (did the supervisor switch?) is a GP CLASSIFIER (sklearn GaussianProcessClassifier, Laplace
  approximation, Rasmussen & Williams 2006 Alg. 3.1/3.2): gp_mean = P(switch) in [0, 1], gp_std = sqrt(p(1-p)), and
  the latent mean / std are kept as '<metric>__latent' for the boundary acquisition. switch_model: regression keeps
  the old +/-1 regression (legacy acquisition).
* Acquisitions (07):
    level_set  (integration3 default, Gotovos et al. IJCAI 2013 "Active Learning for Level Set Estimation";
                straddle of Bryan et al. 2005): score = beta*sigma(x) - |mu(x) - h| on lane_recovery_robustness (z),
                h = recovery threshold (0 = lane boundary), normalised by the target's training std; combined with
                the same straddle on the switch classifier's latent function (P(switch) = 0.5 boundary):
                score = max(rec - penalty*(1 - P(switch)), switch).  "uncertain switch OR recovery near the boundary"
    straddle   (legacy) beta*std_z - |mean_z|, with run.switch_aware_score's penalty from the regressed switch GP
    lower_confidence_bound  mean_z - beta*std_z, minimised (objectives without a boundary)
"""
from __future__ import annotations

import math
import random
import warnings
from typing import Any

import numpy as np

from . import judgment as J

MIN_TRAIN_POINTS = 4
LATENT = "__latent"
DEFAULT_KERNEL = {"length_scale_init": 0.5, "length_scale_bounds": [0.1, 10.0], "constant_bounds": [0.05, 20.0],
                  "noise_init": 1e-2, "noise_bounds": [1e-4, 0.5], "n_restarts": 3}


def _kernel_opts(opts: dict | None) -> dict:
    out = dict(DEFAULT_KERNEL)
    out.update({k: v for k, v in (opts or {}).items() if v is not None})
    return out


def _unit_scaling(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep [0, 1] columns as they are; shift/divide only columns that leave [0, 1]."""
    lo = np.minimum(0.0, X.min(axis=0))
    span = np.maximum(1.0, X.max(axis=0) - lo)
    return lo, span


def _matern(d: int, o: dict):
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    return ConstantKernel(1.0, tuple(o["constant_bounds"])) * Matern(
        length_scale=np.full(d, float(o["length_scale_init"])), length_scale_bounds=tuple(o["length_scale_bounds"]), nu=2.5)


class GPModel:
    """prior_mean None: sklearn normalize_y (prior mean = training mean). A number (integration3, level-set use): the
    prior mean is fixed at that value (the level h), so far from the data the GP says "unknown, on the boundary"
    instead of "like the average training run". The campaign samples failure-heavy regions on purpose, so the training
    mean is strongly negative and an unseen region was predicted as failing (mock sign accuracy 0.57 vs 0.46 safe)."""

    def __init__(self, metric: str, seed: int = 0, kernel_opts: dict | None = None,
                 prior_mean: float | None = None) -> None:
        self.metric = metric
        self.prior_mean = prior_mean
        self.y_off = 0.0
        self.y_div = 1.0
        self.seed = seed
        self.opts = _kernel_opts(kernel_opts)
        self.model = None
        self.lo = self.span = None
        self.n_train = 0
        self.y_std = 1.0

    def fit(self, X: list[list[float]], y: list[float]) -> "GPModel":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import WhiteKernel
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.lo, self.span = _unit_scaling(X)
        Xn = (X - self.lo) / self.span
        o = self.opts
        kernel = _matern(X.shape[1], o) + WhiteKernel(noise_level=float(o["noise_init"]),
                                                      noise_level_bounds=tuple(o["noise_bounds"]))
        fixed = self.prior_mean is not None
        if fixed:
            self.y_off = float(self.prior_mean)
            self.y_div = float(np.sqrt(np.mean((y - self.y_off) ** 2))) or 1.0
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=not fixed,
                                              n_restarts_optimizer=int(o["n_restarts"]), random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Xn, (y - self.y_off) / self.y_div)
        self.n_train = len(y)
        self.y_std = float(np.std(y)) if len(y) > 1 and np.std(y) > 1e-9 else 1.0
        return self

    def predict(self, X: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        Xn = (np.asarray(X, float) - self.lo) / self.span
        mu, sd = self.model.predict(Xn, return_std=True)
        return mu * self.y_div + self.y_off, sd * self.y_div


class SwitchClassifier:
    """P(switch) with a GP classifier (Laplace). One class only in the training data -> constant probability
    (shrunk towards 0.5 by one pseudo-count each side) and prior latent std 1 (nothing learned about the boundary)."""

    def __init__(self, seed: int = 0, kernel_opts: dict | None = None) -> None:
        self.seed = seed
        self.opts = _kernel_opts(kernel_opts)
        self.model = None
        self.const_p: float | None = None
        self.lo = self.span = None
        self.n_train = 0

    def fit(self, X: list[list[float]], labels: list[int]) -> "SwitchClassifier":
        from sklearn.gaussian_process import GaussianProcessClassifier
        X = np.asarray(X, float)
        yl = np.asarray(labels, int)
        self.n_train = len(yl)
        self.lo, self.span = _unit_scaling(X)
        if len(set(yl.tolist())) < 2:
            self.const_p = (yl.sum() + 1.0) / (len(yl) + 2.0)
            return self
        self.model = GaussianProcessClassifier(kernel=_matern(X.shape[1], self.opts),
                                               n_restarts_optimizer=int(self.opts["n_restarts"]), random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit((X - self.lo) / self.span, yl)
        return self

    def predict(self, X: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(P(switch), latent mean, latent std)."""
        n = len(X)
        if self.model is None:
            p = float(self.const_p if self.const_p is not None else 0.5)
            lat = math.log(p / (1 - p))
            return np.full(n, p), np.full(n, lat), np.full(n, 1.0)
        Xn = (np.asarray(X, float) - self.lo) / self.span
        p = self.model.predict_proba(Xn)[:, 1]
        mu_f, var_f = latent_mean_var(self.model, Xn)
        return p, mu_f, np.sqrt(np.maximum(var_f, 0.0))


def latent_mean_var(gpc, Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latent f mean/variance of a binary sklearn GaussianProcessClassifier (GPML Alg. 3.2 lines 4-6)."""
    if hasattr(gpc, "latent_mean_and_variance"):
        return gpc.latent_mean_and_variance(Xn)
    from scipy.linalg import solve
    be = gpc.base_estimator_
    k_star = be.kernel_(be.X_train_, Xn)
    mu = k_star.T.dot(be.y_train_ - be.pi_)
    v = solve(be.L_, be.W_sr_[:, None] * k_star)
    return mu, be.kernel_.diag(Xn) - np.einsum("ij,ij->j", v, v)


class MetricGPs:
    """One GP per active metric, refit per batch on all valid runs where that target is defined.

    switch_metric + switch_model='classifier': that metric's targets (> 0 = switched) go to a SwitchClassifier."""

    def __init__(self, metrics: list[str], seed: int = 0, *, switch_metric: str | None = None,
                 switch_model: str = "classifier", kernel_opts: dict | None = None,
                 min_train_points: int = MIN_TRAIN_POINTS, prior_means: dict[str, float] | None = None) -> None:
        self.metrics = metrics
        self.prior_means = dict(prior_means or {})
        self.seed = seed
        self.switch_metric = switch_metric
        self.switch_model = switch_model
        self.kernel_opts = kernel_opts
        self.min_train_points = int(min_train_points)
        self.models: dict[str, Any] = {}
        self.y_scale: dict[str, float] = {}
        self.n_rows = 0

    def fit(self, rows: list[tuple[list[float], dict[str, float]]]) -> dict[str, int]:
        self.models = {}
        self.y_scale = {}
        self.n_rows = len(rows)
        counts = {}
        for m in self.metrics:
            data = [(x, t[m]) for x, t in rows if m in t and t[m] is not None and math.isfinite(t[m])]
            counts[m] = len(data)
            if len(data) < self.min_train_points:
                continue
            if m == self.switch_metric and self.switch_model == "classifier":
                self.models[m] = SwitchClassifier(self.seed, self.kernel_opts).fit(
                    [d[0] for d in data], [1 if d[1] > 0 else 0 for d in data])
            else:
                gp = GPModel(m, self.seed, self.kernel_opts, self.prior_means.get(m)).fit([d[0] for d in data], [d[1] for d in data])
                self.models[m] = gp
                self.y_scale[m] = gp.y_std
        return counts

    def predict(self, X: list[list[float]]) -> tuple[list[dict], list[dict]]:
        means = [dict() for _ in X]
        stds = [dict() for _ in X]
        if not X:
            return means, stds
        for m, model in self.models.items():
            if isinstance(model, SwitchClassifier):
                p, mu_f, sd_f = model.predict(X)
                for i in range(len(X)):
                    means[i][m] = float(p[i]); stds[i][m] = float(math.sqrt(max(p[i] * (1 - p[i]), 0.0)))
                    means[i][m + LATENT] = float(mu_f[i]); stds[i][m + LATENT] = float(sd_f[i])
                continue
            mu, sd = model.predict(X)
            for i in range(len(X)):
                means[i][m] = float(mu[i]); stds[i][m] = float(sd[i])
        return means, stds


def straddle(mean_z: float, std_z: float, beta: float = 1.96) -> float:
    """Bryan et al. 2005 straddle heuristic: high = uncertain AND near the level set z = 0."""
    return beta * std_z - abs(mean_z)


def level_set_score(mean: float, std: float, beta: float = 1.96, level: float = 0.0) -> float:
    """Gotovos et al. 2013 LSE-style ambiguity for the level set f(x) = h: beta*sigma - |mu - h| (high = unclassified)."""
    return beta * std - abs(mean - level)


def boundary_score(c: dict, metric: str, switch_metric: str | None, *, beta: float, level_z: float = 0.0,
                   y_scale: float = 1.0, switch_penalty: float = 1.0, beta_switch: float | None = None) -> float | None:
    """integration3 default acquisition: max(recovery level-set score - penalty*(1-P(switch)), switch level-set score).

    recovery part: (beta*std_z - |mean_z - h_z|) / y_scale (training std of the target, so both parts are O(1));
    penalised where a switch is unlikely (no recovery target exists without a switch).
    switch part: beta*sd_f - |mu_f| on the classifier's latent function (P(switch) = 0.5 <=> f = 0).
    None when neither model has a prediction (caller falls back to space filling)."""
    gm, gs = c.get("gp_mean") or {}, c.get("gp_std") or {}
    rec = None
    if gm.get(metric) is not None and gs.get(metric) is not None:
        rec = level_set_score(gm[metric], gs[metric], beta, level_z) / max(y_scale, 1e-9)
    sw = None
    p = None
    if switch_metric is not None:
        p = gm.get(switch_metric)
        mu_f, sd_f = gm.get(switch_metric + LATENT), gs.get(switch_metric + LATENT)
        if mu_f is not None and sd_f is not None:
            sw = level_set_score(mu_f, sd_f, beta if beta_switch is None else beta_switch, 0.0)
    if rec is None:
        return sw
    if p is not None and 0.0 <= p <= 1.0:
        rec -= switch_penalty * (1.0 - p)
    return rec if sw is None else max(rec, sw)


def select_batch_scored(candidates: list[dict], score_of, *, batch_size: int, acquisition_fraction: float,
                        min_distance: float, max_per_seed: int, rng: random.Random, maximize: bool) -> list[dict]:
    """Generic 07 batch: acquisition share (diversity + per-seed cap) then random space-filling share."""
    scored = [(score_of(c), c) for c in candidates]
    ranked = [c for s, c in sorted((p for p in scored if p[0] is not None), key=lambda p: p[0], reverse=maximize)]
    n_acq = math.ceil(batch_size * acquisition_fraction)
    picked: list[dict] = []
    per_seed: dict[str, int] = {}
    for c in ranked:
        if len(picked) >= n_acq:
            break
        sid = c["seed_scenario_id"]
        if per_seed.get(sid, 0) >= max_per_seed:
            continue
        if any(math.dist(c["normalized_features"], p["normalized_features"]) < min_distance for p in picked):
            continue
        c["selected_by"] = "acquisition"
        picked.append(c)
        per_seed[sid] = per_seed.get(sid, 0) + 1
    rest = [c for c in candidates if c not in picked]
    rng.shuffle(rest)
    for c in rest[: batch_size - len(picked)]:
        c["selected_by"] = "random_space_filling"
        picked.append(c)
    return picked


def acquisition_score(c: dict, objective_cfg: dict, beta: float) -> float | None:
    metric = objective_cfg["metric"]
    mu, sd = (c.get("gp_mean") or {}).get(metric), (c.get("gp_std") or {}).get(metric)
    if mu is None or sd is None:
        return None
    if objective_cfg.get("acquisition") in ("straddle", "level_set"):
        return straddle(mu, sd, objective_cfg.get("beta", 1.96))
    return -J.lcb(mu, sd, beta)  # maximise -LCB == minimise LCB
