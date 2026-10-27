"""06 GP surrogate (one GP per metric, Matern 5/2 ARD + White, scikit-learn) and 07 acquisitions.

* Targets are run_record.gp_targets[metric] = z = sign*y/scale (judgment.gp_targets); z low = riskier.
* Features are normalised on the training data only (05a feature_engineering.normalization).
* S1 boundary acquisition = straddle: beta*std_z - |mean_z| on lane_recovery_robustness (metric
  'recovery_success'); maximised. The boundary is y = 0 <=> z = 0, so |mean_z| is the distance to it.
* LCB (mean_z - beta*std_z, minimised) stays for objectives without a boundary (judgment.lcb).
"""
from __future__ import annotations

import math
import random
import warnings
from typing import Any

import numpy as np

from . import judgment as J

MIN_TRAIN_POINTS = 4


class GPModel:
    def __init__(self, metric: str, seed: int = 0) -> None:
        self.metric = metric
        self.seed = seed
        self.model = None
        self.lo = self.span = None
        self.n_train = 0

    def fit(self, X: list[list[float]], y: list[float]) -> "GPModel":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.lo = X.min(axis=0)
        self.span = np.where(X.max(axis=0) - self.lo > 1e-12, X.max(axis=0) - self.lo, 1.0)
        Xn = (X - self.lo) / self.span
        d = X.shape[1]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5) \
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                                              random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Xn, y)
        self.n_train = len(y)
        return self

    def predict(self, X: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        Xn = (np.asarray(X, float) - self.lo) / self.span
        mu, sd = self.model.predict(Xn, return_std=True)
        return mu, sd


class MetricGPs:
    """One GP per active metric, refit per batch on all valid runs where that target is defined."""

    def __init__(self, metrics: list[str], seed: int = 0) -> None:
        self.metrics = metrics
        self.seed = seed
        self.models: dict[str, GPModel] = {}

    def fit(self, rows: list[tuple[list[float], dict[str, float]]]) -> dict[str, int]:
        self.models = {}
        counts = {}
        for m in self.metrics:
            data = [(x, t[m]) for x, t in rows if m in t and t[m] is not None and math.isfinite(t[m])]
            counts[m] = len(data)
            if len(data) >= MIN_TRAIN_POINTS:
                self.models[m] = GPModel(m, self.seed).fit([d[0] for d in data], [d[1] for d in data])
        return counts

    def predict(self, X: list[list[float]]) -> tuple[list[dict], list[dict]]:
        means = [dict() for _ in X]
        stds = [dict() for _ in X]
        if not X:
            return means, stds
        for m, gp in self.models.items():
            mu, sd = gp.predict(X)
            for i in range(len(X)):
                means[i][m] = float(mu[i]); stds[i][m] = float(sd[i])
        return means, stds


def straddle(mean_z: float, std_z: float, beta: float = 1.96) -> float:
    """Bryan et al. 2005 straddle heuristic: high = uncertain AND near the level set z = 0."""
    return beta * std_z - abs(mean_z)


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
    if objective_cfg.get("acquisition") == "straddle":
        return straddle(mu, sd, objective_cfg.get("beta", 1.96))
    return -J.lcb(mu, sd, beta)  # maximise -LCB == minimise LCB
