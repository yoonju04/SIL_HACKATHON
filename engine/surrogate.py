"""06 GP surrogate (one GP per metric, Matern 5/2 ARD + White, scikit-learn; 0/1 metrics: GP classifier) and 07 acquisitions.

* Targets are run_record.gp_targets[metric] = z = sign*y/scale (judgment.gp_targets); z low = riskier.
* Features come unit-scaled from the scenario-space ranges (scenarios.numeric_features); the GP does not re-scale them.
* S1 boundary acquisition = straddle: beta*std_z - |mean_z| on the objective metric (integration4: SR4 lane-margin
  score, defined for every run; before: lane_recovery_robustness, switched runs only); maximised. The boundary
  is z = 0, so |mean_z| is the distance to it.
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
# integration4: the old bounds (1e-2..1e2, noise 1e-6) let 4-9 training points pin every length scale to a bound
# and the noise to ~0, so the GP interpolated its few points and returned the prior everywhere else.
LENGTH_SCALE_BOUNDS = (0.1, 10.0)
NOISE_BOUNDS = (1e-3, 1.0)


def _kernel(d: int, length_scale_bounds=LENGTH_SCALE_BOUNDS, noise_bounds=NOISE_BOUNDS, white: bool = True):
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(d), length_scale_bounds=tuple(length_scale_bounds), nu=2.5)
    return k + WhiteKernel(noise_level=0.05, noise_level_bounds=tuple(noise_bounds)) if white else k


class GPModel:
    """GP regression on features that are already unit-scaled by the scenario space (05a numeric_features).

    No re-scaling on the training data: with min/max of a few runs, a fault never seen in training had a constant
    column and new candidates landed far outside the training box."""

    kind = "regression"

    def __init__(self, metric: str, seed: int = 0, **kernel_cfg: Any) -> None:
        self.metric = metric
        self.seed = seed
        self.kernel_cfg = kernel_cfg
        self.model = None
        self.n_train = 0

    def fit(self, X: list[list[float]], y: list[float]) -> "GPModel":
        from sklearn.gaussian_process import GaussianProcessRegressor
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.constant_cols = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) == 0]
        self.model = GaussianProcessRegressor(kernel=_kernel(X.shape[1], **self.kernel_cfg), normalize_y=True,
                                              n_restarts_optimizer=2, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        self.n_train = len(y)
        return self

    def predict(self, X: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.model.predict(np.asarray(X, float), return_std=True)
        return mu, sd

    def diagnostics(self, names: list[str] | None = None) -> dict[str, Any]:
        return _kernel_report(self.model.kernel_, self.n_train, names, self.kind, self.constant_cols)


class GPClassModel:
    """GP classifier for a 0/1 label (switched or not). mean = P(label = 1), std = sqrt(p (1 - p)).

    Regressing +-1 as a number gave predictions like 1.00014 or -0.93 that are not probabilities."""

    kind = "classification"

    def __init__(self, metric: str, seed: int = 0, **kernel_cfg: Any) -> None:
        self.metric = metric
        self.seed = seed
        self.kernel_cfg = kernel_cfg
        self.model = None
        self.constant: float | None = None
        self.n_train = 0

    def fit(self, X: list[list[float]], y: list[float]) -> "GPClassModel":
        from sklearn.gaussian_process import GaussianProcessClassifier
        X = np.asarray(X, float)
        y = (np.asarray(y, float) > 0.5).astype(int)
        self.n_train = len(y)
        self.constant_cols = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) == 0]
        if y.min() == y.max():          # one class only: nothing to separate yet
            self.constant = float(y[0])
            return self
        kcfg = {k: v for k, v in self.kernel_cfg.items() if k != "noise_bounds"}
        self.model = GaussianProcessClassifier(kernel=_kernel(X.shape[1], white=False, **kcfg),
                                               n_restarts_optimizer=2, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict(self, X: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, float)
        p = np.full(len(X), self.constant) if self.model is None else self.model.predict_proba(X)[:, 1]
        return p, np.sqrt(p * (1.0 - p))

    def diagnostics(self, names: list[str] | None = None) -> dict[str, Any]:
        if self.model is None:
            return {"kind": self.kind, "n_train": self.n_train, "constant": self.constant}
        return _kernel_report(self.model.kernel_, self.n_train, names, self.kind, self.constant_cols)


def _kernel_report(k, n_train: int, names: list[str] | None, kind: str, constant_cols: list[int]) -> dict[str, Any]:
    """Fitted amplitude / noise / length scales; columns with no information: constant in training (length scale
    never moves from its start) or length scale on a bound."""
    has_white = hasattr(k, "k2") and hasattr(k.k2, "noise_level")
    prod = k.k1 if has_white else k             # regression: (C * Matern) + White; classification: C * Matern
    amp, mat = float(prod.k1.constant_value), prod.k2
    ls = np.atleast_1d(mat.length_scale).astype(float)
    lo, hi = mat.length_scale_bounds[0] if np.ndim(mat.length_scale_bounds) > 1 else mat.length_scale_bounds
    names = names if names and len(names) == len(ls) else [f"x{i}" for i in range(len(ls))]
    noise = float(k.k2.noise_level) if has_white else None
    return {"kind": kind, "n_train": n_train, "amplitude": amp, "noise": noise,
            "length_scales": {n: round(float(v), 4) for n, v in zip(names, ls)},
            "at_bound": [n for n, v in zip(names, ls) if v <= lo * 1.001 or v >= hi * 0.999],
            "constant_in_training": [names[i] for i in constant_cols]}


class MetricGPs:
    """One GP per active metric, refit per batch on all valid runs where that target is defined.

    Metrics in `classify` (0/1 labels) get a GP classifier instead of a regressor."""

    def __init__(self, metrics: list[str], seed: int = 0, *, classify: tuple[str, ...] | list[str] = (),
                 min_train_points: int = MIN_TRAIN_POINTS, **kernel_cfg: Any) -> None:
        self.metrics = metrics
        self.seed = seed
        self.classify = set(classify)
        self.min_train_points = int(min_train_points)
        self.kernel_cfg = kernel_cfg
        self.models: dict[str, GPModel | GPClassModel] = {}

    def fit(self, rows: list[tuple[list[float], dict[str, float]]]) -> dict[str, int]:
        self.models = {}
        counts = {}
        for m in self.metrics:
            data = [(x, t[m]) for x, t in rows if m in t and t[m] is not None and math.isfinite(t[m])]
            counts[m] = len(data)
            if len(data) >= self.min_train_points:
                cls = GPClassModel if m in self.classify else GPModel
                self.models[m] = cls(m, self.seed, **self.kernel_cfg).fit([d[0] for d in data], [d[1] for d in data])
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

    def diagnostics(self, names: list[str] | None = None) -> dict[str, Any]:
        return {m: gp.diagnostics(names) for m, gp in self.models.items()}


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
