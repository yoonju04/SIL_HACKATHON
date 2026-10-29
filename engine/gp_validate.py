"""Offline GP check on finished campaigns (no CARLA, no LLM): leave-one-out, integration4 GP vs the old GP.

    python -m engine.gp_validate result/result_2 result/result_3 result/result_4 result/result_5 [--ablation X-5]

For every run: train on all the other runs, predict the left-out one. Reported per target
  sr4_margin_score     sign accuracy (harmful < 0 <= harmless), against always guessing the majority sign
  switched_indicator   accuracy (new: P(switch) > 0.5, old: regression mean > 0), same majority baseline
and how many distinct predictions there are (old GP: many candidates got the same value) and the mean std.
The old GP = shared param_u0/u1 columns, min/max re-scaling on the training data, bounds 1e-2..1e2, noise 1e-6,
+-1 regression for the switch label (what ran in result_2..5).
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

from . import scenarios as SC
from .config import REPO_ROOT
from .surrogate import MetricGPs


def legacy_numeric_features(c: dict, space: dict, routes: list[str], fault_pool: list[str]) -> list[float]:
    """05a numeric block as it was before integration4 (one param_u0/u1 pair shared by every fault)."""
    x = [1.0 if c.get("route_id") == r else 0.0 for r in routes]
    lo, hi = space["environment"]["max_speed_kmh"]["range"]
    x.append((c["environment"]["max_speed_kmh"] - lo) / (hi - lo))
    x += [1.0 if c["fault"]["fault_id"] == fid else 0.0 for fid in fault_pool]
    f = SC.faults_by_id(space)[c["fault"]["fault_id"]]
    pu = [SC._param_unit(spec, c["fault"]["params"].get(k)) for k, spec in (f.get("params") or {}).items()]
    pu = [u for u in pu if u is not None]
    x += [pu[0] if pu else 0.5, pu[1] if len(pu) > 1 else 0.5]
    x += [1.0 if c.get("onset_event") == ev else 0.0 for ev in space["fault_timing"]["onset_event"]["values"]]
    olo, ohi = space["fault_timing"]["offset_s"]["range"]
    x.append((c["onset_offset_s"] - olo) / (ohi - olo))
    return x


class LegacyGP:
    """engine/surrogate.GPModel before integration4."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, X, y) -> "LegacyGP":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        X = np.asarray(X, float)
        self.lo = X.min(axis=0)
        self.span = np.where(X.max(axis=0) - self.lo > 1e-12, X.max(axis=0) - self.lo, 1.0)
        d = X.shape[1]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5) \
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit((X - self.lo) / self.span, np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model.predict((np.asarray(X, float) - self.lo) / self.span, return_std=True)


def loo(rows: list[tuple[list[float], float]], fit_predict) -> tuple[list[float], list[float]]:
    mus, sds = [], []
    for i in range(len(rows)):
        train = rows[:i] + rows[i + 1:]
        mu, sd = fit_predict([x for x, _ in train], [y for _, y in train], rows[i][0])
        mus.append(mu); sds.append(sd)
    return mus, sds


def summary(truth: list[bool], guess: list[bool], mus: list[float], sds: list[float]) -> dict:
    major = max(sum(truth), len(truth) - sum(truth)) / len(truth)
    return {"n": len(truth), "accuracy": round(sum(t == g for t, g in zip(truth, guess)) / len(truth), 3),
            "majority_baseline": round(major, 3), "distinct_predictions": len({round(m, 6) for m in mus}),
            "mean_std": round(float(np.mean(sds)), 3)}


def validate(campaigns: list[Path], ablation: str = "X-5", seed: int = 0) -> dict:
    from run import SR4_METRIC, SWITCH_METRIC, Loop, gp_row_targets
    loop = Loop(mode="real", out=REPO_ROOT / "outputs" / "_gp_validate", ablation=ablation, video_on=False, seed=seed,
                cold_start=0, batch_size=1, rounds=0, quiet=True, prior_runs=campaigns)
    data = []
    for c, r in loop.prior:
        x = loop.features(c)[0]
        n_num = len(SC.numeric_features(c, loop.space, loop.route_ids, loop.fault_pool)[0])
        old_x = legacy_numeric_features(c, loop.space, loop.route_ids, loop.fault_pool) + x[n_num:]
        data.append((x, old_x, gp_row_targets(r)))
    out = {"campaigns": [str(c) for c in campaigns], "runs": len(data), "ablation": ablation}

    def new_fp(metric):
        def fp(X, y, xq):
            g = MetricGPs([metric], seed, **loop.gp_kwargs)
            g.fit([(a, {metric: b}) for a, b in zip(X, y)])
            if metric not in g.models:
                return math.nan, math.nan
            mu, sd = g.predict([xq])
            return mu[0][metric], sd[0][metric]
        return fp

    def old_fp(X, y, xq):
        mu, sd = LegacyGP(seed).fit(X, y).predict([xq])
        return float(mu[0]), float(sd[0])

    sr4 = [(d, d[2][SR4_METRIC]) for d in data if d[2].get(SR4_METRIC) is not None]
    truth = [y >= 0 for _, y in sr4]
    m_new, s_new = loo([(d[0], y) for d, y in sr4], new_fp(SR4_METRIC))
    m_old, s_old = loo([(d[1], y) for d, y in sr4], old_fp)
    out[SR4_METRIC] = {"new": summary(truth, [m >= 0 for m in m_new], m_new, s_new),
                       "old": summary(truth, [m >= 0 for m in m_old], m_old, s_old),
                       "harmless_runs": sum(truth)}
    sw = [(d, d[2][SWITCH_METRIC]) for d in data]
    truth = [y > 0.5 for _, y in sw]
    m_new, s_new = loo([(d[0], y) for d, y in sw], new_fp(SWITCH_METRIC))
    m_old, s_old = loo([(d[1], 2 * y - 1) for d, y in sw], old_fp)    # the old GP regressed +-1
    out[SWITCH_METRIC] = {"new": summary(truth, [m > 0.5 for m in m_new], m_new, s_new),
                          "old": summary(truth, [m > 0 for m in m_old], m_old, s_old),
                          "old_out_of_range": sum(abs(m) > 1 for m in m_old), "switched_runs": sum(truth)}
    g = MetricGPs([SR4_METRIC, SWITCH_METRIC], seed, **loop.gp_kwargs)
    g.fit([(d[0], d[2]) for d in data])
    out["fitted_on_all"] = g.diagnostics(loop.features(loop.prior[0][0])[1]) if loop.prior else {}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaigns", nargs="+", type=Path)
    ap.add_argument("--ablation", default="X-5")
    ap.add_argument("--out", type=Path, help="write the report as JSON here too")
    args = ap.parse_args(argv)
    rep = validate(args.campaigns, args.ablation)
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
