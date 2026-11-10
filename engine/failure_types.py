"""05c failure-type stacking layer (laya-judge): which way a switched run fails, per type, multi-label.

Types (pipeline/05c_failure_types/failure_types.yaml):
  jerk            P-04 lateral jerk over the paired baseline > 15 m/s^3 (integration4 limit)
  lat_acc         P-02 lateral acceleration over the baseline > 4.41 * tire_friction_scale m/s^2
  lane_departure  footprint left the lane after the switch (lane_recovery_robustness < 0, recovered_after_departure /
                  unrecovered, departed_at_switch); optionally SR4 footprint departure anywhere in the window
  unsettled       recovery_outcome == unsettled (did not hold the settle band for settle_hold_s)
Labels are read from finished run records (targets only). Inputs: DF columns + J regression probabilities (what the GP
gets) + judge uncertainty (variant disagreement, forward-vs-reverse option order difference).

Models: per-type L2 logistic with class_weight='balanced' (baseline) and a hierarchical Bayesian logistic whose
per-type slopes share a prior (engine.bayes_logistic.HierarchicalLogistic) so a type with 3-6 positives borrows the
shared direction. Decisions use cost thresholds p > c_fa / (c_fa + c_miss). Optional ordinal severity model.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from engine import bayes_logistic as BL

TYPES = ("jerk", "lat_acc", "lane_departure", "unsettled")
DEPARTURE_OUTCOMES = ("recovered_after_departure", "unrecovered")


def load_config(path: Path | None = None) -> dict:
    import yaml
    from engine.config import PIPELINE_DIR
    p = Path(path) if path else PIPELINE_DIR / "05c_failure_types" / "failure_types.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- labels (targets only)
def _f(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def type_labels(rec: dict, cfg: dict | None = None) -> dict[str, bool | None] | None:
    """{type: bool | None} for one run record; None for the whole run when it is not a labelled switched run."""
    cfg = cfg or {}
    tcfg = cfg.get("types") or {}
    if rec.get("sim_status") != "ok":
        return None
    switched = rec.get("switch_time") is not None
    if cfg.get("switched_only", True) and not switched:
        return None
    m = rec.get("metrics") or {}
    env = ((rec.get("parameters") or {}).get("environment") or {})
    fr = _f(env.get("tire_friction_scale")) or _f(m.get("tire_friction_scale")) or 1.0
    jl = float((tcfg.get("jerk") or {}).get("lat_jerk_limit_mps3", 15.0))
    ab = float((tcfg.get("lat_acc") or {}).get("lat_acc_base_mps2", 4.41))
    jerk, acc = _f(m.get("lat_jerk_excess_mps3")), _f(m.get("lat_acc_excess_mps2"))
    out = rec.get("recovery_outcome")
    lrr = _f(rec.get("lane_recovery_robustness", m.get("lane_recovery_robustness")))
    dep_sw = (rec.get("recovery_labels") or {}).get("departed_at_switch")
    dep = None
    if out is not None or lrr is not None:
        dep = bool((lrr is not None and lrr < 0) or out in DEPARTURE_OUTCOMES or dep_sw is True)
        if (tcfg.get("lane_departure") or {}).get("include_sr4_footprint"):
            sr4 = ((rec.get("sr_checks") or {}).get("sr") or {}).get("SR4") or {}
            dep = dep or sr4.get("pass") is False or bool((rec.get("sr_labels") or {}).get("footprint_departure"))
    return {"jerk": None if jerk is None else jerk > jl,
            "lat_acc": None if acc is None else acc > ab * fr,
            "lane_departure": dep,
            "unsettled": None if out is None else out == "unsettled"}


def severity_level(rec: dict, labels: dict[str, bool | None], order: list[str]) -> int | None:
    """Index in the (team-decided) severity order: none < jerk_only < lat_acc < unsettled < departed_recovered <
    unrecovered by default. The worst level that applies wins."""
    if labels is None:
        return None
    out = rec.get("recovery_outcome")
    lvl = "none"
    if labels.get("jerk"):
        lvl = "jerk_only"
    if labels.get("lat_acc"):
        lvl = "lat_acc"
    if labels.get("unsettled"):
        lvl = "unsettled"
    if labels.get("lane_departure"):
        lvl = "departed_recovered"
    if out == "unrecovered":
        lvl = "unrecovered"
    return order.index(lvl) if lvl in order else None


def label_counts(labels: list[dict[str, bool | None]]) -> dict[str, dict[str, int]]:
    return {t: {"positive": sum(1 for l in labels if l and l.get(t) is True),
                "negative": sum(1 for l in labels if l and l.get(t) is False),
                "missing": sum(1 for l in labels if not l or l.get(t) is None)} for t in TYPES}


# ---------------------------------------------------------------- judge uncertainty features
def _tv(p: dict, q: dict) -> float:
    ks = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in ks)


def judge_uncertainty(parsed: dict[str, list[dict]], variant_specs: dict[str, dict] | None = None,
                      qids: list[str] | None = None) -> tuple[list[float], list[str]]:
    """Two state-level columns from the raw judge vectors (engine.judges.parse_answers output):
    judge_disagreement = mean over questions of the mean pairwise total-variation distance between variants;
    judge_order_gap    = mean over questions of TV(mean forward-order variants, mean reverse-order variants).
    Variants whose spec has order: reverse are 'reverse' (en_notes_rev, focus_supervisor, focus_switch_rev)."""
    specs = variant_specs or {}
    dis, gap = [], []
    for qid, js in (parsed or {}).items():
        if qids is not None and qid not in qids:
            continue
        if len(js) >= 2:
            pr = [_tv(a["probs"], b["probs"]) for i, a in enumerate(js) for b in js[i + 1:]]
            dis.append(float(np.mean(pr)))
        fw = [j["probs"] for j in js if (specs.get(j["judge"]) or {}).get("order", "forward") != "reverse"]
        rv = [j["probs"] for j in js if (specs.get(j["judge"]) or {}).get("order") == "reverse"]
        if fw and rv:
            mean = lambda ps: {k: float(np.mean([p.get(k, 0.0) for p in ps])) for k in set().union(*ps)}
            gap.append(_tv(mean(fw), mean(rv)))
    return [float(np.mean(dis)) if dis else 0.0, float(np.mean(gap)) if gap else 0.0], \
        ["judge_disagreement", "judge_order_gap"]


# ---------------------------------------------------------------- inputs
DF_PREFIX = "DF-"
JUDGE_PREFIXES = ("B_", "J1=", "J2=", "J3=", "J4=", "J5=", "J6=", "J7=")


def select_columns(names: list[str], inputs: list[str]) -> list[int]:
    """Indices of the GP feature vector to keep: df (DF-* and their _missing), judges (B_* / J*= columns),
    scenario (everything else the GP has: route, speed, friction, fault one-hot, parameters, onset)."""
    keep = []
    for i, n in enumerate(names):
        is_df = n.startswith(DF_PREFIX)
        is_j = n.startswith(JUDGE_PREFIXES)
        if ("df" in inputs and is_df) or ("judges" in inputs and is_j) or \
                ("scenario" in inputs and not is_df and not is_j):
            keep.append(i)
    return keep


# ---------------------------------------------------------------- models
def cost_thresholds(cfg: dict) -> dict[str, float]:
    c = cfg.get("costs") or {}
    fa = float(c.get("false_alarm", 1.0))
    miss = c.get("miss") or {}
    return {t: fa / (fa + float(miss.get(t, 1.0))) for t in TYPES}


def _prior_shift(y: np.ndarray) -> float:
    pi = float(np.clip(np.mean(y), 1e-3, 1 - 1e-3))
    return math.log(pi / (1 - pi))


class FailureTypeModel:
    """fit(X, Y) with Y (n, T) in TYPES order (NaN = missing); predict(X) -> {probs (n, T), epistemic (n, T)}.
    method logistic_balanced: sklearn per type, class_weight='balanced', prior-corrected; epistemic from a Laplace
    BayesLogistic with the same prior and weights. hier_bayes: HierarchicalLogistic (shared slopes)."""

    def __init__(self, cfg: dict | None = None, method: str | None = None):
        self.cfg = cfg or {}
        mc = self.cfg.get("model") or {}
        self.method = method or mc.get("method", "hier_bayes")
        self.mc = mc
        self.types = list(TYPES)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "FailureTypeModel":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self.std = BL.Standardizer().fit(X)
        Z = self.std.transform(X)
        self.const = {}
        self.models = {}
        C = float(self.mc.get("logistic_C", 0.5))
        if self.method == "logistic_balanced":
            for t, name in enumerate(self.types):
                m = ~np.isnan(Y[:, t])
                y = Y[m, t].astype(int)
                if len(set(y.tolist())) < 2:
                    self.const[name] = float(y.mean()) if len(y) else 0.0
                    continue
                w = BL.balanced_weights(y)
                bl = BL.BayesLogistic(prior_var=C, intercept_var=1e4).fit(Z[m], y, sample_weight=w)
                shift = _prior_shift(y) if self.mc.get("prior_correction", True) else 0.0
                self.models[name] = (bl, shift)
        elif self.method == "hier_bayes":
            hc = self.mc.get("hier") or {}
            SW = None
            if self.mc.get("balanced_weights", False):
                SW = np.ones_like(Y)
                for t in range(Y.shape[1]):
                    m = ~np.isnan(Y[:, t])
                    if len(set(Y[m, t].tolist())) == 2:
                        SW[m, t] = BL.balanced_weights(Y[m, t].astype(int))
            self.shifts = np.array([(_prior_shift(Y[~np.isnan(Y[:, t]), t]) if SW is not None and
                                     self.mc.get("prior_correction", True) else 0.0) for t in range(Y.shape[1])])
            self.hier = BL.HierarchicalLogistic(mu_var=float(hc.get("mu_var", 1.0)), tau=hc.get("tau"),
                                                tau_grid=tuple(hc.get("tau_grid") or (0.1, 0.25, 0.5, 1.0, 2.0)),
                                                intercept_var=float(hc.get("intercept_var", 25.0))).fit(Z, Y, SW)
        else:
            raise ValueError(f"failure_types model.method must be logistic_balanced | hier_bayes, not {self.method!r}")
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Z = self.std.transform(np.asarray(X, dtype=float))
        n, T = len(Z), len(self.types)
        P, E = np.zeros((n, T)), np.zeros((n, T))
        if self.method == "logistic_balanced":
            for t, name in enumerate(self.types):
                if name in self.const:
                    P[:, t] = self.const[name]
                    continue
                bl, shift = self.models[name]
                mu, var = bl.logit_moments(Z)
                P[:, t] = 1 / (1 + np.exp(-((mu + shift) / np.sqrt(1 + math.pi * var / 8))))
                E[:, t] = var
        else:
            mu, var = self.hier.logit_moments(Z)
            mu = mu + self.shifts[None, :]
            P = 1 / (1 + np.exp(-(mu / np.sqrt(1 + math.pi * var / 8))))
            E = var
        return {"probs": P, "epistemic": E}

    def decide(self, P: np.ndarray, thresholds: dict[str, float] | None = None) -> np.ndarray:
        th = thresholds or cost_thresholds(self.cfg)
        return P > np.array([th[t] for t in self.types])[None, :]


class OrdinalSeverityModel:
    """One cumulative-logit model over the configured severity order (team decision)."""

    def __init__(self, cfg: dict | None = None):
        oc = (cfg or {}).get("ordinal") or {}
        self.order = list(oc.get("order") or ["none", "jerk_only", "lat_acc", "unsettled", "departed_recovered", "unrecovered"])
        self.prior_var = float(oc.get("prior_var", 0.5))

    def fit(self, X, y_levels) -> "OrdinalSeverityModel":
        self.std = BL.Standardizer().fit(X)
        self.m = BL.OrdinalLogistic(prior_var=self.prior_var, n_classes=len(self.order)).fit(self.std.transform(X),
                                                                                            np.asarray(y_levels, int))
        return self

    def predict(self, X) -> dict[str, np.ndarray]:
        return self.m.uncertainty(self.std.transform(X))


# ---------------------------------------------------------------- acquisition hook (07)
def rare_type_score(pred: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """weight * max over rare types of P_t(x) * epistemic_t(x): high where a rare failure type is plausible AND the
    model does not know (its logit variance is large)."""
    ac = cfg.get("acquisition") or {}
    idx = [TYPES.index(t) for t in (ac.get("rare_types") or ["lane_departure", "unsettled", "lat_acc"]) if t in TYPES]
    if not idx:
        return np.zeros(len(pred["probs"]))
    s = pred["probs"][:, idx] * pred["epistemic"][:, idx]
    return float(ac.get("weight", 1.0)) * s.max(axis=1)
