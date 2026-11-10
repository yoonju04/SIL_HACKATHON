"""Per-question temperature scaling of saved System One judge answers (Laya first), evaluated on held-out folds.

For each code question (J1 J3 J4 J7 J8 J9; outcome questions J2 J5 J6 are excluded, as in the fine-tune plan), one
temperature T per question is fitted on the training folds: p_T(o) ∝ p(o)^(1/T) for every judge variant, minimising
the NLL of the code answer key (list answers = soft target, uniform over the allowed set: NLL = -sum_a log p_T(a) / |A|).
Held-out ECE (top-label confidence vs. 'argmax in the allowed set'), Brier (vs. the soft target) and NLL before/after.
Temperature never changes a judge's argmax, so raw accuracy is identical by construction; the JudgeRegression on top
(log p_T as inputs) can change, so its LOMO accuracy is reported before/after too. No model is called.

  python tools/laya_calibration_eval.py --raw /workspace/judges_eval_digest/raw_laya_digest.json --out /tmp/laya_cal.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from engine import judges as J  # noqa: E402
from j_regression_offline_eval import load_rows  # noqa: E402

CODE_QS = ["J1", "J3", "J4", "J7", "J8", "J9"]
CLIP = 1e-6


def scaled(probs: dict[str, float], T: float) -> dict[str, float]:
    lg = {o: math.log(max(CLIP, p)) / T for o, p in probs.items()}
    m = max(lg.values())
    e = {o: math.exp(v - m) for o, v in lg.items()}
    s = sum(e.values())
    return {o: v / s for o, v in e.items()}


def items(rows, q, idx):
    """(judge probs, allowed set) for every judge answer of question q in the given states."""
    out = []
    for i in idx:
        r = rows[i]
        allowed = r["key"].get(q)
        if not allowed:
            continue
        for j in r["parsed"].get(q) or []:
            out.append((j["probs"], list(allowed)))
    return out


def nll(its, T):
    return float(np.mean([-np.mean([math.log(max(CLIP, scaled(p, T).get(a, 0.0))) for a in al]) for p, al in its])) if its else 0.0


def fit_T(its) -> float:
    if not its:
        return 1.0
    r = minimize_scalar(lambda lt: nll(its, math.exp(lt)), bounds=(math.log(0.05), math.log(50.0)), method="bounded")
    return float(math.exp(r.x))


def metrics(its, T, bins: int = 10) -> dict:
    if not its:
        return {}
    P = [scaled(p, T) for p, _ in its]
    conf = np.array([max(p.values()) for p in P])
    hit = np.array([max(p, key=p.get) in al for p, (_, al) in zip(P, its)], dtype=float)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (conf >= lo) & ((conf < hi) if b < bins - 1 else (conf <= hi))
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - hit[sel].mean())
    br = np.mean([sum((p.get(o, 0.0) - (1.0 / len(al) if o in al else 0.0)) ** 2 for o in p) for p, (_, al) in zip(P, its)])
    return {"ece": round(float(ece), 3), "brier": round(float(br), 3), "nll": round(nll(its, T), 3),
            "accuracy": round(float(hit.mean()), 3), "mean_conf": round(float(conf.mean()), 3)}


def folds(rows, kind: str, k: int = 5, seed: int = 0):
    n = len(rows)
    if kind == "lomo":
        for m in sorted({r["mode"] for r in rows}):
            te = [i for i in range(n) if rows[i]["mode"] == m]
            yield [i for i in range(n) if i not in te], te
        return
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for f in range(k):
        te = sorted(perm[f::k].tolist())
        yield [i for i in range(n) if i not in te], te


def apply_T(rows, Ts: dict[str, float]):
    out = copy.deepcopy(rows)
    for r in out:
        for q, js in r["parsed"].items():
            if q in Ts:
                for j in js:
                    j["probs"] = scaled(j["probs"], Ts[q])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/judges_eval_digest/raw_laya_digest.json")
    ap.add_argument("--campaign", default=str(ROOT / "result/result_8"))
    ap.add_argument("--out", default="/tmp/laya_cal.json")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "pipeline/05b_judges/judges.yaml").read_text())
    bank_l = J.load_bank()
    rows, bank = load_rows(Path(a.raw), Path(a.campaign), cfg, bank_l)
    rep = {"raw": a.raw, "n_states": len(rows), "per_question": {}}
    for q in CODE_QS:
        res = {}
        for kind in ("kfold5", "lomo"):
            before, after, Tf = [], [], []
            for tr, te in folds(rows, "lomo" if kind == "lomo" else "k"):
                T = fit_T(items(rows, q, tr))
                Tf.append(round(T, 3))
                te_its = items(rows, q, te)
                before += [(p, al, 1.0) for p, al in te_its]
                after += [(p, al, T) for p, al in te_its]
            # pooled held-out metrics: each item scored with its own fold's T
            def pooled(its):
                if not its:
                    return {}
                Ps = [((scaled(p, T)), al) for p, al, T in its]
                return metrics([(p, al) for p, al in Ps], 1.0)
            res[kind] = {"T_per_fold": Tf, "before": pooled(before), "after": pooled(after), "n_items": len(before)}
        res["T_all_states"] = round(fit_T(items(rows, q, range(len(rows)))), 3)
        rep["per_question"][q] = res
        print(q, json.dumps(res), flush=True)
    # regression on top: LOMO accuracy of JudgeRegression with raw vs temperature-scaled judge vectors (T fitted on the
    # training modes only), J1 J4 J7 (single-answer labels)
    from j_regression_offline_eval import evaluate as jeval
    reg_rep = {}
    for q in ("J1", "J4", "J7"):
        base = jeval(rows, bank, cfg, [q], {"inputs": "per_judge", "model": "logistic_l2", "ordinal": []})[q]
        hits, maj = [], []
        lab = [i for i, r in enumerate(rows) if r["parsed"].get(q) and r["key"].get(q) and len(r["key"][q]) == 1]
        for tr, te in folds(rows, "lomo"):
            T = fit_T(items(rows, q, tr))
            cal = apply_T(rows, {q: T})
            trl, tel = [i for i in tr if i in lab], [i for i in te if i in lab]
            if not tel or len({rows[i]["key"][q][0] for i in trl}) < 2:
                continue
            reg = J.JudgeRegression(dict(cfg.get("regression") or {}, inputs="per_judge"), {q: cfg["judges"][q]},
                                    {q: [str(o) for o in bank[q]["answer"]]}, variant_specs=cfg.get("variants") or {})
            reg.fit([({q: cal[i]["parsed"][q]}, {q: cal[i]["key"][q]}) for i in trl], cv=False)
            for i in tel:
                p = reg.predict(q, cal[i]["parsed"][q])
                hits.append(max(p, key=p.get) == rows[i]["key"][q][0])
        reg_rep[q] = {"lomo_regression_raw": base.get("lomo_accuracy"), "lomo_regression_temperature": round(float(np.mean(hits)), 3) if hits else None,
                      "lomo_majority_no_judges": base.get("lomo_majority_no_judges")}
        print(q, reg_rep[q], flush=True)
    rep["regression_on_top"] = reg_rep
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
