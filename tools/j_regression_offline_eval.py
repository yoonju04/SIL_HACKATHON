"""Offline comparison of 05b J-question regressions on saved judge answers (no model is called).

Variants (judges.yaml regression.*): per_judge (v2: V x K log-probs, L2 multinomial) vs compressed (K mean log-probs +
variant std + forward/reverse gap) with L2 multinomial, with the Laplace Bayesian multinomial (epistemic = mutual
information), with cumulative-logit ordinal regression for J2 / J5, and with ridge (L2 least squares, RidgeCV alpha,
scores clipped to [0, 1]; one-vs-rest by default, or ordinal-coded ridge when ridge_ordinal is set). Per question:
leave-one-state-out (LOO), leave-one-fault-mode-out (LOMO) and round-order / forward (train on earlier rounds only)
accuracy, Brier, and one-vs-rest AUROC where it is defined, next to the judge-free fault-mode lookup (LOO) and the
training majority (LOMO). Bayesian variants also report whether the epistemic value ranks the LOMO errors (AUROC; 0.5
= uninformative). Labels: code questions = profile-aware answer key, outcome questions = that run's record; labels
never enter a judge state.

  python tools/j_regression_offline_eval.py --raw /workspace/judges_eval/raw_nemotron.json --out /tmp/jreg_nemotron.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from engine import judges as J  # noqa: E402
from engine.bayes_logistic import roc_auc  # noqa: E402
from judges_offline_eval import run_states  # noqa: E402

VARIANTS = {
    "per_judge_l2":         {"inputs": "per_judge", "model": "logistic_l2", "ordinal": []},
    "compressed_l2":        {"inputs": "compressed", "model": "logistic_l2", "ordinal": []},
    "compressed_l2_ord":    {"inputs": "compressed", "model": "logistic_l2", "ordinal": ["J2", "J5"]},
    "compressed_bayes_ord": {"inputs": "compressed", "model": "bayes_laplace", "ordinal": ["J2", "J5"]},
    "compressed_ridge":     {"inputs": "compressed", "model": "ridge", "ordinal": []},
    "compressed_ridge_ord": {"inputs": "compressed", "model": "ridge", "ordinal": ["J2", "J5"], "ridge_ordinal": True},
}


def load_rows(raw_path: Path, camp: Path, cfg: dict, bank_l: list[dict], context: str | None = None):
    bank = {q["id"]: q for q in bank_l}
    states = run_states(camp, "tight")
    raw = json.loads(raw_path.read_text())
    ctx0 = context or next(iter(raw.values())).get("judge_context", "full")
    _, meta = J.judge_questions(bank_l, cfg, None, context=ctx0)
    rows = []
    for k in sorted(raw):             # sorted run ids: same order (and lookup tie-breaks) as judges_offline_eval
        r = raw[k]
        if k not in states:
            continue
        ctx, rec = states[k]
        rows.append({"run_id": k, "parsed": J.parse_answers(r["response"], meta),
                     "key": J.labels_for(bank_l, ctx, rec), "mode": ctx["fault_mode"],
                     "round": int(rec.get("round", 0))})
    return rows, bank


def brier1(p: dict, t: str) -> float:
    return float(sum((v - (1.0 if o == t else 0.0)) ** 2 for o, v in p.items()))


def auroc_ovr(probs: list[dict], truth: list[str], opts: list[str]) -> float | None:
    """Macro one-vs-rest AUROC over options that have both a positive and a negative in `truth`.
    Binary (or one defined class) returns that one value; multiclass needs at least two defined
    class AUROCs, otherwise None."""
    vals = []
    for o in opts:
        a = roc_auc([p.get(o, 0.0) for p in probs], [t == o for t in truth])
        if a is not None:
            vals.append(a)
    need = 1 if len(opts) <= 2 else 2
    return None if len(vals) < need else float(np.mean(vals))


def evaluate(rows, bank, cfg, qids, variant: dict, seed: int = 0) -> dict:
    rc = dict(copy.deepcopy(cfg.get("regression") or {}), **variant)
    variants_of = {q: list(v) for q, v in cfg["judges"].items()}
    options_of = {q: [str(o) for o in bank[q]["answer"]] for q in variants_of}
    out = {}
    for q in qids:
        lab = [i for i, r in enumerate(rows) if r["parsed"].get(q) and r["key"].get(q) and len(r["key"][q]) == 1]
        if len(lab) < 8 or len({rows[i]["key"][q][0] for i in lab}) < 2:
            out[q] = {"n": len(lab), "note": "too few single-answer labels"}
            continue
        y = {i: rows[i]["key"][q][0] for i in lab}

        def fit_pred(tr, te):
            reg = J.JudgeRegression(rc, {q: variants_of[q]}, {q: options_of[q]}, seed=seed,
                                    variant_specs=cfg.get("variants") or {})
            reg.fit([({q: rows[i]["parsed"][q]}, {q: rows[i]["key"][q]}) for i in tr], cv=False)
            return [(reg.predict(q, rows[i]["parsed"][q]), reg.epistemic(q, rows[i]["parsed"][q])) for i in te]

        res = {"n": len(lab), "labels": dict(Counter(y.values()).most_common())}
        opts = options_of[q]
        # LOO
        P = [fit_pred([j for j in lab if j != i], [i])[0][0] for i in lab]
        yt = [y[i] for i in lab]
        res["loo_accuracy"] = round(float(np.mean([max(p, key=p.get) == t for p, t in zip(P, yt)])), 3)
        res["loo_brier"] = round(float(np.mean([brier1(p, t) for p, t in zip(P, yt)])), 3)
        a = auroc_ovr(P, yt, opts)
        res["loo_auroc"] = None if a is None else round(a, 3)
        look = []
        for i in lab:
            pool = [y[j] for j in lab if j != i and rows[j]["mode"] == rows[i]["mode"]] or [y[j] for j in lab if j != i]
            look.append(Counter(pool).most_common(1)[0][0] == y[i])
        res["fault_mode_lookup_loo"] = round(float(np.mean(look)), 3)
        # LOMO
        hits, maj, br, epi, err, P_lomo, yt_lomo = [], [], [], [], [], [], []
        for m in sorted({rows[i]["mode"] for i in lab}):
            te = [i for i in lab if rows[i]["mode"] == m]
            tr = [i for i in lab if rows[i]["mode"] != m]
            if not tr:
                continue
            if len({y[i] for i in tr}) < 2:
                pr = [({o: float(o == y[tr[0]]) for o in opts}, None) for _ in te]
            else:
                pr = fit_pred(tr, te)
            mj = Counter(y[i] for i in tr).most_common(1)[0][0]
            for (p, e), i in zip(pr, te):
                ok = max(p, key=p.get) == y[i]
                hits.append(ok); maj.append(mj == y[i]); br.append(brier1(p, y[i]))
                P_lomo.append(p); yt_lomo.append(y[i])
                if e is not None:
                    epi.append(e); err.append(not ok)
        res["lomo_accuracy"] = round(float(np.mean(hits)), 3)
        res["lomo_brier"] = round(float(np.mean(br)), 3)
        res["lomo_majority_no_judges"] = round(float(np.mean(maj)), 3)
        a = auroc_ovr(P_lomo, yt_lomo, opts)
        res["lomo_auroc"] = None if a is None else round(a, 3)
        if epi:
            a = roc_auc(epi, err)
            res["lomo_epistemic_auroc_for_errors"] = None if a is None else round(a, 3)
            res["lomo_epistemic_mean_wrong_vs_right"] = [round(float(np.mean([e for e, w in zip(epi, err) if w])), 4) if any(err) else None,
                                                         round(float(np.mean([e for e, w in zip(epi, err) if not w])), 4) if not all(err) else None]
        # round-order / forward: train on earlier rounds only, test the next round (skip a block with
        # fewer than min_labels training rows or fewer than two classes)
        min_tr = int(rc.get("min_labels", 8))
        P_fwd, yt_fwd = [], []
        for rnd in sorted({rows[i]["round"] for i in lab}):
            te = [i for i in lab if rows[i]["round"] == rnd]
            tr = [i for i in lab if rows[i]["round"] < rnd]
            if len(tr) < min_tr or len({y[i] for i in tr}) < 2:
                continue
            for (p, _), i in zip(fit_pred(tr, te), te):
                P_fwd.append(p); yt_fwd.append(y[i])
        if P_fwd:
            res["forward_n"] = len(P_fwd)
            res["forward_accuracy"] = round(float(np.mean([max(p, key=p.get) == t for p, t in zip(P_fwd, yt_fwd)])), 3)
            res["forward_brier"] = round(float(np.mean([brier1(p, t) for p, t in zip(P_fwd, yt_fwd)])), 3)
            a = auroc_ovr(P_fwd, yt_fwd, opts)
            res["forward_auroc"] = None if a is None else round(a, 3)
        res["n_inputs"] = int(len(J.JudgeRegression(rc, {q: variants_of[q]}, {q: options_of[q]},
                                                   variant_specs=cfg.get("variants") or {}).x_row(q, rows[lab[0]]["parsed"][q])))
        out[q] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/judges_eval/raw_nemotron.json")
    ap.add_argument("--campaign", default=str(ROOT / "result/result_8"))
    ap.add_argument("--out", default="/tmp/jreg_eval.json")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--questions", nargs="*", default=["J1", "J3", "J4", "J7", "J2", "J5", "J6"])
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "pipeline/05b_judges/judges.yaml").read_text())
    bank_l = J.load_bank()
    rows, bank = load_rows(Path(a.raw), Path(a.campaign), cfg, bank_l)
    rep = {"raw": a.raw, "n_states": len(rows), "variants": {}}
    for v in a.variants:
        rep["variants"][v] = evaluate(rows, bank, cfg, a.questions, VARIANTS[v])
        print(v, flush=True)
        for q, r in rep["variants"][v].items():
            print(f"  {q} n={r.get('n')} in={r.get('n_inputs')} LOO {r.get('loo_accuracy')} / Brier {r.get('loo_brier')} "
                  f"/ AUROC {r.get('loo_auroc')} (lookup {r.get('fault_mode_lookup_loo')})  "
                  f"LOMO {r.get('lomo_accuracy')} / Brier {r.get('lomo_brier')} / AUROC {r.get('lomo_auroc')} "
                  f"(maj {r.get('lomo_majority_no_judges')})  "
                  f"fwd n={r.get('forward_n')} {r.get('forward_accuracy')} / Brier {r.get('forward_brier')} "
                  f"/ AUROC {r.get('forward_auroc')}  epiAUROC {r.get('lomo_epistemic_auroc_for_errors')}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
