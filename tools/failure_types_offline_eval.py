"""Offline evaluation of the 05c failure-type stacking layer (engine/failure_types.py) on finished campaigns.

Data: switched sim_status-ok runs of result_8..12 (labels from the run records, engine.failure_types.type_labels).
Inputs: the GP's own feature vector re-built offline by run.Loop.features (route events from each campaign's saved
baselines), restricted by failure_types.yaml `inputs`; for result_8 also the J-question regression probabilities
(05b, saved Nemotron System One answers /workspace/judges_eval/raw_nemotron.json) and the two judge-uncertainty
columns. The J regression is refit inside every training fold (outcome questions J2 J5 J6 are labelled by run
outcomes, so an in-sample fit would leak the test run's outcome into its own features).

Evaluation per type: LOO and forward order (train on earlier (campaign, round) blocks only, test on the next block),
recall / precision at the cost threshold, PR-AUC, ROC-AUC, confusion counts; models logistic_balanced vs hier_bayes.

  python tools/failure_types_offline_eval.py --out /workspace/stacking_eval/failure_types.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from engine import bayes_logistic as BL  # noqa: E402
from engine import failure_types as FT  # noqa: E402
from engine import judges as JDG  # noqa: E402

CAMPAIGNS = ["result_8", "result_9", "result_10", "result_11", "result_12"]


def make_loop():
    import run as RUN
    loop = RUN.Loop(mode="mock", out=Path(tempfile.mkdtemp()) / "o", ablation="X-5", video_on=False, seed=0,
                    cold_start=12, batch_size=6, rounds=0, quiet=True, gp_target="safety_margin", judge_b="nemotron")
    loop.flags["use_b"] = False
    return loop


def campaign_rows(loop, camp: Path, ci: int, cfg: dict) -> list[dict]:
    from engine import scenarios as SC
    from engine.monitor import load_av_stack_run
    bases = {}
    for comp in (camp / "runs" / "baselines").glob("*/.complete"):
        k = json.loads(comp.read_text())
        bases[(k["route_id"], float(k["max_speed_kmh"]), float(k.get("tire_friction_scale") or 1.0))] = comp.parent
    cache = {}

    def rev(c):
        env = c["environment"]
        d = bases.get((c["route_id"], float(env["max_speed_kmh"]), float(env.get("tire_friction_scale") or 1.0)))
        if d is None:
            return None
        if d not in cache:
            run = load_av_stack_run(d)
            cache[d] = SC.detect_route_events(run.ticks, run.dt, **(loop.cfg["paired_runs"].get("event_detection") or {}))
        return cache[d]

    loop.route_events = rev
    rows = []
    for ln in (camp / "run_records.jsonl").read_text().splitlines():
        r = json.loads(ln)
        lab = FT.type_labels(r, cfg)
        if lab is None:
            continue
        p = r["parameters"]
        c = {"scenario_id": r["scenario_id"], "route_id": p["route_id"], "environment": p["environment"],
             "fault": p["fault"], "onset_event": p["onset_event"], "onset_offset_s": p["onset_offset_s"]}
        try:
            x, names = loop.features(c)
        except (KeyError, ValueError) as e:
            print(f"skip {camp.name} {r['run_id']}: {e}")
            continue
        rows.append({"campaign": camp.name, "ci": ci, "round": int(r["round"]), "run_id": r["run_id"], "x": x,
                     "names": names, "labels": lab, "mode": p["fault"]["mode"], "rec": r})
    return rows


# ---------------------------------------------------------------- judge features (result_8 only)
class JudgeFeatures:
    def __init__(self, raw_path: Path, camp: Path):
        from judges_offline_eval import run_states
        self.jc = yaml.safe_load((ROOT / "pipeline/05b_judges/judges.yaml").read_text())
        self.bank_l = JDG.load_bank()
        self.bank = {q["id"]: q for q in self.bank_l}
        _, meta = JDG.judge_questions(self.bank_l, self.jc, None, context="full")
        states = run_states(camp, "tight")
        raw = json.loads(raw_path.read_text())
        self.rows = {}
        for k in sorted(raw):
            if k in states:
                ctx, rec = states[k]
                self.rows[k] = (JDG.parse_answers(raw[k]["response"], meta), JDG.labels_for(self.bank_l, ctx, rec))
        self.variants_of = {q: list(v) for q, v in self.jc["judges"].items()}
        self.options_of = {q: [str(o) for o in self.bank[q]["answer"]] for q in self.variants_of}
        self.report_only = {q for q in self.variants_of if self.bank[q].get("regress") is False}

    def fit(self, train_ids: list[str]) -> "JDG.JudgeRegression":
        reg = JDG.JudgeRegression(self.jc.get("regression") or {}, self.variants_of, self.options_of, seed=0,
                                  report_only=self.report_only, variant_specs=self.jc.get("variants") or {})
        return reg.fit([self.rows[k] for k in train_ids if k in self.rows], cv=False)

    def features(self, reg, run_id: str) -> tuple[list[float], list[str]] | None:
        if run_id not in self.rows:
            return None
        parsed, _ = self.rows[run_id]
        ans = JDG.aggregate(parsed, reg, {}, 0.5)
        x, names = JDG.jev_features(ans, self.bank_l)
        u, un = FT.judge_uncertainty(parsed, self.jc.get("variants") or {},
                                     [q for q in self.variants_of if self.bank[q].get("gp_feature", True)])
        return x + u, names + un


# ---------------------------------------------------------------- evaluation
def matrix(rows, inputs, jf=None, reg=None):
    idx = FT.select_columns(rows[0]["names"], inputs)
    X = []
    for r in rows:
        x = [r["x"][i] for i in idx]
        if jf is not None:
            f = jf.features(reg, r["run_id"])
            xs = f[0] if f else []
            if "judges" not in inputs:
                xs = xs[-2:]
            if "judge_uncertainty" not in inputs:
                xs = xs[:-2]
            x = x + xs
        X.append(x)
    return np.array(X, dtype=float)


def Ymat(rows):
    return np.array([[np.nan if r["labels"][t] is None else float(r["labels"][t]) for t in FT.TYPES] for r in rows])


def run_split(rows, train, test, inputs, method, cfg, jf=None):
    reg = jf.fit([rows[i]["run_id"] for i in train]) if jf is not None else None
    Xtr = matrix([rows[i] for i in train], inputs, jf, reg)
    Xte = matrix([rows[i] for i in test], inputs, jf, reg)
    m = FT.FailureTypeModel(cfg, method=method).fit(Xtr, Ymat([rows[i] for i in train]))
    return m.predict(Xte)


def scores(rows, preds: dict[int, np.ndarray], cfg) -> dict:
    th = FT.cost_thresholds(cfg)
    out = {}
    ids = sorted(preds)
    P = np.array([preds[i] for i in ids])
    Y = Ymat([rows[i] for i in ids])
    for t, name in enumerate(FT.TYPES):
        m = ~np.isnan(Y[:, t])
        y = Y[m, t].astype(bool)
        p = P[m, t]
        d = p > th[name]
        tp, fp = int((d & y).sum()), int((d & ~y).sum())
        fn, tn = int((~d & y).sum()), int((~d & ~y).sum())
        ap = BL.pr_auc(p, y)
        out[name] = {"n": int(m.sum()), "positives": int(y.sum()), "threshold": round(th[name], 3),
                     "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                     "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                     "pr_auc": None if ap is None else round(ap, 3),
                     "pr_auc_chance": round(float(y.mean()), 3) if len(y) else None,
                     "roc_auc": None if BL.roc_auc(p, y) is None else round(BL.roc_auc(p, y), 3),
                     "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                     "flag_rate": round(float(d.mean()), 3)}
    return out


def evaluate(rows, inputs, method, cfg, jf=None, min_train: int = 20) -> dict:
    n = len(rows)
    loo = {}
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        loo[i] = run_split(rows, tr, [i], inputs, method, cfg, jf)["probs"][0]
    # forward order: blocks of (campaign index, round); train on every earlier block
    order = sorted({(r["ci"], r["round"]) for r in rows})
    fwd = {}
    for b in order:
        te = [i for i, r in enumerate(rows) if (r["ci"], r["round"]) == b]
        tr = [i for i, r in enumerate(rows) if (r["ci"], r["round"]) < b]
        if len(tr) < min_train:
            continue
        P = run_split(rows, tr, te, inputs, method, cfg, jf)["probs"]
        for i, p in zip(te, P):
            fwd[i] = p
    return {"loo": scores(rows, loo, cfg), "forward": scores(rows, fwd, cfg), "n_forward_test": len(fwd)}


def ordinal_eval(rows, inputs, cfg) -> dict:
    order = list((cfg.get("ordinal") or {}).get("order"))
    lv = np.array([FT.severity_level(r["rec"], r["labels"], order) for r in rows])
    X = matrix(rows, inputs)
    pa, pd = {}, {}
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        if len(set(lv[tr].tolist())) < 2:
            continue
        m = FT.OrdinalSeverityModel(cfg).fit(X[tr], lv[tr])
        p = m.predict(X[[i]])["probs"][0]
        pa[i], pd[i] = 1 - p[0], p[order.index("departed_recovered"):].sum()
    ids = sorted(pa)
    any_f = lv[ids] > 0
    dep = lv[ids] >= order.index("departed_recovered")
    return {"order (TEAM DECISION)": order, "level_counts": {o: int((lv == k).sum()) for k, o in enumerate(order)},
            "loo_pr_auc_any_failure": BL.pr_auc([pa[i] for i in ids], any_f),
            "any_failure_chance": round(float(any_f.mean()), 3),
            "loo_pr_auc_departed_or_worse": BL.pr_auc([pd[i] for i in ids], dep),
            "departed_chance": round(float(dep.mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/stacking_eval/failure_types.json")
    ap.add_argument("--raw", default="/workspace/judges_eval/raw_nemotron.json")
    ap.add_argument("--sr4", action="store_true", help="lane_departure also from SR4 footprint departure (pre-switch too)")
    a = ap.parse_args()
    cfg = FT.load_config()
    if a.sr4:
        cfg["types"]["lane_departure"]["include_sr4_footprint"] = True
    loop = make_loop()
    rows = []
    for ci, c in enumerate(CAMPAIGNS):
        rows += campaign_rows(loop, ROOT / "result" / c, ci, cfg)
    rep = {"label_counts_pooled": FT.label_counts([r["labels"] for r in rows]),
           "label_counts_by_campaign": {c: {t: v["positive"] for t, v in FT.label_counts([r["labels"] for r in rows if r["campaign"] == c]).items()}
                                        for c in CAMPAIGNS},
           "n_switched_labelled": len(rows), "include_sr4_footprint": a.sr4, "cost_thresholds": FT.cost_thresholds(cfg),
           "experiments": {}}
    print(json.dumps(rep["label_counts_pooled"]), flush=True)
    methods = ["logistic_balanced", "hier_bayes"]
    exps = [("pooled_r8_r12", rows, ["df"], None), ("pooled_r8_r12", rows, ["df", "scenario"], None)]
    r8 = [r for r in rows if r["campaign"] == "result_8"]
    jf = JudgeFeatures(Path(a.raw), ROOT / "result" / "result_8")
    exps += [("result_8", r8, ["df"], None), ("result_8", r8, ["df", "judges", "judge_uncertainty"], jf),
             ("result_8", r8, ["df", "judge_uncertainty"], jf)]
    for name, rs, inputs, j in exps:
        for meth in methods:
            key = f"{name} | {'+'.join(inputs)} | {meth}"
            res = evaluate(rs, inputs, meth, cfg, j)
            rep["experiments"][key] = res
            print(key, flush=True)
            for ev in ("loo", "forward"):
                print(f"  {ev}: " + "  ".join(f"{t} pos {d['positives']} rec {d['recall']} prAUC {d['pr_auc']} "
                                              f"(chance {d['pr_auc_chance']}) cm {d['confusion']}"
                                              for t, d in res[ev].items()), flush=True)
            Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    rep["ordinal_pooled_df"] = ordinal_eval(rows, ["df"], cfg)
    print("ordinal", json.dumps(rep["ordinal_pooled_df"]), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
