"""Offline check of the 05b judges (J1..J9, pipeline/05b_judges/) on a finished campaign (default result_8).

One System One request per finished run (its scenario inputs only: fault, onset, speed, road, friction, supervisor
profile; never outcomes). Labels: code questions (J1 J3 J4 J7 J8 J9) from the profile-aware answer key, outcome
questions (J2 J5 J6) from that run's record (engine.judges.outcome_label). Report per question: label class balance,
raw per-judge accuracy, majority / mean of judges, the regression's out-of-fold CV, a judge-free fault-mode lookup
baseline (leave-one-out) and leave-one-fault-mode-out (regression vs the training majority).

  python tools/judges_offline_eval.py --backend nemotron --out /tmp/judges_eval            # Token Factory, parallel
  python tools/judges_offline_eval.py --backend laya --limit 15 --out /tmp/judges_eval      # laya-serve sanity check
Raw answers are cached in <out>/raw_<backend>.json (re-runs only ask what is missing). Keys are never printed."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import judges as J  # noqa: E402
from engine import nemotron as NM  # noqa: E402
from engine.fixtures.synthetic_sim import SUPERVISOR_PROFILES  # noqa: E402

AV = ROOT / "pipeline/08_carla_paired_run/av_stack"


def run_states(camp: Path, sup: str) -> dict[str, tuple[dict, dict]]:
    """{run_id: (judge ctx, record)} for every run with sim_status ok."""
    out = {}
    for ln in (camp / "run_records.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("sim_status") != "ok":
            continue
        p = r.get("parameters") or {}
        f, env = p.get("fault") or {}, p.get("environment") or {}
        ctx = {"fault_mode": f.get("mode"), "fault_params": f.get("params") or {}, "onset_event": p.get("onset_event"),
               "onset_offset_s": p.get("onset_offset_s"), "route_id": p.get("route_id"),
               "max_speed_kmh": env.get("max_speed_kmh"), "tire_friction_scale": env.get("tire_friction_scale", 1.0),
               "planner": "pdm", "controller": "supervised", "supervisor": r.get("supervisor_profile") or sup}
        out[r["run_id"]] = (ctx, r)
    return out


def pick(states: dict, limit: int | None) -> list[str]:
    """Up to `limit` runs, round-robin over fault modes (so a small sanity check still covers every mode)."""
    ids = sorted(states)
    if not limit or limit >= len(ids):
        return ids
    by = {}
    for k in ids:
        by.setdefault(states[k][0]["fault_mode"], []).append(k)
    out, i = [], 0
    while len(out) < limit:
        for m in sorted(by):
            if i < len(by[m]) and len(out) < limit:
                out.append(by[m][i])
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=str(ROOT / "result/result_8"))
    ap.add_argument("--out", default="/tmp/judges_eval")
    ap.add_argument("--backend", default="nemotron", choices=J.BACKENDS)
    ap.add_argument("--limit", type=int, help="ask only this many runs (round-robin over fault modes)")
    ap.add_argument("--parallel", type=int, help="concurrent requests (default: backend 'parallel' or 1)")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    camp, out = Path(a.campaign), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "pipeline/05b_judges/judges.yaml").read_text())
    bank_l = J.load_bank()
    bank = {q["id"]: q for q in bank_l}
    rc = camp / "run_config.json"
    sup = (json.loads(rc.read_text()).get("supervisor") if rc.exists() else None) or "tight"
    qs, meta = J.judge_questions(bank_l, cfg, J.state_views(cfg, a.backend))
    states = run_states(camp, sup)
    todo = pick(states, a.limit)
    raw_path = out / f"raw_{a.backend}.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    if not a.analyse_only:
        bcfg = cfg["backends"][a.backend]
        if a.backend == "nemotron":
            llm = yaml.safe_load((ROOT / "pipeline/llm_models.yaml").read_text())
            cli = J.NemotronSystemOne(bcfg, llm, out / "calls.jsonl")
        else:
            cli = J.SystemOneClient(bcfg, out / "calls.jsonl", backend=a.backend)
        if not cli.reachable():
            raise SystemExit(f"{a.backend} not reachable / no key")
        need = [k for k in todo if k not in raw]

        def ask(k):
            ctx = states[k][0]
            state = J.make_state(cfg, a.backend, AV, ctx, SUPERVISOR_PROFILES.get(ctx["supervisor"]))
            t0 = time.time()
            resp = cli.system_one(state, J.questions_for(qs, meta, bank_l, ctx["fault_mode"]), tag=f"offline:{k}")
            return k, {"context": ctx, "response": resp, "state_chars": len(json.dumps(state)),
                       "wall_s": round(time.time() - t0, 2)}

        par = int(a.parallel or bcfg.get("parallel") or 1)
        with ThreadPoolExecutor(max_workers=par) as ex:
            futs = [ex.submit(ask, k) for k in need]
            for i, f in enumerate(as_completed(futs)):
                try:
                    k, v = f.result()
                except Exception as e:  # noqa: BLE001 - one failed request must not lose the others
                    print(f"[{i + 1}/{len(need)}] error {type(e).__name__}: {str(e)[:160]}", flush=True)
                    continue
                raw[k] = v
                raw_path.write_text(json.dumps(raw))
                u = v["response"].get("usage") or {}
                print(f"[{i + 1}/{len(need)}] {k} {v['wall_s']} s in={u.get('input_tokens')} answers={len(v['response'].get('answers') or {})}",
                      flush=True)
    sel = {k: raw[k] for k in todo if k in raw}
    rep = analyse(sel, states, bank, meta, cfg)
    walls = [r["wall_s"] for r in sel.values()]
    ins = [int((r["response"].get("usage") or {}).get("input_tokens") or 0) for r in sel.values()]
    rep["timing"] = {"requests": len(sel), "wall_s_median": float(np.median(walls)) if walls else None,
                     "input_tokens_median": float(np.median(ins)) if ins else None,
                     "input_tokens_total": int(sum(ins)),
                     "truncated_any": any((r["response"].get("usage") or {}).get("truncated") for r in sel.values())}
    (out / f"report_{a.backend}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(table(rep))


def analyse(raw: dict, states: dict, bank: dict, meta: dict, cfg: dict) -> dict:
    variants_of = {q: list(v) for q, v in cfg["judges"].items()}
    options_of = {q: [str(o) for o in bank[q]["answer"]] for q in variants_of}
    report_only = {q for q in variants_of if bank[q].get("regress") is False}
    rows = []
    for k, r in raw.items():
        ctx, rec = states[k]
        rows.append((k, J.parse_answers(r["response"], meta), J.labels_for(list(bank.values()), ctx, rec), ctx["fault_mode"]))
    reg = J.JudgeRegression(cfg.get("regression") or {}, variants_of, options_of, seed=0,
                            report_only=report_only).fit([(p, kk) for _, p, kk, _ in rows])
    per_q = {}
    for q in variants_of:
        hits, n, maj, mean_hit, answers, labels = Counter(), 0, 0, 0, Counter(), Counter()
        for _, p, kk, _ in rows:
            exp, js = kk.get(q), p.get(q) or []
            if exp is None or not js:
                continue
            n += 1
            labels["|".join(exp)] += 1
            picks = [max(j["probs"], key=j["probs"].get) for j in js]
            for j, pk in zip(js, picks):
                hits[j["judge"]] += pk in exp
            answers.update(picks)
            maj += Counter(picks).most_common(1)[0][0] in exp
            mp = J.mean_probs(js, options_of[q])
            mean_hit += max(mp, key=mp.get) in exp
        per_q[q] = {"label_source": bank[q].get("label_source"), "n": n, "label_balance": dict(labels.most_common()),
                    "raw_per_judge": {v: round(hits[v] / n, 3) for v in variants_of[q]} if n else None,
                    "raw_judge_mean_acc": round(sum(hits.values()) / (n * len(variants_of[q])), 3) if n else None,
                    "majority_vote": round(maj / n, 3) if n else None, "mean_prob": round(mean_hit / n, 3) if n else None,
                    "judge_answer_counts": dict(answers.most_common(6)),
                    "regression_cv": (reg.report or {}).get(q),
                    "loo_checks": None if q in report_only else _loo_checks(rows, q, variants_of[q], options_of[q],
                                                                             cfg.get("regression") or {})}
    return {"n_states": len(rows), "per_question": per_q}


def _loo_checks(rows, q, variants, opts, rc) -> dict | None:
    """Is the regression using the judges, or only recognising the fault mode? Leave-one-out accuracy of (a) the
    regression, (b) the judge-free lookup 'majority label of the other states with the same fault mode', (c) the
    regression with the whole fault mode held out next to the training majority label."""
    from sklearn.linear_model import LogisticRegression
    lab = [(p[q], kk[q][0], m) for _, p, kk, m in rows if p.get(q) and kk.get(q) and len(kk[q]) == 1]
    if len(lab) < 4 or len({t for _, t, _ in lab}) < 2:
        return None
    X = np.array([J._x_row(js, variants, opts, float(rc.get("prob_clip", 1e-3))) for js, _, _ in lab])
    y = np.array([t for _, t, _ in lab])
    modes = np.array([str(m) for _, _, m in lab])

    def fit_pred(tr, te):
        if len(set(y[tr].tolist())) < 2:
            return [Counter(y[tr].tolist()).most_common(1)[0][0]] * len(te)
        return LogisticRegression(C=float(rc.get("C", 0.5)), max_iter=2000).fit(X[tr], y[tr]).predict(X[te]).tolist()

    idx = np.arange(len(y))
    reg_loo = np.mean([fit_pred(idx[idx != i], [i])[0] == y[i] for i in idx])
    look = []
    for i in idx:
        pool = [y[j] for j in idx if j != i and modes[j] == modes[i]] or [y[j] for j in idx if j != i]
        look.append(Counter(pool).most_common(1)[0][0] == y[i])
    lr, lm = [], []
    for m in sorted(set(modes.tolist())):
        te, tr = idx[modes == m], idx[modes != m]
        if len(tr):
            lr += [a == b for a, b in zip(fit_pred(tr, te), y[te])]
            mj = Counter(y[tr].tolist()).most_common(1)[0][0]
            lm += [mj == b for b in y[te]]
    return {"n": int(len(y)), "regression_loo": round(float(reg_loo), 3),
            "fault_mode_lookup_loo_no_judges": round(float(np.mean(look)), 3),
            "regression_leave_one_mode_out": round(float(np.mean(lr)), 3) if lr else None,
            "majority_leave_one_mode_out_no_judges": round(float(np.mean(lm)), 3) if lm else None}


def table(rep: dict) -> str:
    L = [f"states {rep['n_states']}  timing {json.dumps(rep.get('timing'))}",
         "q   src      n  labels                                   judge  maj   regCV  lookup  LOMO reg/maj"]
    for q, d in rep["per_question"].items():
        cv = ((d.get("regression_cv") or {}).get("regression") or {}).get("accuracy")
        lo = d.get("loo_checks") or {}
        L.append(f"{q:3} {d['label_source']:7} {d['n']:3}  {json.dumps(d['label_balance'])[:40]:40} "
                 f"{d['raw_judge_mean_acc']}  {d['majority_vote']}  {cv}  {lo.get('fault_mode_lookup_loo_no_judges')}  "
                 f"{lo.get('regression_leave_one_mode_out')}/{lo.get('majority_leave_one_mode_out_no_judges')}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
