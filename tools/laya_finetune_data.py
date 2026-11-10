"""Export Laya fine-tuning rows (laya-train JSONL {state, questions, gold}) for the 05b code questions.

Soft targets from the code answer key (questions.yaml expected_from_code via engine.nemotron.expected_answers,
supervisor-profile aware): uniform over the allowed set for list answers. Only code questions J1 J3 J4 J7 J8 J9 (the
outcome questions J2 J5 J6 never beat the fault-mode lookup and are left out). States are the judge states the
backend would see (judges.yaml context, default digest for Laya), built from scenario INPUTS only (no outcome).
`--holdout-mode` writes the held-out fault mode to eval.jsonl so the fine-tune is checked on a mode it never saw.

  python tools/laya_finetune_data.py --out /workspace/laya_ft --holdout-mode drift
  laya-train --data /workspace/laya_ft/train.jsonl --eval /workspace/laya_ft/eval.jsonl --base <multilingual dir> \
      --out /workspace/laya_ft/ckpt --loss soft-ce --shuffle-options --freeze-encoder --epochs 2 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from engine import judges as J  # noqa: E402
from engine.fixtures.synthetic_sim import SUPERVISOR_PROFILES  # noqa: E402
from judges_offline_eval import AV, run_states  # noqa: E402

CODE_QS = ("J1", "J3", "J4", "J7", "J8", "J9")


def rows_for(ctx: dict, cfg: dict, bank_l: list[dict], qs: dict, meta: dict, context: str) -> list[dict]:
    key = J.labels_for(bank_l, ctx, None)
    asked = {n: q for n, q in J.questions_for(qs, meta, bank_l, ctx["fault_mode"]).items()
             if meta[n]["qid"] in CODE_QS and key.get(meta[n]["qid"])}
    out = []
    for state, rq in J.make_requests(cfg, "laya", AV, ctx, SUPERVISOR_PROFILES.get(ctx["supervisor"]), asked, meta,
                                     context, False):
        gold = {}
        for name, jq in rq.items():
            m = meta[name]
            allowed = key[m["qid"]]
            if jq["type"] == "noul":
                p_yes = sum(1.0 for a in allowed if a == "yes") / len(allowed)
                gold[name] = {"probabilities": {"true": p_yes, "false": 1.0 - p_yes}}
            else:
                wire_of = {o: w for w, o in (m.get("keys") or {}).items()}
                gold[name] = {"probabilities": {wire_of.get(o, o): (1.0 / len(allowed) if o in allowed else 0.0)
                                                for o in m["options"]}}
        out.append({"state": state, "questions": rq, "gold": gold, "fault_mode": ctx["fault_mode"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=str(ROOT / "result/result_8"))
    ap.add_argument("--out", default="/workspace/laya_ft")
    ap.add_argument("--context", default="digest", choices=J.CONTEXTS)
    ap.add_argument("--holdout-mode", help="fault mode written to eval.jsonl only (leave-one-fault-mode-out check)")
    ap.add_argument("--limit", type=int, help="at most this many states")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "pipeline/05b_judges/judges.yaml").read_text())
    bank_l = J.load_bank()
    qs, meta = J.judge_questions(bank_l, cfg, J.state_views(cfg, "laya"), context=a.context)
    states = run_states(Path(a.campaign), "tight")
    ids = sorted(states)[: a.limit] if a.limit else sorted(states)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "eval": 0}
    with (out / "train.jsonl").open("w") as tr, (out / "eval.jsonl").open("w") as ev:
        for k in ids:
            ctx = states[k][0]           # inputs only; the record is never read here
            for row in rows_for(ctx, cfg, bank_l, qs, meta, a.context):
                fm = row.pop("fault_mode")
                split = "eval" if a.holdout_mode and fm == a.holdout_mode else "train"
                (ev if split == "eval" else tr).write(json.dumps(row, ensure_ascii=False) + "\n")
                n[split] += 1
    print(json.dumps({"rows": n, "states": len(ids), "context": a.context, "holdout_mode": a.holdout_mode}))


if __name__ == "__main__":
    main()
