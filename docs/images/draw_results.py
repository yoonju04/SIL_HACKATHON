"""Draw docs/images/fault_criticality.png from result/result_12/insights/boundaries.json (no hand-typed numbers)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
modes = json.loads((root / "result/result_12/insights/boundaries.json").read_text(encoding="utf-8"))["table"]["modes"]
order = sorted(modes, key=lambda m: (-modes[m]["harmful_runs"] / max(modes[m]["runs"], 1), m))
harm = [100 * modes[m]["harmful_runs"] / modes[m]["runs"] for m in order]
late = [100 * modes[m]["late_or_missed_runs"] / modes[m]["runs"] for m in order]
fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
x = range(len(order)); w = 0.38
ax.bar([i - w / 2 for i in x], harm, w, label="unsafe (safety margin < 0)", color="#d9534f")
ax.bar([i + w / 2 for i in x], late, w, label="late / missed switch (SR F1)", color="#f0ad4e")
for i, m in enumerate(order):
    ax.text(i, max(harm[i], late[i]) + 2, f"n={modes[m]['runs']}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([m.replace("_", "\n") for m in order], fontsize=9)
ax.set_ylabel("% of graded runs"); ax.set_ylim(0, 112)
ax.set_title(f"Fault criticality, {sum(modes[m]['runs'] for m in order)} graded CARLA runs (tight supervisor)", fontsize=11)
ax.legend(fontsize=8.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
out = Path(__file__).with_name("fault_criticality.png")
fig.savefig(out, bbox_inches="tight", facecolor="white"); print(out)
