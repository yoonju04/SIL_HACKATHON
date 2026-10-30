"""Draw docs/images/system_pipeline.png (structure of run.py stages 00-10 on inter_final).

python docs/images/draw_pipeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

LLM, CODE, SIM, DATA = "#d9f2d0", "#e4dcf5", "#d6e8f7", "#fbe7c6"
fig, ax = plt.subplots(figsize=(16, 9.6), dpi=150)
ax.set_xlim(0, 160); ax.set_ylim(0, 96); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=color, ec="#444444", lw=1.3))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=8.6, linespacing=1.35)
    return (x, y, w, h)


def arrow(p, q, text="", color="#333333", style="-|>", rad=0.0, ls="-", tx=0, ty=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, color=color, lw=1.6,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=8.2, color=color,
                ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))


ax.text(80, 94, "AI-driven fault injection loop for MPC → LQR switching safety (run.py)",
        ha="center", va="top", fontsize=15, fontweight="bold")

# row 1: inputs and proposal
box(2, 64, 27, 22, "00–02  Inputs", "Prior knowledge (physics)\nCode-verified safety profile\nof the system under test\nHypotheses + search objective", DATA)
box(35, 64, 28, 22, "03  Nemotron A", "Nemotron 3.5 Lightning\nproposes fault scenarios\n(route, speed, friction,\nfault type / size, onset)\nfrom GP summary + past outcomes", LLM)
box(69, 64, 28, 22, "04  Feasibility gate", "Scenario space, route events,\nevaluation window (G-01..03)\n+ space-filling, perturbed and\ncounterfactual candidates", CODE)
box(103, 64, 26, 22, "05a  Deterministic\nfeatures", "\n\nDF-01..05 computed\nfrom scenario + code", CODE)
box(133, 64, 25, 22, "05b  Nemotron B", "Nemotron 3 Ultra answers\ncode questions about the\ncontroller (cited lines,\nchecked vs. answer key)", LLM)

# row 2: model and selection
box(118, 34, 40, 22, "06  GP surrogate (sklearn)", "Safety margin regression\n(lane / settle / lat. accel. / jerk)\n+ switch-probability classifier\nMatern 5/2 ARD", CODE)
box(72, 34, 38, 22, "07  Orchestrator", "Level-set (straddle) acquisition\nnear the safe/unsafe boundary,\nNemotron share, diversity,\ncontrol-fault cap, budget", CODE)
box(26, 34, 38, 22, "08  CARLA paired runs", "CARLA 0.9.15, Bench2Drive routes\nPDM-Lite planner + MPC / supervisor / LQR\nfault-free baseline (cached) vs fault run\nfault injected into the MPC output", SIM)

# row 3: grading and explanation
box(2, 4, 40, 22, "09  Deterministic monitor", "The ONLY judge of safety (code)\nrecovery verdict, safety margin,\nSR1–SR6 checks, F1 / F2 / F3 / pass\nsafety_spec.yaml P-01..04", CODE)
box(50, 4, 40, 22, "10  Cosmos Reason explainer", "Explains each failed run's front video\nusing monitor events as facts;\nclaims verified vs. hidden monitor\nvalues (±0.5 s), hallucination rate", LLM)
box(98, 4, 36, 22, "Campaign insights", "engine/boundaries.py\ncritical faults + boundary values\n(code table + Cosmos reading)\nreport.md · dashboard", DATA)

# main flow
arrow((29, 75), (35, 75))
arrow((63, 75), (69, 75))
arrow((97, 75), (103, 75))
arrow((97, 70), (133, 70), rad=0.0)
arrow((116, 64), (128, 56.5))
arrow((145.5, 64), (142, 56.5))
arrow((118, 45), (110, 45), "μ, σ")
arrow((72, 45), (64, 45), "batch")
arrow((36, 34), (24, 26.5), "logs + video")
arrow((42, 15), (50, 15), "failed runs")
arrow((90, 15), (98, 15))

# feedback (routed through the gaps between boxes)
def path(pts, color, text="", at=0, ls="-", tx=0, ty=0):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6, linestyle=ls)
    arrow(pts[-2], pts[-1], color=color, ls=ls)
    if text:
        x, y = pts[at]
        ax.text(x + tx, y + ty, text, fontsize=8.2, color=color, style="italic", ha="left", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))


path([(32, 26.5), (32, 30.5), (140, 30.5), (140, 34)], "#1f6fb2", "retrain GP on graded runs", at=2, tx=-40, ty=0)
path([(12, 26.5), (12, 60), (40, 60), (40, 64)], "#1f6fb2", "past outcomes", at=1, ls="--", tx=1, ty=-8)
path([(68, 26.5), (68, 60), (56, 60), (56, 64)], "#2e8b57", "accepted explanations + counterfactual re-runs",
     at=1, tx=-11, ty=1.8)

# legend
for i, (c, t) in enumerate([(LLM, "NVIDIA model (proposes / explains, never grades)"),
                            (CODE, "Deterministic code"), (SIM, "Simulation (CARLA)"), (DATA, "Inputs / outputs")]):
    ax.add_patch(FancyBboxPatch((137, 22 - i * 5), 3.5, 3, boxstyle="round,pad=0.2", fc=c, ec="#444"))
    ax.text(142, 23.5 - i * 5, t, fontsize=8.4, va="center")

out = Path(__file__).with_name("system_pipeline.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
