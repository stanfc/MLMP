"""LR sweep — no-gate monitor on ViT/ImageNet-C (i.i.d. shuffled), truncated to a
common round count. Shows lr=0.001 is the cause of the "collapse"; smaller lrs are stable.
Output: figures/cls_lrsweep.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 24, "axes.labelsize": 30, "axes.titlesize": 30,
                     "xtick.labelsize": 24, "ytick.labelsize": 24,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

ROOT = "/home/stanfc/Desktop/TTA-on-OVSS/DeYO/output"
RUNS = [
    ("lr = 1e-3  (default)", "cls_lrsweep_lr0.001",   "#d62728", 4.2),
    ("lr = 2.5e-4",          "cls_lrsweep_lr0.00025", "#ff7f0e", 3.4),
    ("lr = 1e-4",            "cls_lrsweep_lr0.0001",  "#1f77b4", 3.4),
    ("lr = 5e-5",            "cls_lrsweep_lr0.00005", "#2ca02c", 3.4),
]


def acc(d):
    f = f"{ROOT}/{d}/results_all_rounds.csv"
    return np.array([float(l.split(",")[-1]) for l in open(f).readlines()[1:] if l.strip()]) \
        if os.path.exists(f) else np.array([])


RMAX = min(len(acc(d)) for _, d, _, _ in RUNS)   # common round count
fig, ax = plt.subplots(figsize=(13.5, 8.0))
for lab, d, c, lw in RUNS:
    a = acc(d)[:RMAX]
    ax.plot(np.arange(1, len(a) + 1), a, color=c, lw=lw, marker="o", ms=8)
    print(f"  {lab:22s} R1={a[0]:5.1f}  R{len(a)}={a[-1]:5.1f}")

ax.set_xlim(1, RMAX); ax.set_ylim(0, 65)
ax.set_xlabel("Round   (1 round = 15 ImageNet-C corruptions, no reset)")
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title(f"LR sweep — no-gate, ViT-B/16 ImageNet-C (i.i.d.), to R{RMAX}", fontsize=27, pad=12)
ax.grid(alpha=0.22, lw=0.9)
ax.tick_params(width=1.6, length=8)

h = [Line2D([], [], color=c, lw=lw, marker="o", ms=8) for _, _, c, lw in RUNS]
ax.legend(h, [lab for lab, _, _, _ in RUNS], fontsize=21, loc="center left", framealpha=0.95)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/cls_lrsweep.png", dpi=135, bbox_inches="tight")
print(f"saved -> figures/cls_lrsweep.png  (common R{RMAX})")
