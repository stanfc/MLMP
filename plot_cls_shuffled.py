"""ViT classification, long-horizon continual TTA on ImageNet-C — the two data streams
side by side:  (left) class-ordered / label-shifted,  (right) standard i.i.d. shuffled.
Shows our GDG-PA gate collapses under BOTH, while SAR stays stable.
Output: figures/cls_shuffled_vs_ordered.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 23, "axes.labelsize": 28, "axes.titlesize": 28,
                     "xtick.labelsize": 22, "ytick.labelsize": 22,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

ROOT = "/home/stanfc/Desktop/TTA-on-OVSS/DeYO/output"


def acc(d):
    f = f"{ROOT}/{d}/results_all_rounds.csv"
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f).readlines()[1:] if l.strip()])


PANELS = [
    ("Class-ordered stream (label shift)",
     [("Ours (GDG-PA gate)", "cls_continual_deyo_gate", "#d62728"),
      ("SAR (ICML'23)",       "cls_continual_sar",       "#2ca02c")]),
    ("Standard i.i.d. shuffled stream",
     [("Ours (GDG-PA gate)", "cls_shuf_deyo_gate", "#d62728"),
      ("SAR (ICML'23)",       "cls_shuf_sar",       "#2ca02c")]),
]

fig, axes = plt.subplots(1, 2, figsize=(20, 7.6), sharey=True)
for ax, (title, runs) in zip(axes, PANELS):
    for lab, d, c in runs:
        a = acc(d)
        if len(a):
            ax.plot(np.arange(1, len(a) + 1), a, color=c, lw=4.2, marker="o", ms=9)
            print(f"  {title[:12]:12s} {lab:20s} R1={a[0]:5.1f} last={a[-1]:5.1f}")
    ax.set_xlim(1, 10); ax.set_ylim(0, 65)
    ax.set_xlabel("Round")
    ax.set_title(title, fontsize=25, pad=12)
    ax.grid(alpha=0.22, lw=0.9)
    ax.tick_params(width=1.6, length=8)
axes[0].set_ylabel("Top-1 accuracy (%)")

h = [Line2D([], [], color="#d62728", lw=4.2, marker="o", ms=9),
     Line2D([], [], color="#2ca02c", lw=4.2, marker="o", ms=9)]
fig.tight_layout(rect=[0, 0.09, 1, 1])
fig.legend(h, ["Ours (GDG-PA gate)", "SAR (ICML'23)"], fontsize=23, loc="lower center",
           bbox_to_anchor=(0.5, -0.02), ncol=2, framealpha=0.95, handlelength=2.4, columnspacing=2.5)
fig.suptitle("ViT-B/16 on ImageNet-C — our gate collapses under both streams; SAR stays stable",
             fontsize=25, y=1.02)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/cls_shuffled_vs_ordered.png", dpi=135, bbox_inches="tight")
print("saved -> figures/cls_shuffled_vs_ordered.png")
