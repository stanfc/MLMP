"""ViT classification, long-horizon continual TTA on ImageNet-C (batch 64, 10 rounds).
One round = one pass through all 15 corruptions, no reset.
Output: figures/cls_continual_vit.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 24, "axes.labelsize": 30, "axes.titlesize": 32,
                     "xtick.labelsize": 24, "ytick.labelsize": 24,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

ROOT = "/home/stanfc/Desktop/TTA-on-OVSS"
RUNS = [
    ("SAR   (ICML'23)",        f"{ROOT}/DeYO/output/cls_continual_sar",       "#2ca02c", 3.4),
    ("Ours (GDG-PA gate)",     f"{ROOT}/DeYO/output/cls_continual_deyo_gate", "#d62728", 4.4),
    ("EATA  (ICML'22)",        f"{ROOT}/DeYO/output/cls_continual_eata",      "#1f77b4", 3.0),
    ("MGP   (ETA base)",       f"{ROOT}/tta-373C-MGP/output/cls_continual_mgp_eta", "#8c564b", 3.0),
    ("DeYO  (ICLR'24, no gate)", f"{ROOT}/DeYO/output/cls_continual_monitor",  "#9467bd", 3.0),
    ("TENT  (ICLR'21)",        f"{ROOT}/DeYO/output/cls_continual_tent",      "#ff7f0e", 3.0),
]


def acc(d):
    f = os.path.join(d, "results_all_rounds.csv")
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f).readlines()[1:] if l.strip()])


fig, ax = plt.subplots(figsize=(13.5, 8.2))
for lab, d, c, lw in RUNS:
    a = acc(d)
    if len(a) == 0:
        print(f"  missing {lab}"); continue
    ax.plot(np.arange(1, len(a) + 1), a, color=c, lw=lw, marker="o", ms=7)
    print(f"  {lab:26s} R1={a[0]:5.1f} last(R{len(a)})={a[-1]:5.1f}")

ax.set_xlim(1, 10); ax.set_ylim(0, 65)
ax.set_xlabel("Round   (1 round = all 15 ImageNet-C corruptions, no reset)", fontsize=25)
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("Long-horizon continual TTA — ViT-B/16 on ImageNet-C", fontsize=30, pad=14)
ax.grid(alpha=0.22, lw=0.9)
ax.tick_params(width=1.6, length=8)

h = [Line2D([], [], color=c, lw=lw) for _, _, c, lw in RUNS]
lb = [lab for lab, _, _, _ in RUNS]
fig.tight_layout(rect=[0, 0.13, 1, 1])
fig.legend(h, lb, fontsize=21, loc="lower center", bbox_to_anchor=(0.0, -0.02, 1.0, 0.12),
           mode="expand", ncol=3, framealpha=0.95, handlelength=2.2, columnspacing=1.4)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/cls_continual_vit.png", dpi=135, bbox_inches="tight")
print("saved -> figures/cls_continual_vit.png")
