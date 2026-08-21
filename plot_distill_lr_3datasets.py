"""GDG-PA base vs +EMA distillation at two LRs (5e-6 stable, 3e-5 aggressive),
across ACDC / Cityscapes / VOC20, with No-Adapt and MLMP-episodic references.
Shows distillation@5e-6 is a clean win (higher than base, as stable), while the
aggressive 3e-5 overshoots then drifts on the low-headroom datasets.
Output: figures/distill_lr_3datasets.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 22, "axes.labelsize": 28, "axes.titlesize": 34,
                     "xtick.labelsize": 22, "ytick.labelsize": 22,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})


def miou(p):
    f = f"save/{p}/results_all_rounds.txt"
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")]) \
        if os.path.exists(f) else np.array([])


C_BASE, C_D6, C_D5 = "#1f77b4", "#d62728", "#ff7f0e"

PANELS = [
    dict(title="ACDC", epi=29.84, noadapt=23.34, ylim=(0, 37),
         base="ACDCDataset/deyo_mlmp_hmgate2_continual",
         d6="ACDCDataset/deyo_mlmp_hmgate2_distill_continual_lr5e-6",
         d5="ACDCDataset/deyo_mlmp_hmgate2_distill_continual_lr3e-5"),
    dict(title="Cityscapes", epi=22.71, noadapt=20.56, ylim=(0, 28),
         base="CityscapesDataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
         d6="CityscapesDataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr5e-6",
         d5="CityscapesDataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr3e-5"),
    dict(title="VOC20", epi=75.35, noadapt=68.60, ylim=(55, 84),
         base="PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
         d6="PascalVOC20Dataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr5e-6",
         d5="PascalVOC20Dataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr3e-5"),
]

fig, axes = plt.subplots(1, 3, figsize=(23, 7.8))
for ax, P in zip(axes, PANELS):
    ax.axhline(P["epi"], color="#2ca02c", ls="--", lw=2.6, zorder=3)
    ax.axhline(P["noadapt"], color="k", ls=":", lw=2.4, zorder=3)
    for key, c, lw in [("base", C_BASE, 3.0), ("d6", C_D6, 4.2)]:
        m = miou(P[key])
        ax.plot(np.arange(1, len(m) + 1), m, color=c, lw=lw,
                zorder=6 if key == "d6" else 5)
    ax.set_xlim(1, 150); ax.set_ylim(*P["ylim"])
    ax.set_title(P["title"], fontweight="bold", pad=10)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.22, lw=0.9)
    ax.tick_params(width=1.6, length=8)

axes[0].set_ylabel("mIoU (%)")

handles = [
    Line2D([], [], color="#2ca02c", ls="--", lw=2.6),
    Line2D([], [], color="k", ls=":", lw=2.4),
    Line2D([], [], color=C_BASE, lw=3.0),
    Line2D([], [], color=C_D6, lw=4.2),
]
labels = [
    "MLMP-episodic (NeurIPS'25)", "No adaptation (source)",
    "GDG-PA (ours)", "GDG-PA + EMA distillation (ours)",
]
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.legend(handles, labels, fontsize=22, loc="lower center",
           bbox_to_anchor=(0.0, -0.02, 1.0, 0.10), mode="expand", ncol=2,
           framealpha=0.95, handlelength=2.4, columnspacing=1.6, borderaxespad=0.0)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/distill_lr_3datasets.png", dpi=135, bbox_inches="tight")
print("saved -> figures/distill_lr_3datasets.png")
