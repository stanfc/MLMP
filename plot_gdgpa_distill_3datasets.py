"""GDG-PA (base) vs GDG-PA + EMA teacher-student distillation, on the 3 datasets,
with No-Adapt and MLMP-episodic reference lines. One shared legend.
Output: figures/gdgpa_distill_3datasets.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 15, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14})


def miou(p):
    f = f"save/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


C_BASE = "#1f77b4"    # GDG-PA base
C_DIST = "#d62728"    # GDG-PA + EMA distill

PANELS = [
    dict(title="ACDC", base="ACDCDataset/deyo_mlmp_hmgate2_continual",
         dist="ACDCDataset/deyo_mlmp_hmgate2_distill_continual_lr3e-5",
         epi=29.84, noadapt=23.34, ylim=(0, 37)),
    dict(title="Cityscapes", base="CityscapesDataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
         dist="CityscapesDataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr3e-5",
         epi=22.71, noadapt=20.56, ylim=(0, 28)),
    dict(title="VOC20", base="PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
         dist="PascalVOC20Dataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr3e-5",
         epi=75.35, noadapt=68.60, ylim=(55, 84)),
]

fig, axes = plt.subplots(1, 3, figsize=(22, 7.4))
for ax, P in zip(axes, PANELS):
    ax.axhline(P["epi"], color="#2ca02c", ls="--", lw=2.4, zorder=3)
    ax.axhline(P["noadapt"], color="k", ls=":", lw=2.2, zorder=3)
    mb, md = miou(P["base"]), miou(P["dist"])
    ax.plot(np.arange(1, len(mb) + 1), mb, color=C_BASE, lw=3.0, zorder=5)
    ax.plot(np.arange(1, len(md) + 1), md, color=C_DIST, lw=3.6, zorder=6)
    ax.set_xlim(1, 150); ax.set_ylim(*P["ylim"])
    ax.set_title(P["title"], fontsize=18)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.25)
    print(f"{P['title']:11s} base last={mb[-1]:.1f} | distill peak={md.max():.1f} last={md[-1]:.1f}")

axes[0].set_ylabel("mIoU (%)")

handles = [
    Line2D([], [], color="#2ca02c", ls="--", lw=2.4),
    Line2D([], [], color="k", ls=":", lw=2.2),
    Line2D([], [], color=C_BASE, lw=3.0),
    Line2D([], [], color=C_DIST, lw=3.6),
]
labels = [
    "MLMP-episodic (per-sample reset)",
    "No adaptation (source)",
    "GDG-PA (ours)",
    "GDG-PA + EMA distillation (ours)",
]
fig.legend(handles, labels, fontsize=15, loc="lower center",
           bbox_to_anchor=(0.5, -0.04), ncol=4, framealpha=0.95)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/gdgpa_distill_3datasets.png", dpi=135, bbox_inches="tight")
print("saved -> figures/gdgpa_distill_3datasets.png")
