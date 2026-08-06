"""MGP-DeYO-MLMP across ACDC / Cityscapes / VOC20: peaks high then collapses,
contrasted with our stable GDG-PA+distill and the episodic / no-adapt references.
VOC20 shown to R130 (run still in progress).
Output: figures/mgp_3datasets.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 22, "axes.labelsize": 28, "axes.titlesize": 34,
                     "xtick.labelsize": 22, "ytick.labelsize": 22,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})


def miou(p, cap=None):
    f = f"save/{p}/results_all_rounds.txt"
    m = np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])
    return m[:cap] if cap else m


C_MGP, C_OURS = "#8c564b", "#d62728"
PANELS = [
    dict(title="ACDC", epi=29.84, noadapt=23.34, ylim=(0, 37),
         mgp=("ACDCDataset/mgp_deyo_mlmp_continual", None),
         ours="ACDCDataset/deyo_mlmp_hmgate2_distill_continual_lr5e-6"),
    dict(title="Cityscapes", epi=22.71, noadapt=20.56, ylim=(0, 28),
         mgp=("CityscapesDataset/mgp_deyo_mlmp_continual_5corr_sub100", None),
         ours="CityscapesDataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr5e-6"),
    dict(title="VOC20", epi=75.35, noadapt=68.60, ylim=(0, 82),
         mgp=("PascalVOC20Dataset/mgp_deyo_mlmp_continual_full_15corr", None),
         ours="PascalVOC20Dataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr5e-6"),
]

fig, axes = plt.subplots(1, 3, figsize=(23, 7.8))
for ax, P in zip(axes, PANELS):
    ax.axhline(P["epi"], color="#2ca02c", ls="--", lw=2.6, zorder=3)
    ax.axhline(P["noadapt"], color="k", ls=":", lw=2.4, zorder=3)
    mo = miou(P["ours"])
    ax.plot(np.arange(1, len(mo) + 1), mo, color=C_OURS, lw=4.2, zorder=6)
    mg = miou(P["mgp"][0], P["mgp"][1])
    ax.plot(np.arange(1, len(mg) + 1), mg, color=C_MGP, lw=3.4, zorder=5)
    ax.set_xlim(1, 150); ax.set_ylim(*P["ylim"])
    ax.set_title(P["title"], fontweight="bold", pad=10)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.22, lw=0.9)
    ax.tick_params(width=1.6, length=8)

axes[0].set_ylabel("mIoU (%)")

handles = [
    Line2D([], [], color="#2ca02c", ls="--", lw=2.6),
    Line2D([], [], color="k", ls=":", lw=2.4),
    Line2D([], [], color=C_OURS, lw=4.2),
    Line2D([], [], color=C_MGP, lw=3.4),
]
labels = [
    "MLMP-episodic (NeurIPS'25)", "No adaptation (source)",
    "Ours (GDG-PA + distillation)", "MGP-DeYO-MLMP",
]
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.legend(handles, labels, fontsize=22, loc="lower center",
           bbox_to_anchor=(0.0, -0.02, 1.0, 0.10), mode="expand", ncol=2,
           framealpha=0.95, handlelength=2.4, columnspacing=1.6, borderaxespad=0.0)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/mgp_3datasets.png", dpi=135, bbox_inches="tight")
print("saved -> figures/mgp_3datasets.png")
for P in PANELS:
    mg = miou(P["mgp"][0], P["mgp"][1])
    print(f"  {P['title']:11s} MGP peak={mg.max():.1f}@R{mg.argmax()+1} last={mg[-1]:.1f}")
