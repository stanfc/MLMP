"""ACDC, 150 rounds — all existing CTTA baselines on one figure (for the talk).
Output: figures/baselines_150R_acdc.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 24, "axes.labelsize": 30, "axes.titlesize": 34,
                     "xtick.labelsize": 24, "ytick.labelsize": 24,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})


def miou(p, root="save"):
    f = f"{root}/ACDCDataset/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


NR = int(sys.argv[1]) if len(sys.argv) > 1 else 150   # rounds to show
OUT = f"figures/baselines_{NR}R_acdc.png"

TEKAI = "save_tekai/save"
EPISODIC, NOADAPT = 29.84, 23.34          # MLMP episodic (4-cond mean), No-Adapt

# (label, save_dir, root, colour, linewidth)
CURVES = [
    ("TENT   (ICLR'21)",    "tent_continual_Round150_lr_0.00001", TEKAI,  "#ff7f0e", 3.0),
    ("DeYO   (ICLR'24)",    "deyo_continual",                    "save",  "#e377c2", 3.0),
    ("CLIPArTT (WACV'25)",  "clipartt_continual_K3",             "save",  "#8c564b", 2.8),
    ("EATA   (ICML'22)",    "eata_continual",                    "save",  "#1f77b4", 3.0),
    ("RoTTA  (CVPR'23)",    "rotta_continual",                   "save",  "#9467bd", 3.0),
    ("CoTTA  (CVPR'22)",    "cotta_continual_150R",              "save",  "#17becf", 3.2),
]

fig, ax = plt.subplots(figsize=(15, 8.5))

# reference lines
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=2.6, zorder=3)
ax.axhline(NOADAPT, color="k", ls=":", lw=2.2, zorder=3)

for lab, p, root, c, lw in CURVES:
    m = miou(p, root)
    if len(m) == 0:
        print(f"  MISSING: {lab} ({p})")
        continue
    m = m[:NR]
    ax.plot(np.arange(1, len(m) + 1), m, color=c, lw=lw, zorder=5, label=lab)

ax.set_xlim(1, NR)
ax.set_ylim(0, 36)
ax.set_xlabel("Round  (one round = fog → night → rain → snow)", fontsize=28, labelpad=10)
ax.set_ylabel("mIoU (%)", fontsize=30)
ax.set_title(f"Continual TTA on ACDC  ({NR} rounds, no reset)", fontsize=34)
ax.grid(alpha=0.25, zorder=1)
ax.tick_params(width=1.6, length=8)

h, lb = ax.get_legend_handles_labels()
h = [Line2D([], [], color="#2ca02c", ls="--", lw=2.8),
     Line2D([], [], color="k", ls=":", lw=2.4)] + h
lb = ["MLMP-episodic (NeurIPS'25)", "No adaptation (source)"] + lb
ax.legend(h, lb, fontsize=22, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=3, framealpha=0.95, handlelength=2.4, columnspacing=2.0)

fig.tight_layout(rect=[0, 0.02, 1, 1])
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved ->", OUT)

for lab, p, root, _, _ in CURVES:
    m = miou(p, root)[:NR]
    if len(m):
        print(f"  {lab:20s} R={len(m):3d} mean={m.mean():5.1f} peak={m.max():5.1f} last={m[-1]:5.1f}")
