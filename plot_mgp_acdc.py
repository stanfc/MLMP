"""MGP-DeYO-MLMP on ACDC (150 rounds): peaks high early, then collapses.
Shown against MLMP-episodic / No-Adapt references and our stable GDG-PA+distill.
Output: figures/mgp_acdc.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 24, "axes.labelsize": 30, "axes.titlesize": 32,
                     "xtick.labelsize": 24, "ytick.labelsize": 24,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})


def miou(p):
    f = f"save/{p}/results_all_rounds.txt"
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


EPISODIC, NOADAPT = 29.84, 23.34
mgp = miou("ACDCDataset/mgp_deyo_mlmp_continual")
ours = miou("ACDCDataset/deyo_mlmp_hmgate2_distill_continual_lr5e-6")

fig, ax = plt.subplots(figsize=(13, 7.8))
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=2.8, zorder=3)
ax.axhline(NOADAPT, color="k", ls=":", lw=2.6, zorder=3)
ax.plot(np.arange(1, len(ours) + 1), ours, color="#d62728", lw=4.2, zorder=6)
ax.plot(np.arange(1, len(mgp) + 1), mgp, color="#8c564b", lw=3.4, zorder=5)

ax.set_xlim(1, 150); ax.set_ylim(0, 37)
ax.set_xlabel("Round")
ax.set_ylabel("mIoU (%)")
ax.set_title("Continual TTA on ACDC  (150 rounds, no reset)", fontsize=30, pad=12)
ax.grid(alpha=0.22, lw=0.9)
ax.tick_params(width=1.6, length=8)

handles = [
    Line2D([], [], color="#2ca02c", ls="--", lw=2.8),
    Line2D([], [], color="k", ls=":", lw=2.6),
    Line2D([], [], color="#d62728", lw=4.2),
    Line2D([], [], color="#8c564b", lw=3.4),
]
labels = [
    f"MLMP-episodic (NeurIPS'25)  {EPISODIC:.1f}",
    f"No adaptation (source)  {NOADAPT:.1f}",
    f"Ours (GDG-PA + distill)  peak {ours.max():.1f} → ends {ours[-1]:.1f}",
    f"MGP-DeYO-MLMP  peak {mgp.max():.1f} → ends {mgp[-1]:.1f}",
]
ax.legend(handles, labels, fontsize=18, loc="lower left", framealpha=0.95)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/mgp_acdc.png", dpi=140, bbox_inches="tight")
print(f"saved -> figures/mgp_acdc.png  (mgp peak {mgp.max():.1f}@R{mgp.argmax()+1} -> {mgp[-1]:.1f})")
