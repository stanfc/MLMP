"""Pure TENT-continual vs MLMP-episodic vs No-Adapt on ACDC (150 rounds).
The classic motivation: TENT improves early, then collapses; episodic is strong
but needs per-sample reset; no-adapt is the flat source baseline.
Output: figures/tent_vs_refs_acdc.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 19, "axes.labelsize": 22,
                     "xtick.labelsize": 18, "ytick.labelsize": 18})

TENT = "save_tekai/save/ACDCDataset/tent_continual_Round150_lr_0.00001/results_all_rounds.txt"
EPISODIC, NOADAPT = 29.84, 23.34

m = np.array([float(l.split(",")[-1]) for l in open(TENT) if l.startswith("Round ")])

fig, ax = plt.subplots(figsize=(13, 7.6))
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=3.0, zorder=3)
ax.axhline(NOADAPT, color="k", ls=":", lw=2.6, zorder=3)
ax.plot(np.arange(1, len(m) + 1), m, color="#ff7f0e", lw=3.6, zorder=5)

ax.set_xlim(1, 150)
ax.set_ylim(0, 36)
ax.set_xlabel("Round  (fog → night → rain → snow)")
ax.set_ylabel("mIoU (%)")
ax.set_title("Continual TTA on ACDC  (150 rounds, no reset)", fontsize=23)
ax.grid(alpha=0.25)

handles = [
    Line2D([], [], color="#2ca02c", ls="--", lw=3.0),
    Line2D([], [], color="k", ls=":", lw=2.6),
    Line2D([], [], color="#ff7f0e", lw=3.6),
]
labels = [
    f"MLMP-episodic (per-sample reset)  {EPISODIC:.1f}",
    f"No adaptation (source)  {NOADAPT:.1f}",
    f"TENT-continual  —  peak {m.max():.1f} → ends {m[-1]:.1f}",
]
ax.legend(handles, labels, fontsize=17, loc="upper right", framealpha=0.95)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/tent_vs_refs_acdc.png", dpi=140, bbox_inches="tight")
print("saved -> figures/tent_vs_refs_acdc.png")
