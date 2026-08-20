"""OVSS ACDC at batch 8 (lr linearly scaled 5e-6 -> 4e-5 so per-round drift matches batch 1):
does MGP's collapse go away with 8x cleaner per-step gradients?  (Answer: no.)
Output: figures/bs8_acdc_ours_vs_mgp.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 24, "axes.labelsize": 30, "axes.titlesize": 30,
                     "xtick.labelsize": 24, "ytick.labelsize": 24,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

EPISODIC, NOADAPT = 29.84, 23.34


def miou(p):
    f = f"save/ACDCDataset/{p}/results_all_rounds.txt"
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


ours = miou("deyo_mlmp_hmgate2_continual_bs8")
mgp = miou("mgp_deyo_mlmp_continual_bs8")

fig, ax = plt.subplots(figsize=(13.5, 8.0))
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=2.6, zorder=3)
ax.axhline(NOADAPT, color="k", ls=":", lw=2.4, zorder=3)
ax.plot(np.arange(1, len(ours) + 1), ours, color="#d62728", lw=4.4, zorder=6)
ax.plot(np.arange(1, len(mgp) + 1), mgp, color="#8c564b", lw=3.4, zorder=5)

ax.set_xlim(1, max(len(ours), len(mgp)))
ax.set_ylim(0, 37)
ax.set_xlabel("Round")
ax.set_ylabel("mIoU (%)")
ax.set_title("ACDC @ batch 8 — 8× cleaner gradients do NOT save MGP", fontsize=28, pad=14)
ax.grid(alpha=0.22, lw=0.9)
ax.tick_params(width=1.6, length=8)

h = [Line2D([], [], color="#2ca02c", ls="--", lw=2.6), Line2D([], [], color="k", ls=":", lw=2.4),
     Line2D([], [], color="#d62728", lw=4.4), Line2D([], [], color="#8c564b", lw=3.4)]
lb = ["MLMP-episodic (NeurIPS'25)", "No adaptation (source)",
      f"Ours GDG-PA  (R{len(ours)}: {ours[-1]:.1f})",
      f"MGP          (R{len(mgp)}: {mgp[-1]:.1f})"]
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.legend(h, lb, fontsize=21, loc="lower center", bbox_to_anchor=(0.0, -0.02, 1.0, 0.13),
           mode="expand", ncol=2, framealpha=0.95, handlelength=2.2, columnspacing=1.6)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/bs8_acdc_ours_vs_mgp.png", dpi=135, bbox_inches="tight")
print(f"saved -> figures/bs8_acdc_ours_vs_mgp.png  ours R{len(ours)}={ours[-1]:.1f}  mgp R{len(mgp)}={mgp[-1]:.1f}")
