"""OVSS-side batch-size ablation: does batch size itself (not just block
length = subset_size/batch_size) matter, holding block length CONSTANT at 45
steps/round (subset_size = 45*batch_size: 2880/360/45 for batch=64/8/1)? No
gate (base_rst=0), VOC20 trainval subset.

Result CONTRADICTS the CLIP-classification finding, not confirms it:
classification (block=875 fixed) had batch=1 catastrophically collapse
(R15=0.14) while batch=8 stayed stable (R15=48.65) -- smaller batch was the
danger. Here batch=1/8 both stay stable/improving (R150=71.6/74.7) while
batch=64 collapses from its own peak 78.4 down to 52.8 -- LARGER batch is
the danger. Batch size matters on both, but the direction of the risk
flips between the two setups -- not yet understood why.

Output: figures/v20_batch_ablation.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 20, "axes.labelsize": 24, "axes.titlesize": 19,
                     "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "axes.linewidth": 1.4, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
ROOT = "save/PascalVOC20Dataset/v20_batch_ablation"

ARMS = [
    ("batch=1  (subset=45)",   "adagate_nogate_batch1",  "#1baf7a", 2.6, 5),
    ("batch=8  (subset=360)",  "adagate_nogate_batch8",  "#2a78d6", 2.6, 4),
    ("batch=64 (subset=2880)", "adagate_nogate_batch64", "#e34948", 3.0, 6),
]


def load(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


fig, ax = plt.subplots(figsize=(13, 8.5))
legend_handles = []
for label, key, color, lw, z in ARMS:
    r, m = load(f"{ROOT}/{key}/results_all_rounds.txt")
    ax.plot(r, m, color=color, lw=lw, zorder=z)
    legend_handles.append(Line2D([], [], color=color, lw=lw,
                                 label=f"{label}: R1={m[0]:.1f}->R150={m[-1]:.1f}"))

ax.set_xlabel("Round")
ax.set_ylabel("V20 mean mIoU")
ax.set_xlim(0, 151)
ax.tick_params(width=1.4, length=7)
ax.grid(True, alpha=0.25, lw=0.9)
ax.set_title("No-gate, block length held constant (45 steps/round):\nbatch=64 collapses here -- OPPOSITE of classification, where batch=1 collapsed",
             pad=14)
ax.legend(handles=legend_handles, loc="center left", fontsize=15, framealpha=0.93)

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/v20_batch_ablation.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote v20_batch_ablation.png")
