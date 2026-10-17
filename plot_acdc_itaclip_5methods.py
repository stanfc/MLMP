"""ACDC, ovss_type=itaclip (ported ITACLIP attention), 5-method comparison:
no_adapt / MLMP-episodic (flat references) vs tent / DeYO+MLMP / DeYO+MLMP+AdaGate
(continual trajectories). All continual methods now train all 100 LN tensors
(top_block_exclude bug removed).

Output: figures/acdc_itaclip_5methods.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 16, "axes.labelsize": 19, "axes.titlesize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "axes.linewidth": 1.4, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
ROOT = "save/ACDCDataset"

NO_ADAPT = 25.93  # R1, batch-invariant
MLMP_EPISODIC = (34.33 + 26.98 + 33.00 + 33.22) / 4  # fog/night/rain/snow

ARMS = [
    ("tent",                    "itaclip_tent_continual",         "#1baf7a", 2.2, 4),
    ("DeYO+MLMP",                "itaclip_deyo_mlmp",               "#2a78d6", 2.4, 5),
    ("DeYO+MLMP+AdaGate (ours)", "itaclip_deyo_mlmp_adagate",       "#e34948", 2.8, 6),
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


fig, ax = plt.subplots(figsize=(11, 7.5))
ax.axhline(NO_ADAPT, color="#888888", lw=2.0, ls=":", zorder=2,
           label=f"no_adapt ({NO_ADAPT:.1f})")
ax.axhline(MLMP_EPISODIC, color="#1b9e77", lw=2.0, ls="--", zorder=2,
           label=f"MLMP-episodic ({MLMP_EPISODIC:.1f})")

any_partial = False
for label, key, color, lw, z in ARMS:
    path = f"{ROOT}/{key}/results_all_rounds.txt"
    if not os.path.exists(path):
        continue
    r, m = load(path)
    if len(r) == 0:
        continue
    partial = r[-1] < 150
    any_partial = any_partial or partial
    ax.plot(r, m, color=color, lw=lw, zorder=z,
            label=f"{label}: R1={m[0]:.1f}→R{r[-1]}={m[-1]:.1f}"
                  + (" (running)" if partial else ""))

ax.set_xlabel("Round")
ax.set_ylabel("ACDC mean mIoU")
ax.set_xlim(0, 151)
ax.grid(True, alpha=0.25)
title = "ACDC, ovss_type=itaclip, 100-param LN, 5-method comparison"
if any_partial:
    title += "  (running)"
ax.set_title(title, pad=12)
ax.legend(loc="center right", fontsize=12, framealpha=0.93)

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/acdc_itaclip_5methods.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote acdc_itaclip_5methods.png")
