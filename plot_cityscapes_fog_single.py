#!/usr/bin/env python3
"""
Cityscapes single-corruption (fog only) TENT-continual trajectory, first 100 rounds.
Shows pure TENT drift dynamics without inter-corruption interference.
Baselines: No Adapt (fog) and MLMP-episodic (fog).
"""

from __future__ import annotations
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SAVE_ROOT  = "save/CityscapesDataset"
RUN_DIR    = "tent_continual_single_corruption_fog"
NUM_ROUNDS = 100


def parse_means(path: str):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = [x.strip() for x in lines[0].split(",")]
    mean_idx = header.index("Mean_mIoU")
    rounds, means = [], []
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        m = re.match(r"Round\s+(\d+)", parts[0])
        if not m:
            continue
        rounds.append(int(m.group(1)))
        means.append(float(parts[mean_idx]))
    return rounds, means


def parse_single_miou(path: str) -> float:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("mIoU"):
                continue
            return float(line.split(",")[0].split("+/-")[0].strip())
    return float("nan")


# trajectory
rounds, means = parse_means(os.path.join(SAVE_ROOT, RUN_DIR, "results_all_rounds.txt"))
rounds = rounds[:NUM_ROUNDS]
means  = means[:NUM_ROUNDS]

# baselines (fog only)
no_adapt = parse_single_miou(os.path.join(SAVE_ROOT, "No_Adaptation/round_01/fog/results.txt"))
mlmp_ep  = parse_single_miou(os.path.join(SAVE_ROOT, "mlmp_episodic_weather/02_fog/results.txt"))

# plot
plt.style.use("default")
fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.plot(rounds, means, color="#dc2626", linewidth=2.0, zorder=4)

ax.axhline(no_adapt, linestyle=":",  linewidth=1.5, color="#6b7280", zorder=3)
ax.axhline(mlmp_ep,  linestyle="--", linewidth=2.0, color="#7c3aed", zorder=3)

# y range
all_y = means + [no_adapt, mlmp_ep]
y_lo = max(0, min(all_y) - 2)
y_hi = max(all_y) + 2
ax.set_ylim(y_lo, y_hi)
ax.set_xlim(0, NUM_ROUNDS + 1)

handles = [
    mlines.Line2D([], [], color="#dc2626", linewidth=2.0,
                  label=f"TENT-continual (fog only)  "
                        f"(mean={np.mean(means):.2f}, R{rounds[-1]}={means[-1]:.2f})"),
    mlines.Line2D([], [], color="#6b7280", linewidth=1.5, linestyle=":",
                  label=f"No Adapt (fog) = {no_adapt:.2f}"),
    mlines.Line2D([], [], color="#7c3aed", linewidth=2.0, linestyle="--",
                  label=f"MLMP episodic (fog) = {mlmp_ep:.2f}"),
]
ax.legend(handles=handles, loc="best", frameon=True, facecolor="white",
          edgecolor="#d1d5db", fontsize=10)

ax.set_title(
    f"Cityscapes — TENT-continual on fog only (single corruption, first {NUM_ROUNDS} rounds)\n"
    f"Pure TENT drift dynamics, no inter-corruption interference",
    fontsize=12, pad=10,
)
ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("mIoU on fog (%)", fontsize=11)

fig.tight_layout()
out_png = os.path.join(SAVE_ROOT, "cityscapes_tent_continual_fog_only.png")
out_svg = os.path.join(SAVE_ROOT, "cityscapes_tent_continual_fog_only.svg")
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
print(f"Saved {out_svg}")
