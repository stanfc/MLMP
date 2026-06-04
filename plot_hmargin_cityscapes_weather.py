#!/usr/bin/env python3
"""
Visualise H_margin trajectory for TENT-DivGate weather-subset Cityscapes runs.
Mirrors plot_hmargin_by_threshold.py (ACDC) but for the 2 weather variants.
Colours indicate gate mode: green=aggressive, orange=cautious, red=brake.

Output:
  save/CityscapesDataset/hmargin_weather_threshold_sweep.png
  save/CityscapesDataset/hmargin_weather_threshold_sweep.svg
"""

from __future__ import annotations
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SAVE_ROOT = "save/CityscapesDataset"
# Cityscapes weather-5: 5 conditions × 500 images at batch_size=1.
# 7500 monitor windows × 50 batches = 375000 batches → 150 rounds.
BATCHES_PER_ROUND = 2500

MODE_COLOR = {
    "aggressive": "#16a34a",
    "cautious":   "#f59e0b",
    "brake":      "#dc2626",
}
MODE_ZORDER = {"aggressive": 2, "cautious": 4, "brake": 5}
MODE_SIZE   = {"aggressive": 6, "cautious": 18, "brake": 22}

VARIANTS = [
    # (h_threshold, h_warning, label, dir)
    (1.6, 1.4, "h_thr=1.6", "tent_divgate_continual_weather_threshold_1.6"),
    (1.7, 1.4, "h_thr=1.7", "tent_divgate_continual_weather_threshold_1.7"),
]


def load_log(path: str):
    rounds, h_margins, modes = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("total"):
                continue
            parts = line.split(",")
            total_batches = int(parts[0])
            h_val = float(parts[1])
            mode = parts[2].strip()
            rounds.append(total_batches / BATCHES_PER_ROUND)
            h_margins.append(h_val)
            modes.append(mode)
    return np.array(rounds), np.array(h_margins), modes


def parse_perf(results_path: str):
    """Return (mean, R_last_value, R_last_index) parsed from results_all_rounds.txt."""
    means = []
    last_round, last_val = None, None
    with open(results_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = [x.strip() for x in lines[0].split(",")]
    mean_idx = header.index("Mean_mIoU")
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        m = re.match(r"Round\s+(\d+)", parts[0])
        if not m:
            continue
        v = float(parts[mean_idx])
        means.append(v)
        last_round = int(m.group(1))
        last_val = v
    return float(np.mean(means)), last_val, last_round


def mode_stats(modes):
    from collections import Counter
    c = Counter(modes)
    return c.get("aggressive", 0), c.get("cautious", 0), c.get("brake", 0), len(modes)


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150, sharex=True, sharey=True)
fig.patch.set_facecolor("white")

for ax, (h_thr, h_warn, label, dirname) in zip(axes, VARIANTS):
    log_path = os.path.join(SAVE_ROOT, dirname, "divgate_log.txt")
    res_path = os.path.join(SAVE_ROOT, dirname, "results_all_rounds.txt")
    if not os.path.exists(log_path):
        ax.set_title(f"{label}\n(log not found)")
        continue

    rounds, h_vals, modes = load_log(log_path)
    agg, cau, brk, total = mode_stats(modes)
    mean_miou, last_val, last_round = parse_perf(res_path) if os.path.exists(res_path) \
        else (float("nan"), float("nan"), 0)

    # thin grey connecting line for trajectory shape
    ax.plot(rounds, h_vals, color="#9ca3af", linewidth=0.5, alpha=0.4, zorder=1)

    # scatter coloured by mode
    for mode in ["aggressive", "cautious", "brake"]:
        mask = np.array([m == mode for m in modes])
        if mask.any():
            ax.scatter(
                rounds[mask], h_vals[mask],
                color=MODE_COLOR[mode],
                s=MODE_SIZE[mode],
                zorder=MODE_ZORDER[mode],
                linewidths=0,
            )

    # threshold lines
    ax.axhline(h_thr, color="#1d4ed8", linewidth=1.4, linestyle="-",
               label=f"h_threshold={h_thr}", zorder=3)
    ax.axhline(h_warn, color="#9333ea", linewidth=1.2, linestyle="--",
               label=f"h_warning={h_warn}", zorder=3)

    # round grid lines
    for r in range(25, 151, 25):
        ax.axvline(r, color="#d1d5db", linewidth=0.6, linestyle=":", zorder=0)

    ax.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    # mode-fraction stats (Cityscapes brake actually fires here, unlike ACDC)
    pct = lambda x: 100.0 * x / max(total, 1)
    sub = (f"agg:{agg} ({pct(agg):.1f}%)  "
           f"cau:{cau} ({pct(cau):.1f}%)  "
           f"brk:{brk} ({pct(brk):.1f}%)")
    ax.set_title(
        f"{label}\n{sub}\nmean={mean_miou:.2f}  R{last_round}={last_val:.2f}",
        fontsize=10.5, pad=6,
    )

    # H_margin range on Cityscapes goes up to 2.7 and dips under 1.4 (brake fires)
    ax.set_ylim(1.0, 2.8)
    ax.set_xlim(0, 150)
    ax.set_ylabel("H_margin", fontsize=10)
    ax.set_xlabel("Round", fontsize=10)

# shared legend
legend_handles = [
    mpatches.Patch(color=MODE_COLOR["aggressive"], label="Aggressive (rst=0)"),
    mpatches.Patch(color=MODE_COLOR["cautious"],   label="Cautious (rst=0.01)"),
    mpatches.Patch(color=MODE_COLOR["brake"],      label="Brake (rst=0.05)"),
    plt.Line2D([0], [0], color="#1d4ed8", lw=1.5, linestyle="-",  label="h_threshold"),
    plt.Line2D([0], [0], color="#9333ea", lw=1.2, linestyle="--", label="h_warning=1.4"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    fontsize=9,
    frameon=True,
    facecolor="white",
    edgecolor="#d1d5db",
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "H_margin Trajectory — TENT-DivGate on Cityscapes weather-5 (cau_rst=0.01, h_warning=1.4)\n"
    "Note: Cityscapes H_margin runs ~0.2 nats higher than ACDC (mean ≈ 2.0 vs ≈ 1.79)",
    fontsize=12, y=1.02,
)

fig.tight_layout(rect=[0, 0.06, 1, 1])

out_png = os.path.join(SAVE_ROOT, "hmargin_weather_threshold_sweep.png")
out_svg = os.path.join(SAVE_ROOT, "hmargin_weather_threshold_sweep.svg")
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
print(f"Saved {out_svg}")
