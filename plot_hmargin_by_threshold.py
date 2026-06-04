#!/usr/bin/env python3
"""
Visualise H_margin trajectory for each h_threshold variant.
Colours indicate gate mode: green=aggressive, orange=cautious, red=brake.
2×2 subplot layout, one panel per threshold.

Output:
  save/ACDCDataset/hmargin_threshold_sweep.png
  save/ACDCDataset/hmargin_threshold_sweep.svg
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SAVE_ROOT = "save/ACDCDataset"
BATCHES_PER_ROUND = 406   # 60900 total / 150 rounds

MODE_COLOR = {
    "aggressive": "#16a34a",   # dark green
    "cautious":   "#f59e0b",   # amber
    "brake":      "#dc2626",   # red
}
MODE_ZORDER = {"aggressive": 2, "cautious": 4, "brake": 5}
MODE_SIZE   = {"aggressive": 6, "cautious": 18, "brake": 22}

VARIANTS = [
    # (h_threshold, h_warning, label, dir)
    (1.5, 1.4, "h_thr=1.5", "tent_divgate_continual_cau_threshold_1.5"),
    (1.6, 1.4, "h_thr=1.6 ★ (best)", "tent_divgate_continual_cau_rst_0.01"),
    (1.7, 1.4, "h_thr=1.7", "tent_divgate_continual_cau_threshold_1.7"),
    (1.8, 1.4, "h_thr=1.8", "tent_divgate_continual_cau_threshold_1.8"),
]

PERF = {
    # h_thr → (mean, R150)
    1.5: (30.38, 30.11),
    1.6: (31.58, 31.34),
    1.7: (30.63, 29.74),
    1.8: (30.02, 29.46),
}


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


def mode_stats(modes):
    from collections import Counter
    c = Counter(modes)
    total = len(modes)
    agg = c.get("aggressive", 0)
    cau = c.get("cautious", 0)
    brk = c.get("brake", 0)
    return agg, cau, brk, total


fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=150, sharex=True)
fig.patch.set_facecolor("white")

axes_flat = axes.flatten()  # [top-left, top-right, bottom-left, bottom-right]

for ax, (h_thr, h_warn, label, dirname) in zip(axes_flat, VARIANTS):
    log_path = os.path.join(SAVE_ROOT, dirname, "divgate_log.txt")
    if not os.path.exists(log_path):
        ax.set_title(f"{label}\n(log not found)")
        continue

    rounds, h_vals, modes = load_log(log_path)
    agg, cau, brk, total = mode_stats(modes)
    mean_miou, r150 = PERF[h_thr]

    # --- thin grey connecting line for trajectory shape ---
    ax.plot(rounds, h_vals, color="#9ca3af", linewidth=0.5, alpha=0.4, zorder=1)

    # --- scatter coloured by mode ---
    for mode in ["aggressive", "cautious", "brake"]:
        mask = np.array([m == mode for m in modes])
        if mask.any():
            ax.scatter(
                rounds[mask], h_vals[mask],
                color=MODE_COLOR[mode],
                s=MODE_SIZE[mode],
                zorder=MODE_ZORDER[mode],
                label=mode,
                linewidths=0,
            )

    # --- threshold lines ---
    ax.axhline(h_thr, color="#1d4ed8", linewidth=1.4, linestyle="-",
               label=f"h_threshold={h_thr}", zorder=3)
    ax.axhline(h_warn, color="#9333ea", linewidth=1.2, linestyle="--",
               label=f"h_warning={h_warn}", zorder=3)

    # --- round grid lines (every 25 rounds) ---
    for r in range(25, 151, 25):
        ax.axvline(r, color="#d1d5db", linewidth=0.6, linestyle=":", zorder=0)

    ax.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    # --- title ---
    brake_str = f"  brake:{brk}" if brk > 0 else ""
    ax.set_title(
        f"{label}\n"
        f"agg:{agg}  cau:{cau}{brake_str}  |  mean={mean_miou:.2f}  R150={r150:.2f}",
        fontsize=10.5, pad=6,
    )

    ax.set_ylim(1.0, 2.8)
    ax.set_ylabel("H_margin", fontsize=9)

for ax in axes[1]:
    ax.set_xlabel("Round", fontsize=9)

# --- shared legend ---
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
    "H_margin Trajectory by h_threshold — TENT-DivGate (cau_rst=0.01, h_warning=1.4)\n"
    "Green = aggressive (no restore)   Orange = cautious (rst=0.01)   Red = brake (rst=0.05)",
    fontsize=12, y=1.01,
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

out_png = os.path.join(SAVE_ROOT, "hmargin_threshold_sweep.png")
out_svg = os.path.join(SAVE_ROOT, "hmargin_threshold_sweep.svg")
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
print(f"Saved {out_svg}")
