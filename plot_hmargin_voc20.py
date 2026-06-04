#!/usr/bin/env python3
"""
H_margin trajectory for VOC20 DivGate runs.

Two figures, each with TENT-DivGate (left) vs MLMP-DivGate (right) side-by-side:
  (1) all-15  -> save/PascalVOC20Dataset/hmargin_voc20_all15.{png,svg}
  (2) weather -> save/PascalVOC20Dataset/hmargin_voc20_weather.{png,svg}

Points coloured by gate mode (green=aggressive, orange=cautious, red=brake).
"""

from __future__ import annotations
import os
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SAVE_ROOT = "save/PascalVOC20Dataset"

# VOC20: 1449 val images. batch_size=1, monitor_interval=50.
# all-15: ~22030 batches/round; weather-5: ~7280 batches/round
BATCHES_PER_ROUND = {"all15": 22030, "weather": 7280}

MODE_COLOR  = {"aggressive": "#16a34a", "cautious": "#f59e0b", "brake": "#dc2626"}
MODE_ZORDER = {"aggressive": 2, "cautious": 4, "brake": 5}
MODE_SIZE   = {"aggressive": 6, "cautious": 18, "brake": 22}

PANELS = [
    # (label, color_accent, dir_template)
    ("TENT-DivGate (h_thr=1.6)", "tent_divgate_continual_threshold_1.6"),
    ("MLMP-DivGate (h_thr=1.6)", "mlmp_divgate_continual_threshold_1.6"),
]
H_THRESHOLD = 1.6
H_WARNING   = 1.4


def load_log(path: str, batches_per_round: int):
    rounds, h_margins, modes = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("total"):
                continue
            parts = line.split(",")
            total_batches = int(parts[0])
            rounds.append(total_batches / batches_per_round)
            h_margins.append(float(parts[1]))
            modes.append(parts[2].strip())
    return np.array(rounds), np.array(h_margins), modes


def parse_perf(results_path: str):
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


def plot_split(subset: str):
    """subset in {'all15', 'weather'}"""
    suffix = "_weather" if subset == "weather" else ""
    bpr = BATCHES_PER_ROUND[subset]
    title_tag = "weather subset (5 corruptions)" if subset == "weather" else "all 15 corruptions"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    y_min, y_max = float("inf"), float("-inf")
    max_round = 0

    for ax, (label, base_dir) in zip(axes, PANELS):
        dirname = f"{base_dir}{suffix}"
        log_path = os.path.join(SAVE_ROOT, dirname, "divgate_log.txt")
        res_path = os.path.join(SAVE_ROOT, dirname, "results_all_rounds.txt")
        if not os.path.exists(log_path):
            ax.set_title(f"{label}\n(log not found at {dirname})")
            continue

        rounds, h_vals, modes = load_log(log_path, bpr)
        y_min = min(y_min, h_vals.min())
        y_max = max(y_max, h_vals.max())
        max_round = max(max_round, rounds.max())

        c = Counter(modes)
        agg, cau, brk, total = c.get("aggressive", 0), c.get("cautious", 0), c.get("brake", 0), len(modes)

        mean_miou, last_val, last_round = (
            parse_perf(res_path) if os.path.exists(res_path) else (float("nan"), float("nan"), 0)
        )

        ax.plot(rounds, h_vals, color="#9ca3af", linewidth=0.5, alpha=0.4, zorder=1)
        for mode in ["aggressive", "cautious", "brake"]:
            mask = np.array([m == mode for m in modes])
            if mask.any():
                ax.scatter(rounds[mask], h_vals[mask],
                           color=MODE_COLOR[mode], s=MODE_SIZE[mode],
                           zorder=MODE_ZORDER[mode], linewidths=0)

        ax.axhline(H_THRESHOLD, color="#1d4ed8", linewidth=1.4, linestyle="-", zorder=3)
        ax.axhline(H_WARNING,   color="#9333ea", linewidth=1.2, linestyle="--", zorder=3)

        ax.set_facecolor("#fafafa")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

        pct = lambda x: 100.0 * x / max(total, 1)
        sub = (f"agg:{agg} ({pct(agg):.1f}%)  "
               f"cau:{cau} ({pct(cau):.1f}%)  "
               f"brk:{brk} ({pct(brk):.1f}%)")
        ax.set_title(
            f"{label}\n{sub}\nmean={mean_miou:.2f}  R{last_round}={last_val:.2f}",
            fontsize=10.5, pad=6,
        )
        ax.set_ylabel("H_margin", fontsize=10)
        ax.set_xlabel("Round", fontsize=10)

    # adaptive y range with thresholds visible
    y_lo = min(y_min, H_WARNING) - 0.2
    y_hi = y_max + 0.1
    x_hi = max(max_round, 50) * 1.02
    for ax in axes:
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(0, x_hi)
        # add round grid lines
        tick_step = max(int(x_hi // 6 / 5) * 5, 5)
        for r in range(tick_step, int(x_hi) + 1, tick_step):
            ax.axvline(r, color="#d1d5db", linewidth=0.6, linestyle=":", zorder=0)

    legend_handles = [
        mpatches.Patch(color=MODE_COLOR["aggressive"], label="Aggressive (rst=0)"),
        mpatches.Patch(color=MODE_COLOR["cautious"],   label="Cautious (rst=0.01)"),
        mpatches.Patch(color=MODE_COLOR["brake"],      label="Brake (rst=0.05)"),
        plt.Line2D([0], [0], color="#1d4ed8", lw=1.5, linestyle="-",
                   label=f"h_threshold={H_THRESHOLD}"),
        plt.Line2D([0], [0], color="#9333ea", lw=1.2, linestyle="--",
                   label=f"h_warning={H_WARNING}"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9,
               frameon=True, facecolor="white", edgecolor="#d1d5db",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"H_margin Trajectory — VOC20 DivGate variants — {title_tag}\n"
        f"(cau_rst=0.01, brake_rst=0.05, monitor_interval=50)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_tag = "all15" if subset == "all15" else "weather"
    out_png = os.path.join(SAVE_ROOT, f"hmargin_voc20_{out_tag}.png")
    out_svg = os.path.join(SAVE_ROOT, f"hmargin_voc20_{out_tag}.svg")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    print("=== all-15 corruptions ===")
    plot_split("all15")
    print()
    print("=== weather subset ===")
    plot_split("weather")
