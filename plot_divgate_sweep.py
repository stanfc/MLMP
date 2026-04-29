#!/usr/bin/env python3
"""
Compare TENT-DivGate cautious_rst sweep variants (Mean mIoU over 150 rounds).

Style mirrors the bottom panel of acdc_tent_divgate_150round.png:
  - TENT-continual (no gate)  — dark red reference line
  - DivGate variants          — colour-coded by cautious_rst value
  - No Adapt baseline         — grey dotted
  - MLMP episodic             — purple dashed

Outputs:
  save/ACDCDataset/acdc_divgate_cau_rst_sweep.png
  save/ACDCDataset/acdc_divgate_cau_rst_sweep.svg
"""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


SAVE_ROOT = "save/ACDCDataset"

TENT_FILE = os.path.join(
    SAVE_ROOT, "tent_continual_Round150_lr_0.00001", "results_all_rounds.txt"
)

# (label, path, color)
DIVGATE_VARIANTS = [
    ("DivGate cau_rst=0.001", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_rst_0.001", "results_all_rounds.txt"), "#93c5fd"),  # light blue
    ("DivGate cau_rst=0.003", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_rst_0.003", "results_all_rounds.txt"), "#60a5fa"),  # blue
    ("DivGate cau_rst=0.005", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_rst_0.005", "results_all_rounds.txt"), "#2563eb"),  # medium blue
    ("DivGate cau_rst=0.008", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_rst_0.008", "results_all_rounds.txt"), "#16a34a"),  # green
    ("DivGate cau_rst=0.010", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_rst_0.01",  "results_all_rounds.txt"), "#15803d"),  # dark green
]

# original baseline (h_thr=1.8, h_warn=1.2, cau_rst=0.005) — keep for reference
DIVGATE_BASELINE = (
    "DivGate baseline (h1.8/w1.2/r0.005)",
    os.path.join(SAVE_ROOT, "tent_divgate_continual", "results_all_rounds.txt"),
    "#f97316",  # orange
)

OUTPUT_PNG = os.path.join(SAVE_ROOT, "acdc_divgate_cau_rst_sweep.png")
OUTPUT_SVG = os.path.join(SAVE_ROOT, "acdc_divgate_cau_rst_sweep.svg")

NO_ADAPT_BASELINE = 23.3
MLMP_EPISODIC = 30.6


def parse_means(path: str) -> tuple[list[int], list[float]]:
    with open(path, "r", encoding="utf-8") as f:
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


def overall_stats(means: list[float], rounds: list[int]):
    overall = sum(means) / len(means)
    peak_idx = max(range(len(means)), key=lambda i: means[i])
    return overall, means[peak_idx], rounds[peak_idx], means[-1]


def main():
    tent_rounds, tent_mean = parse_means(TENT_FILE)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # --- TENT-continual reference ---
    ax.plot(tent_rounds, tent_mean,
            color="#dc2626", linewidth=2.0, linestyle="-",
            label="TENT-continual (no gate)", zorder=3)

    # --- DivGate original baseline ---
    bl_label, bl_path, bl_color = DIVGATE_BASELINE
    if os.path.exists(bl_path):
        bl_rounds, bl_mean = parse_means(bl_path)
        ax.plot(bl_rounds, bl_mean,
                color=bl_color, linewidth=1.8, linestyle="--",
                label=bl_label, zorder=3)

    # --- DivGate sweep variants ---
    all_means = [tent_mean]
    for label, path, color in DIVGATE_VARIANTS:
        if not os.path.exists(path):
            print(f"  skip (not found): {path}")
            continue
        r, m = parse_means(path)
        all_means.append(m)
        ax.plot(r, m, color=color, linewidth=2.0, label=label, zorder=4)

    # --- Horizontal references ---
    ax.axhline(NO_ADAPT_BASELINE, linestyle=":", linewidth=1.4,
               color="#6b7280", label=f"No Adapt ({NO_ADAPT_BASELINE:.1f})", zorder=2)
    ax.axhline(MLMP_EPISODIC, linestyle="--", linewidth=1.8,
               color="#7c3aed", label=f"MLMP episodic ({MLMP_EPISODIC:.1f})", zorder=2)

    # --- Y-axis range ---
    flat = [v for m in all_means for v in m]
    y_min = min(flat + [NO_ADAPT_BASELINE]) - 2
    y_max = max(flat + [MLMP_EPISODIC]) + 3
    ax.set_ylim(y_min, y_max)

    # --- Peak + R150 annotations for best two variants ---
    best_two = [
        (DIVGATE_VARIANTS[-2][0], DIVGATE_VARIANTS[-2][2]),
        (DIVGATE_VARIANTS[-1][0], DIVGATE_VARIANTS[-1][2]),
    ]
    paths_last_two = [DIVGATE_VARIANTS[-2][1], DIVGATE_VARIANTS[-1][1]]
    for (lbl, col), path in zip(best_two, paths_last_two):
        if not os.path.exists(path):
            continue
        r, m = parse_means(path)
        ov, pk, pk_r, r150 = overall_stats(m, r)
        short = lbl.split("=")[-1]
        ax.annotate(
            f"r={short} peak {pk:.2f}@R{pk_r}",
            xy=(pk_r, pk),
            xytext=(pk_r + 8, pk + 0.8),
            fontsize=8.5, color=col, ha="left",
            arrowprops=dict(arrowstyle="-", color=col, alpha=0.6, lw=0.8),
        )

    # --- Summary stats text box (lower right) ---
    lines_txt = ["Mean over 150 rounds"]
    tent_ov = sum(tent_mean) / len(tent_mean)
    lines_txt.append(f"  TENT (no gate): {tent_ov:.2f}")
    if os.path.exists(bl_path):
        _, bl_mean = parse_means(bl_path)
        bl_ov = sum(bl_mean) / len(bl_mean)
        lines_txt.append(f"  DivGate (baseline): {bl_ov:.2f}")
    for label, path, _ in DIVGATE_VARIANTS:
        if not os.path.exists(path):
            continue
        _, m = parse_means(path)
        short = label.split("=")[-1]
        ov = sum(m) / len(m)
        r150 = m[-1]
        lines_txt.append(f"  cau_rst={short}: mean={ov:.2f}  R150={r150:.2f}")
    lines_txt.append(f"  MLMP episodic:  {MLMP_EPISODIC:.1f}")

    ax.text(
        0.99, 0.05,
        "\n".join(lines_txt),
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.95),
        family="monospace",
    )

    ax.set_title(
        "TENT-DivGate cautious_rst sweep — Mean mIoU over 150 Rounds\n"
        "(h_threshold=1.6, h_warning=1.4, all new variants)",
        fontsize=13, pad=10,
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean mIoU (%)", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="white",
              edgecolor="#d1d5db", fontsize=8.5, ncol=1)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
