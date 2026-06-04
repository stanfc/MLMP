#!/usr/bin/env python3
"""
Compare TENT-DivGate h_threshold sweep variants (Mean mIoU over 150 rounds).

Style mirrors acdc_divgate_cau_rst_sweep.png:
  - TENT-continual (no gate)  — dark red reference line
  - DivGate variants          — colour-coded by h_threshold value
  - No Adapt baseline         — grey dotted
  - MLMP episodic             — purple dashed

All variants share: h_warning=1.4, cautious_rst=0.01, brake_rst=0.05.

Outputs:
  save/ACDCDataset/acdc_divgate_threshold_sweep.png
  save/ACDCDataset/acdc_divgate_threshold_sweep.svg
"""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt


SAVE_ROOT = "save/ACDCDataset"

TENT_FILE = os.path.join(
    SAVE_ROOT, "tent_continual_Round150_lr_0.00001", "results_all_rounds.txt"
)

# (label, path, color, linestyle)
# h_thr=1.6 uses the confirmed-best cau_rst=0.01 run (h_threshold verified via configurations.txt)
THRESHOLD_VARIANTS = [
    ("h_thr=1.5", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_threshold_1.5", "results_all_rounds.txt"), "#60a5fa", "-"),   # light blue
    ("h_thr=1.6 ★", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_rst_0.01",     "results_all_rounds.txt"), "#16a34a", "-"),  # dark green (best)
    ("h_thr=1.7", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_threshold_1.7", "results_all_rounds.txt"), "#f59e0b", "-"),   # amber
    ("h_thr=1.8", os.path.join(SAVE_ROOT, "tent_divgate_continual_cau_threshold_1.8", "results_all_rounds.txt"), "#9333ea", "-"),   # purple
]

OUTPUT_PNG = os.path.join(SAVE_ROOT, "acdc_divgate_threshold_sweep.png")
OUTPUT_SVG = os.path.join(SAVE_ROOT, "acdc_divgate_threshold_sweep.svg")

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
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # --- TENT-continual reference ---
    all_means = []
    if os.path.exists(TENT_FILE):
        tent_rounds, tent_mean = parse_means(TENT_FILE)
        ax.plot(tent_rounds, tent_mean,
                color="#dc2626", linewidth=2.0, linestyle="-",
                label="TENT-continual (no gate)", zorder=3)
        all_means.append(tent_mean)
    else:
        tent_rounds, tent_mean = [], []
        print(f"  skip (not found): {TENT_FILE}")

    # --- Threshold sweep variants ---
    variant_data = []
    for label, path, color, ls in THRESHOLD_VARIANTS:
        if not os.path.exists(path):
            print(f"  skip (not found): {path}")
            continue
        r, m = parse_means(path)
        all_means.append(m)
        lw = 2.5 if "★" in label else 2.0
        ax.plot(r, m, color=color, linewidth=lw, linestyle=ls, label=label, zorder=4)
        variant_data.append((label, path, color, r, m))

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

    # --- Peak + R150 annotations for each variant (staggered to avoid overlap) ---
    y_offsets = [3.5, 2.0, 0.5, -1.0]  # vertical offsets for 1.5, 1.6, 1.7, 1.8
    x_offsets = [10, 20, 30, 40]
    for i, (label, path, color, r, m) in enumerate(variant_data):
        ov, pk, pk_r, r150 = overall_stats(m, r)
        short = label.replace(" ★", "")
        ax.annotate(
            f"{short} peak {pk:.2f}@R{pk_r}",
            xy=(pk_r, pk),
            xytext=(pk_r + x_offsets[i], pk + y_offsets[i]),
            fontsize=8, color=color, ha="left",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, lw=0.8),
        )

    # --- Summary stats text box (lower right) ---
    lines_txt = ["Mean over 150 rounds  (h_warn=1.4, cau_rst=0.01)"]
    if tent_mean:
        tent_ov = sum(tent_mean) / len(tent_mean)
        lines_txt.append(f"  TENT (no gate):  mean={tent_ov:.2f}")
    for label, path, color, r, m in variant_data:
        ov, pk, pk_r, r150 = overall_stats(m, r)
        short = label.replace(" ★", "")
        lines_txt.append(f"  {short:<10}: mean={ov:.2f}  peak={pk:.2f}@R{pk_r}  R150={r150:.2f}")
    lines_txt.append(f"  MLMP episodic:    mean={MLMP_EPISODIC:.1f} (episodic reset)")

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
        "TENT-DivGate h_threshold sweep — Mean mIoU over 150 Rounds\n"
        "(h_warning=1.4, cautious_rst=0.01, brake_rst=0.05 — all variants)",
        fontsize=13, pad=10,
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean mIoU (%)", fontsize=11)
    ax.legend(loc="upper right", frameon=True, facecolor="white",
              edgecolor="#d1d5db", fontsize=9, ncol=1)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
