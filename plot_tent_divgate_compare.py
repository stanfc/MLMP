#!/usr/bin/env python3
"""
Generate a TENT vs TENT-DivGate comparison chart on ACDC, 150 rounds.

Layout (matches acdc_tent_150round.png style):
  Top    — per-condition trajectories of natural TENT (4 conditions)
  Bottom — Mean mIoU: natural TENT vs TENT-DivGate, with reference
           lines for No Adapt baseline and MLMP-episodic upper bound.

Outputs:
  save/ACDCDataset/acdc_tent_divgate_150round.png
  save/ACDCDataset/acdc_tent_divgate_150round.svg
"""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt


SAVE_ROOT = "save/ACDCDataset"

TENT_FILE = os.path.join(
    SAVE_ROOT, "tent_continual_Round150_lr_0.00001", "results_all_rounds.txt"
)
DIVGATE_FILE = os.path.join(
    SAVE_ROOT, "tent_divgate_continual", "results_all_rounds.txt"
)

OUTPUT_PNG = os.path.join(SAVE_ROOT, "acdc_tent_divgate_150round.png")
OUTPUT_SVG = os.path.join(SAVE_ROOT, "acdc_tent_divgate_150round.svg")

CONDITIONS = ("fog", "night", "rain", "snow")
COND_COLOR = {
    "fog":   "#3b82f6",   # blue
    "night": "#a855f7",   # purple
    "rain":  "#ef4444",   # red
    "snow":  "#f59e0b",   # amber
}

NO_ADAPT_BASELINE = 23.3
MLMP_EPISODIC = 30.6


def parse_all_rounds(path: str):
    """Return rounds list and dict of per-condition lists + mean list."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    header = [item.strip() for item in lines[0].split(",")]
    cond_idx = {c: header.index(c) for c in CONDITIONS}
    mean_idx = header.index("Mean_mIoU")

    rounds = []
    cond_vals = {c: [] for c in CONDITIONS}
    means = []
    for line in lines[1:]:
        parts = [item.strip() for item in line.split(",")]
        m = re.match(r"Round\s+(\d+)", parts[0])
        if not m:
            continue
        rounds.append(int(m.group(1)))
        for c in CONDITIONS:
            cond_vals[c].append(float(parts[cond_idx[c]]))
        means.append(float(parts[mean_idx]))
    return rounds, cond_vals, means


def annotate_peak(ax, rounds, vals, color, label_offset=(0, 0.6)):
    """Mark peak with a vertical dashed line + annotation."""
    peak_idx = max(range(len(vals)), key=lambda i: vals[i])
    peak_r = rounds[peak_idx]
    peak_v = vals[peak_idx]
    ax.axvline(peak_r, linestyle="--", linewidth=1.0, color=color, alpha=0.5)
    ax.annotate(
        f"Peak\n{peak_v:.2f}",
        xy=(peak_r, peak_v),
        xytext=(peak_r + label_offset[0], peak_v + label_offset[1]),
        fontsize=9,
        color=color,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, lw=0.8),
    )
    return peak_r, peak_v


def main():
    tent_rounds, tent_cond, tent_mean = parse_all_rounds(TENT_FILE)
    dg_rounds, dg_cond, dg_mean = parse_all_rounds(DIVGATE_FILE)

    if tent_rounds[: len(dg_rounds)] != dg_rounds:
        raise ValueError("Round indices do not align between TENT and TENT-DivGate.")

    plt.style.use("default")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 9.5), dpi=160,
        gridspec_kw={"height_ratios": [1.0, 1.2], "hspace": 0.32},
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#fafafa")
        ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # ---------------------------------------------------------------
    # Top panel: per-condition trajectories of natural TENT
    # ---------------------------------------------------------------
    for c in CONDITIONS:
        ax_top.plot(
            tent_rounds, tent_cond[c],
            color=COND_COLOR[c], linewidth=1.6,
            label=c.capitalize(),
        )
    ax_top.axhline(
        NO_ADAPT_BASELINE, linestyle=":", linewidth=1.2,
        color="#6b7280",
        label=f"No Adapt baseline ({NO_ADAPT_BASELINE:.1f})",
    )
    ax_top.set_title("TENT-continual on ACDC — 150 Rounds (per condition)",
                     fontsize=14, pad=10)
    ax_top.set_ylabel("mIoU (%)", fontsize=11)
    ax_top.legend(loc="upper right", ncol=5, frameon=True,
                  facecolor="white", edgecolor="#d1d5db", fontsize=9)

    # ---------------------------------------------------------------
    # Bottom panel: Mean mIoU — natural TENT vs TENT-DivGate
    # ---------------------------------------------------------------
    ax_bot.plot(
        tent_rounds, tent_mean,
        color="#dc2626", linewidth=2.2,
        label="TENT-continual (no gate)",
    )
    ax_bot.plot(
        dg_rounds, dg_mean,
        color="#16a34a", linewidth=2.4,
        label="TENT-DivGate",
    )
    ax_bot.axhline(
        NO_ADAPT_BASELINE, linestyle=":", linewidth=1.4,
        color="#6b7280",
        label=f"No Adapt ({NO_ADAPT_BASELINE:.1f})",
    )
    ax_bot.axhline(
        MLMP_EPISODIC, linestyle="--", linewidth=1.6,
        color="#7c3aed",
        label=f"MLMP episodic ({MLMP_EPISODIC:.1f})",
    )

    # Y-axis range first so annotation offsets behave predictably
    y_min = min(min(tent_mean), min(dg_mean), NO_ADAPT_BASELINE) - 2
    y_max = max(max(tent_mean), max(dg_mean), MLMP_EPISODIC) + 3
    ax_bot.set_ylim(y_min, y_max)

    # Peak annotations: place above the curves with leader lines
    tent_peak_idx = max(range(len(tent_mean)), key=lambda i: tent_mean[i])
    dg_peak_idx = max(range(len(dg_mean)), key=lambda i: dg_mean[i])
    tent_peak_r, tent_peak_v = tent_rounds[tent_peak_idx], tent_mean[tent_peak_idx]
    dg_peak_r, dg_peak_v = dg_rounds[dg_peak_idx], dg_mean[dg_peak_idx]

    ax_bot.annotate(
        f"TENT peak\n{tent_peak_v:.2f}@R{tent_peak_r}",
        xy=(tent_peak_r, tent_peak_v),
        xytext=(tent_peak_r + 18, tent_peak_v + 1.8),
        fontsize=9, color="#dc2626", ha="left",
        arrowprops=dict(arrowstyle="-", color="#dc2626", alpha=0.6, lw=0.8),
    )
    ax_bot.annotate(
        f"DivGate peak\n{dg_peak_v:.2f}@R{dg_peak_r}",
        xy=(dg_peak_r, dg_peak_v),
        xytext=(dg_peak_r + 6, dg_peak_v - 4.5),
        fontsize=9, color="#16a34a", ha="left",
        arrowprops=dict(arrowstyle="-", color="#16a34a", alpha=0.6, lw=0.8),
    )

    # Final-round annotations (R150 endpoint)
    ax_bot.annotate(
        f"R150 = {tent_mean[-1]:.2f}",
        xy=(tent_rounds[-1], tent_mean[-1]),
        xytext=(tent_rounds[-1] - 38, tent_mean[-1] - 3.5),
        fontsize=9, color="#dc2626", ha="left",
        arrowprops=dict(arrowstyle="->", color="#dc2626", alpha=0.7, lw=0.9),
    )
    ax_bot.annotate(
        f"R150 = {dg_mean[-1]:.2f}",
        xy=(dg_rounds[-1], dg_mean[-1]),
        xytext=(dg_rounds[-1] - 38, dg_mean[-1] + 2.5),
        fontsize=9, color="#16a34a", ha="left",
        arrowprops=dict(arrowstyle="->", color="#16a34a", alpha=0.7, lw=0.9),
    )

    # Mean over all rounds — placed in lower-right (out of curve area)
    tent_overall = sum(tent_mean) / len(tent_mean)
    dg_overall = sum(dg_mean) / len(dg_mean)
    ax_bot.text(
        0.99, 0.05,
        f"Mean over 150 rounds\n"
        f"  TENT (no gate):  {tent_overall:.2f}\n"
        f"  TENT-DivGate:    {dg_overall:.2f}\n"
        f"  MLMP episodic:   {MLMP_EPISODIC:.1f}",
        transform=ax_bot.transAxes,
        fontsize=9.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.95),
        family="monospace",
    )

    ax_bot.set_title("Mean mIoU — TENT-continual vs TENT-DivGate",
                     fontsize=14, pad=10)
    ax_bot.set_xlabel("Round", fontsize=11)
    ax_bot.set_ylabel("Mean mIoU (%)", fontsize=11)
    ax_bot.legend(loc="upper right", frameon=True, facecolor="white",
                  edgecolor="#d1d5db", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_SVG}")
    print(f"  TENT-continual mean over {len(tent_mean)} rounds: {tent_overall:.2f}")
    print(f"  TENT-DivGate   mean over {len(dg_mean)} rounds: {dg_overall:.2f}")
    print(f"  TENT-DivGate   peak: {max(dg_mean):.2f} @R{dg_rounds[dg_mean.index(max(dg_mean))]}")
    print(f"  TENT-DivGate   R150: {dg_mean[-1]:.2f}")


if __name__ == "__main__":
    main()
