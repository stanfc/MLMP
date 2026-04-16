#!/usr/bin/env python3
"""
Generate a round-vs-mean-mIoU chart for ACDC experiments with matplotlib.

Outputs:
  - save/ACDCDataset/acdc_round_miou.png
  - save/ACDCDataset/acdc_round_miou.svg
"""

from __future__ import annotations

import os
import re
from statistics import mean

import matplotlib.pyplot as plt


SAVE_ROOT = "save/ACDCDataset"

TENT_CONTINUAL_20_ROUND = os.path.join(
    SAVE_ROOT, "tent_continual_Round20_lr_0.00001", "results_all_rounds.txt"
)
MLMP_CONTINUAL_20_ROUND = os.path.join(
    SAVE_ROOT, "mlmp_continual_Round20_batch_1_LR_0.000005", "results_all_rounds.txt"
)
MLMP_EPISODIC_DIR = os.path.join(SAVE_ROOT, "mlmp_batch_1")
OUTPUT_PNG = os.path.join(SAVE_ROOT, "acdc_round_miou.png")
OUTPUT_SVG = os.path.join(SAVE_ROOT, "acdc_round_miou.svg")

CONDITIONS = ("fog", "night", "rain", "snow")


def parse_results_all_rounds(path: str) -> tuple[list[int], list[float]]:
    rounds: list[int] = []
    means: list[float] = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = [item.strip() for item in lines[0].split(",")]
    mean_idx = header.index("Mean_mIoU")

    for line in lines[1:]:
        parts = [item.strip() for item in line.split(",")]
        match = re.match(r"Round\s+(\d+)", parts[0])
        if not match:
            continue
        rounds.append(int(match.group(1)))
        means.append(float(parts[mean_idx]))

    return rounds, means


def parse_episodic_mean(save_dir: str) -> float:
    cond_to_miou: dict[str, float] = {}

    for entry in sorted(os.listdir(save_dir)):
        lower_entry = entry.lower()
        entry_dir = os.path.join(save_dir, entry)
        if not os.path.isdir(entry_dir):
            continue

        for cond in CONDITIONS:
            if cond in lower_entry:
                results_path = os.path.join(entry_dir, "results.txt")
                with open(results_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                cond_to_miou[cond] = float(lines[1].split("+/-")[0].split(",")[0].strip())
                break

    missing = [cond for cond in CONDITIONS if cond not in cond_to_miou]
    if missing:
        raise ValueError(f"Missing episodic results for: {', '.join(missing)}")

    return mean(cond_to_miou.values())


def plot_chart(
    rounds: list[int],
    tent_means: list[float],
    mlmp_means: list[float],
    episodic_mean: float,
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    ax.plot(
        rounds,
        tent_means,
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        color="#2563eb",
        label="TENT continual 20 round",
    )
    ax.plot(
        rounds,
        mlmp_means,
        marker="s",
        linewidth=2.4,
        markersize=5.2,
        color="#ea580c",
        label="MLMP continual 20 round",
    )
    ax.axhline(
        episodic_mean,
        linestyle="--",
        linewidth=2.2,
        color="#6b7280",
        label=f"MLMP episodic ({episodic_mean:.2f})",
    )

    ax.set_title("ACDC Mean mIoU by Round", fontsize=16, pad=14)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Mean mIoU", fontsize=12)
    ax.set_xticks(rounds)
    ax.tick_params(axis="x", labelrotation=0)

    y_all = tent_means + mlmp_means + [episodic_mean]
    y_min = min(y_all)
    y_max = max(y_all)
    y_pad = max(0.8, (y_max - y_min) * 0.12)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#d1d5db")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rounds, tent_means = parse_results_all_rounds(TENT_CONTINUAL_20_ROUND)
    mlmp_rounds, mlmp_means = parse_results_all_rounds(MLMP_CONTINUAL_20_ROUND)
    if rounds != mlmp_rounds:
        raise ValueError("Round indices do not match between continual result files.")

    episodic_mean = parse_episodic_mean(MLMP_EPISODIC_DIR)
    plot_chart(rounds, tent_means, mlmp_means, episodic_mean)

    print(f"Saved chart to {OUTPUT_PNG}")
    print(f"Saved chart to {OUTPUT_SVG}")
    print(f"MLMP episodic mean mIoU: {episodic_mean:.4f}")


if __name__ == "__main__":
    main()
