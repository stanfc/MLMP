#!/usr/bin/env python3
"""
Dark Zurich / Nighttime Driving / DZ_ND_Combined continual methods comparison:
Mean mIoU over rounds.

Generates three figures:
  (1) Dark Zurich            -> save/DarkZurichDataset/dark_zurich_methods.{png,svg}
  (2) Nighttime Driving      -> save/NighttimeDrivingDataset/nighttime_driving_methods.{png,svg}
  (3) DZ + ND Combined       -> save/DZ_ND_Combined/dz_nd_combined_methods.{png,svg}

Methods plotted:
  - No Adapt           (constant baseline, horizontal dotted line)
  - MLMP-episodic      (per-sample reset upper bound, horizontal dashed line)
  - MLMP-continual
  - TENT-continual
  - CoTTA
  - SAR-continual
  - TENT-DivGate
"""

from __future__ import annotations
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def parse_means(path: str) -> tuple[list[int], list[float]]:
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


def parse_episodic_mean(path: str) -> float:
    """Average mIoU across condition rows in an episodic results.txt."""
    vals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("mIoU") or line.startswith("GPU") \
               or line.startswith("Total") or line.startswith("Mean Duration"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                miou = float(parts[1].split("+/-")[0].strip())
                vals.append(miou)
            except ValueError:
                continue
    return float(np.mean(vals)) if vals else float("nan")


def parse_no_adapt(path: str) -> float:
    """No-Adaptation runs are 1 round; take that Mean_mIoU."""
    _, means = parse_means(path)
    return float(np.mean(means)) if means else float("nan")


# Method visual styles (label, color, linewidth)
METHOD_STYLES = {
    "tent_divgate_continual": ("TENT-DivGate (h_thr=1.6)", "#16a34a", 2.4),
    "sar_continual":          ("SAR-continual",            "#dc2626", 2.0),
    "tent_continual":         ("TENT-continual",           "#ef4444", 1.6),
    "mlmp_continual":         ("MLMP-continual",           "#2563eb", 1.6),
}


def collect_method_runs(save_root: str) -> list[dict]:
    runs = []
    for method_key, (label, color, lw) in METHOD_STYLES.items():
        path = os.path.join(save_root, method_key, "results_all_rounds.txt")
        if not os.path.exists(path):
            print(f"  [skip] {method_key} — no results_all_rounds.txt")
            continue
        rounds, means = parse_means(path)
        if len(rounds) < 2:
            print(f"  [skip] {method_key} — only {len(rounds)} round(s)")
            continue
        runs.append(dict(
            name=method_key,
            rounds=rounds, means=means,
            label=label, color=color, lw=lw,
        ))
    return runs


def plot_dataset(save_root: str, title: str, out_basename: str):
    print(f"=== {title} ===")
    runs = collect_method_runs(save_root)

    # baselines
    no_adapt_path = os.path.join(save_root, "No_Adaptation", "results_all_rounds.txt")
    no_adapt = parse_no_adapt(no_adapt_path) if os.path.exists(no_adapt_path) else float("nan")

    ep_path = os.path.join(save_root, "mlmp", "results.txt")
    mlmp_episodic = parse_episodic_mean(ep_path) if os.path.exists(ep_path) else float("nan")

    # plot setup
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    all_y = []
    method_handles = []

    for r in runs:
        ax.plot(r["rounds"], r["means"],
                color=r["color"], linewidth=r["lw"], zorder=4)
        all_y.extend(r["means"])
        last_r, last_v = r["rounds"][-1], r["means"][-1]
        ax.plot(last_r, last_v, "o", color=r["color"], markersize=5, zorder=5)
        method_handles.append(
            mlines.Line2D([], [], color=r["color"], linewidth=r["lw"],
                          label=f"{r['label']}  "
                                f"(mean={np.mean(r['means']):.2f}, "
                                f"R{last_r}={last_v:.2f})")
        )

    if not np.isnan(no_adapt):
        ax.axhline(no_adapt, linestyle=":", linewidth=1.6,
                   color="#6b7280", zorder=3)
    if not np.isnan(mlmp_episodic):
        ax.axhline(mlmp_episodic, linestyle="--", linewidth=2.0,
                   color="#7c3aed", zorder=3)

    # y-axis: only consider non-collapsed values for the floor so the legend is readable
    flat_y = [y for y in all_y if y > 1.0]
    baselines_for_range = [v for v in (no_adapt, mlmp_episodic) if not np.isnan(v)]
    if flat_y or baselines_for_range:
        base_floor = min((flat_y or [0.0]) + baselines_for_range)
        base_top = max((all_y or [0.0]) + baselines_for_range)
        y_lo = max(0, base_floor - 3)
        y_hi = base_top + 3
        if any(y < 1.0 for y in all_y):
            y_lo = -1.0
        ax.set_ylim(y_lo, y_hi)

    # legend
    baseline_handles = []
    if not np.isnan(no_adapt):
        baseline_handles.append(
            mlines.Line2D([], [], color="#6b7280", linewidth=1.6, linestyle=":",
                          label=f"No Adapt = {no_adapt:.2f}")
        )
    if not np.isnan(mlmp_episodic):
        baseline_handles.append(
            mlines.Line2D([], [], color="#7c3aed", linewidth=2.0, linestyle="--",
                          label=f"MLMP episodic = {mlmp_episodic:.2f}  [requires reset]")
        )

    ax.legend(handles=method_handles + baseline_handles,
              loc="best", frameon=True, facecolor="white",
              edgecolor="#d1d5db", fontsize=9, ncol=1)

    ax.set_title(
        f"{title} CTTA\n"
        f"Mean mIoU vs continual round (evaluate-before-adapt)",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean mIoU (%)", fontsize=11)

    fig.tight_layout()
    out_png = os.path.join(save_root, f"{out_basename}.png")
    out_svg = os.path.join(save_root, f"{out_basename}.svg")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")
    print(f"  Saved {out_svg}")


if __name__ == "__main__":
    plot_dataset("save/DarkZurichDataset",
                 "Dark Zurich (night)",
                 "dark_zurich_methods")
    print()
    plot_dataset("save/NighttimeDrivingDataset",
                 "Nighttime Driving (night)",
                 "nighttime_driving_methods")
    print()
    plot_dataset("save/DZ_ND_Combined",
                 "Dark Zurich + Nighttime Driving (combined stream)",
                 "dz_nd_combined_methods")
