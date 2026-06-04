#!/usr/bin/env python3
"""
PascalVOC20 continual methods comparison: Mean mIoU over rounds.

Generates two figures from save/PascalVOC20Dataset/:
  (1) all-15 corruptions  -> voc20_methods_all15.{png,svg}
  (2) weather subset      -> voc20_methods_weather.{png,svg}

Methods plotted:
  - No Adapt           (constant baseline, horizontal dotted line)
  - MLMP-episodic      (per-sample reset upper bound, horizontal dashed line)
  - MLMP-continual
  - MLMP-DivGate
  - TENT-continual
  - TENT-DivGate
"""

from __future__ import annotations
import os
import re
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SAVE_ROOT = "save/PascalVOC20Dataset"


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


# Method visual styles (label, color, linewidth)
METHOD_STYLES = {
    "tent_divgate_continual": ("TENT-DivGate (h_thr=1.6)", "#16a34a", 2.2),
    "mlmp_divgate_continual": ("MLMP-DivGate (h_thr=1.6)", "#9333ea", 2.2),
    "tent_continual":         ("TENT-continual",            "#dc2626", 1.8),
    "mlmp_continual":         ("MLMP-continual",            "#2563eb", 1.8),
    "cotta":                  ("CoTTA",                     "#f59e0b", 1.8),
}


def collect_method_runs(weather: bool) -> list[dict]:
    """Find continual method runs for the requested subset."""
    suffix = "_weather" if weather else ""
    runs = []
    for method_key, (label, color, lw) in METHOD_STYLES.items():
        # Try direct match and threshold-suffixed variant
        candidates = [
            f"{method_key}{suffix}",
            f"{method_key}_threshold_1.6{suffix}",
        ]
        path = None
        for c in candidates:
            p = os.path.join(SAVE_ROOT, c, "results_all_rounds.txt")
            if os.path.exists(p):
                path = p
                break
        if path is None:
            print(f"  [skip] {method_key}{suffix} — no results file found")
            continue
        rounds, means = parse_means(path)
        if len(rounds) < 2:
            print(f"  [skip] {os.path.basename(os.path.dirname(path))} — only {len(rounds)} rounds")
            continue
        runs.append(dict(
            name=os.path.basename(os.path.dirname(path)),
            rounds=rounds, means=means,
            label=label, color=color, lw=lw,
        ))
    return runs


def plot_split(weather: bool):
    title_tag = "weather subset (5 corruptions)" if weather else "all 15 corruptions"
    out_tag = "weather" if weather else "all15"
    suffix = "_weather" if weather else ""

    print(f"=== VOC20 {title_tag} ===")
    runs = collect_method_runs(weather=weather)

    # ── baselines ───────────────────────────────────────────────────────────
    # No Adapt: continual run with all conditions; mean is essentially constant.
    no_adapt_path = os.path.join(SAVE_ROOT, f"No_Adaptation{suffix}", "results_all_rounds.txt")
    if os.path.exists(no_adapt_path):
        _, na_means = parse_means(no_adapt_path)
        no_adapt = float(np.mean(na_means))
    else:
        no_adapt = float("nan")

    # MLMP episodic: single-pass per-condition results.txt
    ep_path = os.path.join(SAVE_ROOT, f"mlmp_episodic{suffix}", "results.txt")
    mlmp_episodic = parse_episodic_mean(ep_path) if os.path.exists(ep_path) else float("nan")

    # ── plot setup ──────────────────────────────────────────────────────────
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
        if last_r < 150:
            ax.plot(last_r, last_v, "o", color=r["color"],
                    markersize=5, zorder=5)
        method_handles.append(
            mlines.Line2D([], [], color=r["color"], linewidth=r["lw"],
                          label=f"{r['label']}  "
                                f"(mean={np.mean(r['means']):.2f}, "
                                f"R{last_r}={last_v:.2f})")
        )

    if not np.isnan(no_adapt):
        ax.axhline(no_adapt, linestyle=":", linewidth=1.5,
                   color="#6b7280", zorder=3)
    if not np.isnan(mlmp_episodic):
        ax.axhline(mlmp_episodic, linestyle="--", linewidth=2.0,
                   color="#7c3aed", zorder=3)

    # ── y-axis range ────────────────────────────────────────────────────────
    flat_y = [y for y in all_y if y > 1.0]
    baselines_for_range = [v for v in (no_adapt, mlmp_episodic) if not np.isnan(v)]
    base_floor = min(baselines_for_range + (flat_y if flat_y else [0.0]))
    base_top = max(baselines_for_range + all_y) if all_y else max(baselines_for_range)
    y_lo = max(0, base_floor - 3)
    y_hi = base_top + 3
    if any(y < 1.0 for y in all_y):
        y_lo = -1.0
    ax.set_ylim(y_lo, y_hi)

    # ── legend ──────────────────────────────────────────────────────────────
    baseline_handles = []
    if not np.isnan(no_adapt):
        baseline_handles.append(
            mlines.Line2D([], [], color="#6b7280", linewidth=1.5, linestyle=":",
                          label=f"No Adapt = {no_adapt:.2f}")
        )
    if not np.isnan(mlmp_episodic):
        baseline_handles.append(
            mlines.Line2D([], [], color="#7c3aed", linewidth=2.0, linestyle="--",
                          label=f"MLMP episodic = {mlmp_episodic:.2f}")
        )

    ax.legend(handles=method_handles + baseline_handles,
              loc="best", frameon=True, facecolor="white",
              edgecolor="#d1d5db", fontsize=9, ncol=1)

    ax.set_title(
        f"PascalVOC20 CTTA — {title_tag}\n"
        f"Mean mIoU vs continual round (evaluate-before-adapt)",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean mIoU (%)", fontsize=11)

    fig.tight_layout()
    out_png = os.path.join(SAVE_ROOT, f"voc20_methods_{out_tag}.png")
    out_svg = os.path.join(SAVE_ROOT, f"voc20_methods_{out_tag}.svg")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")
    print(f"  Saved {out_svg}")


if __name__ == "__main__":
    plot_split(weather=False)
    print()
    plot_split(weather=True)
