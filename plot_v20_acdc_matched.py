#!/usr/bin/env python3
"""
PascalVOC20 (ACDC-matched, 404 imgs/round = 4 corruptions x 101 imgs) continual
methods comparison.

Generates two figures:
  (1) Main methods comparison
        -> save/PascalVOC20Dataset/v20_acdc_matched/v20_acdc_matched_methods.{png,svg}
  (2) TENT-DivGate h_threshold sweep
        -> save/PascalVOC20Dataset/v20_acdc_matched/v20_acdc_matched_threshold_sweep.{png,svg}

Methods in (1):
  - No Adapt            (constant, horizontal dotted line)
  - MLMP-episodic       (per-sample reset upper bound, horizontal dashed line)
  - TENT-DivGate (h=1.6)
  - MLMP-DivGate (h=1.6)
  - SAR-continual
  - CoTTA
  - TENT-continual
  - MLMP-continual

Variants in (2): h_thr in {1.6, 1.7, 1.8, 1.9, 2.0}, all share other gate params.
"""

from __future__ import annotations
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SAVE_ROOT = "save/PascalVOC20Dataset/v20_acdc_matched"


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


# ─── (1) main methods comparison ───────────────────────────────────────────────

METHODS = [
    ("tent_divgate_continual_threshold_1.6",
     "TENT-DivGate (h_thr=1.6)", "#16a34a", 2.4),
    ("mlmp_divgate_continual_threshold_1.6",
     "MLMP-DivGate (h_thr=1.6)", "#9333ea", 2.2),
    ("sar_continual",   "SAR-continual",   "#dc2626", 2.0),
    ("cotta",           "CoTTA",           "#f59e0b", 1.8),
    ("tent_continual",  "TENT-continual",  "#ef4444", 1.6),
    ("mlmp_continual",  "MLMP-continual",  "#2563eb", 1.6),
]


def plot_main_methods():
    print("=== v20_acdc_matched: main methods ===")
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    all_y = []
    method_handles = []

    for key, label, color, lw in METHODS:
        path = os.path.join(SAVE_ROOT, key, "results_all_rounds.txt")
        if not os.path.exists(path):
            print(f"  [skip] {key} — missing")
            continue
        rounds, means = parse_means(path)
        if len(rounds) < 2:
            print(f"  [skip] {key} — only {len(rounds)} round(s)")
            continue
        ax.plot(rounds, means, color=color, linewidth=lw, zorder=4)
        all_y.extend(means)
        last_r, last_v = rounds[-1], means[-1]
        ax.plot(last_r, last_v, "o", color=color, markersize=5, zorder=5)
        method_handles.append(
            mlines.Line2D([], [], color=color, linewidth=lw,
                          label=f"{label}  "
                                f"(mean={np.mean(means):.2f}, "
                                f"R{last_r}={last_v:.2f})")
        )

    # baselines
    na_path = os.path.join(SAVE_ROOT, "No_Adaptation", "results_all_rounds.txt")
    no_adapt = float("nan")
    if os.path.exists(na_path):
        _, na_means = parse_means(na_path)
        no_adapt = float(np.mean(na_means))

    ep_path = os.path.join(SAVE_ROOT, "mlmp_episodic", "results.txt")
    mlmp_episodic = parse_episodic_mean(ep_path) if os.path.exists(ep_path) else float("nan")

    if not np.isnan(no_adapt):
        ax.axhline(no_adapt, linestyle=":", linewidth=1.6,
                   color="#6b7280", zorder=3)
    if not np.isnan(mlmp_episodic):
        ax.axhline(mlmp_episodic, linestyle="--", linewidth=2.0,
                   color="#7c3aed", zorder=3)

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
        "PascalVOC20 (ACDC-matched, 404 imgs/round) CTTA\n"
        "Mean mIoU vs continual round (evaluate-before-adapt)",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean mIoU (%)", fontsize=11)

    fig.tight_layout()
    out_png = os.path.join(SAVE_ROOT, "v20_acdc_matched_methods.png")
    out_svg = os.path.join(SAVE_ROOT, "v20_acdc_matched_methods.svg")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")
    print(f"  Saved {out_svg}")


# ─── (2) TENT-DivGate h_threshold sweep ────────────────────────────────────────

THRESHOLD_VARIANTS = [
    ("h_thr=1.6 ★", "tent_divgate_continual_threshold_1.6", "#16a34a"),
    ("h_thr=1.7",   "tent_divgate_continual_threshold_1.7", "#0ea5e9"),
    ("h_thr=1.8",   "tent_divgate_continual_threshold_1.8", "#f59e0b"),
    ("h_thr=1.9",   "tent_divgate_continual_threshold_1.9", "#9333ea"),
    ("h_thr=2.0",   "tent_divgate_continual_threshold_2.0", "#db2777"),
]


def plot_threshold_sweep():
    print("=== v20_acdc_matched: TENT-DivGate threshold sweep ===")
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    all_y = []
    variant_summaries = []

    # TENT-continual (no gate) reference
    tent_path = os.path.join(SAVE_ROOT, "tent_continual", "results_all_rounds.txt")
    if os.path.exists(tent_path):
        r, m = parse_means(tent_path)
        ax.plot(r, m, color="#dc2626", linewidth=1.8, linestyle="-",
                zorder=3, label="TENT-continual (no gate)")
        all_y.extend(m)
        tent_mean = float(np.mean(m))
    else:
        tent_mean = float("nan")

    handles = []
    if not np.isnan(tent_mean):
        handles.append(mlines.Line2D([], [], color="#dc2626", linewidth=1.8,
                                     label=f"TENT-continual (no gate)  (mean={tent_mean:.2f})"))

    for label, key, color in THRESHOLD_VARIANTS:
        path = os.path.join(SAVE_ROOT, key, "results_all_rounds.txt")
        if not os.path.exists(path):
            print(f"  [skip] {key} — missing")
            continue
        r, m = parse_means(path)
        lw = 2.6 if "★" in label else 2.0
        ax.plot(r, m, color=color, linewidth=lw, zorder=4)
        all_y.extend(m)
        peak_i = int(np.argmax(m))
        ov = float(np.mean(m))
        variant_summaries.append((label, ov, m[peak_i], r[peak_i], m[-1], color))
        handles.append(mlines.Line2D([], [], color=color, linewidth=lw,
                                     label=f"{label}  "
                                           f"(mean={ov:.2f}, peak={m[peak_i]:.2f}@R{r[peak_i]}, R{r[-1]}={m[-1]:.2f})"))

    # baselines
    na_path = os.path.join(SAVE_ROOT, "No_Adaptation", "results_all_rounds.txt")
    no_adapt = float("nan")
    if os.path.exists(na_path):
        _, na_means = parse_means(na_path)
        no_adapt = float(np.mean(na_means))
    ep_path = os.path.join(SAVE_ROOT, "mlmp_episodic", "results.txt")
    mlmp_episodic = parse_episodic_mean(ep_path) if os.path.exists(ep_path) else float("nan")

    if not np.isnan(no_adapt):
        ax.axhline(no_adapt, linestyle=":", linewidth=1.6,
                   color="#6b7280", zorder=2)
        handles.append(mlines.Line2D([], [], color="#6b7280", linewidth=1.6, linestyle=":",
                                     label=f"No Adapt = {no_adapt:.2f}"))
    if not np.isnan(mlmp_episodic):
        ax.axhline(mlmp_episodic, linestyle="--", linewidth=2.0,
                   color="#7c3aed", zorder=2)
        handles.append(mlines.Line2D([], [], color="#7c3aed", linewidth=2.0, linestyle="--",
                                     label=f"MLMP episodic = {mlmp_episodic:.2f}"))

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

    ax.legend(handles=handles, loc="lower left", frameon=True,
              facecolor="white", edgecolor="#d1d5db", fontsize=8.5, ncol=1)

    # summary box
    lines_txt = ["Mean / peak / R150  (h_warn=1.4, cau_rst=0.01)"]
    for label, ov, pk, pk_r, last_v, _ in variant_summaries:
        short = label.replace(" ★", "")
        lines_txt.append(f"  {short:<10}: mean={ov:.2f}  peak={pk:.2f}@R{pk_r}  R150={last_v:.2f}")
    if not np.isnan(mlmp_episodic):
        lines_txt.append(f"  MLMP-ep   : {mlmp_episodic:.2f} (reset)")
    if not np.isnan(no_adapt):
        lines_txt.append(f"  No-Adapt  : {no_adapt:.2f}")

    ax.text(
        0.99, 0.04,
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
        "PascalVOC20 (ACDC-matched) — TENT-DivGate h_threshold sweep\n"
        "(h_warning=1.4, cautious_rst=0.01, brake_rst=0.05 — all variants)",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean mIoU (%)", fontsize=11)

    fig.tight_layout()
    out_png = os.path.join(SAVE_ROOT, "v20_acdc_matched_threshold_sweep.png")
    out_svg = os.path.join(SAVE_ROOT, "v20_acdc_matched_threshold_sweep.svg")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")
    print(f"  Saved {out_svg}")


if __name__ == "__main__":
    plot_main_methods()
    print()
    plot_threshold_sweep()
