#!/usr/bin/env python3
"""
Cityscapes continual methods comparison: Mean mIoU over rounds.

Generates two figures from save/CityscapesDataset/:
  (1) all-15 corruptions  -> cityscapes_methods_all15.{png,svg}
  (2) weather subset      -> cityscapes_methods_weather.{png,svg}

Folder name convention: contains 'weather' -> weather-5 run; else -> all-15.
Runs without results_all_rounds.txt (incomplete) are skipped with a warning.

Baselines:
  - No Adapt        : computed from save/CityscapesDataset/No_Adaptation/round_01
                      (all-15 plot uses the 10 corruptions actually run; weather
                      plot uses R1 of tent_continual_weather as a clean proxy
                      since brightness/contrast are missing from No_Adaptation).
  - MLMP-episodic   : mean of save/CityscapesDataset/mlmp_episodic{,_weather}/results.txt
  - TENT-continual  : trajectory from tent_continual_weather (weather plot).
                      All-15 trajectory unavailable (only 1 corruption × 1 round
                      ran); mlmp_continual serves as the analog 'naive continual'
                      collapse curve.
"""

from __future__ import annotations
import os
import re
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SAVE_ROOT = "save/CityscapesDataset"

WEATHER_CONDS = {"snow", "frost", "fog", "brightness", "contrast"}


# ── parsers ───────────────────────────────────────────────────────────────────

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


def parse_episodic_mean(path: str, only_conds: set[str] | None = None) -> float:
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
            cond = parts[0]
            if only_conds is not None and cond not in only_conds:
                continue
            try:
                miou = float(parts[1].split("+/-")[0].strip())
                vals.append(miou)
            except ValueError:
                continue
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def parse_no_adapt(no_adapt_dir: str, only_conds: set[str] | None = None
                   ) -> tuple[float, list[str]]:
    """
    Read No_Adaptation/round_01/<cond>/results.txt for each condition;
    each file is a single-line mIoU value (no class breakdown).
    Returns (mean_miou, used_conds).
    """
    base = os.path.join(no_adapt_dir, "round_01")
    if not os.path.isdir(base):
        return float("nan"), []
    used, vals = [], []
    for cond in sorted(os.listdir(base)):
        if only_conds is not None and cond not in only_conds:
            continue
        p = os.path.join(base, cond, "results.txt")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            data = [l.strip() for l in f if l.strip()]
        if len(data) < 2:
            continue
        try:
            miou = float(data[1].split(",")[0].split("+/-")[0].strip())
            vals.append(miou)
            used.append(cond)
        except ValueError:
            continue
    if not vals:
        return float("nan"), []
    return float(np.mean(vals)), used


# ── method collection ─────────────────────────────────────────────────────────

# Visual style per method "kind" — keyed by substring match against folder name.
# Order matters: first match wins.
KIND_STYLES = [
    # (substring,            label_template,                   color,     lw,  alpha, ls)
    ("mlmp_divgate_continual",
     "MLMP-DivGate ({hp})",  "#9333ea", 2.4, 1.0, "-"),
    ("tent_divgate_continual",
     "TENT-DivGate ({hp})",  "#16a34a", 2.0, 1.0, "-"),
    ("mlmp_continual",
     "MLMP-continual",       "#2563eb", 2.0, 1.0, "-"),
    ("tent_continual",
     "TENT-continual",       "#dc2626", 2.0, 1.0, "-"),
]

# Per-h_threshold tint adjustment for DivGate variants (so 1.6/1.7/etc are
# visually distinguishable while staying within the same color family).
DIVGATE_TINTS = {
    "tent_divgate": {
        "1.5": "#86efac",
        "1.6": "#16a34a",
        "1.7": "#15803d",
        "1.8": "#14532d",
    },
    "mlmp_divgate": {
        "1.6": "#9333ea",
        "1.7": "#6b21a8",
    },
}


def hp_suffix_from_dirname(name: str) -> str:
    """Pull a short hyperparameter tag out of folder name for the legend."""
    m = re.search(r"threshold[_\-]?(\d+\.?\d*)", name)
    if m:
        return f"h_thr={m.group(1)}"
    m = re.search(r"hthr[_\-]?(\d+\.?\d*)", name)
    if m:
        return f"h_thr={m.group(1)}"
    return ""


def style_for(folder_name: str) -> dict:
    """Determine plot style for a given run folder."""
    for substr, label_tpl, color, lw, alpha, ls in KIND_STYLES:
        if substr not in folder_name:
            continue
        hp = hp_suffix_from_dirname(folder_name)
        # tint by h_threshold for divgate variants
        if "tent_divgate_continual" in folder_name and hp.startswith("h_thr="):
            t = DIVGATE_TINTS["tent_divgate"].get(hp.split("=")[1])
            if t:
                color = t
        if "mlmp_divgate_continual" in folder_name and hp.startswith("h_thr="):
            t = DIVGATE_TINTS["mlmp_divgate"].get(hp.split("=")[1])
            if t:
                color = t
        label = label_tpl.format(hp=hp) if "{hp}" in label_tpl else label_tpl
        # remove empty parens if no hp
        label = re.sub(r"\s*\(\s*\)\s*$", "", label)
        return dict(label=label, color=color, lw=lw, alpha=alpha, ls=ls)
    return dict(label=folder_name, color="#666666", lw=1.2, alpha=0.6, ls="-")


def collect_runs(weather: bool, min_rounds: int = 5) -> list[dict]:
    """Find all runs in SAVE_ROOT matching the requested split (weather/all-15)."""
    runs = []
    for d in sorted(glob(os.path.join(SAVE_ROOT, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        is_weather = "weather" in name.lower()
        if weather != is_weather:
            continue
        # episodic / no-adapt runs are baselines, handled separately
        if "episodic" in name or name.lower().startswith("no_adapt") \
           or name == "No_Adaptation":
            continue
        path = os.path.join(d, "results_all_rounds.txt")
        if not os.path.exists(path):
            print(f"  skip (no results_all_rounds.txt): {name}")
            continue
        rounds, means = parse_means(path)
        if len(rounds) < min_rounds:
            print(f"  skip (only {len(rounds)} rounds): {name}")
            continue
        runs.append(dict(name=name, path=path, rounds=rounds, means=means,
                         **style_for(name)))
    return runs


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_split(weather: bool):
    title_tag = "weather subset (5 corruptions)" if weather else "all 15 corruptions"
    out_tag = "weather" if weather else "all15"

    runs = collect_runs(weather=weather)

    # baselines
    if weather:
        no_adapt = 21.32  # R1 of tent_continual_weather (= source eval pre-drift)
        no_adapt_label = "No Adapt (R1)"
        episodic_path = os.path.join(SAVE_ROOT, "mlmp_episodic_weather/results.txt")
    else:
        # No_Adaptation run only completed 10/15 corruptions
        no_adapt, used = parse_no_adapt(os.path.join(SAVE_ROOT, "No_Adaptation"))
        no_adapt_label = f"No Adapt ({len(used)}/15 conds)"
        episodic_path = os.path.join(SAVE_ROOT, "mlmp_episodic/results.txt")

    mlmp_episodic = parse_episodic_mean(episodic_path)

    # plot
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    all_y = []
    method_handles = []

    for r in runs:
        ax.plot(r["rounds"], r["means"],
                color=r["color"], linewidth=r["lw"], alpha=r["alpha"],
                linestyle=r["ls"], zorder=4)
        all_y.extend(r["means"])
        # mark endpoint if run is short (< 150 rounds)
        last_r, last_v = r["rounds"][-1], r["means"][-1]
        if last_r < 150:
            ax.plot(last_r, last_v, "o", color=r["color"],
                    markersize=5, alpha=0.85, zorder=5)
        method_handles.append(
            mlines.Line2D([], [], color=r["color"], linewidth=r["lw"],
                          alpha=r["alpha"], linestyle=r["ls"],
                          label=f"{r['label']}  "
                                f"(mean={np.mean(r['means']):.2f}, "
                                f"R{last_r}={last_v:.2f})")
        )

    # horizontal baselines
    if not np.isnan(no_adapt):
        ax.axhline(no_adapt, linestyle=":", linewidth=1.5,
                   color="#6b7280", zorder=3)
    if not np.isnan(mlmp_episodic):
        ax.axhline(mlmp_episodic, linestyle="--", linewidth=2.0,
                   color="#7c3aed", zorder=3)

    # y-axis: use 1.0 floor cushion if some runs collapsed near zero
    flat_y = [y for y in all_y if y > 1.0]
    base_floor = min([no_adapt, mlmp_episodic] + (flat_y if flat_y else [0.0]))
    base_top = max([no_adapt, mlmp_episodic] + all_y) if all_y else mlmp_episodic
    y_lo = max(0, base_floor - 3)
    y_hi = base_top + 3
    # if any run truly collapsed (< 1.0), include zero
    if any(y < 1.0 for y in all_y):
        y_lo = -1.0
    ax.set_ylim(y_lo, y_hi)

    # legend
    baseline_handles = []
    if not np.isnan(no_adapt):
        baseline_handles.append(
            mlines.Line2D([], [], color="#6b7280", linewidth=1.5, linestyle=":",
                          label=f"{no_adapt_label} = {no_adapt:.2f}")
        )
    if not np.isnan(mlmp_episodic):
        baseline_handles.append(
            mlines.Line2D([], [], color="#7c3aed", linewidth=2.0, linestyle="--",
                          label=f"MLMP episodic = {mlmp_episodic:.2f}")
        )

    ax.legend(handles=method_handles + baseline_handles,
              loc="lower left" if weather else "upper right",
              frameon=True, facecolor="white", edgecolor="#d1d5db",
              fontsize=8.5, ncol=1)

    ax.set_title(
        f"Cityscapes CTTA — {title_tag}\n"
        f"Mean mIoU vs continual round (evaluate-before-adapt)",
        fontsize=12, pad=10,
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean mIoU (%)", fontsize=11)

    fig.tight_layout()
    out_png = os.path.join(SAVE_ROOT, f"cityscapes_methods_{out_tag}.png")
    out_svg = os.path.join(SAVE_ROOT, f"cityscapes_methods_{out_tag}.svg")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    print("=== all-15 corruptions ===")
    plot_split(weather=False)
    print()
    print("=== weather subset ===")
    plot_split(weather=True)
