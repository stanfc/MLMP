"""Exp 1 plots: box plot, mean±CI bar, |g_sup| vs cos scatter.

Usage:
    python -m experiments.h1_validation.exp1_gradient_cosine.plot \
        --csv experiments/h1_validation/results/exp1/all.csv \
        --out_dir experiments/h1_validation/results/exp1/figs/
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NATIVE_COLORS = ["#0ea5e9", "#0891b2", "#0d9488", "#059669", "#16a34a", "#65a30d"]
SYNTHETIC_COLORS = ["#f97316", "#ea580c", "#dc2626", "#db2777", "#c026d3", "#9333ea"]


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cos"] = float(r["cos"])
        r["norm_tent"] = float(r["norm_tent"])
        r["norm_sup"] = float(r["norm_sup"])
        r["idx"] = int(r["idx"])
        r["n_pixels"] = int(r["n_pixels"])
    return rows


def group_by_condition(rows: list[dict]) -> "OrderedDict[tuple[str, str], dict]":
    """Group rows by (dataset, condition); preserve native-first order."""
    groups: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
    for r in rows:
        key = (r["dataset"], r["condition"])
        if key not in groups:
            groups[key] = {"kind": r["kind"], "cosines": [], "norms_sup": []}
        groups[key]["cosines"].append(r["cos"])
        groups[key]["norms_sup"].append(r["norm_sup"])
    # sort: native first, then synthetic; within kind preserve insertion order
    sorted_keys = sorted(groups.keys(), key=lambda k: (groups[k]["kind"] != "native", k))
    return OrderedDict((k, groups[k]) for k in sorted_keys)


def color_for(idx_within_kind: int, kind: str) -> str:
    palette = NATIVE_COLORS if kind == "native" else SYNTHETIC_COLORS
    return palette[idx_within_kind % len(palette)]


def plot_boxplot(groups, out_path: str):
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(groups) + 4), 6), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.0, color="#dc2626", linewidth=1.0, zorder=2)
    labels, data, colors = [], [], []
    nidx = sidx = 0
    for (ds, cond), g in groups.items():
        labels.append(f"{ds}\n{cond}")
        data.append(g["cosines"])
        if g["kind"] == "native":
            colors.append(color_for(nidx, "native"))
            nidx += 1
        else:
            colors.append(color_for(sidx, "synthetic"))
            sidx += 1
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("cos(g_tent, g_sup)", fontsize=11)
    ax.set_title("Exp 1 — Per-image cosine of TENT gradient vs supervised CE gradient\n"
                 "(positive ⇒ entropy minimization aligned with correctness)",
                 fontsize=11, pad=10)
    handles = [
        Line2D([], [], marker="s", linestyle="", color=NATIVE_COLORS[0], label="native"),
        Line2D([], [], marker="s", linestyle="", color=SYNTHETIC_COLORS[0], label="synthetic"),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, facecolor="white",
              edgecolor="#d1d5db", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_bar(groups, out_path: str):
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(groups) + 4), 6), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.0, color="#dc2626", linewidth=1.0, zorder=2)
    labels, means, ci_lows, ci_highs, colors = [], [], [], [], []
    nidx = sidx = 0
    for (ds, cond), g in groups.items():
        arr = np.array(g["cosines"])
        m = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        labels.append(f"{ds}\n{cond}")
        means.append(m)
        ci_lows.append(1.96 * se)
        ci_highs.append(1.96 * se)
        if g["kind"] == "native":
            colors.append(color_for(nidx, "native"))
            nidx += 1
        else:
            colors.append(color_for(sidx, "synthetic"))
            sidx += 1
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=[ci_lows, ci_highs], color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean cos(g_tent, g_sup) ± 95% CI", fontsize=11)
    ax.set_title("Exp 1 — Mean gradient cosine per (dataset, condition)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_scatter(groups, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0.0, color="#dc2626", linewidth=1.0)
    for (ds, cond), g in groups.items():
        col = "#0ea5e9" if g["kind"] == "native" else "#dc2626"
        ax.scatter(np.log10(np.array(g["norms_sup"]) + 1e-12), g["cosines"],
                   c=col, alpha=0.55, s=22)
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#0ea5e9", label="native"),
        Line2D([], [], marker="o", linestyle="", color="#dc2626", label="synthetic"),
    ]
    ax.legend(handles=handles, loc="best")
    ax.set_xlabel("log10 ||g_sup||", fontsize=11)
    ax.set_ylabel("cos(g_tent, g_sup)", fontsize=11)
    ax.set_title("Exp 1 — Per-image |g_sup| vs cosine alignment\n"
                 "(small |g_sup| on synthetic ⇒ supervised loss has little to fix)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_gsup_city_vs_voc(rows, out_path: str):
    """Cityscapes vs VOC20: scatter of cos vs log10||g_sup|| with marginal
    ||g_sup|| distributions on top. Focuses on the per-image g_sup difference
    between the two synthetic datasets (Cityscapes has larger g_sup = more
    headroom; VOC20 is bunched at small g_sup = source near-optimal)."""
    series = OrderedDict([
        ("Cityscapes", {"color": "#dc2626", "x": [], "y": []}),
        ("VOC20_matched", {"color": "#2563eb", "x": [], "y": []}),
    ])
    for r in rows:
        ds = r["dataset"]
        if ds in series:
            series[ds]["x"].append(np.log10(r["norm_sup"] + 1e-12))
            series[ds]["y"].append(r["cos"])

    fig = plt.figure(figsize=(9, 7), dpi=160)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.04)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_top)

    # shared bins across both datasets so the histograms are comparable
    all_x = np.concatenate([np.array(s["x"]) for s in series.values()])
    bins = np.linspace(all_x.min(), all_x.max(), 46)

    for label, s in series.items():
        x = np.array(s["x"]); y = np.array(s["y"])
        med = np.median(x)
        # main scatter
        ax.scatter(x, y, c=s["color"], alpha=0.35, s=16,
                   edgecolors="none", label=f"{label}  (n={len(x)})")
        # marginal ||g_sup|| distribution (density) on top
        ax_top.hist(x, bins=bins, color=s["color"], alpha=0.45, density=True)
        ax_top.axvline(med, color=s["color"], linewidth=1.6, linestyle="-")
        ax.axvline(med, color=s["color"], linewidth=1.2, linestyle="--", alpha=0.8)

    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xlabel(r"$\log_{10}\,\|g_{sup}\|$   (supervised gradient magnitude = headroom)", fontsize=11)
    ax.set_ylabel(r"$\cos(g_{tent},\, g_{sup})$   (direction alignment)", fontsize=11)
    ax.legend(loc="lower right", framealpha=0.9)

    ax_top.set_facecolor("#fafafa")
    ax_top.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax_top.set_ylabel("density", fontsize=9)
    ax_top.tick_params(labelbottom=False)
    # annotate the two medians on the marginal
    cmed = np.median(series["Cityscapes"]["x"])
    vmed = np.median(series["VOC20_matched"]["x"])
    ax_top.set_title(
        "Exp 1 — Cityscapes vs VOC20: per-image gradient alignment & headroom\n"
        rf"median $\|g_{{sup}}\|$:  Cityscapes $10^{{{cmed:.2f}}}{{=}}{10**cmed:.2f}$   "
        rf"vs  VOC20 $10^{{{vmed:.2f}}}{{=}}{10**vmed:.2f}$   "
        "(right ⇒ more to fix; up ⇒ TENT points the right way)",
        fontsize=10, pad=8)

    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/h1_validation/results/exp1/all.csv")
    ap.add_argument("--out_dir", default="experiments/h1_validation/results/exp1/figs/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    groups = group_by_condition(rows)
    print(f"[plot] {len(rows)} rows across {len(groups)} (dataset, condition) pairs")
    plot_boxplot(groups, os.path.join(args.out_dir, "cosine_boxplot"))
    plot_bar(groups, os.path.join(args.out_dir, "cosine_bar"))
    plot_scatter(groups, os.path.join(args.out_dir, "norm_vs_cos_scatter"))
    plot_gsup_city_vs_voc(rows, os.path.join(args.out_dir, "gsup_city_vs_voc"))
    print(f"[plot] Saved 4 figure pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
