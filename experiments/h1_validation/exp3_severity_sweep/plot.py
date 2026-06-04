"""Exp 3 plots: severity sweep across the full 15 ImageNet-C corruptions,
designed to compare the VOC20 and Cityscapes *trends* despite their ~4x
absolute-mIoU gap.

Three figures:
  1. severity_retention_grid  — MAIN. 3x5 small-multiples (one per corruption),
     y = retention % (mIoU normalized to its own severity-1 value). Normalizing
     puts both datasets on a 0-100% scale so the decay *shape* is directly
     comparable; VOC solid, Cityscapes dashed. Flat line ⇒ CLIP pretraining
     already covered this corruption (H1); steep line ⇒ genuinely OOD.
  2. severity_decay_panels    — absolute mIoU, two side-by-side panels (each
     with its own y-axis) so the raw numbers remain visible. Native source-mIoU
     reference lines on the Cityscapes panel (shared 19-class scale).
  3. relative_drop_grouped    — grouped bar of (sev1-sev5)/sev1 per corruption,
     VOC vs Cityscapes side by side, sorted by mean drop. One-glance summary.

Native datasets (no severity axis) appear as horizontal reference lines with
values hard-coded below — sourced from save/{Dataset}/No_Adaptation/.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Native source mIoU references (constants, see spec §7.5). Cityscapes-19 scale.
NATIVE_REFS = {
    "ACDC": 23.34,
    "DarkZurich": 20.24,
    "NighttimeDriving": 31.01,
}

# Category-ordered 15 ImageNet-C corruptions (must match
# common/datasets.py IMAGENET_C_15). Drives subplot order + category coloring.
CORRUPTION_ORDER = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
# Category of each corruption → a base color (used in the absolute panel).
CATEGORY = {
    "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur",
    "zoom_blur": "blur",
    "snow": "weather", "frost": "weather", "fog": "weather",
    "brightness": "weather", "contrast": "weather",
    "elastic_transform": "digital", "pixelate": "digital",
    "jpeg_compression": "digital",
}
CATEGORY_COLOR = {
    "noise": "#dc2626", "blur": "#ea580c",
    "weather": "#2563eb", "digital": "#16a34a",
}

DS_KEYS = ["VOC20_matched", "Cityscapes"]
DS_STYLE = {
    "VOC20_matched": dict(linestyle="-", marker="o", label="VOC20"),
    "Cityscapes": dict(linestyle="--", marker="s", label="Cityscapes"),
}


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["severity"] = int(r["severity"])
        r["miou"] = float(r["miou"])
    return rows


def _series(rows):
    """(dataset, corruption) -> {severity: miou}."""
    s = defaultdict(dict)
    for r in rows:
        s[(r["dataset"], r["corruption"])][r["severity"]] = r["miou"]
    return s


def _present_corruptions(series):
    """Corruptions present in the CSV, kept in CORRUPTION_ORDER order."""
    have = {corr for (_ds, corr) in series}
    ordered = [c for c in CORRUPTION_ORDER if c in have]
    # tolerate any corruption not in the canonical list (appended at the end)
    ordered += sorted(c for c in have if c not in CORRUPTION_ORDER)
    return ordered


# --------------------------------------------------------------------------- #
# Figure 1 — normalized retention small-multiples (MAIN trend comparison)
# --------------------------------------------------------------------------- #
def plot_retention_grid(rows, out_path):
    series = _series(rows)
    corrs = _present_corruptions(series)
    n = len(corrs)
    ncols = 5
    nrows = max(1, math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows),
                             dpi=160, sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")

    for i, corr in enumerate(corrs):
        ax = axes[i // ncols][i % ncols]
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        ax.axhline(100, color="#9ca3af", linewidth=0.8, zorder=1)
        for ds in DS_KEYS:
            sev_map = series.get((ds, corr))
            if not sev_map or 1 not in sev_map:
                continue
            xs = sorted(sev_map)
            base = sev_map[1]
            ys = [sev_map[s] / base * 100 for s in xs]
            st = DS_STYLE[ds]
            ax.plot(xs, ys, color="#111827", linewidth=1.8,
                    linestyle=st["linestyle"], marker=st["marker"],
                    markersize=4, zorder=4)
        cat_c = CATEGORY_COLOR.get(CATEGORY.get(corr, ""), "#111827")
        ax.set_title(corr, fontsize=9, color=cat_c, fontweight="bold")
        ax.set_xticks([1, 2, 3, 4, 5])

    # hide any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    for r in range(nrows):
        axes[r][0].set_ylabel("retention % (vs sev1)", fontsize=9)
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("severity", fontsize=9)

    handles = [Line2D([], [], color="#111827", linestyle=DS_STYLE[ds]["linestyle"],
                      marker=DS_STYLE[ds]["marker"], label=DS_STYLE[ds]["label"])
               for ds in DS_KEYS]
    fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=True,
               facecolor="white", edgecolor="#d1d5db")
    fig.suptitle("Exp 3 — Severity decay shape (mIoU normalized to severity 1)\n"
                 "flat ⇒ CLIP already robust (H1)  ·  steep ⇒ genuinely OOD",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — absolute mIoU, two panels (own y-axis each)
# --------------------------------------------------------------------------- #
def plot_decay_panels(rows, out_path):
    series = _series(rows)
    corrs = _present_corruptions(series)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=160, squeeze=False)
    fig.patch.set_facecolor("white")

    for ax, ds in zip(axes[0], DS_KEYS):
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        for corr in corrs:
            sev_map = series.get((ds, corr))
            if not sev_map:
                continue
            xs = sorted(sev_map)
            ys = [sev_map[s] for s in xs]
            color = CATEGORY_COLOR.get(CATEGORY.get(corr, ""), "#475569")
            ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.6,
                    color=color, alpha=0.85, label=corr, zorder=4)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Corruption severity", fontsize=11)
        ax.set_title(DS_STYLE[ds]["label"], fontsize=12, fontweight="bold")

    axes[0][0].set_ylabel("Source mIoU (no adaptation)", fontsize=11)

    # native references on the Cityscapes panel (shared 19-class scale)
    city_ax = axes[0][1]
    for name, v in NATIVE_REFS.items():
        city_ax.axhline(v, color="#6b7280", linestyle=":", linewidth=1.2, zorder=2)
        city_ax.annotate(f"{name} native={v:.1f}", xy=(5, v), fontsize=7,
                         color="#6b7280", va="bottom", ha="right")

    # one shared legend (corruptions colored by category)
    cat_handles = [Line2D([], [], color=c, linewidth=2.5, label=cat)
                   for cat, c in CATEGORY_COLOR.items()]
    corr_handles = [Line2D([], [], color=CATEGORY_COLOR[CATEGORY[c]], linewidth=1.6,
                           label=c) for c in corrs]
    axes[0][0].legend(handles=corr_handles, loc="best", fontsize=7, ncol=2,
                      frameon=True, facecolor="white", edgecolor="#d1d5db")
    fig.suptitle("Exp 3 — Source mIoU vs severity (absolute scale; note panels "
                 "have different y-ranges)", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — grouped relative-drop bar (VOC vs Cityscapes per corruption)
# --------------------------------------------------------------------------- #
def plot_relative_drop_grouped(rows, out_path):
    series = _series(rows)
    corrs = _present_corruptions(series)

    def rel_drop(ds, corr):
        sm = series.get((ds, corr))
        if not sm or 1 not in sm or 5 not in sm or sm[1] <= 0:
            return None
        return (sm[1] - sm[5]) / sm[1] * 100

    items = []
    for corr in corrs:
        v = rel_drop("VOC20_matched", corr)
        c = rel_drop("Cityscapes", corr)
        mean = np.mean([x for x in (v, c) if x is not None]) if (v is not None or c is not None) else 0
        items.append((corr, v, c, mean))
    items.sort(key=lambda x: x[3])  # ascending: most robust first

    if not items:
        print("[plot] skip relative_drop_grouped — need sev1 and sev5 rows")
        return

    labels = [it[0] for it in items]
    voc = [it[1] if it[1] is not None else 0 for it in items]
    city = [it[2] if it[2] is not None else 0 for it in items]
    xs = np.arange(len(items))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(9, 0.65 * len(items) + 4), 5.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.bar(xs - w / 2, voc, width=w, color="#0ea5e9", label="VOC20", alpha=0.9)
    ax.bar(xs + w / 2, city, width=w, color="#dc2626", label="Cityscapes", alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Relative mIoU drop sev1→5  (%)", fontsize=11)
    ax.set_title("Exp 3 — Relative mIoU drop per corruption (sorted by mean)\n"
                 "small ⇒ CLIP already robust (H1 supported)  ·  large ⇒ OOD",
                 fontsize=11, pad=10)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/h1_validation/results/exp3/all.csv")
    ap.add_argument("--out_dir", default="experiments/h1_validation/results/exp3/figs/")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    print(f"[plot] {len(rows)} rows")
    plot_retention_grid(rows, os.path.join(args.out_dir, "severity_retention_grid"))
    plot_decay_panels(rows, os.path.join(args.out_dir, "severity_decay_panels"))
    plot_relative_drop_grouped(rows, os.path.join(args.out_dir, "relative_drop_grouped"))
    print(f"[plot] Saved 3 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
