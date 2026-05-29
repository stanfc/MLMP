"""Exp 3 plots: severity decay curves + relative-drop bar.

Native datasets (no severity axis) appear as horizontal reference lines with
values hard-coded below — sourced from save/{Dataset}/No_Adaptation/ runs at
the time of the design spec (2026-05-28).
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Native source mIoU references (constants, see spec §7.5)
NATIVE_REFS = {
    "ACDC": 23.34,
    "DarkZurich": 20.24,
    "NighttimeDriving": 31.01,
}

VOC_COLORS = {
    "snow": "#0ea5e9", "fog": "#0891b2", "frost": "#16a34a",
    "contrast": "#9333ea", "brightness": "#ea580c",
}
CITY_COLORS = {
    "snow": "#0284c7", "fog": "#0e7490", "frost": "#15803d",
    "contrast": "#7e22ce", "brightness": "#c2410c",
}


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["severity"] = int(r["severity"])
        r["miou"] = float(r["miou"])
    return rows


def plot_decay(rows, out_path):
    series = defaultdict(list)
    for r in rows:
        series[(r["dataset"], r["corruption"])].append((r["severity"], r["miou"]))
    for k in series:
        series[k].sort()

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    handles = []
    for (ds, corr), pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        is_voc = ds == "VOC20_matched"
        color = (VOC_COLORS if is_voc else CITY_COLORS).get(corr, "#475569")
        ls = "-" if is_voc else "--"
        ax.plot(xs, ys, marker="o", color=color, linewidth=2.0, linestyle=ls, zorder=4)
        handles.append(Line2D([], [], color=color, linewidth=2.0, linestyle=ls,
                              label=f"{ds}/{corr}"))

    native_handles = []
    for name, v in NATIVE_REFS.items():
        ax.axhline(v, color="#6b7280", linestyle=":", linewidth=1.2, zorder=2)
        native_handles.append(Line2D([], [], color="#6b7280", linestyle=":",
                                     label=f"{name} (native, source={v:.2f})"))

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Corruption severity", fontsize=11)
    ax.set_ylabel("Source mIoU (no adaptation)", fontsize=11)
    ax.set_title("Exp 3 — Source mIoU vs synthetic corruption severity\n"
                 "(flat curve ⇒ CLIP pretraining covered this; steep ⇒ OOD)",
                 fontsize=11, pad=10)
    ax.legend(handles=handles + native_handles, loc="best",
              fontsize=8, frameon=True, facecolor="white", edgecolor="#d1d5db")
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_relative_drop(rows, out_path):
    by_pair: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        by_pair[(r["dataset"], r["corruption"])][r["severity"]] = r["miou"]
    items = []
    for (ds, corr), sev_map in by_pair.items():
        if 1 not in sev_map or 5 not in sev_map:
            continue
        m1, m5 = sev_map[1], sev_map[5]
        drop = (m1 - m5) / m1 if m1 > 0 else 0.0
        items.append((ds, corr, drop, m1, m5))
    items.sort(key=lambda x: x[2])

    if not items:
        print("[plot] skip relative_drop_bar — need rows at both sev=1 and sev=5")
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(items) + 4), 5.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    labels = [f"{ds}/{corr}" for ds, corr, _, _, _ in items]
    drops = [d for _, _, d, _, _ in items]
    colors = ["#0ea5e9" if ds == "VOC20_matched" else "#dc2626" for ds, _, _, _, _ in items]
    xs = np.arange(len(items))
    ax.bar(xs, drops, color=colors, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("(miou@sev1 − miou@sev5) / miou@sev1", fontsize=11)
    ax.set_title("Exp 3 — Relative mIoU drop from severity 1 → 5\n"
                 "(small drop ⇒ CLIP already robust ⇒ H1 supported)",
                 fontsize=11, pad=10)
    handles = [
        Line2D([], [], marker="s", linestyle="", color="#0ea5e9", label="VOC20_matched"),
        Line2D([], [], marker="s", linestyle="", color="#dc2626", label="Cityscapes"),
    ]
    ax.legend(handles=handles, loc="best")
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
    plot_decay(rows, os.path.join(args.out_dir, "severity_decay_curves"))
    plot_relative_drop(rows, os.path.join(args.out_dir, "relative_drop_bar"))
    print(f"[plot] Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
