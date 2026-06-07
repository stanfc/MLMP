#!/usr/bin/env python
"""
plot_smooth_vs_source.py — compare tent_divgate_SMOOTH_ANCHOR vs the
source-reset tent_divgate_continual baseline across datasets.

Produces:
  save/_compare/smooth_vs_source_trajectories.{png,svg}  (Round vs Mean mIoU grid)
  save/_compare/smooth_vs_source_summary.{png,svg}       (grouped bar of all-round mean)

Run from repo root:  python plot_smooth_vs_source.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (label, source-reset dir, smooth-anchor dir)
PAIRS = [
    ("ACDC",        "save/ACDCDataset/tent_divgate_continual_cau_rst_0.01",
                    "save/ACDCDataset/smooth_anchor_ceil1.6_floor1.4"),
    ("VOC20",       "save/PascalVOC20Dataset/tent_divgate_continual_threshold_1.6_weather",
                    "save/PascalVOC20Dataset/tent_divgate_smooth_anchor_ceil1.6_floor1.4_weather"),
    ("Cityscapes",  "save/CityscapesDataset/tent_divgate_continual_weather_threshold_2.0",
                    "save/CityscapesDataset/tent_divgate_smooth_anchor_ceil2.0_floor1.4_weather"),
    ("Dark Zurich", "save/DarkZurichDataset/tent_divgate_continual",
                    "save/DarkZurichDataset/tent_divgate_smooth_anchor"),
    ("Nighttime Driving", "save/NighttimeDrivingDataset/tent_divgate_continual",
                    "save/NighttimeDrivingDataset/tent_divgate_smooth_anchor"),
    ("DZ+ND Combined", "save/DZ_ND_Combined/tent_divgate_continual",
                    "save/DZ_ND_Combined/tent_divgate_smooth_anchor"),
]

SRC_C, SM_C = "#1f77b4", "#d62728"   # blue = source-reset, red = smooth-anchor
OUT = "save/_compare"


def load_means(run_dir):
    """results_all_rounds.txt -> list of per-round Mean_mIoU (last column)."""
    path = os.path.join(run_dir, "results_all_rounds.txt")
    if not os.path.isfile(path):
        return []
    out, first = [], True
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:                      # header
                first = False
                continue
            try:
                out.append(float(line.split(",")[-1]))
            except ValueError:
                pass
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = []
    for label, src_dir, sm_dir in PAIRS:
        src = load_means(src_dir)
        sm = load_means(sm_dir)
        data.append((label, src, sm))

    # ---------- Figure 1: trajectories ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (label, src, sm) in zip(axes.flat, data):
        if src:
            ax.plot(range(1, len(src) + 1), src, color=SRC_C, lw=1.6,
                    label=f"source-reset (n={len(src)})")
        if sm:
            ax.plot(range(1, len(sm) + 1), sm, color=SM_C, lw=1.6, ls="--",
                    label=f"smooth-anchor (n={len(sm)})")
        # fair all-round mean over the common (shorter) horizon
        n = min(len(src), len(sm)) if src and sm else 0
        sub = ""
        if n:
            ms, mm = sum(src[:n]) / n, sum(sm[:n]) / n
            sub = f"   mean@{n}R: src={ms:.2f}  smooth={mm:.2f}  Δ={mm-ms:+.2f}"
        ax.set_title(f"{label}{sub}", fontsize=10)
        ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.suptitle("Smooth-anchor vs source-reset DivGate (matched gate geometry)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/smooth_vs_source_trajectories.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: summary grouped bar (all-round mean over common horizon) ----------
    labels, src_means, sm_means, ns = [], [], [], []
    for label, src, sm in data:
        if not (src and sm):
            continue
        n = min(len(src), len(sm))
        labels.append(f"{label}\n(@{n}R)")
        src_means.append(sum(src[:n]) / n)
        sm_means.append(sum(sm[:n]) / n)
        ns.append(n)
    x = range(len(labels)); w = 0.38
    fig2, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar([i - w/2 for i in x], src_means, w, color=SRC_C, label="source-reset")
    b2 = ax.bar([i + w/2 for i in x], sm_means, w, color=SM_C, label="smooth-anchor")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.2,
                    f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    # delta annotations
    for i, (s, m) in enumerate(zip(src_means, sm_means)):
        ax.text(i, max(s, m) + 1.8, f"Δ={m-s:+.2f}", ha="center", fontsize=9,
                color=(SM_C if m < s else "green"), fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("All-round mean mIoU (common horizon)")
    ax.set_title("Smooth-anchor vs source-reset — all-round mean mIoU", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    for ext in ("png", "svg"):
        fig2.savefig(f"{OUT}/smooth_vs_source_summary.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    # ---------- console summary ----------
    print(f"{'dataset':<20}{'n':>6}{'src':>9}{'smooth':>9}{'delta':>9}")
    for label, src, sm in data:
        if not (src and sm):
            print(f"{label:<20}{'--':>6}  (incomplete: src={len(src)} sm={len(sm)})")
            continue
        n = min(len(src), len(sm))
        ms, mm = sum(src[:n])/n, sum(sm[:n])/n
        print(f"{label:<20}{n:>6}{ms:>9.2f}{mm:>9.2f}{mm-ms:>+9.2f}")
    print(f"\nFigures -> {OUT}/smooth_vs_source_trajectories.{{png,svg}}, "
          f"{OUT}/smooth_vs_source_summary.{{png,svg}}")


if __name__ == "__main__":
    main()
