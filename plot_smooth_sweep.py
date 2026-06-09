#!/usr/bin/env python
"""
plot_smooth_sweep.py — ACDC smooth-anchor hyperparameter sweep vs source-reset.

Figures -> save/_compare/:
  smooth_sweep_bars.{png,svg}        all configs' all-round mean, sorted, with
                                     source-reset + matched reference lines
  smooth_sweep_trajectories.{png,svg} Round vs mean for key configs
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "save/ACDCDataset"
OUT = "save/_compare"
SOURCE = "tent_divgate_continual_cau_rst_0.01"     # source-reset baseline
MATCHED = "smooth_anchor_ceil1.6_floor1.4"          # matched-geometry smooth (loses)

# label : dir : family
CONFIGS = [
    ("source-reset (baseline)", SOURCE, "ref"),
    ("matched (ceil1.6/floor1.4)", MATCHED, "ref"),
    ("A: ceil1.8", "smooth_sweep_A_ceil1.8", "ceil"),
    ("A: ceil2.0", "smooth_sweep_A_ceil2.0", "ceil"),
    ("B: lag300", "smooth_sweep_B_lag300", "lag"),
    ("B: lag1000", "smooth_sweep_B_lag1000", "lag"),
    ("C: ceil2.0+lag1000", "smooth_sweep_C_ceil2.0_lag1000", "combo"),
    ("D: ceil1.8+floor1.55", "smooth_sweep_D_ceil1.8_floor1.55", "floor"),
    ("win: mon10", "smooth_sweep_win_mon10", "window"),
    ("win: mon25", "smooth_sweep_win_mon25", "window"),
    ("decoup: buf50/mon10", "smooth_sweep_decoup_buf50_mon10", "window"),
    ("decoup: buf50/mon5", "smooth_sweep_decoup_buf50_mon5", "window"),
]
FAM_COLOR = {"ref": "#555555", "ceil": "#1f77b4", "lag": "#2ca02c",
             "combo": "#9467bd", "floor": "#d62728", "window": "#ff7f0e"}


def load_means(dirname):
    path = os.path.join(BASE, dirname, "results_all_rounds.txt")
    if not os.path.isfile(path):
        return []
    out, first = [], True
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False; continue
            try:
                out.append(float(line.split(",")[-1]))
            except ValueError:
                pass
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []   # (label, family, means)
    for label, d, fam in CONFIGS:
        m = load_means(d)
        if m:
            rows.append((label, fam, m))
        else:
            print(f"  (skip, no results yet: {d})")

    src_mean = next((sum(m)/len(m) for l, f, m in rows if l.startswith("source-reset")), None)
    matched_mean = next((sum(m)/len(m) for l, f, m in rows if l.startswith("matched")), None)

    # ---------- Figure 1: sorted horizontal bars ----------
    stats = [(l, f, sum(m)/len(m), m[-1]) for l, f, m in rows]
    stats.sort(key=lambda x: x[2])
    fig, ax = plt.subplots(figsize=(11, 7))
    ys = range(len(stats))
    ax.barh(list(ys), [s[2] for s in stats],
            color=[FAM_COLOR[s[1]] for s in stats])
    for i, s in enumerate(stats):
        ax.text(s[2] + 0.03, i, f"{s[2]:.2f} (R150 {s[3]:.1f})", va="center", fontsize=8)
    ax.set_yticks(list(ys)); ax.set_yticklabels([s[0] for s in stats], fontsize=9)
    if src_mean is not None:
        ax.axvline(src_mean, color="#555555", ls="--", lw=1.4,
                   label=f"source-reset = {src_mean:.2f}")
    if matched_mean is not None:
        ax.axvline(matched_mean, color="#d62728", ls=":", lw=1.2, alpha=0.7,
                   label=f"matched-smooth = {matched_mean:.2f}")
    ax.set_xlabel("All-round mean mIoU (ACDC, 150R)")
    ax.set_title("Smooth-anchor sweep on ACDC — recovering the deficit vs source-reset",
                 fontweight="bold")
    ax.set_xlim(left=min(s[2] for s in stats) - 0.5)
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/smooth_sweep_bars.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: trajectories of key configs ----------
    KEY = ["source-reset (baseline)", "matched (ceil1.6/floor1.4)",
           "D: ceil1.8+floor1.55", "C: ceil2.0+lag1000", "A: ceil2.0"]
    KEYC = {"source-reset (baseline)": "#555555", "matched (ceil1.6/floor1.4)": "#d62728",
            "D: ceil1.8+floor1.55": "#2ca02c", "C: ceil2.0+lag1000": "#9467bd",
            "A: ceil2.0": "#1f77b4"}
    fig2, ax = plt.subplots(figsize=(11, 6))
    for label, fam, m in rows:
        if label in KEY:
            ls = "--" if label.startswith("matched") else "-"
            ax.plot(range(1, len(m)+1), m, color=KEYC[label], lw=1.7, ls=ls,
                    label=f"{label} (mean {sum(m)/len(m):.2f})")
    ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU")
    ax.set_title("ACDC trajectories: D (raise floor) holds, matched decays", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig2.tight_layout()
    for ext in ("png", "svg"):
        fig2.savefig(f"{OUT}/smooth_sweep_trajectories.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nFigures -> {OUT}/smooth_sweep_bars.{{png,svg}}, smooth_sweep_trajectories.{{png,svg}}")


if __name__ == "__main__":
    main()
