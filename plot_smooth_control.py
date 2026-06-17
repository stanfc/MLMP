#!/usr/bin/env python
"""
plot_smooth_control.py — the confound control: is smooth-anchor genuinely better
than source-reset, or was the earlier "win" just a poorly-tuned source-reset gate?

We re-ran source-reset on VOC20 (subset-101, weather-5, 150R) at RAISED thresholds,
including one config (SR-match) whose gate geometry is IDENTICAL to the winning
smooth config (ceil2.3/floor2.2, rst0.01) so the ONLY difference is the restoration
TARGET (frozen source vs recent snapshot).

Outputs (save/_compare/control_*):
  control_voc20_bars         all configs, mean + R150, smooth highlighted
  control_voc20_trajectories smooth floor2.2 vs SR-match vs SR-best
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "save/_compare"
V = "save/PascalVOC20Dataset"

# label : (dir, kind)   kind in {"orig","sr","smooth"}
RUNS = [
    ("source-reset\norig (h1.6/1.4)",       f"{V}/tent_divgate_continual_sub101_weather",      "orig"),
    ("source-reset\nt2.0 (h2.0/1.8)",       f"{V}/srctrl_t2.0_w1.8_r0.01",                     "sr"),
    ("source-reset\nMATCH (h2.3/2.2)",      f"{V}/srctrl_match_h2.3_w2.2_r0.01",               "sr"),
    ("source-reset\nt2.5+brake (h2.5/2.2)", f"{V}/srctrl_t2.5_w2.2_brake0.05",                 "sr"),
    ("SMOOTH-ANCHOR\nfloor2.2 (ceil2.3)",   f"{V}/tdsa_Dtune_sub101_ceil2.3_floor2.2",         "smooth"),
]


def load_means(d):
    p = os.path.join(d, "results_all_rounds.txt")
    if not os.path.isfile(p):
        return []
    out, first = [], True
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        if first:
            first = False
            continue
        try:
            out.append(float(line.split(",")[-1]))
        except ValueError:
            pass
    return out


def _save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def bars():
    labels, mean_v, rl_v, kinds = [], [], [], []
    for lab, d, kind in RUNS:
        m = load_means(d)
        if not m:
            continue
        labels.append(lab); mean_v.append(np.mean(m)); rl_v.append(m[-1]); kinds.append(kind)
    cmap = {"orig": "#bbbbbb", "sr": "#1f77b4", "smooth": "#2ca02c"}
    cols = [cmap[k] for k in kinds]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, mean_v, w, color=cols, label="all-round mean")
    b2 = ax.bar(x + w/2, rl_v, w, color=cols, alpha=0.45, label="R150")
    smooth_mean = mean_v[kinds.index("smooth")]
    ax.axhline(smooth_mean, color="#2ca02c", ls="--", lw=1.4, label=f"smooth-anchor mean = {smooth_mean:.2f}")
    for b, v in list(zip(b1, mean_v)) + list(zip(b2, rl_v)):
        ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mIoU (VOC20 subset-101, weather-5, 150R)")
    ax.set_ylim(55, 68)
    ax.set_title("CONTROL: a gate-matched source-reset TIES smooth-anchor (64.62 vs 64.70);\n"
                 "a tuned source-reset (t2.5+brake) BEATS it (65.92). The lever is gate-band tuning, not the anchor target.",
                 fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); _save(fig, "control_voc20_bars")


def trajectories():
    keep = [("source-reset MATCH (h2.3/2.2)", f"{V}/srctrl_match_h2.3_w2.2_r0.01", "#1f77b4", "-"),
            ("source-reset t2.5+brake (best)", f"{V}/srctrl_t2.5_w2.2_brake0.05", "#9467bd", "-"),
            ("smooth-anchor floor2.2",         f"{V}/tdsa_Dtune_sub101_ceil2.3_floor2.2", "#2ca02c", "-"),
            ("source-reset orig (h1.6/1.4)",   f"{V}/tent_divgate_continual_sub101_weather", "#bbbbbb", "--")]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for lab, d, c, ls in keep:
        m = load_means(d)
        if not m:
            continue
        ax.plot(range(1, len(m)+1), m, color=c, ls=ls, lw=1.9, label=f"{lab}  (mean {np.mean(m):.2f})")
    ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU")
    ax.set_title("VOC20: smooth-anchor and gate-matched source-reset are indistinguishable",
                 fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, "control_voc20_trajectories")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bars()
    trajectories()
    print("Control figures written to", OUT)
