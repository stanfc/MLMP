#!/usr/bin/env python
"""
plot_smooth_generality.py — does the ACDC "D principle" (raise floor →
retreat-to-source sooner → smooth-anchor beats source-reset) generalise to
VOC20 and Cityscapes?

For each dataset we ran a floor mini-sweep at fixed ceil, on a weather/4-corr
subset-101 stream (150 rounds), plus a subset-matched source-reset baseline.

Outputs (save/_compare/gen_*):
  gen_voc20_trajectories     source-reset vs floor sweep, round vs mean mIoU
  gen_voc20_bars             all-round mean + R150 bars
  gen_cityscapes_trajectories
  gen_cityscapes_bars
  gen_floor_vs_miou          THE money panel: floor -> mean mIoU, both datasets
  gen_hmargin_vs_floor       floor -> H_margin median (mechanism: high floor keeps model healthy)

Run from repo root:  python plot_smooth_generality.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "save/_compare"

# dataset -> list of (label, dir, floor, is_baseline)
DSETS = {
    "VOC20": {
        "title": "VOC20 (subset-101, weather-5, 150R)",
        "runs": [
            ("source-reset (h1.6/1.4)", "save/PascalVOC20Dataset/tent_divgate_continual_sub101_weather", None, True),
            ("smooth floor1.8",         "save/PascalVOC20Dataset/tdsa_Dtune_sub101_ceil2.3_floor1.8",    1.8, False),
            ("smooth floor2.0",         "save/PascalVOC20Dataset/tdsa_Dtune_sub101_ceil2.3_floor2.0",    2.0, False),
            ("smooth floor2.2",         "save/PascalVOC20Dataset/tdsa_Dtune_sub101_ceil2.3_floor2.2",    2.2, False),
        ],
    },
    "Cityscapes": {
        "title": "Cityscapes (subset-101, 4-corr, 150R)",
        "runs": [
            ("source-reset (h2.4/2.1)", "save/CityscapesDataset/tent_divgate_continual_sub101_4corr_thr2.4", None, True),
            ("smooth floor1.9",         "save/CityscapesDataset/tent_divgate_smooth_anchor_Dtune_city_ceil2.4_floor1.9", 1.9, False),
            ("smooth floor2.1",         "save/CityscapesDataset/tent_divgate_smooth_anchor_Dtune_city_ceil2.4_floor2.1", 2.1, False),
            ("smooth floor2.3",         "save/CityscapesDataset/tent_divgate_smooth_anchor_Dtune_city_ceil2.4_floor2.3", 2.3, False),
        ],
    },
}


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


def load_hmargin_median(d):
    ep = os.path.join(d, "entropy_log.csv")
    if not os.path.isfile(ep):
        return None
    v = []
    with open(ep, newline="") as f:
        for row in csv.DictReader(f):
            try:
                v.append(float(row["h_margin"]))
            except (KeyError, ValueError):
                pass
    return float(np.median(v)) if v else None


def _save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def trajectories(dset_key, fname):
    cfg = DSETS[dset_key]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    greens = ["#a1d99b", "#41ab5d", "#006d2c"]
    gi = 0
    base_mean = None
    for label, d, floor, is_base in cfg["runs"]:
        m = load_means(d)
        if not m:
            continue
        if is_base:
            base_mean = np.mean(m)
            ax.plot(range(1, len(m) + 1), m, color="#333", ls="--", lw=2.2,
                    label=f"{label}  (mean {np.mean(m):.2f}, R150 {m[-1]:.2f})")
        else:
            ax.plot(range(1, len(m) + 1), m, color=greens[gi % 3], lw=1.8,
                    label=f"{label}  (mean {np.mean(m):.2f}, R150 {m[-1]:.2f})")
            gi += 1
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean mIoU")
    ttl = f"{cfg['title']}: higher floor → retreat-to-source sooner → holds longer"
    ax.set_title(ttl, fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, fname)


def bars(dset_key, fname):
    cfg = DSETS[dset_key]
    labels, mean_v, rl_v, base = [], [], [], None
    for label, d, floor, is_base in cfg["runs"]:
        m = load_means(d)
        if not m:
            continue
        labels.append(label.replace("smooth ", "").replace("source-reset ", "src-reset\n"))
        mean_v.append(np.mean(m))
        rl_v.append(m[-1])
        if is_base:
            base = np.mean(m)
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = ["#888" if "src-reset" in l else ("#2ca02c" if (base and v > base) else "#d62728")
            for l, v in zip(labels, mean_v)]
    b1 = ax.bar(x - w / 2, mean_v, w, color=cols, label="all-round mean")
    b2 = ax.bar(x + w / 2, rl_v, w, color=cols, alpha=0.45, label="R150")
    if base:
        ax.axhline(base, color="#333", ls="--", lw=1.3, label=f"source-reset mean = {base:.2f}")
    for b, v in list(zip(b1, mean_v)) + list(zip(b2, rl_v)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mIoU")
    ax.set_title(f"{cfg['title']}", fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, fname)


def floor_vs_miou():
    """The money panel: floor (relative to H median) -> mean mIoU, both datasets."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cols = {"VOC20": "#1f77b4", "Cityscapes": "#ff7f0e"}
    for dk, cfg in DSETS.items():
        xs, ys, base = [], [], None
        for label, d, floor, is_base in cfg["runs"]:
            m = load_means(d)
            if not m:
                continue
            if is_base:
                base = np.mean(m)
                continue
            xs.append(floor)
            ys.append(np.mean(m))
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        ax.plot(xs, ys, "o-", color=cols[dk], lw=2, ms=8, label=f"{dk} smooth-anchor")
        if base is not None:
            ax.axhline(base, color=cols[dk], ls="--", lw=1.2, alpha=0.7,
                       label=f"{dk} source-reset = {base:.2f}")
    ax.set_xlabel("h_floor  (retreat-to-source threshold; higher = retreat sooner)")
    ax.set_ylabel("All-round mean mIoU (150R)")
    ax.set_title("Generality of the D principle: raising the floor monotonically lifts mIoU\n"
                 "and overtakes source-reset on both datasets", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "gen_floor_vs_miou")


def hmargin_vs_floor():
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = {"VOC20": "#1f77b4", "Cityscapes": "#ff7f0e"}
    for dk, cfg in DSETS.items():
        xs, ys = [], []
        for label, d, floor, is_base in cfg["runs"]:
            if is_base:
                continue
            hm = load_hmargin_median(d)
            if hm is None:
                continue
            xs.append(floor)
            ys.append(hm)
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        ax.plot(xs, ys, "s-", color=cols[dk], lw=2, ms=8, label=dk)
    ax.set_xlabel("h_floor")
    ax.set_ylabel("H_margin median (model health)")
    ax.set_title("Mechanism: a higher floor keeps the model in a healthier (less collapsed) state",
                 fontweight="bold", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "gen_hmargin_vs_floor")


def main():
    os.makedirs(OUT, exist_ok=True)
    trajectories("VOC20", "gen_voc20_trajectories")
    bars("VOC20", "gen_voc20_bars")
    trajectories("Cityscapes", "gen_cityscapes_trajectories")
    bars("Cityscapes", "gen_cityscapes_bars")
    floor_vs_miou()
    hmargin_vs_floor()
    print("All generality figures written to", OUT)


if __name__ == "__main__":
    main()
