"""1) Stats: when does H_margin's running peak (h_max) actually stop updating,
   across all 4 dataset protocols, for both hmargin_uncapped and hmargin_scaled?
2) Figure: round-level mIoU vs round-mean H_margin, V20-15corr (the dataset where
   the peak keeps moving for the longest -- most dynamic case), colored by round.

Output: figures/hmargin_peak_stats.txt, figures/miou_vs_hmargin_v20_15corr.{png,svg}
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18, "axes.labelsize": 21, "axes.titlesize": 17,
                     "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "axes.linewidth": 1.3, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load_gate_log(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_results(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


ARMS = [("hmargin_uncapped", "ABmad05_ecdf_grow_hmargin", "#2ca02c"),
        ("hmargin_scaled", "ABmad05_ecdf_grow_hmscaled", "#d62728")]

# ---------------------------------------------------------- 1) peak-timing stats
DATASETS = [
    ("V20-15corr", "save/PascalVOC20Dataset/v20_15corr"),
    ("V20-5corr", "save/PascalVOC20Dataset/v20_acdc_matched"),
    ("ACDC", "save/ACDCDataset"),
    ("Cityscapes-5corr", "save/CityscapesDataset"),
]
lines = [f"{'dataset':18s} {'arm':16s} {'windows':>8s} {'last_peak_win':>14s} {'%_through':>10s} {'n_new_highs':>12s}"]
for dsname, root in DATASETS:
    for armname, key, _ in ARMS:
        gl = load_gate_log(f"{root}/adagate_{key}/gate_log.csv")
        hs = [row["h_margin"] for row in gl]
        running_max, last_peak_idx, n_updates = -1, 0, 0
        for i, h in enumerate(hs):
            if h >= running_max:
                running_max, last_peak_idx, n_updates = h, i, n_updates + 1
        pct = 100 * last_peak_idx / len(hs)
        lines.append(f"{dsname:18s} {armname:16s} {len(hs):8d} {last_peak_idx:14d} {pct:9.1f}% {n_updates:12d}")
stats_txt = "\n".join(lines)
print(stats_txt)
with open(f"{OUT}/hmargin_peak_stats.txt", "w") as f:
    f.write(stats_txt + "\n")

# ---------------------------------------------------------- 2) mIoU vs H_margin, V20-15corr
root = "save/PascalVOC20Dataset/v20_15corr"
WINDOWS_PER_ROUND = 30  # 1500 img/round / monitor_interval=50

fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), sharey=True)
for ax, (armname, key, color) in zip(axes, ARMS):
    gl = load_gate_log(f"{root}/adagate_{key}/gate_log.csv")
    h_by_round = {}
    for i, row in enumerate(gl):
        rnd = i // WINDOWS_PER_ROUND + 1
        h_by_round.setdefault(rnd, []).append(row["h_margin"])
    rounds_h = sorted(h_by_round)
    h_mean = np.array([np.mean(h_by_round[r]) for r in rounds_h])

    r_m, miou = load_results(f"{root}/adagate_{key}/results_all_rounds.txt")
    # align by round number (both start at round 1)
    common = sorted(set(rounds_h) & set(r_m))
    h_plot = np.array([h_mean[rounds_h.index(r)] for r in common])
    m_plot = np.array([miou[list(r_m).index(r)] for r in common])

    sc = ax.scatter(h_plot, m_plot, c=common, cmap="viridis", s=22, zorder=3)
    ax.set_xlabel("mean H_margin (this round)")
    ax.set_title(f"{armname}", pad=10)
    ax.grid(True, alpha=0.25, lw=0.8)
    ax.tick_params(width=1.3, length=6)

axes[0].set_ylabel("VOC20 mIoU")
cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Round")
fig.suptitle("V20-15corr: mIoU vs H_margin per round (color = round number)", y=1.02, fontsize=19)

for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/miou_vs_hmargin_v20_15corr.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote miou_vs_hmargin_v20_15corr.png")
