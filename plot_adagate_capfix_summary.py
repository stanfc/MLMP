"""AdaGate VOC20 tail-decline fix: two figures.

  1. VOC20 mIoU-vs-round for all 5 arms (the target problem, real trajectories).
  2. Grouped bar chart: Delta-mean vs flagship, for each of the 4 growing variants,
     across all 3 datasets (the trade-off summary: VOC20 benefit vs ACDC/Cityscapes cost).

Outputs: figures/adagate_capfix_voc20_trajectories.{png,svg}
         figures/adagate_capfix_tradeoff_summary.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 24, "axes.labelsize": 30, "axes.titlesize": 23,
                     "xtick.labelsize": 22, "ytick.labelsize": 22,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


ARMS = [
    ("flagship (no cap fix)",              "ABmad05_ecdf",              "#7f7f7f", 2, 3.0),
    ("growing (A)",                        "ABmad05_ecdf_grow",         "#ff7f0e", 4, 2.4),
    ("growing_scaled (prop.1)",            "ABmad05_ecdf_grow_scaled",  "#1f77b4", 5, 2.4),
    ("growing_hmargin (prop.2)",           "ABmad05_ecdf_grow_hmargin", "#2ca02c", 6, 2.4),
    ("growing_hmargin_scaled (prop.3)",    "ABmad05_ecdf_grow_hmscaled","#d62728", 7, 3.0),
]

# ============================================================ Figure 1: VOC20 trajectories
fig, ax = plt.subplots(figsize=(15, 9))
legend_handles = []
for label, key, color, z, lw in ARMS:
    path = f"save/PascalVOC20Dataset/v20_acdc_matched/adagate_{key}/results_all_rounds.txt"
    r, m = load(path)
    ax.plot(r, m, color=color, lw=lw, zorder=z)
    legend_handles.append(Line2D([], [], color=color, lw=lw,
                                 label=f"{label}: mean={m.mean():.2f}"))

ax.set_xlabel("Round")
ax.set_ylabel("VOC20 mIoU")
ax.set_xlim(0, 152)
ax.set_ylim(71, 80)
ax.tick_params(width=1.6, length=8)
ax.grid(True, alpha=0.25, lw=1.0)
ax.set_title("VOC20: the target problem -- does the fix keep the ceiling AND stop the tail decline?", pad=14)
ax.legend(handles=legend_handles, loc="lower right", fontsize=18, framealpha=0.92)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/adagate_capfix_voc20_trajectories.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote adagate_capfix_voc20_trajectories.png")

# ============================================================ Figure 2: trade-off summary
DATASETS = [
    ("ACDC", "save/ACDCDataset/adagate_{}/results_all_rounds.txt"),
    ("VOC20", "save/PascalVOC20Dataset/v20_acdc_matched/adagate_{}/results_all_rounds.txt"),
    ("Cityscapes", "save/CityscapesDataset/adagate_{}/results_all_rounds.txt"),
]
VARIANTS = ARMS[1:]  # exclude flagship itself (it's the zero-reference line)

deltas = {}  # (ds_name, variant_label) -> delta mean
flagship_means = {}
for ds_name, tmpl in DATASETS:
    _, m = load(tmpl.format("ABmad05_ecdf"))
    flagship_means[ds_name] = m.mean()
    for label, key, color, z, lw in VARIANTS:
        _, mv = load(tmpl.format(key))
        deltas[(ds_name, label)] = mv.mean() - m.mean()

fig, ax = plt.subplots(figsize=(16, 9))
n_ds = len(DATASETS)
n_var = len(VARIANTS)
bar_w = 0.18
x = np.arange(n_ds)

for i, (label, key, color, z, lw) in enumerate(VARIANTS):
    offsets = (i - (n_var - 1) / 2) * bar_w
    vals = [deltas[(ds_name, label)] for ds_name, _ in DATASETS]
    ax.bar(x + offsets, vals, width=bar_w * 0.92, color=color, label=label, zorder=3)

ax.axhline(0, color="#444444", lw=1.8, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels([f"{name}\n(flagship mean={flagship_means[name]:.2f})" for name, _ in DATASETS])
ax.set_ylabel(r"$\Delta$ mean mIoU vs flagship")
ax.tick_params(width=1.6, length=8)
ax.grid(True, alpha=0.25, lw=1.0, axis="y")
ax.set_title("Trade-off: VOC20 gain vs ACDC/Cityscapes cost, per variant", pad=14)
ax.legend(loc="lower left", fontsize=17, framealpha=0.92, ncol=1)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/adagate_capfix_tradeoff_summary.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote adagate_capfix_tradeoff_summary.png")
