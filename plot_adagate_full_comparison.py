"""Full comparison: no-adapt, MLMP episodic, old GDG-PA (ctrl), AdaGate flagship,
and the 4 shallow-cap variants, on all 3 datasets. ACDC now uses the FULL dataset
(400 img/round, subset bug fixed); VOC20/Cityscapes use 5corr + subset_size=100
(unchanged, always correct).

Output: figures/adagate_full_comparison_3datasets.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 21, "axes.labelsize": 26, "axes.titlesize": 20,
                     "xtick.labelsize": 19, "ytick.labelsize": 19,
                     "axes.linewidth": 1.5, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load_continual(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


def load_episodic_mean(path):
    vals = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(("mIoU", "GPU", "Total", "Mean")):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vals.append(float(parts[1].split("+/-")[0].strip()))
        except (ValueError, IndexError):
            continue
    return np.mean(vals) if vals else None


ARMS = [
    ("ctrl (old GDG-PA)",     "adagate_ctrl",                     "#9467bd", 2.6, 4),
    ("flagship",               "adagate_ABmad05_ecdf",             "#7f7f7f", 3.2, 5),
    ("growing (A)",            "adagate_ABmad05_ecdf_grow",        "#ff7f0e", 2.2, 3),
    ("growing_scaled (P1)",    "adagate_ABmad05_ecdf_grow_scaled", "#1f77b4", 2.2, 3),
    ("growing_hmargin (P2)",   "adagate_ABmad05_ecdf_grow_hmargin","#2ca02c", 2.2, 3),
    ("growing_hmscaled (P3)",  "adagate_ABmad05_ecdf_grow_hmscaled","#d62728", 3.2, 6),
]

PANELS = [
    ("ACDC (full, 400/round)", "save/ACDCDataset", "acdc_no_adapt",
     "mlmp_episodic_matched", (27, 33)),
    ("VOC20 (5corr, sub100)", "save/PascalVOC20Dataset/v20_acdc_matched", "no_adapt",
     "mlmp_episodic_5corr", (70, 80)),
    ("Cityscapes (5corr, sub100)", "save/CityscapesDataset", "cityscapes_no_adapt",
     "mlmp_episodic_5corr", (21, 25)),
]

fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))

for ax, (title, root, noadapt_dir, episodic_dir, ylim) in zip(axes, PANELS):
    _, m_na = load_continual(f"{root}/{noadapt_dir}/results_all_rounds.txt")
    na_mean = m_na.mean()
    ep_mean = load_episodic_mean(f"{root}/{episodic_dir}/results.txt")

    ax.axhline(na_mean, color="black", ls=":", lw=2.0, zorder=2)
    ax.axhline(ep_mean, color="#555555", ls="-.", lw=2.0, zorder=2)

    legend_handles = [
        Line2D([], [], color="black", ls=":", lw=2.0, label=f"no-adapt: {na_mean:.2f}"),
        Line2D([], [], color="#555555", ls="-.", lw=2.0, label=f"MLMP episodic: {ep_mean:.2f}"),
    ]
    for label, key, color, lw, z in ARMS:
        r, m = load_continual(f"{root}/{key}/results_all_rounds.txt")
        ax.plot(r, m, color=color, lw=lw, zorder=z)
        legend_handles.append(Line2D([], [], color=color, lw=lw, label=f"{label}: {m.mean():.2f}"))

    ax.set_xlabel("Round")
    ax.set_ylabel("mIoU")
    ax.set_xlim(0, 152)
    ax.set_ylim(*ylim)
    ax.tick_params(width=1.5, length=7)
    ax.grid(True, alpha=0.25, lw=0.9)
    ax.set_title(title, pad=10)
    ax.legend(handles=legend_handles, loc="lower right", fontsize=13, framealpha=0.92)

fig.suptitle("AdaGate variants vs old GDG-PA, no-adapt, and MLMP episodic (mean = legend value)",
            fontsize=22, y=1.03)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/adagate_full_comparison_3datasets.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote adagate_full_comparison_3datasets.png")
