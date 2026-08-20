"""Mid-stream difficulty jump: rounds 1-75 use easy5corr (snow/frost/fog/
brightness/contrast), rounds 76-150 switch to hard5corr (defocus_blur/
glass_blur/gaussian_noise/zoom_blur/elastic_transform). Does the gate handle
a difficulty jump mid-run, not just a fixed corruption set for 150 rounds?
ACDC excluded (real weather, no corruption concept -- can't do this switch).

Output: figures/easy2hard_{v20,cityscapes}.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 20, "axes.labelsize": 24, "axes.titlesize": 19,
                     "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "axes.linewidth": 1.4, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

ARMS = [
    ("GDG-PA",           "adagate_ctrl",                    "#4a3aa7", 2.2, 4),
    ("ABmad05_ecdf",     "adagate_ABmad05_ecdf",             "#888888", 2.2, 4),
    ("gradnorm_uncapped", "adagate_ABmad05_ecdf_grow",        "#eb6834", 2.2, 4),
    ("gradnorm_scaled",  "adagate_ABmad05_ecdf_grow_scaled", "#2a78d6", 3.0, 6),
]


def load(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


DATASETS = [
    ("v20", "save/PascalVOC20Dataset/v20_easy2hard", "V20 (easy5corr -> hard5corr)"),
    ("cityscapes", "save/CityscapesDataset/cityscapes_easy2hard", "Cityscapes (easy5corr -> hard5corr)"),
]

for key, root, title in DATASETS:
    fig, ax = plt.subplots(figsize=(14, 8.5))
    legend_handles = []
    for label, arm_dir, color, lw, z in ARMS:
        r, m = load(f"{root}/{arm_dir}/results_all_rounds.txt")
        ax.plot(r, m, color=color, lw=lw, zorder=z)
        legend_handles.append(Line2D([], [], color=color, lw=lw,
                                     label=f"{label}: R75={m[74]:.1f}->R76={m[75]:.1f}->R150={m[-1]:.1f}"))

    ax.axvline(75.5, color="black", lw=1.6, ls="--", alpha=0.6, zorder=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("mean mIoU")
    ax.set_xlim(0, 151)
    ax.tick_params(width=1.4, length=7)
    ax.grid(True, alpha=0.25, lw=0.9)
    ymin, ymax = ax.get_ylim()
    ax.text(75.5, ymax - 0.04 * (ymax - ymin), " easy5corr | hard5corr ",
            ha="center", va="top", fontsize=13, color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_title(f"{title}: mid-stream difficulty jump @ R76", pad=14)
    ax.legend(handles=legend_handles, loc="lower left", fontsize=14, framealpha=0.93)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/easy2hard_{key}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote easy2hard_{key}.png")
