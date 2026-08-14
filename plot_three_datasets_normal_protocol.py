"""Same presentation as plot_v20_15corr_collapse_check.py, but for the ORIGINAL
(non-stress) protocol on all three datasets: ACDC (full, 4 real weather
conditions), VOC20 (5corr, sub100), Cityscapes (5corr, sub100). One figure per
dataset -- same arms, same colors/labels, same reference lines.

Output: figures/normal_protocol_{acdc,v20_5corr,cityscapes_5corr}.{png,svg}
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


# same arms, same colors, same labels as the v20_15corr figure
ARMS = [
    ("GDG-PA",                "adagate_ctrl",                      "#9467bd", 2.6, 4),
    ("ABmad05_ecdf",          "adagate_ABmad05_ecdf",              "#7f7f7f", 3.0, 5),
    ("gradnorm_uncapped",     "adagate_ABmad05_ecdf_grow",         "#ff7f0e", 2.2, 3),
    ("gradnorm_scaled",       "adagate_ABmad05_ecdf_grow_scaled",  "#1f77b4", 2.2, 3),
]

DATASETS = [
    dict(key="acdc", root="save/ACDCDataset", no_adapt="acdc_no_adapt",
         episodic="mlmp_episodic_matched", ylabel="ACDC mIoU (4 real weather, full)",
         title="ACDC (full, 4 real weather conditions, standard protocol)", ylim=(28, 33)),
    dict(key="v20_5corr", root="save/PascalVOC20Dataset/v20_acdc_matched", no_adapt="no_adapt",
         episodic="mlmp_episodic_5corr", ylabel="VOC20 mIoU (5corr, sub100)",
         title="VOC20 (5corr, sub100, standard protocol)", ylim=(69, 80)),
    dict(key="cityscapes_5corr", root="save/CityscapesDataset", no_adapt="cityscapes_no_adapt",
         episodic="mlmp_episodic_5corr", ylabel="Cityscapes mIoU (5corr, sub100)",
         title="Cityscapes (5corr, sub100, standard protocol)", ylim=(21, 25)),
]

for ds in DATASETS:
    fig, ax = plt.subplots(figsize=(15, 9.5))

    _, m_na = load_continual(f"{ds['root']}/{ds['no_adapt']}/results_all_rounds.txt")
    na_mean = m_na.mean()
    ep_mean = load_episodic_mean(f"{ds['root']}/{ds['episodic']}/results.txt")

    ax.axhline(na_mean, color="black", ls=":", lw=2.0, zorder=2)
    ax.axhline(ep_mean, color="#555555", ls="-.", lw=2.0, zorder=2)

    legend_handles = [
        Line2D([], [], color="black", ls=":", lw=2.0, label=f"no-adapt: {na_mean:.2f}"),
        Line2D([], [], color="#555555", ls="-.", lw=2.0, label=f"MLMP episodic: {ep_mean:.2f}"),
    ]
    for label, key, color, lw, z in ARMS:
        r, m = load_continual(f"{ds['root']}/{key}/results_all_rounds.txt")
        ax.plot(r, m, color=color, lw=lw, zorder=z, solid_capstyle="round")
        legend_handles.append(Line2D([], [], color=color, lw=lw,
                                     label=f"{label}: mean={m.mean():.2f} (150R done)"))

    ax.set_xlabel("Round")
    ax.set_ylabel(ds["ylabel"])
    ax.set_xlim(0, 152)
    ax.set_ylim(*ds["ylim"])
    ax.tick_params(width=1.4, length=7)
    ax.grid(True, alpha=0.25, lw=0.9)
    ax.set_title(ds["title"], pad=14)
    ax.legend(handles=legend_handles, loc="lower right", fontsize=15, framealpha=0.93)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/normal_protocol_{ds['key']}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote normal_protocol_{ds['key']}.png")
