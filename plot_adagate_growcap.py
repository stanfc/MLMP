"""AdaGate ABmad05_ecdf + shallow_cap_mode=growing (Phase S, fix #1 for the VOC20
tail-decline issue) -- mIoU vs round, one panel per dataset. The old flagship
(no growing cap) trajectory was not preserved on this machine (only its summary
stats survive in EXPERIMENT_STATUS.md), so it is shown as reference
mean/peak/tail annotations, not a plotted curve -- only real per-round data is
plotted as a line.

Output: figures/adagate_growcap_3datasets.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 22, "axes.labelsize": 26, "axes.titlesize": 22,
                     "xtick.labelsize": 20, "ytick.labelsize": 20,
                     "axes.linewidth": 1.5, "font.family": "DejaVu Sans"})

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


# (label, path, old_flagship_mean, old_flagship_peaklast_delta, ylim)
PANELS = [
    ("ACDC", "save/ACDCDataset/adagate_ABmad05_ecdf_grow/results_all_rounds.txt",
     31.31, -0.25, (28, 33)),
    ("VOC20 (v20_acdc_matched)",
     "save/PascalVOC20Dataset/v20_acdc_matched/adagate_ABmad05_ecdf_grow/results_all_rounds.txt",
     77.80, -0.35, (71, 80)),
    ("Cityscapes", "save/CityscapesDataset/adagate_ABmad05_ecdf_grow/results_all_rounds.txt",
     23.97, -0.05, (21, 25)),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))
C_NEW = "#d62728"

for ax, (name, path, old_mean, old_decay, ylim) in zip(axes, PANELS):
    r, m = load(path)
    peak_i = m.argmax()
    new_mean = m.mean()
    new_decay = m[peak_i] - m[-1]

    ax.plot(r, m, color=C_NEW, lw=2.6, zorder=5)
    ax.scatter([r[peak_i]], [m[peak_i]], color=C_NEW, s=70, zorder=6, marker="^")
    ax.scatter([r[-1]], [m[-1]], color=C_NEW, s=70, zorder=6)

    ax.axhline(old_mean, color="#7f7f7f", ls=":", lw=1.8, zorder=2)

    ax.set_xlabel("Round")
    ax.set_ylabel(name.split()[0] + (" mIoU" if "20" not in name.split()[0] and name!="Cityscapes" else " mIoU"))
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 152)
    ax.tick_params(width=1.5, length=7)
    ax.grid(True, alpha=0.25, lw=0.9)
    ax.set_title(name, pad=10)

    legend_handles = [
        Line2D([], [], color=C_NEW, lw=2.6, marker="^", ms=9,
              label=f"growing cap: mean={new_mean:.2f}, peak→last={new_decay:+.2f}"),
        Line2D([], [], color="#7f7f7f", ls=":", lw=1.8,
              label=f"old flagship (ref): mean={old_mean:.2f}, peak→last={old_decay:+.2f}"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=13.5, framealpha=0.92)
    print(f"{name}: new mean={new_mean:.2f} (old {old_mean:.2f}, {new_mean-old_mean:+.2f})  "
          f"peak->last new={new_decay:+.2f} (old {old_decay:+.2f})")

fig.suptitle("AdaGate ABmad05_ecdf + shallow_cap_mode=growing: tail decline fixed, ceiling lower",
            fontsize=22, y=1.03)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/adagate_growcap_3datasets.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  wrote {OUT}/adagate_growcap_3datasets.png")
