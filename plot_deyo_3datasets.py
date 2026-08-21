"""Pure DeYO (continual) on ACDC / VOC20-C / Cityscapes-C, with MLMP-episodic
and No-Adapt reference lines. Same 3-panel layout as deyo_uaml_3version.png.
Output: figures/deyo_3datasets.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 14})


def miou(p):
    f = f"save/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


PANELS = [
    ("ACDC (4 cond)", "ACDCDataset/deyo_continual", 29.84, 23.34, (0, 36)),
    ("VOC20-C (5corr sub100)", "PascalVOC20Dataset/deyo_continual_5corr_sub100", 75.35, 68.60, (0, 80)),
    ("Cityscapes-C (5corr sub100)", "CityscapesDataset/deyo_continual_5corr_sub100", 22.71, 20.56, (0, 27)),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
for ax, (title, p, epi, noadapt, ylim) in zip(axes, PANELS):
    m = miou(p)
    ax.axhline(epi, color="#9467bd", ls=":", lw=2.2, zorder=3)
    ax.axhline(noadapt, color="k", ls="-", lw=1.6, zorder=3)
    ax.plot(np.arange(1, len(m) + 1), m, color="#ff7f0e", lw=2.8, zorder=5,
            label=f"DeYO  pk={m.max():.1f}@R{m.argmax()+1}, last={m[-1]:.1f}")
    ax.set_xlim(1, 150); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.25)
    h = [Line2D([], [], color="#ff7f0e", lw=2.8),
         Line2D([], [], color="#9467bd", ls=":", lw=2.2),
         Line2D([], [], color="k", lw=1.6)]
    lb = [f"DeYO  pk={m.max():.1f}, last={m[-1]:.1f}",
          f"MLMP episodic ({epi:.1f})", f"No Adapt ({noadapt:.1f})"]
    ax.legend(h, lb, fontsize=11, loc="upper right", framealpha=0.95)

axes[0].set_ylabel("Mean mIoU (%)")
fig.suptitle("Pure DeYO (continual, no reset) across three datasets", fontsize=18)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/deyo_3datasets.png", dpi=135, bbox_inches="tight")
print("saved -> figures/deyo_3datasets.png")
