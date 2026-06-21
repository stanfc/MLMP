"""
Gate comparison across 3 datasets (whatever progress exists):
no-gate vs composite vs grad-slope, + MLMP-episodic / No-Adapt baselines.
VOC20 panel also shows the slope_window variants (sw10/sw50/sw100).

Output: figures/all_gates_3datasets.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save"


def rd(path):
    r, m = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path) as f:
        next(f)
        for ln in f:
            ln = ln.strip()
            if ln.startswith("Round "):
                x = [t.strip() for t in ln.split(",")]
                try:
                    r.append(int(x[0].split()[1])); m.append(float(x[-1]))
                except (IndexError, ValueError):
                    pass
    return np.array(r), np.array(m)


def epi(path):
    if not os.path.exists(path):
        return None
    v = [float(l.split(",")[1].split("+/-")[0]) for l in open(path)
         if "+/-" in l and "Duration" not in l]
    return float(np.mean(v)) if v else None


def na_mean(path):
    _, m = rd(path)
    return float(m.mean()) if len(m) else None


PANELS = [
    {
        "title": "ACDC (4 cond)",
        "curves": [
            ("ACDCDataset/deyo_mlmp_continual", "#d62728", "no-gate"),
            ("ACDCDataset/deyo_mlmp_gradslope_continual", "#2ca02c", "GradSlope sw10"),
            ("ACDCDataset/deyo_mlmp_gradslope_continual_sw100", "#ff7f0e", "GradSlope sw100"),
            ("ACDCDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300", "#000000", "GradSlope sw10+lag300"),
        ],
        "epi": "../save_tekai/save/ACDCDataset/mlmp_episodic_step_1/results.txt",
        "na": "../save_tekai/save/ACDCDataset/No_Adaptation/results_all_rounds.txt",
    },
    {
        "title": "Cityscapes-C (5corr sub100)",
        "curves": [
            ("CityscapesDataset/deyo_mlmp_continual_5corr_sub100", "#d62728", "no-gate"),
            ("CityscapesDataset/deyo_mlmp_gradslope_continual_5corr_sub100", "#2ca02c", "GradSlope sw10"),
            ("CityscapesDataset/deyo_mlmp_gradslope_continual_sw100_5corr_sub100", "#ff7f0e", "GradSlope sw100"),
            ("CityscapesDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100", "#000000", "GradSlope sw10+lag300"),
        ],
        "epi": "CityscapesDataset/mlmp_episodic_sub100_step_1/results.txt",
        "na": "CityscapesDataset/No_Adaptation_sub100/results_all_rounds.txt",
    },
    {
        "title": "VOC20-C (5corr sub100)",
        "curves": [
            ("PascalVOC20Dataset/deyo_mlmp_continual_5corr_sub100", "#d62728", "no-gate"),
            ("PascalVOC20Dataset/deyo_mlmp_gradslope_continual_5corr_sub100", "#2ca02c", "GradSlope sw10"),
            ("PascalVOC20Dataset/deyo_mlmp_gradslope_continual_sw50_5corr_sub100", "#9467bd", "GradSlope sw50"),
            ("PascalVOC20Dataset/deyo_mlmp_gradslope_continual_sw100_5corr_sub100", "#ff7f0e", "GradSlope sw100"),
            ("PascalVOC20Dataset/deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100", "#000000", "GradSlope sw10+lag300"),
        ],
        "epi": "PascalVOC20Dataset/mlmp_episodic_step_1/results.txt",
        "na": "PascalVOC20Dataset/No_Adaptation_sub100/results_all_rounds.txt",
    },
]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
for ax, P in zip(axes, PANELS):
    ann = []
    for sub, color, label in P["curves"]:
        r, m = rd(f"{SAVE}/{sub}/results_all_rounds.txt")
        if len(m) == 0:
            continue
        done = "" if r[-1] >= 150 else f" (R{r[-1]})"
        ax.plot(r, m, color=color, lw=1.8, label=label + done)
        ann.append(f"{label}: pk {m.max():.1f}@R{r[np.argmax(m)]}, last {m[-1]:.1f}")
    e = epi(f"{SAVE}/{P['epi']}")
    if e is not None:
        ax.axhline(e, color="gray", ls="--", lw=1.2, label=f"MLMP episodic ({e:.1f})")
    n = na_mean(f"{SAVE}/{P['na']}")
    if n is not None:
        ax.axhline(n, color="black", ls=":", lw=1.2, label=f"No Adapt ({n:.1f})")
    ax.set_title(P["title"], fontsize=11)
    ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU (%)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=7.5)
    ax.text(0.98, 0.02, "\n".join(ann), transform=ax.transAxes, fontsize=6.8,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

fig.suptitle("DeYO+MLMP gate comparison (current progress) — no-gate vs GradSlope variants",
             fontsize=13, y=1.02)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
out = "figures/all_gates_3datasets.png"
fig.savefig(out, dpi=125, bbox_inches="tight")
print(f"saved -> {out}")
