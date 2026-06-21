"""
DeYO+MLMP gate ablation, full 150-round, three datasets.
Compares: no-gate / DivGate / SmoothAnchor continual curves,
with MLMP-episodic and No-Adapt as flat reference lines.

Output: figures/deyo_mlmp_smooth_full.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save"


def read_rounds(path):
    """Return (rounds, mean_miou) from a results_all_rounds.txt (Mean_mIoU = last col)."""
    rounds, means = [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line or not line.startswith("Round "):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                rnum = int(parts[0].split()[1])
            except (IndexError, ValueError):
                continue
            rounds.append(rnum)
            means.append(float(parts[-1]))
    return np.array(rounds), np.array(means)


def episodic_mean(path):
    """Mean over per-condition mIoU rows in an episodic results.txt."""
    vals = []
    with open(path) as f:
        for line in f:
            if "+/-" in line and "Duration" not in line:
                # format: "<cond>, <miou> +/- 0.00, ..."
                try:
                    vals.append(float(line.split(",")[1].split("+/-")[0]))
                except (IndexError, ValueError):
                    pass
    return float(np.mean(vals)) if vals else None


def no_adapt_mean(path):
    """No-adapt is constant across rounds; take mean of Mean_mIoU column."""
    _, m = read_rounds(path)
    return float(m.mean()) if len(m) else None


# (title, ymin, ymax, paths)
DATASETS = [
    {
        "title": "ACDC (4 cond)",
        "nogate":  f"{SAVE}/ACDCDataset/deyo_mlmp_continual/results_all_rounds.txt",
        "divgate": f"{SAVE}/ACDCDataset/deyo_mlmp_divgate_continual/results_all_rounds.txt",
        "smooth":  f"{SAVE}/ACDCDataset/deyo_mlmp_smooth_anchor_continual_recal_h2.0_1.6/results_all_rounds.txt",
        "episodic": f"{SAVE}/../save_tekai/save/ACDCDataset/mlmp_episodic_step_1/results.txt",
        "noadapt":  f"{SAVE}/../save_tekai/save/ACDCDataset/No_Adaptation/results_all_rounds.txt",
        "baseline_note": "",
    },
    {
        "title": "Cityscapes-C (5corr sub100)",
        "nogate":  f"{SAVE}/CityscapesDataset/deyo_mlmp_continual_5corr_sub100/results_all_rounds.txt",
        "divgate": f"{SAVE}/CityscapesDataset/deyo_mlmp_divgate_continual_5corr_sub100/results_all_rounds.txt",
        "smooth":  f"{SAVE}/CityscapesDataset/deyo_mlmp_smooth_anchor_continual_recal_h2.1_1.7_5corr_sub100/results_all_rounds.txt",
        "episodic": f"{SAVE}/CityscapesDataset/mlmp_episodic_sub100_step_1/results.txt",
        "noadapt":  f"{SAVE}/CityscapesDataset/No_Adaptation_sub100/results_all_rounds.txt",
        "baseline_note": "",
    },
    {
        "title": "VOC20-C (5corr sub100)",
        "nogate":  f"{SAVE}/PascalVOC20Dataset/deyo_mlmp_continual_5corr_sub100/results_all_rounds.txt",
        "divgate": f"{SAVE}/PascalVOC20Dataset/deyo_mlmp_divgate_continual_5corr_sub100/results_all_rounds.txt",
        "smooth":  f"{SAVE}/PascalVOC20Dataset/deyo_mlmp_smooth_anchor_continual_5corr_sub100/results_all_rounds.txt",
        "episodic": f"{SAVE}/PascalVOC20Dataset/mlmp_episodic_step_1/results.txt",
        "noadapt":  f"{SAVE}/PascalVOC20Dataset/No_Adaptation_sub100/results_all_rounds.txt",
        "baseline_note": "",
    },
]

COL = {
    "nogate":  ("#d62728", "DeYO+MLMP (no gate)"),
    "divgate": ("#2ca02c", "DeYO+MLMP + DivGate"),
    "smooth":  ("#1f77b4", "DeYO+MLMP + SmoothAnchor"),
    "episodic": ("#7f7f7f", "MLMP episodic"),
    "noadapt":  ("#000000", "No Adapt"),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
handles_labels = {}

for ax, ds in zip(axes, DATASETS):
    annot = []
    for key in ("nogate", "divgate", "smooth"):
        r, m = read_rounds(ds[key])
        color, label = COL[key]
        (h,) = ax.plot(r, m, color=color, lw=1.8, label=label)
        handles_labels[label] = h
        if len(m):
            annot.append(f"{label.split('+ ')[-1] if '+ ' in label else label}: "
                         f"pk {m.max():.1f}@R{r[np.argmax(m)]}, last {m[-1]:.1f}")

    ep = episodic_mean(ds["episodic"])
    if ep is not None:
        color, label = COL["episodic"]
        h = ax.axhline(ep, color=color, ls="--", lw=1.4, label=label)
        handles_labels[label] = h

    na = no_adapt_mean(ds["noadapt"])
    if na is not None:
        color, label = COL["noadapt"]
        h = ax.axhline(na, color=color, ls=":", lw=1.4, label=label)
        handles_labels[label] = h

    ax.set_title(ds["title"] + (f"  {ds['baseline_note']}" if ds["baseline_note"] else ""),
                 fontsize=12)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean mIoU (%)")
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, "\n".join(annot), transform=ax.transAxes,
            fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

fig.suptitle("DeYO+MLMP: no gate vs DivGate vs SmoothAnchor (150 rounds, recalibrated SmoothAnchor)",
             fontsize=14, y=1.02)
fig.legend(handles_labels.values(), handles_labels.keys(),
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=10)
fig.tight_layout()
out = "figures/deyo_mlmp_smooth_full.png"
os.makedirs("figures", exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved -> {out}")
