"""p9/p10 (Observation 2): mIoU vs grad_norm, 3 datasets — faithful to the original
signal_trends panels (blue = grad_norm left axis, red = mIoU right axis,
title "grad_norm (rho=...)"). Source = the no-restore MONITOR runs.

Output: figures/p9_gradnorm_vs_miou.png
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans"})

SAVE = "save"
# left->right order as on the slide
PANELS = [
    ("V20",        "PascalVOC20Dataset/deyo_mlmp_continual_monitor_5corr_sub100"),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_continual_monitor_5corr_sub100"),
    ("ACDC",       "ACDCDataset/deyo_mlmp_continual_monitor"),
]


def read_miou(path):
    r, m = [], []
    with open(path) as f:
        next(f)
        for line in f:
            if line.startswith("Round "):
                p = [x.strip() for x in line.split(",")]
                r.append(int(p[0].split()[1])); m.append(float(p[-1]))
    return np.array(r), np.array(m)


def read_gradnorm(path):
    by_round = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row["grad_norm"]
            if v not in ("", "nan"):
                by_round.setdefault(int(row["round"]), []).append(float(v))
    rounds = np.array(sorted(by_round))
    return rounds, np.array([np.mean(by_round[r]) for r in rounds])


def spearman(a, b):
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


# match the original panel colours exactly
C_SIG, C_MIOU = "tab:blue", "tab:red"

fig, axes = plt.subplots(1, 3, figsize=(24, 6.6))
for ax, (name, sub) in zip(axes, PANELS):
    gr, gn = read_gradnorm(f"{SAVE}/{sub}/signals_log.csv")
    mr, mi = read_miou(f"{SAVE}/{sub}/results_all_rounds.txt")
    common = np.intersect1d(gr, mr)
    gn = gn[np.searchsorted(gr, common)]; mi = mi[np.searchsorted(mr, common)]
    rho = spearman(gn, mi)

    ax.plot(common, gn, color=C_SIG, lw=2.6)
    ax.set_ylabel("grad_norm", color=C_SIG, fontsize=26)
    ax.tick_params(axis="y", labelcolor=C_SIG, labelsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlabel("Round", fontsize=26)

    axb = ax.twinx()
    axb.plot(common, mi, color=C_MIOU, lw=2.6, alpha=0.85)
    axb.set_ylabel("mIoU (%)", color=C_MIOU, fontsize=26)
    axb.tick_params(axis="y", labelcolor=C_MIOU, labelsize=20)

    ax.set_title(f"mIoU vs. grad_norm ({name})\ngrad_norm  (ρ={rho:+.2f})", fontsize=24)

fig.tight_layout(w_pad=6.0)
fig.subplots_adjust(wspace=0.42)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/p9_gradnorm_vs_miou.png", dpi=140, bbox_inches="tight")
print("saved -> figures/p9_gradnorm_vs_miou.png")
