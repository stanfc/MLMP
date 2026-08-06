"""ViT-seg (NA-CLIP ViT-L/14) GDG-PA gate on ACDC, 150 rounds — mIoU vs two gate
signals, faithful to the project's signal_trends panels (signal on the left axis,
mIoU red on the right axis, Spearman rho in the title). Two figures.

Outputs: figures/seg_voc20full_gradnorm_vs_miou.png
         figures/seg_voc20full_hmargin_vs_miou.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 26, "axes.labelsize": 32, "axes.titlesize": 28,
                     "xtick.labelsize": 25, "ytick.labelsize": 25,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

RUN = "save/PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_full_15corr"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

miou = np.array([float(l.split(",")[-1]) for l in
                 open(f"{RUN}/results_all_rounds.txt").readlines()[1:] if l.strip()])
n = len(miou)
rounds = np.arange(1, n + 1)

g = np.genfromtxt(f"{RUN}/gate_log.csv", delimiter=",", names=True)
tb = g["total_batches"].astype(float)
bpr = tb.max() / n
rw = np.clip(np.ceil(tb / bpr).astype(int), 1, n)
per_round = lambda col: np.array([g[col][rw == r].mean() if (rw == r).any() else np.nan
                                  for r in rounds])


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ar = np.argsort(np.argsort(a[m])); br = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ar, br)[0, 1])


C_MIOU = "#d62728"
SPECS = [
    ("grad_norm", "gradient norm", "#8c564b", (6.7, 8.4),
     "seg_voc20full_gradnorm_vs_miou",
     r"ViT-seg / VOC20 full-15corr 150R: mIoU vs gradient norm"),
    ("h_margin", r"$H_{\mathrm{margin}}$", "#1f77b4", (2.85, 3.05),
     "seg_voc20full_hmargin_vs_miou",
     r"ViT-seg / VOC20 full-15corr 150R: mIoU vs $H_{\mathrm{margin}}$"),
]

for col, ylab, c_sig, ylim, out, title in SPECS:
    sig = per_round(col)
    rho = spearman(sig, miou)
    fig, ax = plt.subplots(figsize=(13.5, 8.0))

    ax.plot(rounds, sig, color=c_sig, lw=3.4, zorder=5)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylab, color=c_sig)
    ax.set_ylim(*ylim)
    ax.set_xlim(1, n)
    ax.tick_params(axis="y", labelcolor=c_sig, width=1.6, length=8)
    ax.tick_params(axis="x", width=1.6, length=8)
    ax.grid(True, alpha=0.25, lw=1.0)

    ax2 = ax.twinx()
    ax2.plot(rounds, miou, color=C_MIOU, lw=3.4, zorder=6)
    ax2.set_ylabel("mIoU", color=C_MIOU)
    ax2.set_ylim(72.0, 78.5)
    ax2.tick_params(axis="y", labelcolor=C_MIOU, width=1.6, length=8)

    ax.set_title(f"{title}  ($\\rho={rho:+.2f}$)", pad=14)
    ax.legend(handles=[Line2D([], [], color=c_sig, lw=3.4, label=ylab),
                       Line2D([], [], color=C_MIOU, lw=3.4, label="mIoU")],
              loc="upper right", fontsize=23, framealpha=0.92)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{out}.png   rho={rho:+.3f}")
