"""No-gate monitor (ViT/ImageNet-C, i.i.d. shuffled stream, 150 rounds, base_rst=0):
two twin-axis figures — accuracy vs H_margin, accuracy vs grad_norm.
Signals averaged per round; Spearman rho vs accuracy in each title.
Outputs: figures/cls_monitor_hmargin_vs_acc.png , figures/cls_monitor_gradnorm_vs_acc.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 26, "axes.labelsize": 32, "axes.titlesize": 29,
                     "xtick.labelsize": 25, "ytick.labelsize": 25,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

RUN = "/home/stanfc/Desktop/TTA-on-OVSS/DeYO/output/cls_shuf_monitor"
g = np.genfromtxt(f"{RUN}/gate_log.csv", delimiter=",", names=True)
acc = np.array([float(l.split(",")[-1]) for l in
                open(f"{RUN}/results_all_rounds.csv").readlines()[1:] if l.strip()])

rounds = np.arange(1, len(acc) + 1)
tb = g["total_batches"].astype(float)
bpr = tb.max() / len(acc)
rw = np.clip(np.ceil(tb / bpr).astype(int), 1, len(acc))
per_round = lambda col: np.array([g[col][rw == r].mean() if (rw == r).any() else np.nan
                                  for r in rounds])


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    ar = np.argsort(np.argsort(a[m])); br = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ar, br)[0, 1])


C_ACC = "#d62728"
SPECS = [
    ("h_margin",  r"$H_{\mathrm{margin}}$", "#1f77b4", "cls_monitor_hmargin_vs_acc",
     r"No-gate monitor: accuracy vs $H_{\mathrm{margin}}$ (ViT, i.i.d. stream)"),
    ("grad_norm", "gradient norm",          "#8c564b", "cls_monitor_gradnorm_vs_acc",
     "No-gate monitor: accuracy vs gradient norm (ViT, i.i.d. stream)"),
]

for col, ylab, c_sig, out, title in SPECS:
    sig = per_round(col)
    rho = spearman(sig, acc)
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    ax.plot(rounds, sig, color=c_sig, lw=3.4, zorder=5)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylab, color=c_sig)
    ax.tick_params(axis="y", labelcolor=c_sig, width=1.6, length=8)
    ax.tick_params(axis="x", width=1.6, length=8)
    ax.set_xlim(1, len(acc))
    ax.grid(alpha=0.22, lw=0.9)

    axr = ax.twinx()
    axr.plot(rounds, acc, color=C_ACC, lw=3.4, zorder=6)
    axr.set_ylabel("Top-1 accuracy (%)", color=C_ACC)
    axr.tick_params(axis="y", labelcolor=C_ACC, width=1.6, length=8)

    ax.set_title(f"{title}\n(ρ = {rho:+.2f})", fontsize=27, pad=12)
    ax.legend([Line2D([], [], color=c_sig, lw=3.4), Line2D([], [], color=C_ACC, lw=3.4)],
              [f"{ylab} (left)", "accuracy (right)"],
              fontsize=22, loc="center right", framealpha=0.95)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{out}.png", dpi=135, bbox_inches="tight")
    print(f"saved -> figures/{out}.png   {col} rho={rho:+.2f}  ({np.nanmin(sig):.2f}..{np.nanmax(sig):.2f})")
print(f"  acc: {acc[0]:.1f} -> {acc[-1]:.1f}")
