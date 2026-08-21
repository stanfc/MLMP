"""ViT classification (ImageNet-C, class-ordered stream): how the two gate signals
(H_margin, grad_norm) track our method's accuracy over the long horizon.
Top: H_margin + grad_norm per gate window.  Bottom: our top-1 accuracy per round.
Output: figures/cls_signals_vs_acc.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 22, "axes.labelsize": 26, "axes.titlesize": 28,
                     "xtick.labelsize": 21, "ytick.labelsize": 21,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

RUN = "/home/stanfc/Desktop/TTA-on-OVSS/DeYO/output/cls_continual_deyo_gate"
g = np.genfromtxt(f"{RUN}/gate_log.csv", delimiter=",", names=True)
acc = np.array([float(l.split(",")[-1]) for l in
                open(f"{RUN}/results_all_rounds.csv").readlines()[1:] if l.strip()])

tb = g["total_batches"].astype(float)
bpr = tb.max() / len(acc)                      # batches per round
gr = tb / bpr                                  # gate windows -> round axis
C_H, C_G, C_A = "#1f77b4", "#8c564b", "#d62728"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                               gridspec_kw={"height_ratios": [1.15, 1]})

# ---- top: the two gate signals ----
ax1.plot(gr, g["h_margin"], color=C_H, lw=3.0)
ax1.set_ylabel(r"$H_{\mathrm{margin}}$", color=C_H)
ax1.tick_params(axis="y", labelcolor=C_H, width=1.6, length=7)
ax1b = ax1.twinx()
ax1b.plot(gr, g["grad_norm"], color=C_G, lw=2.4, alpha=0.85)
ax1b.set_ylabel("grad_norm", color=C_G)
ax1b.tick_params(axis="y", labelcolor=C_G, width=1.6, length=7)
ax1.set_title("Gate signals vs. accuracy — ViT-B/16, ImageNet-C (class-ordered stream)",
              fontsize=25, pad=12)
ax1.grid(alpha=0.22, lw=0.9)
ax1.legend([Line2D([], [], color=C_H, lw=3.0), Line2D([], [], color=C_G, lw=2.4)],
           [r"$H_{\mathrm{margin}}$ (left)", "grad_norm (right)"],
           fontsize=19, loc="upper right", framealpha=0.95)

# ---- bottom: our accuracy ----
ax2.plot(np.arange(1, len(acc) + 1), acc, color=C_A, lw=4.0, marker="o", ms=8)
ax2.set_ylabel("Top-1 acc (%)", color=C_A)
ax2.tick_params(axis="y", labelcolor=C_A, width=1.6, length=7)
ax2.tick_params(axis="x", width=1.6, length=7)
ax2.set_xlabel("Round   (1 round = 15 ImageNet-C corruptions)")
ax2.set_ylim(0, 65)
ax2.grid(alpha=0.22, lw=0.9)
ax2.legend([Line2D([], [], color=C_A, lw=4.0)], ["Ours (GDG-PA gate)"],
           fontsize=19, loc="lower left", framealpha=0.95)

ax1.set_xlim(0, len(acc))
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/cls_signals_vs_acc.png", dpi=135, bbox_inches="tight")
print(f"saved -> figures/cls_signals_vs_acc.png  (windows={len(gr)}, rounds={len(acc)}, bpr={bpr:.0f})")
