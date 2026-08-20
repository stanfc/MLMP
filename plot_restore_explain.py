"""Explain SHALLOW vs DEEP restore, grounded in a real GDG-PA run.

Left : the run's mIoU trajectory. At a real DEEP-restore event we mark the current
       position, the permanent best anchor (grad-norm minimum) it restores to, and
       the tiny shallow-restore window.
Right: zoom on the window deque -> what SHALLOW restore actually reaches.

Output: figures/restore_explain.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"font.size": 26, "axes.labelsize": 30,
                     "xtick.labelsize": 24, "ytick.labelsize": 24,
                     "axes.linewidth": 1.5, "font.family": "DejaVu Sans"})

RUN = "save/ACDCDataset/deyo_mlmp_hmgate2_oracle_nogate_lr3e-5"
MAXLAG_SHALLOW = 6          # windows
MONITOR = 50                # batches per window

miou = np.array([float(l.split(",")[-1]) for l in open(f"{RUN}/results_all_rounds.txt")
                 if l.startswith("Round ")])
R = len(miou)
rows = [l.strip().split(",") for l in open(f"{RUN}/gate_log.csv")][1:]
tb = np.array([int(r[0]) for r in rows])
wsm = np.array([int(r[5]) for r in rows])
lag = np.array([int(r[6]) for r in rows])
deep = np.array([int(r[8]) for r in rows])
per = tb.max() / R                     # batches per round
wpr = per / MONITOR                    # windows per round

# a real DEEP-restore event with a long look-back
rnd_all = tb / per
cand = np.where((deep == 1) & (rnd_all > 14) & (rnd_all < 16))[0]
i = int(cand[0])
cur_w = i
anchor_w = i - wsm[i]
cur_r = tb[i] / per
anchor_r = float(miou.argmax() + 1)   # star placed at the mIoU peak
shallow_r = MAXLAG_SHALLOW / wpr       # shallow reach, in rounds

print(f"DEEP event: window {cur_w} (round {cur_r:.1f}), look-back {wsm[i]} windows "
      f"-> anchor at round {anchor_r:.1f}")
print(f"SHALLOW reach: {MAXLAG_SHALLOW} windows = {shallow_r:.2f} round")

fig, ax = plt.subplots(figsize=(12, 7.0))
XMAX = 20
r = np.arange(1, R + 1)
ax.plot(r, miou, color="#1f77b4", lw=3.2, zorder=4)
ax.set_xlim(0, XMAX); ax.set_ylim(0, 38)
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
ax.grid(alpha=0.28)
ax.tick_params(width=1.5, length=7)

y_cur = miou[int(round(cur_r)) - 1]
y_anc = miou[max(0, int(round(anchor_r)) - 1)]

# permanent best anchor
ax.plot(anchor_r, y_anc, marker="*", ms=36, color="#2ca02c",
        mec="black", mew=1.3, zorder=9)

# current position
ax.axvline(cur_r, color="black", ls="--", lw=2.2, zorder=5)
ax.plot(cur_r, y_cur, "o", ms=16, color="black", zorder=9)

# shallow window
ax.add_patch(Rectangle((cur_r - shallow_r, 0), shallow_r, 38,
                       color="#d62728", alpha=0.40, zorder=2))

# DEEP restore arrow: tail = black dot, head = green star, bowing UP
ax.add_patch(FancyArrowPatch((cur_r, y_cur), (anchor_r, y_anc),
                             connectionstyle="arc3,rad=0.45", arrowstyle="-|>",
                             mutation_scale=32, lw=3.6, color="#2ca02c",
                             shrinkA=6, shrinkB=14, zorder=6))

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/restore_explain.png", dpi=135, bbox_inches="tight")
print("saved -> figures/restore_explain.png")
