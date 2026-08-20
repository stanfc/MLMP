"""p8 (Observation 1): Model collapse <=> over-confidence, visualised via H_margin.
H_margin = entropy of the class-marginal prediction distribution (over 50-batch window).
When the model collapses onto 1-2 classes, the marginal concentrates -> H_margin drops,
and mIoU drops with it. Uses the NO-GATE collapse run (per project rule: grad/H_margin
observation figures must come from an un-gated collapse run, not the gated method).

Output: figures/p8_hmargin_collapse.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 28, "axes.labelsize": 32, "axes.titlesize": 34,
                     "xtick.labelsize": 25, "ytick.labelsize": 25,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})

RUN = "save/ACDCDataset/deyo_mlmp_continual_monitor"  # no-restore monitor observation run

# --- per-round H_margin from signals_log.csv
rows = np.genfromtxt(f"{RUN}/signals_log.csv", delimiter=",", names=True)
rnd = rows["round"].astype(int)
hm = rows["h_margin"]
rounds = np.arange(1, rnd.max() + 1)
hm_r = np.array([np.nanmean(hm[rnd == r]) for r in rounds])

# --- per-round mIoU
miou = np.array([float(l.split(",")[-1]) for l in open(f"{RUN}/results_all_rounds.txt")
                 if l.startswith("Round ")])

# mark the mIoU peak
peak = int(rounds[miou.argmax()])

C_H, C_M = "#1f77b4", "#d62728"
fig, ax = plt.subplots(figsize=(14, 8.0))

ax.plot(rounds, hm_r, color=C_H, lw=4.2, zorder=5)
ax.set_xlabel("Round")
ax.set_ylabel(r"$H_{\mathrm{margin}}$", color=C_H)
ax.tick_params(axis="y", labelcolor=C_H, width=1.6, length=8)
ax.tick_params(axis="x", width=1.6, length=8)
ax.set_xlim(1, rounds.max())
ax.grid(alpha=0.22, lw=0.9)

axr = ax.twinx()
axr.plot(rounds, miou, color=C_M, lw=4.2, zorder=5)
axr.set_ylabel("mIoU", color=C_M)
axr.tick_params(axis="y", labelcolor=C_M, width=1.6, length=8)

ax.axvline(peak, color="k", ls="--", lw=2.4, alpha=0.6)
# threshold = H_margin value where the peak line meets the blue curve
h_thr = hm_r[peak - 1]
ax.axhline(h_thr, color="#2ca02c", ls="--", lw=3.0, alpha=0.9, zorder=6)

ax.set_title(r"Collapse $\Leftrightarrow$ over-confidence: $H_{\mathrm{margin}}$ and mIoU fall together",
             fontsize=30, pad=30)
handles = [Line2D([], [], color=C_H, lw=4.2), Line2D([], [], color=C_M, lw=4.2)]
ax.legend(handles, [r"$H_{\mathrm{margin}}$ (left axis)", "mIoU (right axis)"],
          fontsize=27, loc="upper right", framealpha=0.95)

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/p8_hmargin_collapse.png", dpi=140, bbox_inches="tight")
print(f"saved -> figures/p8_hmargin_collapse.png  (mIoU peak R{peak})")
