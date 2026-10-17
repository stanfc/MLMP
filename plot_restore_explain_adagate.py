"""AdaGate restore mechanism -- two schematics with the SAME layout, one per branch.
Same real ACDC no-gate collapse trajectory in both; only the restore arrow differs.

  restore_explain_deep.png    : COLLAPSE regime -> DEEP restore.
      windowed H_margin < h_drop_ratio * its running peak  ->  restore straight to
      the PERMANENT best anchor (LN snapshot at the grad-norm minimum, never
      evicted). look-back = windows_since_min, i.e. all the way back to the
      healthy peak however many windows have elapsed.

  restore_explain_shallow.png : HEALTHY regime -> SHALLOW restore.
      grad-norm slope worsening (robust MAD z-score z > trend_thr) but H_margin
      still healthy  ->  hop back only lag windows into the window buffer, where
      lag = ceil(cap * u),  u = ecdf rank of z,
      cap = ceil(windows_since_min * u_sev)   (growing_scaled -- grows with
      time-since-best, scaled by severity; no fixed number).
      A small pull to a RECENT window, NOT the permanent anchor (the star).

Trajectory grounded in: save/ACDCDataset/deyo_mlmp_hmgate2_oracle_nogate_lr3e-5
Visual language: tick marks, no numbers, no axis titles.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({"font.size": 26, "axes.linewidth": 1.5, "font.family": "DejaVu Sans"})

BLUE, GREEN = "#1f77b4", "#2ca02c"
MONITOR = 50
os.makedirs("figures", exist_ok=True)

RUN = "save/ACDCDataset/deyo_mlmp_hmgate2_oracle_nogate_lr3e-5"
miou = np.array([float(l.split(",")[-1]) for l in open(f"{RUN}/results_all_rounds.txt")
                 if l.startswith("Round ")])
R = len(miou)
rows = list(csv.DictReader(open(f"{RUN}/gate_log.csv")))
tb = np.array([int(r["total_batches"]) for r in rows])
per = tb.max() / R
wpr = per / MONITOR
deep = np.array([int(r["deep"]) for r in rows])
rnd_all = tb / per
i = int(np.where((deep == 1) & (rnd_all > 14) & (rnd_all < 16))[0][0])

cur_r = tb[i] / per                          # a real DEEP event, deep in collapse
anchor_r = float(miou.argmax() + 1)          # permanent best anchor (grad-min ~ peak)
shallow_target_r = cur_r - 1.4               # shallow: a small hop to a recent window
rr = np.arange(1, R + 1)


def yat(x):
    return float(np.interp(x, rr, miou))


def base_axes():
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.plot(rr, miou, color=BLUE, lw=3.2, zorder=4)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 38)
    ax.tick_params(width=1.5, length=7, labelbottom=False, labelleft=False)
    ax.grid(alpha=0.28)
    ax.axvline(cur_r, color="black", ls="--", lw=2.2, zorder=5)
    ax.plot(cur_r, yat(cur_r), "o", ms=16, color="black", zorder=9)
    ax.plot(anchor_r, yat(anchor_r), marker="*", ms=36, color=GREEN,
            mec="black", mew=1.3, zorder=9)
    return fig, ax


# ---------- DEEP: long arc -> the star ----------
fig, ax = base_axes()
ax.add_patch(FancyArrowPatch((cur_r, yat(cur_r)), (anchor_r, yat(anchor_r)),
                             connectionstyle="arc3,rad=0.45", arrowstyle="-|>",
                             mutation_scale=32, lw=3.6, color=GREEN,
                             shrinkA=6, shrinkB=14, zorder=6))
fig.tight_layout()
fig.savefig("figures/restore_explain_deep.png", dpi=135, bbox_inches="tight")
plt.close(fig)

# ---------- SHALLOW: short arc -> a recent window (green dot), NOT the star ----------
fig, ax = base_axes()
ax.plot(shallow_target_r, yat(shallow_target_r), "o", ms=15, color=GREEN,
        mec="black", mew=1.3, zorder=9)
ax.add_patch(FancyArrowPatch((cur_r, yat(cur_r) + 1.6),
                             (shallow_target_r, yat(shallow_target_r) + 1.6),
                             connectionstyle="arc3,rad=0.9", arrowstyle="-|>",
                             mutation_scale=32, lw=3.6, color=GREEN,
                             shrinkA=2, shrinkB=4, zorder=6))
fig.tight_layout()
fig.savefig("figures/restore_explain_shallow.png", dpi=135, bbox_inches="tight")
plt.close(fig)

print(f"DEEP event round {cur_r:.1f}, anchor round {anchor_r:.1f}")
print("saved -> figures/restore_explain_deep.png , figures/restore_explain_shallow.png")
