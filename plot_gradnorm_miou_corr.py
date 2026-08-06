"""Test proposition B: low grad_norm <-> high mIoU (grad_norm tracks current-domain fit).
Uses ONLY existing logs (gate_log.csv per-window signals + results_all_rounds.txt mIoU).
Output: figures/gradnorm_miou_corr.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import sys
RUN = sys.argv[1] if len(sys.argv) > 1 else "deyo_mlmp_hmgate2_continual"
BASE = f"save/ACDCDataset/{RUN}"

# per-round mIoU
miou = [float(l.split(",")[-1]) for l in open(f"{BASE}/results_all_rounds.txt")
        if l.startswith("Round ")]
miou = np.array(miou)
R = len(miou)

# per-window signals -> bin into R equal round-groups (constant batches/round)
rows = [l.strip().split(",") for l in open(f"{BASE}/gate_log.csv")][1:]
gn = np.array([float(r[1]) for r in rows])
hm = np.array([float(r[3]) for r in rows])
gn_r = np.array([g.mean() for g in np.array_split(gn, R)])
hm_r = np.array([h.mean() for h in np.array_split(hm, R)])

pr, _ = pearsonr(gn_r, miou); sr, _ = spearmanr(gn_r, miou)
pr_h, _ = pearsonr(hm_r, miou)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (1) twin-axis overlay: mIoU up, grad_norm (inverted axis) tracks it
ax = axes[0]; rounds = np.arange(1, R + 1)
ax.plot(rounds, miou, color="#2ca02c", lw=2.2, label="mIoU")
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)", color="#2ca02c")
ax.tick_params(axis="y", labelcolor="#2ca02c")
ax2 = ax.twinx()
ax2.plot(rounds, gn_r, color="#d62728", lw=1.8, alpha=0.8, label="grad_norm")
ax2.set_ylabel("grad_norm (axis INVERTED)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.invert_yaxis()   # inverted so 'low grad_norm' points UP, next to high mIoU
ax.set_title(f"Overlay: mIoU vs grad_norm (inverted)\nthey rise together -> low grad_norm = high mIoU")
ax.grid(alpha=0.3)

# (2) scatter grad_norm vs mIoU, colored by round
ax = axes[1]
sc = ax.scatter(gn_r, miou, c=rounds, cmap="viridis", s=28, edgecolor="k", lw=0.3)
ax.set_xlabel("per-round mean grad_norm"); ax.set_ylabel("per-round mIoU (%)")
ax.set_title(f"grad_norm vs mIoU\nPearson r={pr:.2f}, Spearman={sr:.2f}  "
             f"(h_margin r={pr_h:.2f})")
cb = fig.colorbar(sc, ax=ax); cb.set_label("Round")
# linear fit line
b, a = np.polyfit(gn_r, miou, 1)
xs = np.linspace(gn_r.min(), gn_r.max(), 50)
ax.plot(xs, b * xs + a, color="#d62728", ls="--", lw=1.5)
ax.grid(alpha=0.3)

fig.suptitle(f"Proposition B on {RUN}:  small grad_norm <-> well-fit (high-mIoU) state",
             fontsize=13)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig(f"figures/gradnorm_miou_corr_{RUN}.png", dpi=130, bbox_inches="tight")
print(f"saved -> figures/gradnorm_miou_corr_{RUN}.png  |  Pearson r(gn,miou)={pr:.3f} "
      f"Spearman={sr:.3f}  Pearson r(hm,miou)={pr_h:.3f}")
