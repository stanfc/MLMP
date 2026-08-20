"""Zoomed view of cos(g_entropy, g_oracle) over the FIRST 10 rounds, at per-window
(50-batch) granularity -- every logged point, not per-round means.
Output: figures/cos_zoom_r0_10.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

RUN = "ACDCDataset/deyo_mlmp_hmgate2_oracle_nogate_lr3e-5"
NR = 10                                    # rounds to show

# ---- oracle log (per window) ----
rows = [l.strip().split(",") for l in open(f"save/{RUN}/oracle_log.csv")][1:]
tb = np.array([int(r[0]) for r in rows])
cos = np.array([float(r[2]) for r in rows])
gn = np.array([float(r[1]) for r in rows])

# ---- entropy log (per batch) -> map batch index to (round, condition) ----
etb, eround, econd = [], [], []
for l in open(f"save/{RUN}/entropy_log.csv"):
    x = l.strip().split(",")
    if not x[0].isdigit():
        continue
    etb.append(int(x[0])); eround.append(int(x[1])); econd.append(x[2])
etb = np.array(etb); eround = np.array(eround); econd = np.array(econd)

# continuous round coordinate for every batch: round + progress within round
xcont = np.empty(len(etb), dtype=float)
for r in np.unique(eround):
    m = eround == r
    n = m.sum()
    xcont[m] = r + np.arange(n) / n          # r .. r+1

# for each oracle window, take its position from the matching batch index
idx = np.searchsorted(etb, tb).clip(0, len(etb) - 1)
w_x = xcont[idx]
w_cond = econd[idx]

keep = w_x < (1 + NR)                        # rounds 1..NR
w_x, w_cos, w_gn, w_cond = w_x[keep], cos[keep], gn[keep], w_cond[keep]

# ---- mIoU (per round) for the peak marker ----
miou = np.array([float(l.split(",")[-1])
                 for l in open(f"save/{RUN}/results_all_rounds.txt")
                 if l.startswith("Round ")])
pk = int(miou.argmax()) + 1

fig, ax = plt.subplots(figsize=(14, 6.4))

# domain shading
COL = {"fog": "#cfe6f5", "night": "#c9ccd6", "rain": "#d7f0c8", "snow": "#ffe6c2"}
seen = set()
for i in range(len(w_x) - 1):
    c = w_cond[i]
    ax.axvspan(w_x[i], w_x[i + 1], color=COL.get(c, "#eeeeee"), alpha=0.55, lw=0,
               label=c if c not in seen else None)
    seen.add(c)

# zero line = the sign flip that matters
ax.axhline(0, color="k", ls="--", lw=1.6, zorder=4,
           label="cos = 0  (above: entropy HELPS / below: entropy FIGHTS the true task)")

# every logged window point
ax.plot(w_x, w_cos, color="#7d5bbe", lw=1.0, alpha=0.5, zorder=5)
ax.scatter(w_x, w_cos, c=np.where(w_cos >= 0, "#1f77b4", "#d62728"),
           s=34, zorder=6, edgecolor="k", linewidth=0.4,
           label="per-window cos (50-batch mean)")

# round gridlines
for r in range(1, NR + 2):
    ax.axvline(r, color="gray", lw=0.6, alpha=0.5, zorder=3)
ax.axvline(pk, color="#2ca02c", ls=":", lw=2.4, zorder=7,
           label=f"R{pk}: mIoU PEAK ({miou.max():.1f})")

# first sign flip
neg = np.where(w_cos < 0)[0]
if len(neg):
    fx = w_x[neg[0]]
    ax.annotate(f"first cos<0\n@ round {fx:.2f}", xy=(fx, w_cos[neg[0]]),
                xytext=(fx + 0.45, 0.16), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2), zorder=8)

ax.set_xlim(1, 1 + NR)
ax.set_xticks(np.arange(1, NR + 2))
ax.set_xlabel("Round (continuous — each round is one fog→night→rain→snow pass)")
ax.set_ylabel("cos(g_entropy, g_oracle)")
ax.set_title("ACDC no-gate, ROUNDS 1–10 — every logged window, not round means\n"
             f"the entropy gradient flips from agreeing with the true task to FIGHTING it, "
             f"right around the mIoU peak (R{pk})", fontsize=12)
ax.grid(alpha=0.25, axis="y")
h, lb = ax.get_legend_handles_labels()
ax.legend(h, lb, fontsize=8.5, loc="lower left", ncol=2)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/cos_zoom_r0_10.png", dpi=140, bbox_inches="tight")
print(f"saved -> figures/cos_zoom_r0_10.png   ({len(w_x)} window points over {NR} rounds)")
print(f"cos: R1 start {w_cos[0]:+.3f} | min {w_cos.min():+.3f} | at end of R{NR} {w_cos[-1]:+.3f}")
