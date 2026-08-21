"""Ceiling-raiser experiments on top of GDG-PA, ACDC. Output: figures/ceiling_acdc.png"""
import os
import numpy as np
import matplotlib.pyplot as plt


def rd(p):
    r, m = [], []
    f = f"save/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([]), np.array([])
    for ln in open(f):
        if ln.startswith("Round "):
            x = [t.strip() for t in ln.split(",")]
            r.append(int(x[0].split()[1])); m.append(float(x[-1]))
    return np.array(r), np.array(m)


LINES = [
    ("GDG-PA (base)", "ACDCDataset/deyo_mlmp_hmgate2_continual", "#000000", 2.6),
    ("① EMA-eval + flip-TTA", "ACDCDataset/deyo_mlmp_hmgate2_emaeval_continual", "#2ca02c", 2.2),
    ("③ text-align", "ACDCDataset/deyo_mlmp_hmgate2_textalign_continual", "#1f77b4", 1.6),
    ("⑤ ratchet anchor", "ACDCDataset/deyo_mlmp_hmgate2_ratchet_continual", "#ff7f0e", 1.6),
    ("④ logit-adjust", "ACDCDataset/deyo_mlmp_hmgate2_logitadj_continual", "#d62728", 1.6),
]
EPISODIC, NOADAPT = 29.84, 23.34

fig, ax = plt.subplots(figsize=(11, 6.5))
for name, p, c, lw in LINES:
    r, m = rd(p)
    if len(m) == 0:
        continue
    ax.plot(r, m, color=c, lw=lw, label=f"{name}: last {m[-1]:.1f}")
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=1.2, alpha=0.6, label=f"MLMP-episodic {EPISODIC:.1f}")
ax.axhline(NOADAPT, color="gray", ls=":", lw=1.2, label=f"No-Adapt {NOADAPT:.1f}")

ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
ax.set_title("ACDC — raising the ceiling on top of GDG-PA (base last 31.8)\n"
             "only ① EMA-eval+TTA helps (+0.8 → 32.6); logit-adjust hurts badly",
             fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ceiling_acdc.png", dpi=130, bbox_inches="tight")
print("saved -> figures/ceiling_acdc.png")
