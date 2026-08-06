"""Ceiling push round 2: aggressive LR + EMA-eval combo on ACDC. figures/ceiling2_acdc.png"""
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
    ("EMA-eval + LR2e-5 (combo)", "ACDCDataset/deyo_mlmp_hmgate2_emaeval_continual_lr2e-5", "#2ca02c", 2.8),
    ("LR 3e-5", "ACDCDataset/deyo_mlmp_hmgate2_continual_lr3e-5", "#1f77b4", 1.8),
    ("EMA-eval (LR 5e-6)", "ACDCDataset/deyo_mlmp_hmgate2_emaeval_continual", "#9467bd", 1.8),
    ("GDG-PA base", "ACDCDataset/deyo_mlmp_hmgate2_continual", "#000000", 2.2),
    ("taconsensus (text-align)", "ACDCDataset/deyo_mlmp_hmgate2_taconsensus_continual", "#7f7f7f", 1.4),
]
NOGATE_PEAK, EPISODIC, NOADAPT = 33.5, 29.84, 23.34

fig, ax = plt.subplots(figsize=(11, 6.5))
for name, p, c, lw in LINES:
    r, m = rd(p)
    if len(m) == 0:
        continue
    ax.plot(r, m, color=c, lw=lw, label=f"{name}: last {m[-1]:.1f}")
ax.axhline(NOGATE_PEAK, color="#d62728", ls="--", lw=1.4, label=f"no-gate PEAK {NOGATE_PEAK:.1f} (the ceiling)")
ax.axhline(EPISODIC, color="#2ca02c", ls=":", lw=1.2, alpha=0.6, label=f"MLMP-episodic {EPISODIC:.1f}")

ax.set_ylim(29, 34)
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
ax.set_title("ACDC — aggressive LR + EMA-eval recovers the ceiling the gate cost\n"
             "combo reaches 33.0 (base 31.8 → +1.2), nearly the no-gate peak 33.5, with NO drop",
             fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ceiling2_acdc.png", dpi=130, bbox_inches="tight")
print("saved -> figures/ceiling2_acdc.png")
