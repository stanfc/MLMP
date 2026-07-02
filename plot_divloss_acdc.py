"""divloss (marginal-diversity loss term) lambda sweep on ACDC vs no-adapt / no-gate / GDG-PA.
Output: figures/divloss_acdc.png"""
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
    ("divloss λ=0.1", "ACDCDataset/deyo_mlmp_divloss_continual_lam0.1", "#ff7f0e", 1.8),
    ("divloss λ=0.5", "ACDCDataset/deyo_mlmp_divloss_continual_lam0.5", "#d62728", 1.8),
    ("divloss λ=1.0", "ACDCDataset/deyo_mlmp_divloss_continual_lam1.0", "#8c564b", 1.8),
    ("no-gate DeYO+MLMP", "ACDCDataset/deyo_mlmp_continual", "#9467bd", 1.5),
    ("GDG-PA", "ACDCDataset/deyo_mlmp_hmgate2_continual", "#000000", 2.4),
]
NOADAPT, EPISODIC = 23.34, 29.84

fig, ax = plt.subplots(figsize=(11, 6.5))
for name, p, c, lw in LINES:
    r, m = rd(p)
    if len(m) == 0:
        continue
    ax.plot(r, m, color=c, lw=lw,
            label=f"{name}: pk{m.max():.1f} last{m[-1]:.1f} drop{m.max()-m[-1]:.1f}")
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=1.4, label=f"MLMP-episodic {EPISODIC:.1f}")
ax.axhline(NOADAPT, color="gray", ls=":", lw=1.6, label=f"No-Adapt {NOADAPT:.1f}")

ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
ax.set_title("ACDC — marginal-diversity loss term (divloss) fails to prevent collapse\n"
             "all λ collapse toward no-gate; higher λ worse. Restore-based GDG-PA holds.",
             fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="best")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/divloss_acdc.png", dpi=130, bbox_inches="tight")
print("saved -> figures/divloss_acdc.png")
