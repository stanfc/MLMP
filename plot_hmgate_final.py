"""HMGate (winner) vs no-gate + baselines, 3 datasets. figures/hmgate_final.png"""
import os
import numpy as np
import matplotlib.pyplot as plt

S = "save"


def rd(p):
    r, m = [], []
    if not os.path.exists(p):
        return np.array([]), np.array([])
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith("Round "):
            x = [t.strip() for t in ln.split(",")]
            try:
                r.append(int(x[0].split()[1])); m.append(float(x[-1]))
            except (IndexError, ValueError):
                pass
    return np.array(r), np.array(m)


def epi(p):
    if not os.path.exists(p):
        return None
    v = [float(l.split(",")[1].split("+/-")[0]) for l in open(p)
         if "+/-" in l and "Duration" not in l]
    return float(np.mean(v)) if v else None


PANELS = [
    ("ACDC (4 cond)", "ACDCDataset/deyo_mlmp_continual",
     "ACDCDataset/deyo_mlmp_hmgate_continual",
     "../save_tekai/save/ACDCDataset/mlmp_episodic_step_1/results.txt",
     "../save_tekai/save/ACDCDataset/No_Adaptation/results_all_rounds.txt"),
    ("Cityscapes-C (5corr sub100)", "CityscapesDataset/deyo_mlmp_continual_5corr_sub100",
     "CityscapesDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "CityscapesDataset/mlmp_episodic_sub100_step_1/results.txt",
     "CityscapesDataset/No_Adaptation_sub100/results_all_rounds.txt"),
    ("VOC20-C (5corr sub100)", "PascalVOC20Dataset/deyo_mlmp_continual_5corr_sub100",
     "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalVOC20Dataset/mlmp_episodic_step_1/results.txt",
     "PascalVOC20Dataset/No_Adaptation_sub100/results_all_rounds.txt"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for ax, (title, ng, hm, ep, na) in zip(axes, PANELS):
    r, m = rd(f"{S}/{ng}/results_all_rounds.txt")
    ax.plot(r, m, color="#d62728", lw=1.8, label=f"no-gate (last {m[-1]:.1f})")
    r2, m2 = rd(f"{S}/{hm}/results_all_rounds.txt")
    ax.plot(r2, m2, color="#1f77b4", lw=2.4,
            label=f"HMGate (ours) pk{m2.max():.1f} last{m2[-1]:.1f}")
    e = epi(f"{S}/{ep}")
    if e:
        ax.axhline(e, color="#7f7f7f", ls="--", lw=1.4, label=f"MLMP episodic ({e:.1f})")
    _, mna = rd(f"{S}/{na}")
    if len(mna):
        ax.axhline(mna.mean(), color="black", ls=":", lw=1.3, label=f"No Adapt ({mna.mean():.1f})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU (%)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

fig.suptitle("DeYO+MLMP + H-margin-regime gate (HMGate) — no drop, near peak, beats MLMP-episodic",
             fontsize=13, y=1.02)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/hmgate_final.png", dpi=130, bbox_inches="tight")
print("saved -> figures/hmgate_final.png")
