"""LCoTTA+MLMP vs GDG / no-gate / baselines on V20 and Cityscapes (5corr sub100).
Output: figures/lcotta_compare.png"""
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


def epi(d):
    f = f"save/{d}/results.txt"
    if not os.path.exists(f):
        return None
    vals = []
    for ln in open(f):
        q = [t.strip() for t in ln.split(",")]
        if len(q) >= 2 and "+/-" in q[1] and q[0].lower() != "original":
            vals.append(float(q[1].split("+/-")[0]))
    return float(np.mean(vals)) if vals else None


def na(d):
    r, m = rd(d)
    return float(m[0]) if len(m) else None


PANELS = [
    ("VOC20-C (5corr sub100)",
     "PascalVOC20Dataset/lcotta_mlmp_continual_5corr_sub100",
     "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalVOC20Dataset/deyo_mlmp_continual_5corr_sub100",
     "PascalVOC20Dataset/mlmp_episodic_step_1", "PascalVOC20Dataset/No_Adaptation_sub100"),
    ("Cityscapes-C (5corr sub100)",
     "CityscapesDataset/lcotta_mlmp_continual_5corr_sub100",
     "CityscapesDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "CityscapesDataset/deyo_mlmp_continual_5corr_sub100",
     "CityscapesDataset/mlmp_episodic_sub100_step_1", "CityscapesDataset/No_Adaptation_sub100"),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (name, lc, gdg, ng, ep, nad) in zip(axes, PANELS):
    rl, ml = rd(lc)
    rg, mg = rd(gdg)
    rn, mn = rd(ng)
    if len(mn):
        ax.plot(rn, mn, color="#9467bd", lw=1.4, alpha=0.85,
                label=f"no-gate (MLMP): last{mn[-1]:.1f} drop{mn.max()-mn[-1]:.1f}")
    if len(mg):
        ax.plot(rg, mg, color="#d62728", lw=1.7,
                label=f"GDG 0.9: last{mg[-1]:.1f} drop{mg.max()-mg[-1]:.1f}")
    if len(ml):
        ax.plot(rl, ml, color="#000000", lw=2.4,
                label=f"LCoTTA+MLMP: pk{ml.max():.1f} last{ml[-1]:.1f} drop{ml.max()-ml[-1]:.1f}")
    e = epi(ep); a = na(nad)
    if e is not None:
        ax.axhline(e, color="#2ca02c", ls="--", lw=1.3, label=f"episodic {e:.1f}")
    if a is not None:
        ax.axhline(a, color="gray", ls=":", lw=1.3, label=f"no-adapt {a:.1f}")
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")

fig.suptitle("LCoTTA+MLMP (black) vs GDG / no-gate — subspace projection caps V20 low "
             "and fails to hold Cityscapes", fontsize=13, y=1.01)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/lcotta_compare.png", dpi=130, bbox_inches="tight")
print("saved -> figures/lcotta_compare.png")
