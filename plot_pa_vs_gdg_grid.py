"""Clean 3x3 grid (hdr_compare layout): GradDivGate-PA (black) vs GDG 0.9 (red),
+ episodic/no-adapt baselines. Shows PA stability across all datasets.
Output: figures/pa_vs_gdg_grid.png"""
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


def baseline(d):
    if d is None:
        return None
    p = d if d.startswith(("save", ".save")) else f"save/{d}"
    rar = f"{p}/results_all_rounds.txt"
    if os.path.exists(rar):
        r, m = rd(d if not d.startswith(("save", ".save")) else d.split("save/")[-1])
        return float(m[0]) if len(m) else None
    f = f"{p}/results.txt"
    if not os.path.exists(f):
        return None
    vals = []
    for ln in open(f):
        q = [t.strip() for t in ln.split(",")]
        if len(q) >= 2 and "+/-" in q[1] and q[0].lower() != "original":
            try:
                vals.append(float(q[1].split("+/-")[0]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else None


# (title, PA dir, GDG0.9 dir, episodic, no-adapt)
PANELS = [
    ("ACDC", "ACDCDataset/deyo_mlmp_hmgate2_continual",
     "ACDCDataset/deyo_mlmp_hmgate_continual",
     "save_tekai/save/ACDCDataset/mlmp_episodic_step_1", "save_tekai/save/ACDCDataset/No_Adaptation"),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
     "CityscapesDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "CityscapesDataset/mlmp_episodic_sub100_step_1", "CityscapesDataset/No_Adaptation_sub100"),
    ("VOC20", "PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
     "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalVOC20Dataset/mlmp_episodic_step_1", "PascalVOC20Dataset/No_Adaptation_sub100"),
    ("VOC21", "PascalVOC21Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
     "PascalVOC21Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalVOC21Dataset/mlmp_episodic_5corr_sub100", "PascalVOC21Dataset/No_Adaptation_sub100"),
    ("PContext59", "PascalContext59Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
     "PascalContext59Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalContext59Dataset/mlmp_episodic_5corr_sub100", "PascalContext59Dataset/No_Adaptation_sub100"),
    ("PContext60", "PascalContext60Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
     "PascalContext60Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalContext60Dataset/mlmp_episodic_5corr_sub100", "PascalContext60Dataset/No_Adaptation_sub100"),
    ("COCO-Object", "COCOObjectDataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
     "COCOObjectDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "COCOObjectDataset/mlmp_episodic_5corr_sub100", "COCOObjectDataset/No_Adaptation_sub100"),
    ("COCO-Stuff", "COCOStuffDataset/deyo_mlmp_hmgate2_continual_5corr_sub100",
     "COCOStuffDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "COCOStuffDataset/mlmp_episodic_5corr_sub100", "COCOStuffDataset/No_Adaptation_sub100"),
    ("VOC20 full+15corr", "PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_full_15corr",
     "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr",
     ".save/PascalVOC20Dataset/mlmp", ".save/PascalVOC20Dataset/No_Adaptation"),
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.reshape(-1)
for ax, (name, pa, gdg, ep, na) in zip(axes, PANELS):
    rp, mp = rd(pa)
    rg, mg = rd(gdg)
    rn, mn = rd(gdg.replace("_hmgate", ""))   # no-gate
    if len(mn):
        ax.plot(rn, mn, color="#9467bd", lw=1.3, alpha=0.85,
                label=f"no-gate: last{mn[-1]:.1f} drop{mn.max()-mn[-1]:.1f}")
    if len(mg):
        d = "" if rg[-1] >= 150 else f" R{rg[-1]}"
        ax.plot(rg, mg, color="#d62728", lw=1.7,
                label=f"GDG 0.9{d}: last{mg[-1]:.1f} drop{mg.max()-mg[-1]:.1f}")
    if len(mp):
        d = "" if rp[-1] >= 150 else f" R{rp[-1]}"
        ax.plot(rp, mp, color="#000000", lw=2.4,
                label=f"GDG-PA{d}: last{mp[-1]:.1f} drop{mp.max()-mp[-1]:.1f}")
    e = baseline(ep); a = baseline(na)
    if e is not None:
        ax.axhline(e, color="#2ca02c", ls="--", lw=1.3, label=f"episodic {e:.1f}")
    if a is not None:
        ax.axhline(a, color="gray", ls=":", lw=1.3, label=f"no-adapt {a:.1f}")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="best")

fig.suptitle("GradDivGate-PA (black, permanent anchor) vs GDG 0.9 (red, rolling anchor) "
             "— PA only differs on heavy full runs (VOC20-full)", fontsize=13, y=1.005)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/pa_vs_gdg_grid.png", dpi=120, bbox_inches="tight")
print("saved -> figures/pa_vs_gdg_grid.png")
