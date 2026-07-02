"""HMGate h_drop_ratio 0.9 vs 0.95, all datasets, + no-adapt / MLMP-episodic
baseline lines. figures/hdr_compare.png"""
import os
import numpy as np
import matplotlib.pyplot as plt


def rd(p):
    """continual results_all_rounds.txt -> (rounds, mean_miou)."""
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


def _resultstxt_mean(path):
    """episodic/no-adapt results.txt: avg of per-corruption mIoU, EXCLUDING 'original'
    (HMGate full+15corr runs the 15 corruptions only)."""
    if not os.path.exists(path):
        return None
    vals = []
    for ln in open(path):
        parts = [t.strip() for t in ln.split(",")]
        if len(parts) >= 2 and "+/-" in parts[1] and parts[0].lower() != "original":
            try:
                vals.append(float(parts[1].split("+/-")[0]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else None


def baseline(d):
    """Constant baseline value. Accepts a continual dir (results_all_rounds.txt -> R1 mean)
    or an episodic/no-adapt dir (results.txt -> mean over corruptions)."""
    if d is None:
        return None
    p = d if d.startswith(("save", ".save")) else f"save/{d}"
    rar = f"{p}/results_all_rounds.txt"
    if os.path.exists(rar):
        r, m = rd(rar)
        return float(m[0]) if len(m) else None
    return _resultstxt_mean(f"{p}/results.txt")


# (title, old_dir(0.9), new_dir(0.95), noadapt_dir, episodic_dir) -- dirs under save/ unless prefixed
PANELS = [
    ("ACDC", "ACDCDataset/deyo_mlmp_hmgate_continual",
     "ACDCDataset/deyo_mlmp_hmgate_continual_hdr095",
     "save_tekai/save/ACDCDataset/No_Adaptation",
     "save_tekai/save/ACDCDataset/mlmp_episodic_step_1"),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "CityscapesDataset/deyo_mlmp_hmgate_continual_5corr_sub100_hdr095",
     "CityscapesDataset/No_Adaptation_sub100",
     "CityscapesDataset/mlmp_episodic_sub100_step_1"),
    ("VOC20", "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_5corr_sub100_hdr095",
     "PascalVOC20Dataset/No_Adaptation_sub100",
     "PascalVOC20Dataset/mlmp_episodic_step_1"),
    ("VOC21", "PascalVOC21Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalVOC21Dataset/deyo_mlmp_hmgate_continual_5corr_sub100_hdr095",
     "PascalVOC21Dataset/No_Adaptation_sub100",
     "PascalVOC21Dataset/mlmp_episodic_5corr_sub100"),
    ("PContext59", "PascalContext59Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalContext59Dataset/deyo_mlmp_hmgate_continual_5corr_sub100_hdr095",
     "PascalContext59Dataset/No_Adaptation_sub100",
     "PascalContext59Dataset/mlmp_episodic_5corr_sub100"),
    ("PContext60", "PascalContext60Dataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "PascalContext60Dataset/deyo_mlmp_hmgate_continual_5corr_sub100_hdr095",
     "PascalContext60Dataset/No_Adaptation_sub100",
     "PascalContext60Dataset/mlmp_episodic_5corr_sub100"),
    ("COCO-Object", "COCOObjectDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "COCOObjectDataset/deyo_mlmp_hmgate_continual_5corr_sub100_hdr095",
     "COCOObjectDataset/No_Adaptation_sub100",
     "COCOObjectDataset/mlmp_episodic_5corr_sub100"),
    ("COCO-Stuff", "COCOStuffDataset/deyo_mlmp_hmgate_continual_5corr_sub100",
     "COCOStuffDataset/deyo_mlmp_hmgate_continual_5corr_sub100_hdr095",
     "COCOStuffDataset/No_Adaptation_sub100",
     "COCOStuffDataset/mlmp_episodic_5corr_sub100"),
    ("VOC20 full+15corr", "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr",
     "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr_hdr095",
     ".save/PascalVOC20Dataset/No_Adaptation",  # full val, 15 corr (excl. original)
     ".save/PascalVOC20Dataset/mlmp"),          # episodic MLMP, full val, 15 corr
]

S = "save"
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.reshape(-1)
for ax, (name, old, new, na_dir, ep_dir) in zip(axes, PANELS):
    ro, mo = rd(f"{S}/{old}/results_all_rounds.txt")
    rm, mm = rd(f"{S}/{old}_hdr092/results_all_rounds.txt")  # 0.92 (in progress)
    rn, mn = rd(f"{S}/{new}/results_all_rounds.txt")
    rg, mg = rd(f"{S}/{old.replace('_hmgate', '')}/results_all_rounds.txt")  # no-gate
    r2, m2 = rd(f"{S}/{old.replace('hmgate', 'hmgate2')}/results_all_rounds.txt")  # hmgate2 (perm anchor)
    if len(mg):
        ax.plot(rg, mg, color="#9467bd", lw=1.3, ls="-", alpha=0.85,
                label=f"no-gate: pk{mg.max():.1f} last{mg[-1]:.1f} drop{mg.max()-mg[-1]:.1f}")
    if len(m2):
        d = "" if (len(r2) and r2[-1] >= 150) else f" R{r2[-1]}"
        ax.plot(r2, m2, color="#000000", lw=2.2,
                label=f"GradDivGate-PA{d}: pk{m2.max():.1f} last{m2[-1]:.1f} drop{m2.max()-m2[-1]:.1f}")
    if len(mo):
        ax.plot(ro, mo, color="#d62728", lw=1.7,
                label=f"GDG 0.9: pk{mo.max():.1f} last{mo[-1]:.1f} drop{mo.max()-mo[-1]:.1f}")
    if len(mm):
        d = "" if (len(rm) and rm[-1] >= 150) else f" R{rm[-1]}"
        ax.plot(rm, mm, color="#ff7f0e", lw=1.7,
                label=f"GDG 0.92{d}: pk{mm.max():.1f} last{mm[-1]:.1f} drop{mm.max()-mm[-1]:.1f}")
    if len(mn):
        d = "" if (len(rn) and rn[-1] >= 150) else f" R{rn[-1]}"
        ax.plot(rn, mn, color="#1f77b4", lw=1.9,
                label=f"GDG 0.95{d}: pk{mn.max():.1f} last{mn[-1]:.1f} drop{mn.max()-mn[-1]:.1f}")
    na = baseline(na_dir)
    ep = baseline(ep_dir)
    if ep is not None:
        ax.axhline(ep, color="#2ca02c", ls="--", lw=1.4, label=f"MLMP-episodic {ep:.1f}")
    if na is not None:
        ax.axhline(na, color="gray", ls=":", lw=1.4, label=f"No-Adapt {na:.1f}")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="best")

fig.suptitle("GradDivGate-PA (black, permanent best anchor) vs GradDivGate h_drop_ratio "
             "0.9/0.92/0.95 & no-gate, vs No-Adapt (gray:) / MLMP-episodic (green--)",
             fontsize=13, y=1.005)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/hdr_compare.png", dpi=120, bbox_inches="tight")
print("saved -> figures/hdr_compare.png")
