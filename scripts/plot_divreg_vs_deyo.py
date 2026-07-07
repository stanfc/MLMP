"""Per-dataset trajectory: our diversity loss vs plain DeYO+MLMP (no diversity),
with no-adapt / SHOT / MLMP-episodic baselines.  -> save/_compare/divreg_vs_deyo_{ds}.png

plain-DeYO source (學長, un-foldered) verified bit-identical to our lambda=0:
  ACDC   save/deyo_mlmp_continual                       (R150 6.04 == our lam0)
  VOC20  save/deyo_mlmp_continual_monitor_5corr_sub100  (R150 74.29 == our lam0)
  Citys  (學長 file not identifiable) -> our divreg_lam0.0 (== plain DeYO, term skipped)
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/tekai324/MLMP"

CFG = {
 "ACDC": dict(
    ours ="save/ACDCDataset/deyo_mlmp_divreg_lam0.3",
    deyo ="save/deyo_mlmp_continual",
    shot ="save/ACDCDataset/shot_continual",
    no_adapt=23.34, episodic=30.57),
 "VOC20": dict(
    ours ="save/PascalVOC20Dataset/v20_acdc_matched/deyo_mlmp_divreg_lam0.3",
    deyo ="save/deyo_mlmp_continual_monitor_5corr_sub100",
    shot ="save/PascalVOC20Dataset/v20_acdc_matched/shot_continual",
    no_adapt=71.65, episodic=76.21),
 "Cityscapes": dict(
    ours ="save/CityscapesDataset/deyo_mlmp_divreg_lam0.3",
    deyo ="save/CityscapesDataset/deyo_mlmp_divreg_lam0.0",   # == plain DeYO
    shot ="save/CityscapesDataset/shot_continual",
    no_adapt=18.47, episodic=20.01),
}

def load(path):
    R, M = [], []
    p = os.path.join(ROOT, path, "results_all_rounds.txt")
    if not os.path.exists(p):
        return None, None
    for line in open(p):
        line = line.strip()
        if not line.startswith("Round"):
            continue
        q = [x.strip() for x in line.replace("Round", "").split(",")]
        try:
            R.append(int(q[0])); M.append(float(q[-1]))
        except Exception:
            pass
    i = np.argsort(R)
    return np.array(R)[i], np.array(M)[i]

for ds, c in CFG.items():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # trajectories
    for key, lbl, col, lw in [
        ("ours", "DeYO+MLMP + DivReg (ours, λ=0.3)", "#377eb8", 2.8),
        ("deyo", "DeYO+MLMP (no diversity)",          "#e41a1c", 2.2),
        ("shot", "SHOT (IM loss)",                    "#984ea3", 1.8),
    ]:
        R, M = load(c[key])
        if R is not None:
            ax.plot(R, M, color=col, lw=lw, label=lbl, zorder=3)
    # baseline reference lines
    ax.axhline(c["no_adapt"], ls="--", color="#555555", lw=1.4,
               label=f"No adaptation ({c['no_adapt']:.1f})")
    ax.axhline(c["episodic"], ls=":", color="#4daf4a", lw=1.8,
               label=f"MLMP episodic ({c['episodic']:.1f})")
    ax.set_title(f"{ds}: diversity loss vs plain DeYO+MLMP (150 rounds)")
    ax.set_xlabel("Round"); ax.set_ylabel("mean mIoU")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    ax.set_xlim(0, 150)
    plt.tight_layout()
    out = os.path.join(ROOT, f"save/_compare/divreg_vs_deyo_{ds}.png")
    plt.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)
