"""Per-dataset trajectory: our method WITH restoration (DivReg + composite gate)
vs composite-gate-alone, no-adapt, MLMP-episodic.
  -> save/_compare/divregcomp_vs_baselines_{ds}.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/tekai324/MLMP"

CFG = {
 "ACDC": dict(
    ours="save/ACDCDataset/deyo_mlmp_divreg0.3_composite_cc0.70_rst0.02_gm4",
    comp="save/ACDCDataset/deyo_mlmp_composite_cc0.70_rst0.02_gm4",
    no_adapt=23.34, episodic=30.57),
 "VOC20": dict(
    ours="save/PascalVOC20Dataset/v20_acdc_matched/deyo_mlmp_divreg0.3_composite_cc0.58_rst0.005_gm4",
    comp="save/PascalVOC20Dataset/v20_acdc_matched/deyo_mlmp_composite_cc0.60_rst0.005_gm4",
    no_adapt=71.65, episodic=76.21),
 "Cityscapes": dict(  # cc0.52 is the better config (still running -> line to current progress)
    ours="save/CityscapesDataset/deyo_mlmp_divreg0.3_composite_cc0.52_rst0.005_gm4",
    comp="save/CityscapesDataset/deyo_mlmp_composite_cc0.66_rst0.005_gm4",
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
    for key, lbl, col, lw in [
        ("ours", "Ours + restoration (DivReg + Composite gate)", "#377eb8", 2.8),
        ("comp", "Composite gate (no diversity)",                "#ff7f00", 2.2),
    ]:
        R, M = load(c[key])
        if R is not None:
            ax.plot(R, M, color=col, lw=lw, label=lbl, zorder=3)
    ax.axhline(c["no_adapt"], ls="--", color="#555555", lw=1.4,
               label=f"No adaptation ({c['no_adapt']:.1f})")
    ax.axhline(c["episodic"], ls=":", color="#4daf4a", lw=1.8,
               label=f"MLMP episodic ({c['episodic']:.1f})")
    ax.set_title(f"{ds}: our loss + restoration vs composite gate (150 rounds)")
    ax.set_xlabel("Round"); ax.set_ylabel("mean mIoU")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    ax.set_xlim(0, 150)
    plt.tight_layout()
    out = os.path.join(ROOT, f"save/_compare/divregcomp_vs_baselines_{ds}.png")
    plt.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)
