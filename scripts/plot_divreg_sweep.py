"""Performance-trajectory figures for the deyo_mlmp_divreg lambda sweep.

Writes to save/_compare/:
  - divreg_{ACDC,VOC20,Cityscapes}.png   (one per dataset, annotated)
  - divreg_lambda_sweep.png              (combined 1x3)

Run:  conda run -n MLMP python scripts/plot_divreg_sweep.py
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/tekai324/MLMP"
DS = {
    "ACDC":       ("save/ACDCDataset",                        31.8, "DeYO-DivGate 31.8"),
    "VOC20":      ("save/PascalVOC20Dataset/v20_acdc_matched",77.4, "DeYO-DivGate 77.4"),
    "Cityscapes": ("save/CityscapesDataset",                 23.6, "DeYO-DivGate 23.6"),
}
LAMS = [0.0, 0.1, 0.3, 1.0, 3.0]
COL  = dict(zip(LAMS, ["#777777", "#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"]))

def load(path):
    R, M = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round"):
            continue
        p = [x.strip() for x in line.replace("Round", "").split(",")]
        try:
            R.append(int(p[0])); M.append(float(p[-1]))
        except Exception:
            pass
    i = np.argsort(R)
    return np.array(R)[i], np.array(M)[i]

def plot_one(ax, name, sub, bar, barlbl, legend=True):
    for lam in LAMS:
        rp = os.path.join(ROOT, sub, f"deyo_mlmp_divreg_lam{lam}", "results_all_rounds.txt")
        if not os.path.exists(rp):
            continue
        R, M = load(rp)
        lw = 2.6 if lam in (0.0, 0.3) else 1.5
        lbl = f"λ={lam}" + (" (control=DeYO base)" if lam == 0 else "")
        ax.plot(R, M, color=COL[lam], lw=lw, label=lbl, zorder=3 if lam in (0.0,0.3) else 2)
        pk = int(np.argmax(M))
        ax.scatter([R[pk]], [M[pk]], color=COL[lam], s=22, zorder=4, edgecolor="white", linewidth=0.6)
    ax.axhline(bar, ls="--", color="black", lw=1, alpha=0.7)
    ax.text(0.99, bar, f"{barlbl}", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8, color="black")
    ax.set_title(f"{name}  —  DeYO-MLMP + λ·(−H_margin) diversity loss", fontsize=11)
    ax.set_xlabel("Round"); ax.set_ylabel("mean mIoU"); ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=8, loc="lower left" if name != "VOC20" else "lower center")

os.makedirs(os.path.join(ROOT, "save/_compare"), exist_ok=True)

# individual figures
for name, (sub, bar, barlbl) in DS.items():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_one(ax, name, sub, bar, barlbl)
    plt.tight_layout()
    out = os.path.join(ROOT, f"save/_compare/divreg_{name}.png")
    plt.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)

# combined
fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
for ax, (name, (sub, bar, barlbl)) in zip(axes, DS.items()):
    plot_one(ax, name, sub, bar, barlbl)
plt.tight_layout()
out = os.path.join(ROOT, "save/_compare/divreg_lambda_sweep.png")
plt.savefig(out, dpi=110); plt.close(fig)
print("saved", out)
