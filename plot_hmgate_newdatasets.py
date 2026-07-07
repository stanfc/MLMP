"""HMGate trajectories on the new datasets (no baselines yet). figures/hmgate_newdatasets.png"""
import os
import numpy as np
import matplotlib.pyplot as plt


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


RUNS = [
    ("VOC21", "save/PascalVOC21Dataset/deyo_mlmp_hmgate_continual_5corr_sub100"),
    ("PContext59", "save/PascalContext59Dataset/deyo_mlmp_hmgate_continual_5corr_sub100"),
    ("PContext60", "save/PascalContext60Dataset/deyo_mlmp_hmgate_continual_5corr_sub100"),
    ("COCO-Object", "save/COCOObjectDataset/deyo_mlmp_hmgate_continual_5corr_sub100"),
    ("COCO-Stuff", "save/COCOStuffDataset/deyo_mlmp_hmgate_continual_5corr_sub100"),
    ("VOC20 full+15corr", "save/PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.reshape(-1)
for ax, (name, d) in zip(axes, RUNS):
    r, m = rd(f"{d}/results_all_rounds.txt")
    if len(m) == 0:
        ax.set_title(f"{name} [no data]"); ax.axis("off"); continue
    ax.plot(r, m, color="#1f77b4", lw=1.9)
    done = "" if r[-1] >= 150 else f" (R{r[-1]})"
    ax.set_title(f"{name}{done}  pk{m.max():.1f}@R{r[np.argmax(m)]} "
                 f"last{m[-1]:.1f} drop{m.max()-m[-1]:.1f}", fontsize=10)
    ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)"); ax.grid(alpha=0.3)

fig.suptitle("HMGate on new datasets (no baselines yet) — checking the no-drop property",
             fontsize=13, y=1.01)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/hmgate_newdatasets.png", dpi=125, bbox_inches="tight")
print("saved -> figures/hmgate_newdatasets.png")
