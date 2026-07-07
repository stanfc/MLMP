"""full+15corr method comparison: GradDivGate-PA vs GDG 0.95/0.9 vs no-gate.
The key result: does the permanent anchor (PA) flatten the post-peak drift?
Outputs figures/v20full_compare.png and figures/cityscapes_full_compare.png"""
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


# (title, dir, color, lw)
PANELS = {
    "v20full_compare": dict(
        title="VOC20 full+15corr",
        lines=[
            ("GradDivGate-PA (perm anchor)",
             "PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_full_15corr", "#000000", 2.6),
            ("GDG 0.95 (rolling anchor)",
             "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr_hdr095", "#1f77b4", 1.9),
            ("GDG 0.9 (rolling anchor)",
             "PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr", "#d62728", 1.9),
            ("no-gate", "PascalVOC20Dataset/deyo_mlmp_continual_full_15corr", "#9467bd", 1.5),
        ],
        episodic=77.56, noadapt=69.0),
    "cityscapes_full_compare": dict(
        title="Cityscapes full+15corr",
        lines=[
            ("GradDivGate-PA (perm anchor)",
             "CityscapesDataset/deyo_mlmp_hmgate2_continual_full_15corr", "#000000", 2.6),
            ("GDG 0.95 (rolling anchor)",
             "CityscapesDataset/deyo_mlmp_hmgate_continual_full_15corr_hdr095", "#1f77b4", 1.9),
            ("no-gate", "CityscapesDataset/deyo_mlmp_continual_full_15corr", "#9467bd", 1.5),
        ],
        episodic=23.04, noadapt=21.63),   # episodic lr1e-3 (corrected; old lr0.01 gave broken 10.86)
}

for out, cfg in PANELS.items():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    any_line = False
    for name, p, c, lw in cfg["lines"]:
        r, m = rd(p)
        if len(m) == 0:
            continue
        any_line = True
        ax.plot(r, m, color=c, lw=lw,
                label=f"{name}: pk{m.max():.1f} last{m[-1]:.1f} (R{r[-1]}) drop{m.max()-m[-1]:.1f}")
    if cfg["episodic"] is not None:
        ax.axhline(cfg["episodic"], color="#2ca02c", ls="--", lw=1.4,
                   label=f"MLMP-episodic {cfg['episodic']:.1f}")
    if cfg["noadapt"] is not None:
        ax.axhline(cfg["noadapt"], color="gray", ls=":", lw=1.4,
                   label=f"No-Adapt {cfg['noadapt']:.1f}")
    ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
    ax.set_title(f"{cfg['title']} — permanent anchor (PA) vs rolling-anchor GDG vs no-gate",
                 fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{out}.png", dpi=130, bbox_inches="tight")
    print(f"saved -> figures/{out}.png" + ("" if any_line else "  [no data!]"))
