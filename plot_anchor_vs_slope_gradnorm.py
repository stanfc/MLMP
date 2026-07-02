"""
GradAnchor (method2) vs GradSlope sw10+maxlag300: grad_norm AND mIoU overlaid,
per dataset. Shows why GradAnchor holds (grad_norm stays controlled) while
sw10+maxlag300 lets grad_norm blow up (collapse).

Left axis = grad_norm (blue family), right axis = mIoU (red family).
Solid = GradAnchor, dashed = GradSlope sw10+maxlag300.

Outputs: figures/anchor_vs_slope_gradnorm_{ACDC,Cityscapes}.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save"
DSETS = [
    ("ACDC", "ACDCDataset/deyo_mlmp_gradanchor_continual",
             "ACDCDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300", 406),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_gradanchor_continual_5corr_sub100",
                   "CityscapesDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100", 500),
]


def read_miou(path):
    r, m = [], []
    with open(path) as f:
        next(f)
        for ln in f:
            ln = ln.strip()
            if ln.startswith("Round "):
                x = [t.strip() for t in ln.split(",")]
                try:
                    r.append(int(x[0].split()[1])); m.append(float(x[-1]))
                except (IndexError, ValueError):
                    pass
    return np.array(r), np.array(m)


def read_gradnorm(path, bpr):
    g = {}
    with open(path) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 2:
                continue
            try:
                tb = int(p[0]); gn = float(p[1])
            except ValueError:
                continue
            g.setdefault((tb - 1) // bpr + 1, []).append(gn)
    rounds = sorted(g)
    return np.array(rounds), np.array([np.mean(g[r]) for r in rounds])


for name, anchor_sub, slope_sub, bpr in DSETS:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    axr = ax.twinx()

    for sub, gcol, mcol, style, tag in [
        (anchor_sub, "#1f77b4", "#d62728", "-",  "GradAnchor"),
        (slope_sub,  "#1f77b4", "#d62728", "--", "sw10+lag300"),
    ]:
        gp = f"{SAVE}/{sub}/gate_log.csv"
        mp = f"{SAVE}/{sub}/results_all_rounds.txt"
        if not (os.path.exists(gp) and os.path.exists(mp)):
            continue
        gr, gn = read_gradnorm(gp, bpr)
        mr, mi = read_miou(mp)
        ax.plot(gr, gn, color=gcol, ls=style, lw=1.4, alpha=0.8,
                label=f"grad_norm ({tag})")
        axr.plot(mr, mi, color=mcol, ls=style, lw=2.0,
                 label=f"mIoU ({tag})")

    ax.set_xlabel("Round")
    ax.set_ylabel("grad_norm", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    axr.set_ylabel("mIoU (%)", color="#d62728")
    axr.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title(f"{name}: GradAnchor (solid) vs GradSlope sw10+lag300 (dashed)",
                 fontsize=11)
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/anchor_vs_slope_gradnorm_{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
