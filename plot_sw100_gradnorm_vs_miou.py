"""
sw100 GradSlope: grad_norm (gate signal) vs mIoU per round, 3 datasets.
Left axis = grad_norm (window-mean, from gate_log.csv), right axis = mIoU.
Shaded where the gate is braking (lag > 0).

Output: figures/sw100_gradnorm_vs_miou.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save"
PANELS = [
    ("ACDC",       "ACDCDataset/deyo_mlmp_gradslope_continual_sw100", 406),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_gradslope_continual_sw100_5corr_sub100", 500),
    ("VOC20",      "PascalVOC20Dataset/deyo_mlmp_gradslope_continual_sw100_5corr_sub100", 500),
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


def read_gate(path, bpr):
    """gate_log: total_batches,grad_norm,grad_slope,lag,rst -> per-round mean."""
    g, lag = {}, {}
    with open(path) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 5:
                continue
            try:
                tb = int(p[0]); gn = float(p[1]); lg = float(p[3])
            except ValueError:
                continue
            rd = (tb - 1) // bpr + 1
            g.setdefault(rd, []).append(gn)
            lag.setdefault(rd, []).append(lg)
    rounds = sorted(g)
    gmean = np.array([np.mean(g[r]) for r in rounds])
    lmean = np.array([np.mean(lag[r]) for r in rounds])
    return np.array(rounds), gmean, lmean


fig, axes = plt.subplots(1, 3, figsize=(19, 5))
for ax, (name, sub, bpr) in zip(axes, PANELS):
    gp = f"{SAVE}/{sub}/gate_log.csv"
    mp = f"{SAVE}/{sub}/results_all_rounds.txt"
    if not (os.path.exists(gp) and os.path.exists(mp)):
        ax.set_title(f"{name} [missing]"); continue
    gr, gn, lg = read_gate(gp, bpr)
    mr, mi = read_miou(mp)

    # grad_norm (left, blue)
    ax.plot(gr, gn, color="#1f77b4", lw=1.5, label="grad_norm")
    ax.set_ylabel("grad_norm", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xlabel("Round")

    # mIoU (right, red)
    axr = ax.twinx()
    axr.plot(mr, mi, color="#d62728", lw=1.8, label="mIoU")
    axr.set_ylabel("mIoU (%)", color="#d62728")
    axr.tick_params(axis="y", labelcolor="#d62728")

    # shade braking (lag>0)
    brake = gr[lg > 0]
    for b in brake:
        ax.axvspan(b - 0.5, b + 0.5, color="orange", alpha=0.12, lw=0)
    n_brake = int((lg > 0).sum())
    last_r = int(mr[-1]) if len(mr) else 0
    ax.set_title(f"{name}  (to R{last_r}, braking rounds={n_brake})", fontsize=11)
    ax.grid(alpha=0.25)

fig.suptitle("GradSlope sw100: grad_norm (blue) vs mIoU (red) — orange = gate braking (lag>0)",
             fontsize=13, y=1.02)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
out = "figures/sw100_gradnorm_vs_miou.png"
fig.savefig(out, dpi=125, bbox_inches="tight")
print(f"saved -> {out}")
