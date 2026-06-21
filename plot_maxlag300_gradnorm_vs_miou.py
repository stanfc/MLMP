"""
sw10+maxlag300 GradSlope: grad_norm vs mIoU per round, ONE FIGURE PER DATASET.
Left axis = grad_norm (window-mean, gate_log.csv), right axis = mIoU.
Orange shading = gate braking (lag > 0).

Outputs: figures/maxlag300_gradnorm_vs_miou_{ACDC,Cityscapes,VOC20}.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save"
RUNS = [
    ("ACDC",       "ACDCDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300", 406),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100", 500),
    ("VOC20",      "PascalVOC20Dataset/deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100", 500),
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
    return (np.array(rounds),
            np.array([np.mean(g[r]) for r in rounds]),
            np.array([np.mean(lag[r]) for r in rounds]))


for name, sub, bpr in RUNS:
    gp = f"{SAVE}/{sub}/gate_log.csv"
    mp = f"{SAVE}/{sub}/results_all_rounds.txt"
    if not (os.path.exists(gp) and os.path.exists(mp)):
        print(f"[skip] {name}: missing logs")
        continue
    gr, gn, lg = read_gate(gp, bpr)
    mr, mi = read_miou(mp)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gr, gn, color="#1f77b4", lw=1.5, label="grad_norm")
    ax.set_ylabel("grad_norm", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xlabel("Round")

    axr = ax.twinx()
    axr.plot(mr, mi, color="#d62728", lw=1.9, label="mIoU")
    axr.set_ylabel("mIoU (%)", color="#d62728")
    axr.tick_params(axis="y", labelcolor="#d62728")

    for b in gr[lg > 0]:
        ax.axvspan(b - 0.5, b + 0.5, color="orange", alpha=0.12, lw=0)

    n_brake = int((lg > 0).sum())
    last_r = int(mr[-1]) if len(mr) else 0
    pk = mi.max(); pkr = mr[int(np.argmax(mi))]
    ax.set_title(f"{name}  sw10+maxlag300  (to R{last_r}, "
                 f"mIoU pk {pk:.1f}@R{pkr} last {mi[-1]:.1f}, braking rounds={n_brake})",
                 fontsize=11)
    ax.grid(alpha=0.25)
    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2 + [plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.2)],
              l1 + l2 + ["braking (lag>0)"], loc="best", fontsize=8)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/maxlag300_gradnorm_vs_miou_{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
