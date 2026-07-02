"""
Diagnose why HMGate still drops on COCO-Object, COCO-Stuff, VOC20(full+15corr):
plot mIoU / H_margin / grad_norm per round (stacked, shared x), + collapse-regime shading.

Question: does H_margin DROP (collapse type -> deep restore should have engaged) or
stay HIGH while mIoU falls (uniform type -> HMGate keeps it shallow -> can't hold)?

Output: figures/dropping_diag_{name}.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

RUNS = [
    ("COCO-Object", "save/COCOObjectDataset/deyo_mlmp_hmgate_continual_5corr_sub100", 500),
    ("COCO-Stuff", "save/COCOStuffDataset/deyo_mlmp_hmgate_continual_5corr_sub100", 500),
    ("VOC20 full+15corr", "save/PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr", 21735),
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
    """per-round mean of grad_norm, h_margin, collapse-fraction."""
    g, h, c = {}, {}, {}
    with open(path) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 8:
                continue
            try:
                tb = int(p[0]); gn = float(p[1]); hm = float(p[3]); col = int(p[4])
            except ValueError:
                continue
            rd = (tb - 1) // bpr + 1
            g.setdefault(rd, []).append(gn)
            h.setdefault(rd, []).append(hm)
            c.setdefault(rd, []).append(col)
    rounds = sorted(g)
    return (np.array(rounds),
            np.array([np.mean(g[r]) for r in rounds]),
            np.array([np.mean(h[r]) for r in rounds]),
            np.array([np.mean(c[r]) for r in rounds]))


for name, d, bpr in RUNS:
    mr, mi = read_miou(f"{d}/results_all_rounds.txt")
    gr, gn, hm, cf = read_gate(f"{d}/gate_log.csv", bpr)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    # mIoU
    axes[0].plot(mr, mi, color="#d62728", lw=2)
    axes[0].set_ylabel("mIoU (%)", color="#d62728")
    axes[0].set_title(f"{name}: mIoU pk{mi.max():.1f}@R{mr[np.argmax(mi)]} "
                      f"last{mi[-1]:.1f} drop{mi.max()-mi[-1]:.1f}", fontsize=11)
    # H_margin + collapse shading
    axes[1].plot(gr, hm, color="#2ca02c", lw=1.6)
    axes[1].axhline(0.9 * hm.max(), color="gray", ls="--", lw=1,
                    label=f"0.9*max ({0.9*hm.max():.2f}) = collapse threshold")
    axes[1].set_ylabel("H_margin", color="#2ca02c")
    axes[1].legend(fontsize=8, loc="best")
    # grad_norm
    axes[2].plot(gr, gn, color="#1f77b4", lw=1.6)
    axes[2].set_ylabel("grad_norm", color="#1f77b4")
    axes[2].set_xlabel("Round")

    # shade collapse-regime rounds across all panels
    for b in gr[cf > 0.5]:
        for ax in axes:
            ax.axvspan(b - 0.5, b + 0.5, color="orange", alpha=0.10, lw=0)
    for ax in axes:
        ax.grid(alpha=0.25)

    n_col = int((cf > 0.5).sum())
    fig.suptitle(f"{name} — orange = collapse regime (deep restore), {n_col}/{len(gr)} rounds",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    safe = name.split()[0].replace("-", "")
    out = f"figures/dropping_diag_{safe}.png"
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}  (H_margin range {hm.min():.2f}-{hm.max():.2f}, "
          f"collapse rounds {n_col}/{len(gr)})")
