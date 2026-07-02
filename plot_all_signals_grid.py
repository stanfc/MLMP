"""Two 3x3 grids (same layout as hdr_compare): per-dataset mIoU vs H_margin and
mIoU vs grad_norm, dual-axis. All datasets to R150; VOC20 full to R40.
Outputs figures/all_miou_vs_hmargin.png and figures/all_miou_vs_gradnorm.png."""
import os
import numpy as np
import matplotlib.pyplot as plt

# (title, dir, RMAX) -- GDG (hmgate) runs; VOC20 full uses the 0.95 run.
RUNS = [
    ("ACDC", "save/ACDCDataset/deyo_mlmp_hmgate_continual", 150),
    ("Cityscapes", "save/CityscapesDataset/deyo_mlmp_hmgate_continual_5corr_sub100", 150),
    ("VOC20", "save/PascalVOC20Dataset/deyo_mlmp_hmgate_continual_5corr_sub100", 150),
    ("VOC21", "save/PascalVOC21Dataset/deyo_mlmp_hmgate_continual_5corr_sub100", 150),
    ("PContext59", "save/PascalContext59Dataset/deyo_mlmp_hmgate_continual_5corr_sub100", 150),
    ("PContext60", "save/PascalContext60Dataset/deyo_mlmp_hmgate_continual_5corr_sub100", 150),
    ("COCO-Object", "save/COCOObjectDataset/deyo_mlmp_hmgate_continual_5corr_sub100", 150),
    ("COCO-Stuff", "save/COCOStuffDataset/deyo_mlmp_hmgate_continual_5corr_sub100", 150),
    ("VOC20 full+15corr",
     "save/PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr_hdr095", 40),
]
# gate_log column index of the signal
SIG_COL = {"H_margin": 3, "grad_norm": 1}


def read_miou(path, rmax):
    r, m = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    for ln in open(path):
        if ln.startswith("Round "):
            x = [t.strip() for t in ln.split(",")]
            rr = int(x[0].split()[1])
            if rr <= rmax:
                r.append(rr); m.append(float(x[-1]))
    return np.array(r), np.array(m)


def read_signal(path, col, bpr, rmax):
    d = {}
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(",")
            if len(p) <= col or not p[0].isdigit():
                continue
            try:
                tb = int(p[0]); v = float(p[col])
            except ValueError:
                continue
            rd = (tb - 1) // bpr + 1
            if rd > rmax:
                continue
            d.setdefault(rd, []).append(v)
    r = np.array(sorted(d))
    return r, np.array([np.mean(d[k]) for k in r])


def build(sig_name, sig_color, out):
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    axes = axes.reshape(-1)
    for ax, (title, d, rmax) in zip(axes, RUNS):
        mr, mi = read_miou(f"{d}/results_all_rounds.txt", rmax)
        if len(mr) == 0:
            ax.set_title(f"{title} [no data]"); ax.axis("off"); continue
        last_tb = 0
        for ln in open(f"{d}/gate_log.csv"):
            p = ln.strip().split(",")
            if p and p[0].isdigit():
                last_tb = max(last_tb, int(p[0]))
        bpr = max(1, int(round(last_tb / mr[-1])))
        sr, sv = read_signal(f"{d}/gate_log.csv", SIG_COL[sig_name], bpr, rmax)
        pk_r = int(mr[np.argmax(mi)])
        ax.plot(mr, mi, color="#d62728", lw=1.8)
        ax.set_ylabel("mIoU (%)", color="#d62728", fontsize=8)
        ax.tick_params(axis="y", labelcolor="#d62728", labelsize=7)
        ax.axvline(pk_r, color="gray", ls="--", lw=0.9, alpha=0.6)
        ax2 = ax.twinx()
        if len(sr):
            ax2.plot(sr, sv, color=sig_color, lw=1.6)
        ax2.set_ylabel(sig_name, color=sig_color, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=sig_color, labelsize=7)
        ax.set_xlim(0, rmax); ax.grid(alpha=0.25)
        ax.set_xlabel("Round", fontsize=8)
        ax.set_title(f"{title}  (peak mIoU @R{pk_r})", fontsize=10)
    fig.suptitle(f"GradDivGate: mIoU (red) vs {sig_name} ({sig_color}) per dataset "
                 f"— all R150, VOC20-full R40", fontsize=14, y=1.005)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"saved -> {out}")


build("H_margin", "#2ca02c", "figures/all_miou_vs_hmargin.png")
build("grad_norm", "#1f77b4", "figures/all_miou_vs_gradnorm.png")
