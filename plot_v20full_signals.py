"""mIoU vs H_margin and mIoU vs grad_norm, dual-axis, per run.
Outputs figures/{tag}_miou_vs_hmargin.png and figures/{tag}_miou_vs_gradnorm.png."""
import os
import numpy as np
import matplotlib.pyplot as plt

# (tag, title, dir, RMAX)
RUNS = [
    ("v20full", "VOC20 full+15corr (GDG 0.95)",
     "save/PascalVOC20Dataset/deyo_mlmp_hmgate_continual_full_15corr_hdr095", 40),
    ("acdc", "ACDC (GDG 0.9)",
     "save/ACDCDataset/deyo_mlmp_hmgate_continual", 150),
]


def read_miou(path, rmax):
    r, m = [], []
    for ln in open(path):
        if ln.startswith("Round "):
            x = [t.strip() for t in ln.split(",")]
            rr = int(x[0].split()[1])
            if rr <= rmax:
                r.append(rr); m.append(float(x[-1]))
    return np.array(r), np.array(m)


def read_gate(path, bpr, rmax):
    g, h = {}, {}
    with open(path) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(",")
            if len(p) < 8:
                continue
            try:
                tb = int(p[0]); gn = float(p[1]); hm = float(p[3])
            except ValueError:
                continue
            rd = (tb - 1) // bpr + 1
            if rd > rmax:
                continue
            g.setdefault(rd, []).append(gn)
            h.setdefault(rd, []).append(hm)
    rr = np.array(sorted(g))
    return rr, np.array([np.mean(g[r]) for r in rr]), np.array([np.mean(h[r]) for r in rr])


def dual(mr, mi, sr, sv, sig_name, sig_color, title, rmax, pk_r, out, sig_span=None):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(mr, mi, color="#d62728", lw=2.2, label="mIoU")
    ax1.set_xlabel("Round"); ax1.set_ylabel("mIoU (%)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.axvline(pk_r, color="gray", ls="--", lw=1, alpha=0.7, label=f"mIoU peak @R{pk_r}")
    ax2 = ax1.twinx()
    ax2.plot(sr, sv, color=sig_color, lw=2.0, label=sig_name)
    ax2.set_ylabel(sig_name, color=sig_color); ax2.tick_params(axis="y", labelcolor=sig_color)
    if sig_span is not None:                       # fixed span, centred on this run's data
        mid = (sv.min() + sv.max()) / 2
        ax2.set_ylim(mid - sig_span / 2, mid + sig_span / 2)
    ax1.set_xlim(0, rmax); ax1.grid(alpha=0.3)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="best", fontsize=9)
    fig.suptitle(f"{title}: mIoU vs {sig_name}  (R0-{rmax})", fontsize=12)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"saved -> {out}")


# Pass 1: gather each run's data
data = []
for tag, title, d, rmax in RUNS:
    mr, mi = read_miou(f"{d}/results_all_rounds.txt", rmax)
    last_tb = 0
    for ln in open(f"{d}/gate_log.csv"):
        p = ln.strip().split(",")
        if len(p) >= 8 and p[0].isdigit():
            last_tb = max(last_tb, int(p[0]))
    bpr = int(round(last_tb / mr[-1]))
    gr, gn, hm = read_gate(f"{d}/gate_log.csv", bpr, rmax)
    data.append((tag, title, rmax, mr, mi, gr, gn, hm))

# common grad_norm span (largest run's range + 12% margin) so the two grad_norm
# figures share a scale -> ACDC's small wiggle reads as noise vs V20's real V-shape.
GSPAN = max(gn.max() - gn.min() for *_, gn, _ in data) * 1.12

for tag, title, rmax, mr, mi, gr, gn, hm in data:
    pk_r = int(mr[np.argmax(mi)])
    dual(mr, mi, gr, hm, "H_margin", "#2ca02c", title, rmax, pk_r,
         f"figures/{tag}_miou_vs_hmargin.png")
    dual(mr, mi, gr, gn, "grad_norm", "#1f77b4", title, rmax, pk_r,
         f"figures/{tag}_miou_vs_gradnorm.png", sig_span=GSPAN)
