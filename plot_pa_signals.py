"""GDG-PA (hmgate2): mIoU vs H_margin and mIoU vs grad_norm for V20-full and ACDC.
grad_norm axes share a span so the two are comparable.
Outputs figures/pa_{tag}_miou_vs_{hmargin,gradnorm}.png"""
import os
import numpy as np
import matplotlib.pyplot as plt

# (tag, title, dir, RMAX)
RUNS = [
    ("v20full", "VOC20 full+15corr (GDG-PA)",
     "save/PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_full_15corr", 120),
    ("acdc", "ACDC (GDG-PA)",
     "save/ACDCDataset/deyo_mlmp_hmgate2_continual", 150),
]
# gate_log.csv cols: total_batches,grad_norm,grad_slope,h_margin,collapse,...,deep
SIG_COL = {"H_margin": 3, "grad_norm": 1}


def read_miou(path, rmax):
    r, m = [], []
    for ln in open(path):
        if ln.startswith("Round "):
            x = [t.strip() for t in ln.split(",")]
            rr = int(x[0].split()[1])
            if rr <= rmax:
                r.append(rr); m.append(float(x[-1]))
    return np.array(r), np.array(m)


def read_sig(path, col, bpr, rmax):
    d = {}
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


def dual(mr, mi, sr, sv, sig, color, title, rmax, pk_r, out, span=None):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(mr, mi, color="#d62728", lw=2.2, label="mIoU")
    ax1.set_xlabel("Round"); ax1.set_ylabel("mIoU (%)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.axvline(pk_r, color="gray", ls="--", lw=1, alpha=0.7, label=f"mIoU peak @R{pk_r}")
    ax2 = ax1.twinx()
    ax2.plot(sr, sv, color=color, lw=2.0, label=sig)
    ax2.set_ylabel(sig, color=color); ax2.tick_params(axis="y", labelcolor=color)
    if span is not None and len(sv):
        mid = (sv.min() + sv.max()) / 2
        ax2.set_ylim(mid - span / 2, mid + span / 2)
    ax1.set_xlim(0, rmax); ax1.grid(alpha=0.3)
    l1, a1 = ax1.get_legend_handles_labels(); l2, a2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, a1 + a2, loc="best", fontsize=9)
    fig.suptitle(f"{title}: mIoU vs {sig}  (R0-{rmax})", fontsize=12)
    fig.tight_layout(); os.makedirs("figures", exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight"); print(f"saved -> {out}")


data = []
for tag, title, d, rmax in RUNS:
    mr, mi = read_miou(f"{d}/results_all_rounds.txt", rmax)
    last_tb = max(int(p[0]) for p in (l.strip().split(",") for l in open(f"{d}/gate_log.csv"))
                  if p and p[0].isdigit())
    bpr = max(1, int(round(last_tb / mr[-1])))
    gr, gn = read_sig(f"{d}/gate_log.csv", SIG_COL["grad_norm"], bpr, rmax)
    hr, hm = read_sig(f"{d}/gate_log.csv", SIG_COL["H_margin"], bpr, rmax)
    data.append((tag, title, rmax, mr, mi, gr, gn, hr, hm))

GSPAN = max(gn.max() - gn.min() for *_, gn, _, _ in data) * 1.12
for tag, title, rmax, mr, mi, gr, gn, hr, hm in data:
    pk_r = int(mr[np.argmax(mi)])
    dual(mr, mi, hr, hm, "H_margin", "#2ca02c", title, rmax, pk_r,
         f"figures/pa_{tag}_miou_vs_hmargin.png")
    dual(mr, mi, gr, gn, "grad_norm", "#1f77b4", title, rmax, pk_r,
         f"figures/pa_{tag}_miou_vs_gradnorm.png", span=GSPAN)
