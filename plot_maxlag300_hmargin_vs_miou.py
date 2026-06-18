"""
sw10+maxlag300: h_margin (evaluate-based, entropy_log.csv) vs mIoU per round,
ONE FIGURE PER DATASET. Left axis = h_margin, right axis = mIoU.

Outputs: figures/maxlag300_hmargin_vs_miou_{ACDC,Cityscapes,VOC20}.png
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save"
RUNS = [
    ("ACDC",       "ACDCDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300"),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100"),
    ("VOC20",      "PascalVOC20Dataset/deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100"),
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


def read_hmargin(path):
    """entropy_log.csv -> per-round mean h_margin."""
    by = {}
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                rd = int(row["round"]); h = float(row["h_margin"])
            except (KeyError, ValueError):
                continue
            by.setdefault(rd, []).append(h)
    rounds = sorted(by)
    return np.array(rounds), np.array([np.mean(by[r]) for r in rounds])


for name, sub in RUNS:
    ep = f"{SAVE}/{sub}/entropy_log.csv"
    mp = f"{SAVE}/{sub}/results_all_rounds.txt"
    if not (os.path.exists(ep) and os.path.exists(mp)):
        print(f"[skip] {name}: missing logs")
        continue
    hr, hm = read_hmargin(ep)
    mr, mi = read_miou(mp)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hr, hm, color="#2ca02c", lw=1.5, label="h_margin (evaluate)")
    ax.set_ylabel("h_margin", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_xlabel("Round")

    axr = ax.twinx()
    axr.plot(mr, mi, color="#d62728", lw=1.9, label="mIoU")
    axr.set_ylabel("mIoU (%)", color="#d62728")
    axr.tick_params(axis="y", labelcolor="#d62728")

    last_r = int(mr[-1]) if len(mr) else 0
    pk = mi.max(); pkr = mr[int(np.argmax(mi))]
    ax.set_title(f"{name}  sw10+maxlag300  (to R{last_r}, "
                 f"mIoU pk {pk:.1f}@R{pkr} last {mi[-1]:.1f})", fontsize=11)
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/maxlag300_hmargin_vs_miou_{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
