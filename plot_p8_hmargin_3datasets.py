"""H_margin vs mIoU, 3 datasets — same layout/style as p9_gradnorm_vs_miou
(blue = H_margin left axis, red = mIoU right axis, Spearman rho in the title).
Source = the no-restore MONITOR runs (no gate, so no circularity).

Output: figures/p8_hmargin_vs_miou.png
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans"})

SAVE = "save"
PANELS = [
    ("V20",        "PascalVOC20Dataset/deyo_mlmp_continual_monitor_5corr_sub100"),
    ("Cityscapes", "CityscapesDataset/deyo_mlmp_continual_monitor_5corr_sub100"),
    ("ACDC",       "ACDCDataset/deyo_mlmp_continual_monitor"),
]


def read_miou(path):
    r, m = [], []
    with open(path) as f:
        next(f)
        for line in f:
            if line.startswith("Round "):
                p = [x.strip() for x in line.split(",")]
                r.append(int(p[0].split()[1])); m.append(float(p[-1]))
    return np.array(r), np.array(m)


def read_signal(path, col):
    by_round = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row[col]
            if v not in ("", "nan"):
                by_round.setdefault(int(row["round"]), []).append(float(v))
    rounds = np.array(sorted(by_round))
    return rounds, np.array([np.mean(by_round[r]) for r in rounds])


def spearman(a, b):
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


C_SIG, C_MIOU = "tab:blue", "tab:red"

fig, axes = plt.subplots(1, 3, figsize=(24, 6.6))
for ax, (name, sub) in zip(axes, PANELS):
    gr, sg = read_signal(f"{SAVE}/{sub}/signals_log.csv", "h_margin")
    mr, mi = read_miou(f"{SAVE}/{sub}/results_all_rounds.txt")
    common = np.intersect1d(gr, mr)
    sg = sg[np.searchsorted(gr, common)]; mi = mi[np.searchsorted(mr, common)]
    rho = spearman(sg, mi)

    ax.plot(common, sg, color=C_SIG, lw=2.6)
    ax.set_ylabel(r"$H_{\mathrm{margin}}$", color=C_SIG, fontsize=26)
    ax.tick_params(axis="y", labelcolor=C_SIG, labelsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlabel("Round", fontsize=26)

    axb = ax.twinx()
    axb.plot(common, mi, color=C_MIOU, lw=2.6, alpha=0.85)
    axb.set_ylabel("mIoU (%)", color=C_MIOU, fontsize=26)
    axb.tick_params(axis="y", labelcolor=C_MIOU, labelsize=20)

    ax.set_title(f"mIoU vs. $H_{{\\mathrm{{margin}}}}$ ({name})\n"
                 f"$H_{{\\mathrm{{margin}}}}$  (ρ={rho:+.2f})", fontsize=24)
    print(f"  {name:11s} rho={rho:+.2f}  H:{sg.min():.2f}..{sg.max():.2f}  mIoU:{mi.max():.1f}->{mi[-1]:.1f}")

fig.tight_layout(w_pad=6.0)
fig.subplots_adjust(wspace=0.42)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/p8_hmargin_vs_miou.png", dpi=140, bbox_inches="tight")
print("saved -> figures/p8_hmargin_vs_miou.png")
