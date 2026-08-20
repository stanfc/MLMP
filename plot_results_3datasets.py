"""RESULT slide: ours vs all protocol-matched baselines vs no-adapt vs MLMP-episodic,
on ACDC / Cityscapes / VOC20.
Output: figures/results_3datasets.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 30, "axes.labelsize": 34, "axes.titlesize": 40,
                     "xtick.labelsize": 26, "ytick.labelsize": 26,
                     "axes.linewidth": 1.6, "font.family": "DejaVu Sans"})
TEKAI = "save_tekai/save"

# publication venue tags for the legend (match p7 style)
VENUE = {"MLMP-continual": "MLMP-continual (NeurIPS'25)", "DeYO": "DeYO (ICLR'24)",
         "EATA": "EATA (ICML'22)", "RoTTA": "RoTTA (CVPR'23)"}


def miou(ds, p, root="save"):
    f = f"{root}/{ds}/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


def episodic(ds, p):
    """mean mIoU over the corruptions listed in an episodic results.txt"""
    f = f"save/{ds}/{p}/results.txt"
    v = []
    for l in open(f):
        parts = l.split(",")
        if len(parts) >= 2 and not l.startswith("mIoU") and "+/-" in parts[1]:
            try:
                v.append(float(parts[1].split("+/-")[0]))
            except ValueError:
                pass
    return float(np.mean(v))


OURS = "#d62728"          # ours — red, thick
BASE_C = {"TENT": "#ff7f0e", "MLMP-continual": "#8c564b", "DeYO": "#e377c2",
          "CLIPArTT": "#7f7f7f", "EATA": "#1f77b4", "RoTTA": "#9467bd",
          "CoTTA": "#17becf"}

PANELS = [
    dict(title="ACDC", ds="ACDCDataset",
         ours="deyo_mlmp_hmgate2_continual",   # base GDG-PA (LR5e-6) — the stable one
         noadapt=23.34, epi=29.84, ylim=(0, 37),
         base=[("MLMP-continual", "mlmp_continual", "save"),
               ("DeYO", "deyo_continual", "save"),
               ("EATA", "eata_continual", "save"),
               ("RoTTA", "rotta_continual", "save")]),
    dict(title="Cityscapes", ds="CityscapesDataset",
         ours="deyo_mlmp_hmgate2_continual_5corr_sub100",   # base GDG-PA (LR5e-6)
         noadapt=20.56, epi=None, epi_run="mlmp_episodic_sub100_step_1", ylim=(0, 28),
         base=[("MLMP-continual", "mlmp_continual_5corr_sub100", "save"),
               ("DeYO", "deyo_continual_5corr_sub100", "save"),
               ("EATA", "eata_continual_5corr_sub100", "save"),
               ("RoTTA", "rotta_continual_5corr_sub100", "save")]),
    dict(title="VOC20", ds="PascalVOC20Dataset",
         ours="deyo_mlmp_hmgate2_continual_5corr_sub100",   # base GDG-PA (LR5e-6)
         noadapt=68.60, epi=None, epi_run="mlmp_episodic_step_1", ylim=(0, 82),
         base=[("MLMP-continual", "mlmp_continual_5corr_sub100", "save"),
               ("DeYO", "deyo_continual_5corr_sub100", "save"),
               ("EATA", "eata_continual_5corr_sub100", "save"),
               ("RoTTA", "rotta_continual_5corr_sub100", "save")]),
]

fig, axes = plt.subplots(1, 3, figsize=(22, 7.6))

for ax, P in zip(axes, PANELS):
    epi = P["epi"] if P["epi"] is not None else episodic(P["ds"], P["epi_run"])
    ax.axhline(epi, color="#2ca02c", ls="--", lw=2.4, zorder=3)
    ax.axhline(P["noadapt"], color="k", ls=":", lw=2.2, zorder=3)

    for name, p, root in P["base"]:
        m = miou(P["ds"], p, root)
        if len(m) == 0:
            print(f"  MISSING {P['ds']} {name}")
            continue
        ax.plot(np.arange(1, len(m) + 1), m, color=BASE_C[name], lw=2.6,
                alpha=0.9, zorder=4, label=name)

    m = miou(P["ds"], P["ours"])
    ax.plot(np.arange(1, len(m) + 1), m, color=OURS, lw=5.5, zorder=6, label="Ours")

    ax.set_xlim(1, 150); ax.set_ylim(*P["ylim"])
    ax.set_title(P["title"], fontsize=40, fontweight="bold", pad=10)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.22, lw=0.9)
    ax.tick_params(width=1.6, length=8)

    print(f"{P['ds']:22s} ours mean={m.mean():.2f} last={m[-1]:.2f} | "
          f"episodic={epi:.2f} no-adapt={P['noadapt']:.2f}")

axes[0].set_ylabel("mIoU (%)")

handles = [Line2D([], [], color="#2ca02c", ls="--", lw=2.8),
           Line2D([], [], color="k", ls=":", lw=2.6)] + \
          [Line2D([], [], color=BASE_C[n], lw=2.8)
           for n in ["MLMP-continual", "DeYO", "EATA", "RoTTA"]] + \
          [Line2D([], [], color=OURS, lw=5.5)]
labels = ["MLMP-episodic (NeurIPS'25)", "No adaptation (source)"] + \
         [VENUE[n] for n in ["MLMP-continual", "DeYO", "EATA", "RoTTA"]] + ["Ours (proposed)"]
fig.tight_layout(rect=[0, 0.18, 1, 1])
# legend spans the FULL figure width (mode="expand"), 3 columns -> 3 rows, roomy so no overlap
fig.legend(handles, labels, fontsize=25, loc="lower center",
           bbox_to_anchor=(0.02, -0.04, 0.96, 0.16), mode="expand", ncol=3,
           framealpha=0.95, handlelength=2.0, handletextpad=0.6,
           columnspacing=1.4, borderaxespad=0.0)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/results_3datasets.png", dpi=135, bbox_inches="tight")
print("saved -> figures/results_3datasets.png")
