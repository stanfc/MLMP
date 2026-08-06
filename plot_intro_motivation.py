"""INTRO / TALK figure: why the continual setting matters and why it is hard.
ACDC, 4 conditions (fog/night/rain/snow), 150 rounds, evaluate-before-adapt.

The core tension:
  * episodic MLMP is strong -- but it RESETS the model for every sample (not deployable)
  * remove the reset and the SAME method collapses to 1.5 mIoU
  * existing CTTA methods are either (i) improve-then-collapse or (ii) stable-but-stuck

Output: figures/intro_motivation_acdc.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})


def miou(p, root="save"):
    f = f"{root}/ACDCDataset/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


EPISODIC, NOADAPT = 29.84, 23.34

# (label, save_dir, colour, linewidth, group)
TEKAI = "save_tekai/save"
COLLAPSE = [
    ("MLMP  (reset removed)", "mlmp_continual", "#d62728", 3.2, "save"),
    ("TENT  (ICLR'21)", "tent_continual_Round150_lr_0.00001", "#ff7f0e", 2.4, TEKAI),
    ("DeYO  (ICLR'24)", "deyo_continual", "#e377c2", 2.0, "save"),
    ("CLIPArTT", "clipartt_continual_K3", "#8c564b", 1.8, "save"),
]
STUCK = [
    ("EATA  (ICML'22)",  "eata_continual",  "#1f77b4", 2.0, "save"),
    ("SAR   (ICLR'23)",  "sar_continual",   "#17becf", 2.0, "save"),
    ("RoTTA (CVPR'23)",  "rotta_continual", "#9467bd", 2.0, "save"),
    ("CoTTA (CVPR'22)  [only 10R available]", "cotta_step_1", "#bcbd22", 3.4, TEKAI),
]

fig, ax = plt.subplots(figsize=(13.5, 7.4))

# ---- reference bands -------------------------------------------------------
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=2.6, zorder=3)
ax.axhline(NOADAPT, color="k", ls=":", lw=2.0, zorder=3)

# ---- the two failure families ---------------------------------------------
for lab, p, c, lw, root in COLLAPSE:
    m = miou(p, root)
    if len(m) == 0:
        continue
    ax.plot(np.arange(1, len(m) + 1), m, color=c, lw=lw, zorder=5,
            label=f"{lab}  —  peak {m.max():.1f} → ends {m[-1]:.1f}")
for lab, p, c, lw, root in STUCK:
    m = miou(p, root)
    if len(m) == 0:
        continue
    ax.plot(np.arange(1, len(m) + 1), m, color=c, lw=lw, ls="-", alpha=0.9, zorder=4,
            label=f"{lab}  —  ends {m[-1]:.1f}")

# ---- no annotations: the speaker narrates ---------------------------------
ax.set_xlim(1, 150)
ax.set_ylim(0, 39)
ax.set_xlabel("Round  (one round = fog -> night -> rain -> snow)", fontsize=14)
ax.set_ylabel("mIoU (%)", fontsize=14)
ax.set_title("Continual TTA on ACDC  (150 rounds, no reset)", fontsize=16)
ax.grid(alpha=0.25, zorder=1)

# reference lines go in the legend, not as side text
from matplotlib.lines import Line2D
h, lb = ax.get_legend_handles_labels()
h = [Line2D([], [], color="#2ca02c", ls="--", lw=2.6),
     Line2D([], [], color="k", ls=":", lw=2.0)] + h
lb = [f"MLMP episodic (per-sample reset)  {EPISODIC:.1f}",
      f"No adaptation (source)  {NOADAPT:.1f}"] + lb
ax.legend(h, lb, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=3, framealpha=0.95)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/intro_motivation_acdc.png", dpi=140, bbox_inches="tight")
print("saved -> figures/intro_motivation_acdc.png")

for lab, p, _, _, root in COLLAPSE + STUCK:
    m = miou(p, root)
    if len(m):
        print(f"  {lab:26s} R={len(m):3d} mean={m.mean():5.1f} peak={m.max():5.1f} last={m[-1]:5.1f}")
