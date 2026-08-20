"""OVSS gated batch-size follow-up: does deyo_mlmp_adagate_continual
(gradnorm_scaled = shallow_cap_mode=growing_scaled) hold up at batch=8/64 the
way it does at batch=1 (flagship v20_acdc_matched run, R150=78.4, stable)?
The no-gate observation run (v20_batch_ablation.png) showed ALL THREE batch
sizes collapse (negative tail slope) when ungated -- this tests whether the
gate rescues batch=8/64 too, or only batch=1.

Caveat: batch=1 reference uses --split val --subset_size=100 (flagship
v20_acdc_matched config, block=100/round); batch=8/64 use --ann_file
trainval.txt with no subset cap (block=2913/8=364, 2913/64=45 steps/round)
per the standing "don't control block length" instruction. Not a strict
block-matched comparison, but that's expected -- we're testing whether the
gate generalizes across batch size given its natural dataset chunking, not
holding block length constant.

Output: figures/v20_deyo_adagate_batch_ablation.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 20, "axes.labelsize": 24, "axes.titlesize": 17,
                     "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "axes.linewidth": 1.4, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

ARMS = [
    ("batch=1  (flagship, subset=100/round)",
     "save/PascalVOC20Dataset/v20_acdc_matched/adagate_ABmad05_ecdf_grow_scaled",
     "#1baf7a", 2.6, 5),
    ("batch=8  (trainval, ~364 steps/round)",
     "save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch8",
     "#2a78d6", 2.6, 4),
    ("batch=64 (trainval, ~45 steps/round)",
     "save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch64",
     "#e34948", 3.0, 6),
]


def load(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


fig, ax = plt.subplots(figsize=(13, 8.5))
legend_handles = []
any_partial = False
for label, root, color, lw, z in ARMS:
    path = f"{root}/results_all_rounds.txt"
    if not os.path.exists(path):
        continue
    r, m = load(path)
    if len(r) == 0:
        continue
    if r[-1] < 150:
        any_partial = True
    ax.plot(r, m, color=color, lw=lw, zorder=z)
    legend_handles.append(Line2D([], [], color=color, lw=lw,
                                 label=f"{label}: R1={m[0]:.1f}->R{r[-1]}={m[-1]:.1f}"
                                       + (" (running)" if r[-1] < 150 else "")))

ax.set_xlabel("Round")
ax.set_ylabel("V20 mean mIoU")
ax.set_xlim(0, 151)
ax.tick_params(width=1.4, length=7)
ax.grid(True, alpha=0.25, lw=0.9)
title = ("OVSS: deyo_mlmp_adagate (gradnorm_scaled) at batch=1/8/64\n"
         "does the gate that stabilizes batch=1 also hold at batch=8/64?")
if any_partial:
    title += "  (some arms still running -- partial)"
ax.set_title(title, pad=14)
ax.legend(handles=legend_handles, loc="center left", fontsize=15, framealpha=0.93)

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/v20_deyo_adagate_batch_ablation.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote v20_deyo_adagate_batch_ablation.png")
