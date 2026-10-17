"""V20 batch=1/8/64, in the style of the user's reference Cityscapes chart:
"ours" (deyo_mlmp_adagate, gradnorm_scaled) vs "DeYO+MLMP" (no gate/restore)
trajectories, with MLMP-episodic and no-adaptation as flat reference lines.

Filled with whatever is ALREADY COMPLETE AT R150 right now, per explicit
request, rather than waiting on the clean fulldata reruns (still in
progress -- see bash/_v20_batch_ablation_nogate_fulldata.sh and
bash/_v20_deyo_adagate_batch1_clean.sh). Known caveats, to fix once those
land:
- no_adapt: batch-invariant (frozen model), R1=70.41 taken as the final value
  (only 1/150 rounds run, but no_adapt doesn't change round to round).
- mlmp_episodic: only 3/5 corruptions done (snow/frost/fog); reference line
  uses their mean (73.13) -- NOT the true 5-corruption episodic upper bound.
- no-gate ("DeYO+MLMP") batch=1/8/64: NOW using the new fulldata rerun
  (full trainval.txt, no subset cap, matches the gated arms' data config) --
  still in progress as of last check, partial trajectories.
- gated ("ours") batch=1: the OLD flagship result (v20_acdc_matched,
  --split val --subset_size=100), NOT the new clean trainval.txt rerun (in
  progress, only R4 so far). Different underlying image subset than the
  no-gate batch=1 line above and than gated batch=8/64 below -- the R1
  mismatch this whole investigation started from. batch=8/64 gated lines ARE
  on full trainval.txt already (complete, no caveat).

Output: figures/v20_batch_ablation_with_refs.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "axes.linewidth": 1.4, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

NO_ADAPT = 70.41
MLMP_EPISODIC = (75.00 + 69.04 + 75.36 + 79.38 + 75.27) / 5  # complete, all 5 corruptions

PANELS = [
    ("batch = 1",
     "save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch1_fulldata/results_all_rounds.txt",
     "save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch1_clean/results_all_rounds.txt"),
    ("batch = 8",
     "save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch8_fulldata/results_all_rounds.txt",
     "save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch8/results_all_rounds.txt"),
    ("batch = 64",
     # old subset-capped run (subset=2880) is close enough to the full 2913
     # that the mismatch barely matters here -- use it since it's already
     # complete at R150, instead of waiting on the fulldata rerun (partial).
     "save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch64/results_all_rounds.txt",
     "save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch64/results_all_rounds.txt"),
]
# gated ("ours") lines are h_max_mode=lifetime (method of record). batch=8/64
# complete at R150; batch=1 is the clean trainval.txt rerun, resumed from its
# R101 checkpoint back to R150.


def load(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True)

for ax, (title, nogate_path, gated_path) in zip(axes, PANELS):
    ax.axhline(MLMP_EPISODIC, color="#1b9e77", lw=2.2, ls="--", zorder=2)
    ax.axhline(NO_ADAPT, color="#7b3294", lw=2.2, ls=":", zorder=2)

    r_no, m_no = load(nogate_path)
    ax.plot(r_no, m_no, color="#2166ac", lw=2.4, zorder=3)
    nogate_partial = r_no[-1] < 150

    gated_partial = False
    if os.path.exists(gated_path):
        r_g, m_g = load(gated_path)
        if len(r_g) > 0:
            gated_partial = r_g[-1] < 150
            ax.plot(r_g, m_g, color="#d6272f", lw=2.8, zorder=4)

    tag = ""
    if nogate_partial and gated_partial:
        tag = "  (both partial)"
    elif nogate_partial:
        tag = "  (no-gate partial)"
    elif gated_partial:
        tag = "  (ours partial)"
    ax.set_title(title + tag, pad=10)
    ax.set_xlabel("round")
    ax.grid(True, alpha=0.25, lw=0.8)

axes[0].set_ylabel("mean mIoU")

legend_handles = [
    Line2D([], [], color="#d6272f", lw=2.8, label="ours (gradnorm-scaled)"),
    Line2D([], [], color="#2166ac", lw=2.4, label="DeYO+MLMP (no gate/restore)"),
    Line2D([], [], color="#1b9e77", lw=2.2, ls="--",
           label=f"MLMP-episodic ({MLMP_EPISODIC:.1f})"),
    Line2D([], [], color="#7b3294", lw=2.2, ls=":", label=f"no adaptation ({NO_ADAPT:.1f})"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=13,
           bbox_to_anchor=(0.5, -0.06), framealpha=0.93)

fig.suptitle("V20 (PascalVOC20) — no-gate now on the fulldata rerun (in progress); gated batch=1 resuming to R150",
             fontsize=15, y=1.02)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/v20_batch_ablation_with_refs.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote v20_batch_ablation_with_refs.png")
