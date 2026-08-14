"""V20, 100-sub, 15-corruption (3x the img/round of the 5corr protocol): does more
corruption diversity + more total adaptation steps induce collapse anywhere in the
150-round trajectory? All 8 arms: no-adapt, MLMP episodic, ctrl (old GDG-PA),
flagship, and the 4 shallow_cap_mode variants.

growing_scaled / growing_hmargin are still running (restarted after an earlier
session teardown killed them mid-run) -- plotted as partial, dashed, faded lines.

Output: figures/v20_15corr_collapse_check.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 20, "axes.labelsize": 24, "axes.titlesize": 21,
                     "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "axes.linewidth": 1.4, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
ROOT = "save/PascalVOC20Dataset/v20_15corr"


def load_continual(path):
    r, m = [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("Round "):
            continue
        parts = [p.strip() for p in line.split(",")]
        r.append(int(parts[0].split()[1])); m.append(float(parts[-1]))
    return np.array(r), np.array(m)


def load_episodic_mean(path):
    vals = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(("mIoU", "GPU", "Total", "Mean")):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vals.append(float(parts[1].split("+/-")[0].strip()))
        except (ValueError, IndexError):
            continue
    return np.mean(vals) if vals else None


# fixed categorical order/colors, consistent with the earlier 3-dataset comparison figure
# Label naming: <what "distance since healthy" is measured against>_<uncapped|scaled>.
#   fixed_cap        : original flagship -- SHALLOW restore reach hard-capped at 6 windows
#   gradnorm_uncapped: reach = time since grad_norm's own minimum, no cap (A)
#   gradnorm_scaled  : same, but scaled by how severe today's drift is (P1)
#   hmargin_uncapped : reach = time since H_margin's own peak, no cap (P2)
#   hmargin_scaled   : reach = time since H_margin peak x severity -- the winner (P3)
ARMS = [
    # label,                    key,                                color,     lw,  z,  done
    ("GDG-PA",                  "adagate_ctrl",                      "#9467bd", 2.6, 4, True),
    ("ABmad05_ecdf",             "adagate_ABmad05_ecdf",              "#7f7f7f", 3.0, 5, True),
    ("gradnorm_uncapped",        "adagate_ABmad05_ecdf_grow",         "#ff7f0e", 2.2, 3, True),
    ("gradnorm_scaled",          "adagate_ABmad05_ecdf_grow_scaled",  "#1f77b4", 2.2, 3, True),
]

fig, ax = plt.subplots(figsize=(15, 9.5))

_, m_na = load_continual(f"{ROOT}/no_adapt/results_all_rounds.txt")
na_mean = m_na.mean()
ep_mean = load_episodic_mean(f"{ROOT}/mlmp_episodic/results.txt")

ax.axhline(na_mean, color="black", ls=":", lw=2.0, zorder=2)
ax.axhline(ep_mean, color="#555555", ls="-.", lw=2.0, zorder=2)

legend_handles = [
    Line2D([], [], color="black", ls=":", lw=2.0, label=f"no-adapt: {na_mean:.2f}"),
    Line2D([], [], color="#555555", ls="-.", lw=2.0, label=f"MLMP episodic: {ep_mean:.2f}"),
]

for label, key, color, lw, z, done in ARMS:
    r, m = load_continual(f"{ROOT}/{key}/results_all_rounds.txt")
    n_done = len(r)
    if done:
        ax.plot(r, m, color=color, lw=lw, zorder=z, solid_capstyle="round")
        legend_handles.append(Line2D([], [], color=color, lw=lw,
                                     label=f"{label}: mean={m.mean():.2f} (150R done)"))
    else:
        ax.plot(r, m, color=color, lw=lw, zorder=z, ls="--", alpha=0.55)
        ax.scatter([r[-1]], [m[-1]], color=color, s=45, zorder=z + 1, alpha=0.8)
        legend_handles.append(Line2D([], [], color=color, lw=lw, ls="--", alpha=0.7,
                                     label=f"{label}: R{n_done}/150 so far (running)"))

ax.set_xlabel("Round")
ax.set_ylabel("VOC20 mIoU (15 corruptions, sub100)")
ax.set_xlim(0, 152)
ax.set_ylim(65, 80)
ax.tick_params(width=1.4, length=7)
ax.grid(True, alpha=0.25, lw=0.9)
ax.set_title("V20 sub100, 15-corruption (3x img/round of the 5corr protocol):\nany sign of collapse from more corruption diversity / more adaptation steps?",
             pad=14)
ax.legend(handles=legend_handles, loc="lower right", fontsize=13.5, framealpha=0.92)

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/v20_15corr_collapse_check.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote v20_15corr_collapse_check.png")
