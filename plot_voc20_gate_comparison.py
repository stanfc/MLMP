"""
VOC20-C gate comparison: no-gate (gate前) vs composite / grad-slope gates (gate後),
with No-Adapt and MLMP-episodic reference lines.

Output: figures/voc20_gate_comparison.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save/PascalVOC20Dataset"


def read_rounds(path):
    r, m = [], []
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line.startswith("Round "):
                continue
            p = [x.strip() for x in line.split(",")]
            try:
                r.append(int(p[0].split()[1])); m.append(float(p[-1]))
            except (IndexError, ValueError):
                pass
    return np.array(r), np.array(m)


def episodic_mean(path):
    vals = []
    with open(path) as f:
        for line in f:
            if "+/-" in line and "Duration" not in line:
                try:
                    vals.append(float(line.split(",")[1].split("+/-")[0]))
                except (IndexError, ValueError):
                    pass
    return float(np.mean(vals)) if vals else None


CURVES = [
    ("deyo_mlmp_continual_5corr_sub100",                "#d62728", "DeYO+MLMP (no gate / before)"),
    ("deyo_mlmp_composite_gate_continual_5corr_sub100", "#1f77b4", "+ Composite gate (conf+grad)"),
    ("deyo_mlmp_gradslope_continual_5corr_sub100",      "#2ca02c", "+ GradSlope gate"),
]

fig, ax = plt.subplots(figsize=(9, 5.5))
annot = []
for sub, color, label in CURVES:
    r, m = read_rounds(f"{SAVE}/{sub}/results_all_rounds.txt")
    ax.plot(r, m, color=color, lw=1.8, label=label)
    annot.append(f"{label.split('(')[0].strip()}: pk {m.max():.1f}@R{r[np.argmax(m)]}, "
                 f"last {m[-1]:.1f}, mean {m.mean():.1f}")

ep = episodic_mean(f"{SAVE}/mlmp_episodic_step_1/results.txt")
if ep is not None:
    ax.axhline(ep, color="#7f7f7f", ls="--", lw=1.4, label=f"MLMP episodic ({ep:.1f})")
_, na = read_rounds(f"{SAVE}/No_Adaptation_sub100/results_all_rounds.txt")
na = float(na.mean())
ax.axhline(na, color="black", ls=":", lw=1.4, label=f"No Adapt ({na:.1f})")

ax.set_title("VOC20-C (5corr sub100): no gate vs gated — 150 rounds", fontsize=13)
ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU (%)")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
ax.text(0.98, 0.02, "\n".join(annot), transform=ax.transAxes, fontsize=8,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
out = "figures/voc20_gate_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved -> {out}")
for a in annot:
    print(" ", a)
print(f"  MLMP episodic={ep:.1f}  No-Adapt={na:.1f}")
