"""VOC20-C: all gate variants vs baselines. Output figures/voc20_all_variants.png"""
import os
import numpy as np
import matplotlib.pyplot as plt

S = "save/PascalVOC20Dataset"


def rd(p):
    r, m = [], []
    if not os.path.exists(p):
        return np.array([]), np.array([])
    with open(p) as f:
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


def epi(p):
    v = [float(l.split(",")[1].split("+/-")[0]) for l in open(p)
         if "+/-" in l and "Duration" not in l]
    return float(np.mean(v)) if v else None


curves = [
    ("deyo_mlmp_continual_5corr_sub100", "#d62728", "no-gate"),
    ("deyo_mlmp_gradslope_continual_5corr_sub100", "#2ca02c", "GradSlope sw10"),
    ("deyo_mlmp_gradslope_continual_sw50_5corr_sub100", "#9467bd", "GradSlope sw50"),
    ("deyo_mlmp_gradslope_continual_sw100_5corr_sub100", "#ff7f0e", "GradSlope sw100"),
    ("deyo_mlmp_gradslope_continual_sw10_maxlag300_5corr_sub100", "#000000", "GradSlope sw10+maxlag300"),
]

fig, ax = plt.subplots(figsize=(10, 6))
ann = []
for sub, c, lab in curves:
    r, m = rd(f"{S}/{sub}/results_all_rounds.txt")
    if len(m) == 0:
        continue
    lw = 2.4 if "maxlag300" in sub else 1.6
    done = "" if r[-1] >= 150 else f" (R{r[-1]})"
    ax.plot(r, m, color=c, lw=lw, label=lab + done)
    ann.append(f"{lab}: pk {m.max():.1f}@R{r[np.argmax(m)]}, last {m[-1]:.1f}, mean {m.mean():.1f}")

e = epi(f"{S}/mlmp_episodic_step_1/results.txt")
ax.axhline(e, color="gray", ls="--", lw=1.2, label=f"MLMP episodic ({e:.1f})")
_, na = rd(f"{S}/No_Adaptation_sub100/results_all_rounds.txt")
na = float(na.mean())
ax.axhline(na, color="black", ls=":", lw=1.2, label=f"No Adapt ({na:.1f})")

ax.set_title("VOC20-C: all gate variants (150 rounds)", fontsize=13)
ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU (%)")
ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
ax.text(0.98, 0.02, "\n".join(ann), transform=ax.transAxes, fontsize=7,
        va="bottom", ha="right", bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/voc20_all_variants.png", dpi=130, bbox_inches="tight")
print("saved -> figures/voc20_all_variants.png")
for a in ann:
    print(" ", a)
