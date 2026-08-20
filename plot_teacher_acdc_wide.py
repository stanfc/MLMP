"""Wider-scale version of the teacher-usage comparison so MLMP-episodic is visible.
Output: figures/teacher_acdc_wide.png  (does NOT overwrite teacher_acdc.png)
Same four lines as plot_teacher_acdc.py, y-range extended down to include the
MLMP-episodic reference (29.84).
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def rd(p):
    r, m = [], []
    f = f"save/ACDCDataset/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([]), np.array([])
    for ln in open(f):
        if ln.startswith("Round "):
            x = [t.strip() for t in ln.split(",")]
            r.append(int(x[0].split()[1])); m.append(float(x[-1]))
    return np.array(r), np.array(m)


LINES = [
    ("distill  (EMA teacher-student TRAINING, LR3e-5)",
     "deyo_mlmp_hmgate2_distill_continual_lr3e-5", "#2ca02c", 2.6, "-"),
    ("inference  (EMA teacher at EVAL only, LR3e-5)",
     "deyo_mlmp_hmgate2_emaeval_continual_lr3e-5", "#ff7f0e", 2.2, "-"),
    ("original  (no teacher, LR3e-5)",
     "deyo_mlmp_hmgate2_continual_lr3e-5", "#000000", 2.2, "-"),
    ("original GDG-PA base (LR5e-6, starting point)",
     "deyo_mlmp_hmgate2_continual", "#7f7f7f", 1.6, ":"),
]
NOGATE_PEAK, EPISODIC = 33.5, 29.84

fig, ax = plt.subplots(figsize=(11, 6.5))
for name, p, c, lw, ls in LINES:
    r, m = rd(p)
    if len(m) == 0:
        continue
    ax.plot(r, m, color=c, lw=lw, ls=ls,
            label=f"{name}: mean {m.mean():.2f}, last {m[-1]:.1f}")
ax.axhline(NOGATE_PEAK, color="#d62728", ls="--", lw=1.3,
           label=f"no-gate PEAK {NOGATE_PEAK:.1f} (the ceiling)")
ax.axhline(EPISODIC, color="#1f77b4", ls="-.", lw=1.6,
           label=f"MLMP-episodic {EPISODIC:.1f} (needs per-sample reset)")

ax.set_ylim(29.3, 34.3)
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
ax.set_title("ACDC — EMA teacher usage vs the MLMP-episodic upper bound\n"
             "even 'no teacher' (LR3e-5) already clears episodic 29.8; distill reaches "
             "the no-gate ceiling 33.5 (+3.6 over episodic), no reset, no collapse",
             fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/teacher_acdc_wide.png", dpi=130, bbox_inches="tight")
print("saved -> figures/teacher_acdc_wide.png")
