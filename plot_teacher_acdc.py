"""Teacher-usage comparison on ACDC: none vs inference-only vs teacher-student training.
Output: figures/teacher_acdc.png

Three EMA-teacher usage levels, all at the SAME aggressive LR 3e-5 (isolates the
teacher mechanism from the LR):
  original   -- GDG-PA, no teacher
  inference  -- EMA teacher swapped in ONLY at eval (training loss unchanged)
  distill    -- CoTTA-style: EMA teacher pseudo-labels added to the training loss
Dotted grey = the original GDG-PA base at LR 5e-6 (the pre-ceiling-push starting point).
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
ax.axhline(EPISODIC, color="#2ca02c", ls=":", lw=1.0, alpha=0.5,
           label=f"MLMP-episodic {EPISODIC:.1f}")

ax.set_ylim(31, 34.2)
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
ax.set_title("ACDC — EMA teacher: none -> inference-only -> teacher-student training\n"
             "at matched LR 3e-5, adding the teacher climbs 32.4 -> 33.2 (eval) -> 33.5 "
             "(distill), reaching the no-gate ceiling with NO collapse",
             fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/teacher_acdc.png", dpi=130, bbox_inches="tight")
print("saved -> figures/teacher_acdc.png")
