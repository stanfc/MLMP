"""Text-centric mechanisms on top of GDG-PA, ACDC. Output: figures/text_acdc.png

Answers the 'CLIP must use its text embedding' concern empirically:
  (A) repel      -- push visual features AWAY from top-k confusable WRONG-class text anchors
  (B) srcdistill -- distill toward the frozen SOURCE (text-manifold-preserving) model
Both are stable but neither raises the ceiling; the productive lever is eval-side (EMA-eval).
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
    ("EMA-eval (productive lever, ref)", "deyo_mlmp_hmgate2_emaeval_continual", "#2ca02c", 2.0, "-"),
    ("GDG-PA base", "deyo_mlmp_hmgate2_continual", "#000000", 2.6, "-"),
    ("(B) srcdistill  (text-manifold anchor)", "deyo_mlmp_hmgate2_srcdistill_continual", "#1f77b4", 1.8, "-"),
    ("(A) repel  (push from confusable text)", "deyo_mlmp_hmgate2_repel_continual", "#d62728", 1.8, "-"),
]
EPISODIC, NOADAPT = 29.84, 23.34

fig, ax = plt.subplots(figsize=(11, 6.5))
for name, p, c, lw, ls in LINES:
    r, m = rd(p)
    if len(m) == 0:
        continue
    ax.plot(r, m, color=c, lw=lw, ls=ls,
            label=f"{name}: mean {m.mean():.2f}, last {m[-1]:.1f}")
ax.axhline(EPISODIC, color="#2ca02c", ls="--", lw=1.1, alpha=0.5, label=f"MLMP-episodic {EPISODIC:.1f}")

ax.set_ylim(29, 33.2)
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)")
ax.set_title("ACDC — text-centric mechanisms on top of GDG-PA (base mean 31.5)\n"
             "text repulsion / text-manifold anchor are STABLE but neutral (≈base); "
             "eval-side EMA-eval is the lever that moves the ceiling (+0.8)",
             fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/text_acdc.png", dpi=130, bbox_inches="tight")
print("saved -> figures/text_acdc.png")
