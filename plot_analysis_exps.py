"""Two analysis experiments (GDG-PA, V20 5corr sub100):
 (1) OOD probe: ID (train 5corr) vs OOD (held-out 5corr, frozen) — same trend?
 (2) resample: fresh random 100-subset each round vs fixed subset — overfit to the 100?
Output: figures/analysis_exps.png"""
import os
import numpy as np
import matplotlib.pyplot as plt


def rd(p, fn="results_all_rounds.txt"):
    r, m = [], []
    f = f"save/{p}/{fn}"
    if not os.path.exists(f):
        return np.array([]), np.array([])
    for ln in open(f):
        if ln.startswith("Round "):
            x = [t.strip() for t in ln.split(",")]
            r.append(int(x[0].split()[1])); m.append(float(x[-1]))
    return np.array(r), np.array(m)


E1 = "PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100_oodprobe"
E2 = "PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100_resample"
FIXED = "PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100"  # plain GDG-PA, fixed subset

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- Exp1: ID vs OOD ---
ax = axes[0]
r_id, m_id = rd(E1)
r_ood, m_ood = rd(E1, "results_ood.txt")
ax.plot(r_id, m_id, color="#d62728", lw=2.2,
        label=f"ID  (train 5corr): R1 {m_id[0]:.1f} → R{r_id[-1]} {m_id[-1]:.1f}")
ax.plot(r_ood, m_ood, color="#1f77b4", lw=2.2,
        label=f"OOD (held-out 5corr, frozen): R1 {m_ood[0]:.1f} → R{r_ood[-1]} {m_ood[-1]:.1f}")
ax.set_title("Exp1 — does adaptation improve held-out corruptions too?\n"
             "(OOD rising with ID ⇒ overall model improvement, not corruption-fitting)",
             fontsize=11)
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

# --- Exp2: resample vs fixed ---
ax = axes[1]
r_fx, m_fx = rd(FIXED)
r_rs, m_rs = rd(E2)
if len(m_fx):
    ax.plot(r_fx, m_fx, color="#000000", lw=2.2,
            label=f"fixed 100-subset: last {m_fx[-1]:.1f}")
if len(m_rs):
    ax.plot(r_rs, m_rs, color="#ff7f0e", lw=1.8, alpha=0.9,
            label=f"resampled each round: mean {m_rs.mean():.1f} last {m_rs[-1]:.1f}")
ax.axhline(75.3, color="#2ca02c", ls="--", lw=1.3, label="MLMP-episodic 75.3")
ax.axhline(68.6, color="gray", ls=":", lw=1.3, label="No-Adapt 68.6")
ax.set_title("Exp2 — overfit to the specific 100 images?\n"
             "(resampled staying well above No-Adapt ⇒ genuine adaptation, not memorization)",
             fontsize=11)
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

fig.suptitle("GDG-PA analysis on VOC20-C (5corr sub100)", fontsize=13, y=1.02)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/analysis_exps.png", dpi=130, bbox_inches="tight")
print("saved -> figures/analysis_exps.png")
