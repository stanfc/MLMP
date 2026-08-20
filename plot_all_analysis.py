"""Full analysis figure set.

Fig1 gate_ablation_acdc      -- C3: the winner recipe with vs without the gate
Fig2 nogate_mechanism_acdc   -- B1: mIoU / grad_norm / cos(g_ent,g_oracle) / |g_oracle|
Fig3 domain_sawtooth_acdc    -- grad_norm vs stream, shaded by domain
Fig4 recipe_multidataset     -- base vs distill vs no-gate on ACDC / VOC20 / Cityscapes
Fig5 oracle_multidataset     -- the grad_norm/oracle story across all 3 datasets

All collapse-motivation panels are drawn on NO-GATE runs (never the gated method).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
S = "save"


def miou(p):
    f = f"{S}/{p}/results_all_rounds.txt"
    if not os.path.exists(f):
        return np.array([])
    return np.array([float(l.split(",")[-1]) for l in open(f) if l.startswith("Round ")])


def gate(p):
    """gate_log.csv -> dict of arrays (per window)."""
    f = f"{S}/{p}/gate_log.csv"
    if not os.path.exists(f):
        return None
    rows = [l.strip().split(",") for l in open(f)][1:]
    return dict(tb=np.array([int(r[0]) for r in rows]),
                gn=np.array([float(r[1]) for r in rows]),
                hm=np.array([float(r[3]) for r in rows]))


def oracle(p):
    f = f"{S}/{p}/oracle_log.csv"
    if not os.path.exists(f):
        return None
    rows = [l.strip().split(",") for l in open(f)][1:]
    return dict(tb=np.array([int(r[0]) for r in rows]),
                gn=np.array([float(r[1]) for r in rows]),
                cos=np.array([float(r[2]) for r in rows]),
                gor=np.array([float(r[4]) for r in rows]),
                hm=np.array([float(r[5]) for r in rows]))


def per_round(arr, R):
    """bin a per-window array into R per-round means."""
    return np.array([a.mean() for a in np.array_split(arr, R)])


def smooth(a, k=15):
    if len(a) < k:
        return a
    return np.convolve(a, np.ones(k) / k, mode="same")


def cond_map(p):
    """map cumulative batch index -> condition name, from entropy_log.csv."""
    f = f"{S}/{p}/entropy_log.csv"
    if not os.path.exists(f):
        return None
    tb, cond = [], []
    for l in open(f):
        x = l.strip().split(",")
        if not x[0].isdigit():
            continue
        tb.append(int(x[0])); cond.append(x[2])
    return np.array(tb), np.array(cond)


# paths
ACDC_NOGATE = "ACDCDataset/deyo_mlmp_hmgate2_oracle_nogate_lr3e-5"
ACDC_DISTILL_NOGATE = "ACDCDataset/deyo_mlmp_hmgate2_distill_nogate_lr3e-5"
ACDC_DISTILL = "ACDCDataset/deyo_mlmp_hmgate2_distill_continual_lr3e-5"
ACDC_BASE = "ACDCDataset/deyo_mlmp_hmgate2_continual"
V20_DISTILL = "PascalVOC20Dataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr3e-5"
V20_NOGATE = "PascalVOC20Dataset/deyo_mlmp_hmgate2_oracle_nogate_5corr_sub100_lr3e-5"
V20_BASE = "PascalVOC20Dataset/deyo_mlmp_hmgate2_continual_5corr_sub100"
CITY_DISTILL = "CityscapesDataset/deyo_mlmp_hmgate2_distill_continual_5corr_sub100_lr3e-5"
CITY_NOGATE = "CityscapesDataset/deyo_mlmp_hmgate2_oracle_nogate_5corr_sub100_lr3e-5"
CITY_BASE = "CityscapesDataset/deyo_mlmp_hmgate2_continual_5corr_sub100"

# ============================== FIG 1: gate ablation ==============================
fig, ax = plt.subplots(figsize=(11, 6.2))
for name, p, c, lw in [
    ("distill + GATE (ours)", ACDC_DISTILL, "#2ca02c", 2.8),
    ("GDG-PA base + gate (LR5e-6)", ACDC_BASE, "#7f7f7f", 1.6),
    ("distill, GATE OFF (base_rst=0)", ACDC_DISTILL_NOGATE, "#d62728", 2.4),
    ("no-gate, no teacher (LR3e-5)", ACDC_NOGATE, "#ff7f0e", 1.8),
]:
    m = miou(p)
    if len(m) == 0:
        continue
    ax.plot(np.arange(1, len(m) + 1), m, color=c, lw=lw,
            label=f"{name}: peak {m.max():.1f}, last {m[-1]:.1f}, mean {m.mean():.1f}")
ax.axhline(23.34, color="k", ls=":", lw=1.1, label="No-Adapt 23.3")
ax.set_xlabel("Round"); ax.set_ylabel("mIoU (%)"); ax.set_ylim(0, 37)
ax.set_title("ACDC — the GATE is the ENABLER, not the scorer\n"
             "identical recipe (EMA teacher-student + LR3e-5): gate OFF peaks HIGHER (34.4@R6) "
             "then collapses to 3.1; gate ON holds 33.5 for 150 rounds", fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="center right")
fig.tight_layout(); fig.savefig("figures/gate_ablation_acdc.png", dpi=130, bbox_inches="tight")
print("saved -> figures/gate_ablation_acdc.png")

# ====================== FIG 2: no-gate mechanism (ACDC) ==========================
o = oracle(ACDC_NOGATE); m = miou(ACDC_NOGATE); R = len(m)
gn_r, cos_r, gor_r, hm_r = (per_round(o["gn"], R), per_round(o["cos"], R),
                            per_round(o["gor"], R), per_round(o["hm"], R))
rounds = np.arange(1, R + 1)

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9))

ax = axes[0, 0]
ax.plot(rounds, m, color="#2ca02c", lw=2.2)
ax.set_ylabel("mIoU (%)", color="#2ca02c"); ax.tick_params(axis="y", labelcolor="#2ca02c")
a2 = ax.twinx(); a2.plot(rounds, gn_r, color="#d62728", lw=1.8)
a2.set_ylabel("grad_norm", color="#d62728"); a2.tick_params(axis="y", labelcolor="#d62728")
ax.set_xlabel("Round"); ax.grid(alpha=0.3)
_pk = m.argmax(); _on = int(np.diff(m).argmin()) + 1
ax.axvline(_pk + 1, color="#2ca02c", ls=":", lw=1.6,
           label=f"R{_pk+1}  mIoU PEAK (grad_norm at its MINIMUM)")
ax.axvline(_on + 1, color="#d62728", ls=":", lw=1.6,
           label=f"R{_on+1}  COLLAPSE ONSET (steepest mIoU drop)")
ax.legend(fontsize=8, loc="center right")
ax.set_title(f"(a) grad_norm MINIMUM ({gn_r[_pk]:.1f}) lands on the mIoU PEAK (R{_pk+1}), then RISES at onset\n"
             f"-> pinning the best anchor at the grad-min is exactly right")

ax = axes[0, 1]
ax.plot(rounds, cos_r, color="#9467bd", lw=1.0, alpha=0.35)
ax.plot(rounds, smooth(cos_r, 11), color="#9467bd", lw=2.4, label="cos(g_entropy, g_oracle)")
ax.axhline(0, color="k", ls="--", lw=1.2)
ax.set_xlabel("Round"); ax.set_ylabel("cosine similarity")
pk = m.argmax(); on = int(np.diff(m).argmin()) + 1
ax.axvline(pk + 1, color="#2ca02c", ls=":", lw=1.6, label=f"R{pk+1} mIoU PEAK")
ax.axvline(on + 1, color="#d62728", ls=":", lw=1.6, label=f"R{on+1} COLLAPSE ONSET")
ax.set_title("(b) at collapse ONSET the entropy gradient turns AGAINST the true task\n"
             f"cos at mIoU peak (R{pk+1}) = {cos_r[pk]:+.2f}  ->  at onset (R{on+1}) = {cos_r[on]:+.2f}")
ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1, 0]
ax.plot(rounds, gor_r, color="#8c564b", lw=2.0, label="|g_oracle| (true-task grad)")
ax.plot(rounds, gn_r, color="#d62728", lw=2.0, label="|g_entropy| (what we use)")
ax.set_xlabel("Round"); ax.set_ylabel("gradient norm")
ax.set_title("(c) the TRUE-task gradient explodes (4.6 -> 55)\n"
             "the model becomes badly wrong while its own signal stays small")
ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1, 1]
sc = ax.scatter(gn_r, m, c=cos_r, cmap="coolwarm_r", s=30, edgecolor="k", lw=0.3,
                vmin=-0.3, vmax=0.3)
ax.set_xlabel("per-round grad_norm"); ax.set_ylabel("per-round mIoU (%)")
pr = np.corrcoef(gn_r, m)[0, 1]
ax.set_title(f"(d) grad_norm vs mIoU (colour = oracle cosine), r={pr:.2f}\n"
             f"healthy = low grad_norm + cos near 0 (top-left); collapsed = red, high grad_norm")
cb = fig.colorbar(sc, ax=ax); cb.set_label("cos(g_ent, g_oracle)")
ax.grid(alpha=0.3)

fig.suptitle("ACDC NO-GATE (natural collapse) — why grad_norm is the right restore signal",
             fontsize=14)
fig.tight_layout(); fig.savefig("figures/nogate_mechanism_acdc.png", dpi=130, bbox_inches="tight")
print("saved -> figures/nogate_mechanism_acdc.png")

# ============ FIG 3: peak-vs-onset summary across the 3 datasets =================
# NOTE: the "domain sawtooth" hypothesis did NOT hold -- grad_norm's motion is
# dominated by the collapse trend, not by fog/night/rain/snow switching. Dropped.
DS = [("ACDC", ACDC_NOGATE), ("VOC20", V20_NOGATE), ("Cityscapes", CITY_NOGATE)]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
names, gpk, gon, cpk, con = [], [], [], [], []
for nm, p in DS:
    oo = oracle(p); mm = miou(p)
    if oo is None or len(mm) == 0:
        continue
    Rj = len(mm)
    g_r = per_round(oo["gn"], Rj); c_r = per_round(oo["cos"], Rj)
    pk = mm.argmax(); on = int(np.diff(mm).argmin()) + 1
    names.append(nm); gpk.append(g_r[pk]); gon.append(g_r[on])
    cpk.append(c_r[pk]); con.append(c_r[on])
x = np.arange(len(names)); w = 0.35

ax = axes[0]
ax.bar(x - w/2, gpk, w, color="#2ca02c", label="at mIoU PEAK (healthy)")
ax.bar(x + w/2, gon, w, color="#d62728", label="at collapse ONSET")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("grad_norm")
ax.set_title("grad_norm is LOW (~6) at the peak on every dataset\nand RISES at collapse onset -> slope is the trigger")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip(gpk, gon)):
    ax.text(i - w/2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + w/2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=9)

ax = axes[1]
ax.bar(x - w/2, cpk, w, color="#2ca02c", label="at mIoU PEAK")
ax.bar(x + w/2, con, w, color="#d62728", label="at collapse ONSET")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("cos(g_entropy, g_oracle)")
ax.set_title("the entropy gradient turns AGAINST the true task at onset\n(cos plunges negative on every dataset)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip(cpk, con)):
    ax.text(i - w/2, a, f"{a:+.2f}", ha="center", va="top" if a < 0 else "bottom", fontsize=9)
    ax.text(i + w/2, b, f"{b:+.2f}", ha="center", va="top", fontsize=9)

fig.suptitle("Why the gate is built the way it is: grad-min = best anchor, grad-RISE = restore trigger",
             fontsize=13)
fig.tight_layout(); fig.savefig("figures/peak_vs_onset.png", dpi=130, bbox_inches="tight")
print("saved -> figures/peak_vs_onset.png")

# ================== FIG 4: recipe across the 3 datasets ==========================
PANELS = [
    ("ACDC", ACDC_BASE, ACDC_DISTILL, ACDC_NOGATE, (0, 37)),
    ("VOC20 (5corr sub100)", V20_BASE, V20_DISTILL, V20_NOGATE, (60, 84)),
    ("Cityscapes (5corr sub100)", CITY_BASE, CITY_DISTILL, CITY_NOGATE, (0, 28)),
]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
for ax, (title, pb, pd_, pn, ylim) in zip(axes, PANELS):
    for name, p, c, lw in [("GDG-PA base", pb, "#000000", 2.0),
                           ("+ distill recipe (ours)", pd_, "#2ca02c", 2.6),
                           ("NO GATE", pn, "#d62728", 1.8)]:
        mm = miou(p)
        if len(mm) == 0:
            continue
        ax.plot(np.arange(1, len(mm) + 1), mm, color=c, lw=lw,
                label=f"{name}: mean {mm.mean():.1f}, last {mm[-1]:.1f}")
    ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylim(*ylim)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
axes[0].set_ylabel("mIoU (%)")
fig.suptitle("The recipe across datasets — the gate prevents collapse everywhere; "
             "the teacher adds on ACDC (+2.0) and VOC20 (+1.8), is flat on Cityscapes",
             fontsize=13)
fig.tight_layout(); fig.savefig("figures/recipe_multidataset.png", dpi=130, bbox_inches="tight")
print("saved -> figures/recipe_multidataset.png")

# ============ FIG 5: oracle/grad_norm story across the 3 datasets ================
fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))
for j, (title, pn) in enumerate([("ACDC", ACDC_NOGATE), ("VOC20", V20_NOGATE),
                                 ("Cityscapes", CITY_NOGATE)]):
    oo = oracle(pn); mm = miou(pn)
    if oo is None or len(mm) == 0:
        continue
    Rj = len(mm); rj = np.arange(1, Rj + 1)
    g_r = per_round(oo["gn"], Rj); c_r = per_round(oo["cos"], Rj)

    ax = axes[0, j]
    ax.plot(rj, mm, color="#2ca02c", lw=2.0)
    ax.set_ylabel("mIoU (%)" if j == 0 else "", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    a2 = ax.twinx(); a2.plot(rj, g_r, color="#d62728", lw=1.6)
    a2.set_ylabel("grad_norm" if j == 2 else "", color="#d62728")
    a2.tick_params(axis="y", labelcolor="#d62728")
    pk = mm.argmax(); on = int(np.diff(mm).argmin()) + 1
    ax.axvline(pk + 1, color="#2ca02c", ls=":", lw=1.3)
    ax.axvline(on + 1, color="#d62728", ls=":", lw=1.3)
    ax.set_title(f"{title} NO-GATE — grad_norm {g_r[pk]:.1f} at peak -> {g_r[on]:.1f} at onset\n(green=peak, red=onset; raw magnitude is NOT monotone -> use the SLOPE)")
    ax.grid(alpha=0.3); ax.set_xlabel("Round")

    ax = axes[1, j]
    ax.plot(rj, c_r, color="#9467bd", lw=0.9, alpha=0.3)
    ax.plot(rj, smooth(c_r, 11), color="#9467bd", lw=2.2)
    ax.axhline(0, color="k", ls="--", lw=1.1)
    ax.set_xlabel("Round"); ax.set_ylabel("cos(g_ent, g_oracle)" if j == 0 else "")
    ax.axvline(pk + 1, color="#2ca02c", ls=":", lw=1.3)
    ax.axvline(on + 1, color="#d62728", ls=":", lw=1.3)
    ax.set_title(f"cos: {c_r[pk]:+.2f} at peak -> {c_r[on]:+.2f} at onset")
    ax.grid(alpha=0.3)
fig.suptitle("The signal generalizes: on all 3 datasets grad_norm is ~6 at the mIoU peak, RISES at "
             "collapse onset, and the entropy gradient turns AGAINST the true-task gradient (cos<0)",
             fontsize=13)
fig.tight_layout(); fig.savefig("figures/oracle_multidataset.png", dpi=130, bbox_inches="tight")
print("saved -> figures/oracle_multidataset.png")
