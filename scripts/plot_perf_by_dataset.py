#!/usr/bin/env python
"""Per-dataset performance comparison figures (one PNG per dataset).

Each figure shows:
  * No-Adapt        - horizontal reference (model never changes -> flat)
  * DeYO-DivGate    - horizontal reference (mean; no per-round curve on this machine)
  * Episodic        - horizontal reference (upper bound, needs per-sample reset)
  * GradSlope       - our runs (all configs faint, best bold) -> real per-round trajectory
  * Composite       - our runs (all configs faint, best bold) -> real per-round trajectory

Output: save/_compare/perf_{VOC20,ACDC,Cityscapes}.png
"""
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE, OUT = "save", "save/_compare"
os.makedirs(OUT, exist_ok=True)

REF = {
    "VOC20":      dict(noadapt=68.6, divgate=77.4, episodic=76.21, ylim=(67, 79.5),
                       dir="PascalVOC20Dataset/v20_acdc_matched"),
    "ACDC":       dict(noadapt=23.34, divgate=31.8, episodic=29.84, ylim=(22, 34.5),
                       dir="ACDCDataset"),
    "Cityscapes": dict(noadapt=20.6, divgate=23.6, episodic=20.0, ylim=(19.5, 24.6),
                       dir="CityscapesDataset"),
}


def rounds(path):
    ms = []
    if not os.path.exists(path):
        return ms
    for ln in open(path):
        ln = ln.strip()
        if ln.lower().startswith("round "):
            try:
                ms.append(float(ln.split(",")[-1]))
            except (ValueError, IndexError):
                pass
    return ms


def runs_of(ds, method):
    out = {}
    for d in sorted(glob.glob(os.path.join(SAVE, REF[ds]["dir"], f"deyo_mlmp_{method}_*"))):
        ms = rounds(os.path.join(d, "results_all_rounds.txt"))
        if len(ms) >= 30:   # skip barely-started runs
            out[os.path.basename(d)] = ms
    return out


def best_key(d):
    return max(d, key=lambda k: sum(d[k]) / len(d[k])) if d else None


def plot_ds(ds):
    r = REF[ds]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    gs = runs_of(ds, "gradslope")
    comp = runs_of(ds, "composite")

    # faint: every config we ran
    for ms in gs.values():
        ax.plot(range(1, len(ms) + 1), ms, color="#f0a868", lw=0.7, alpha=0.35)
    for ms in comp.values():
        ax.plot(range(1, len(ms) + 1), ms, color="#3b7a57", lw=0.7, alpha=0.30)

    # bold: best of each method
    gk, ck = best_key(gs), best_key(comp)
    if gk:
        m = gs[gk]; mean = sum(m) / len(m)
        ax.plot(range(1, len(m) + 1), m, color="#e07b20", lw=2.2,
                label=f"GradSlope best  (mean {mean:.1f})")
    if ck:
        m = comp[ck]; mean = sum(m) / len(m)
        ax.plot(range(1, len(m) + 1), m, color="#1f6b43", lw=2.2,
                label=f"Composite best  (mean {mean:.1f})")

    # horizontal references
    ax.axhline(r["divgate"], ls="--", color="#9b8d3a", lw=1.6, label=f"DeYO-DivGate ref ({r['divgate']})")
    ax.axhline(r["episodic"], ls=":", color="#7aa6c2", lw=1.5, label=f"Episodic ({r['episodic']})")
    ax.axhline(r["noadapt"], ls=":", color="#999999", lw=1.5, label=f"No-Adapt ({r['noadapt']})")

    if r.get("ylim"):
        ax.set_ylim(*r["ylim"])
    ax.set_title(f"{ds} — performance over 150 rounds", fontweight="bold")
    ax.set_xlabel("round"); ax.set_ylabel("mIoU")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    n_runs = len(gs) + len(comp)
    ax.text(0.01, 0.99, f"faint lines = all {n_runs} sweep runs", transform=ax.transAxes,
            fontsize=7, va="top", color="#666")
    fig.tight_layout()
    p = os.path.join(OUT, f"perf_{ds}.png")
    fig.savefig(p, dpi=150); print("wrote", p, f"({len(gs)} gradslope + {len(comp)} composite runs)")


if __name__ == "__main__":
    for ds in ("VOC20", "ACDC", "Cityscapes"):
        plot_ds(ds)
    print("\nfigures in", OUT)
