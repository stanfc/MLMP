#!/usr/bin/env python
"""Presentation figures for the gradslope/composite gate session (2026-06-22).

Reads results_all_rounds.txt of the runs launched this session and produces
slide-ready PNGs in save/_compare/:
  session_headline_bars.png   - best method per dataset vs divgate/episodic/no-adapt
  session_trajectories.png    - mIoU vs round, best composite vs best gradslope (3 panels)
  session_acdc_restore_lever.png - ACDC composite restore-strength lever (rst 0.01->0.04 holds peak)
  session_voc20_holds_peak.png   - VOC20: composite holds peak, gradslope caps low

Labels are in English to avoid missing-CJK-font tofu; add Chinese captions in the slides.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "save"
OUT = "save/_compare"
os.makedirs(OUT, exist_ok=True)

# reference constants (divgate/episodic/no-adapt; divgate from 學長, same protocol)
REF = {
    "VOC20":      dict(divgate=77.4, episodic=76.21, noadapt=68.6),
    "ACDC":       dict(divgate=31.8, episodic=29.84, noadapt=23.34),
    "Cityscapes": dict(divgate=23.6, episodic=20.0,  noadapt=20.6),
}
DDIR = {"VOC20": "PascalVOC20Dataset/v20_acdc_matched",
        "ACDC": "ACDCDataset", "Cityscapes": "CityscapesDataset"}


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


def load(ds, cfg):
    return rounds(os.path.join(SAVE, DDIR[ds], cfg, "results_all_rounds.txt"))


def mean(ms):
    return sum(ms) / len(ms) if ms else float("nan")


# best run per dataset (chosen from the analysis)
BEST_COMP = {"VOC20": "deyo_mlmp_composite_cc0.68_rst0.01_gm4",
             "ACDC": "deyo_mlmp_composite_cc0.78_rst0.04_gm4",
             "Cityscapes": "deyo_mlmp_composite_cc0.66_rst0.01_gm4"}
BEST_GS = {"VOC20": "deyo_mlmp_gradslope_dz0.008_rst0.01",
           "ACDC": "deyo_mlmp_gradslope_dz0.008_rst0.01",
           "Cityscapes": "deyo_mlmp_gradslope_dz0.008_rst0.01"}

DS_ORDER = ["VOC20", "ACDC", "Cityscapes"]


# ---------- Figure 1: headline bars ----------
def fig_headline():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, ds in zip(axes, DS_ORDER):
        comp = mean(load(ds, BEST_COMP[ds]))
        gs = mean(load(ds, BEST_GS[ds]))
        r = REF[ds]
        names = ["No-Adapt", "Episodic", "GradSlope", "DivGate", "Composite"]
        vals = [r["noadapt"], r["episodic"], gs, r["divgate"], comp]
        colors = ["#bbbbbb", "#7aa6c2", "#f0a868", "#9b9b6f", "#3b7a57"]
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(ds, fontweight="bold")
        ax.set_ylabel("mean mIoU (150R)")
        lo = min(v for v in vals if v == v)
        ax.set_ylim(lo - (max(vals) - lo) * 0.6, max(vals) + (max(vals) - lo) * 0.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.axhline(r["divgate"], ls="--", color="#9b9b6f", lw=1, alpha=0.7)
    fig.suptitle("Composite gate matches/beats the DivGate bar on all 3 datasets",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, "session_headline_bars.png")
    fig.savefig(p, dpi=150); print("wrote", p)


# ---------- Figure 2: trajectories (3 panels) ----------
def fig_trajectories():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, ds in zip(axes, DS_ORDER):
        comp = load(ds, BEST_COMP[ds]); gs = load(ds, BEST_GS[ds])
        if comp:
            ax.plot(range(1, len(comp) + 1), comp, color="#3b7a57", lw=1.8,
                    label=f"Composite ({mean(comp):.1f})")
        if gs:
            ax.plot(range(1, len(gs) + 1), gs, color="#f0a868", lw=1.5,
                    label=f"GradSlope ({mean(gs):.1f})")
        r = REF[ds]
        ax.axhline(r["divgate"], ls="--", color="#9b9b6f", lw=1.3, label=f"DivGate {r['divgate']}")
        ax.axhline(r["episodic"], ls=":", color="#7aa6c2", lw=1.3, label=f"Episodic {r['episodic']}")
        ax.set_title(ds, fontweight="bold"); ax.set_xlabel("round"); ax.set_ylabel("mIoU")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    fig.suptitle("mIoU trajectory (150 rounds): Composite reaches & holds the peak",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, "session_trajectories.png")
    fig.savefig(p, dpi=150); print("wrote", p)


# ---------- Figure 3: ACDC restore-strength lever ----------
def fig_acdc_lever():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    variants = [("deyo_mlmp_composite_cc0.78_rst0.01_gm4", "rst 0.01 (weak)", "#d98880"),
                ("deyo_mlmp_composite_cc0.78_rst0.02_gm4", "rst 0.02", "#e8b04b"),
                ("deyo_mlmp_composite_cc0.78_rst0.04_gm4", "rst 0.04 (holds!)", "#3b7a57")]
    for cfg, lab, c in variants:
        ms = load("ACDC", cfg)
        if ms:
            ax.plot(range(1, len(ms) + 1), ms, color=c, lw=1.8,
                    label=f"Composite {lab}  mean={mean(ms):.1f}")
    gs = load("ACDC", BEST_GS["ACDC"])
    if gs:
        ax.plot(range(1, len(gs) + 1), gs, color="#888", lw=1.2, ls="-.",
                label=f"GradSlope  mean={mean(gs):.1f}")
    ax.axhline(REF["ACDC"]["divgate"], ls="--", color="#9b9b6f", lw=1.3, label="DivGate 31.8")
    ax.set_title("ACDC: stronger restore (rst) holds Composite's 33.5 peak", fontweight="bold")
    ax.set_xlabel("round"); ax.set_ylabel("mIoU"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "session_acdc_restore_lever.png")
    fig.savefig(p, dpi=150); print("wrote", p)


# ---------- Figure 4: VOC20 holds peak ----------
def fig_voc20():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    comp = load("VOC20", BEST_COMP["VOC20"]); gs = load("VOC20", BEST_GS["VOC20"])
    if comp:
        ax.plot(range(1, len(comp) + 1), comp, color="#3b7a57", lw=1.8,
                label=f"Composite (mean {mean(comp):.1f}, holds peak 78.7)")
    if gs:
        ax.plot(range(1, len(gs) + 1), gs, color="#f0a868", lw=1.6,
                label=f"GradSlope (mean {mean(gs):.1f}, peak capped)")
    r = REF["VOC20"]
    ax.axhline(r["divgate"], ls="--", color="#9b9b6f", lw=1.3, label=f"DivGate {r['divgate']}")
    ax.axhline(r["episodic"], ls=":", color="#7aa6c2", lw=1.3, label=f"Episodic {r['episodic']}")
    ax.axhline(r["noadapt"], ls=":", color="#bbbbbb", lw=1.2, label=f"No-Adapt {r['noadapt']}")
    ax.set_title("VOC20 (uniform-drift): mean_conf trigger lets Composite hold the peak",
                 fontweight="bold")
    ax.set_xlabel("round"); ax.set_ylabel("mIoU"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "session_voc20_holds_peak.png")
    fig.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    fig_headline()
    fig_trajectories()
    fig_acdc_lever()
    fig_voc20()
    print("\nAll figures in", OUT)
