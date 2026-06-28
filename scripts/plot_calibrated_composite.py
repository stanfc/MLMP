#!/usr/bin/env python
"""Figures for the conf_ceil-calibration experiment (2026-06-25).

Shows that calibrating composite's conf_ceil to the GATE-INTERNAL mean_conf scale
makes the gate engage near the peak, capping over-confidence and removing the ACDC
oscillation -> high AND stable.

  cal_acdc_fix.png    - ACDC trajectory: old (uncalibrated, oscillates) vs calibrated (smooth)
  cal_conf_lever.png  - ACDC: mean & steady-state std vs conf_ceil (the lever)
  cal_headline.png    - updated per-dataset best bars
"""
import os, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE, OUT = "save", "save/_compare"
os.makedirs(OUT, exist_ok=True)
A = "save/ACDCDataset"


def rounds(d):
    f = os.path.join(SAVE, d, "results_all_rounds.txt") if not d.startswith("save/") else os.path.join(d, "results_all_rounds.txt")
    return [float(l.split(",")[-1]) for l in open(f) if l.lower().startswith("round ")] if os.path.exists(f) else []


def tail_std(ms):
    t = ms[29:] if len(ms) > 40 else ms
    return st.pstdev(t) if t else 0.0


# ---------- Fig 1: ACDC calibration fix ----------
def fig_acdc_fix():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    series = [
        (f"{A}/deyo_mlmp_composite_cc0.78_rst0.04_gm4", "Composite UNcalibrated (conf_ceil 0.78)", "#d62828", 2.0),
        (f"{A}/deyo_mlmp_composite_cc0.70_rst0.02_gm4", "Composite CALIBRATED (conf_ceil 0.70, rst0.02)", "#1f6b43", 2.4),
        (f"{A}/deyo_mlmp_gradslope_dz0.008_rst0.01", "GradSlope", "#e07b20", 1.4),
    ]
    for d, lab, c, lw in series:
        ms = rounds(d)
        if ms:
            ax.plot(range(1, len(ms) + 1), ms, color=c, lw=lw,
                    label=f"{lab}  (mean {sum(ms)/len(ms):.1f}, std {tail_std(ms):.2f})")
    ax.axhline(31.8, ls="--", color="#9b8d3a", lw=1.4, label="DeYO-DivGate ref (31.8)")
    ax.axhline(29.84, ls=":", color="#7aa6c2", lw=1.3, label="Episodic (29.84)")
    ax.set_ylim(28, 34.2)
    ax.set_title("ACDC: calibrated conf_ceil cuts oscillation 3x (std 1.46->0.49) & raises the floor",
                 fontweight="bold", fontsize=10.5)
    ax.set_xlabel("round"); ax.set_ylabel("mIoU"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout(); p = os.path.join(OUT, "cal_acdc_fix.png")
    fig.savefig(p, dpi=150); print("wrote", p)


# ---------- Fig 2: conf_ceil lever (ACDC, rst0.02 series) ----------
def fig_conf_lever():
    # rst0.02 across conf_ceil; include old hold variant cc0.78_rst0.02
    ccs = [0.68, 0.70, 0.72, 0.78]
    means, stds = [], []
    for cc in ccs:
        ms = rounds(f"{A}/deyo_mlmp_composite_cc{cc}_rst0.02_gm4")
        means.append(sum(ms)/len(ms) if ms else float("nan"))
        stds.append(tail_std(ms) if ms else float("nan"))
    x = range(len(ccs))
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    b = ax1.bar([i - 0.18 for i in x], means, width=0.36, color="#3b7a57", label="mean mIoU")
    ax1.set_ylabel("mean mIoU", color="#3b7a57"); ax1.set_ylim(30, 33)
    for i, v in zip(x, means):
        ax1.text(i - 0.18, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(31.8, ls="--", color="#9b8d3a", lw=1.3)
    ax1.text(len(ccs)-1, 31.85, "DivGate 31.8", fontsize=7, color="#9b8d3a", ha="right")
    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], stds, width=0.36, color="#d98880", label="steady-state std")
    ax2.set_ylabel("steady-state std (lower=stabler)", color="#d98880"); ax2.set_ylim(0, 2.0)
    for i, v in zip(x, stds):
        ax2.text(i + 0.18, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(list(x)); ax1.set_xticklabels([f"{c}\n{'(calib)' if c<0.78 else '(uncalib)'}" for c in ccs])
    ax1.set_xlabel("conf_ceil  (gate-internal mean_conf peak ≈ 0.70)")
    ax1.set_title("ACDC conf_ceil lever (rst0.02): lower conf_ceil -> higher mean, much lower std",
                  fontweight="bold", fontsize=10)
    fig.tight_layout(); p = os.path.join(OUT, "cal_conf_lever.png")
    fig.savefig(p, dpi=150); print("wrote", p)


# ---------- Fig 3: updated headline ----------
def fig_headline():
    data = {
        "VOC20": dict(noadapt=68.6, episodic=76.21, divgate=77.4,
                      comp=rounds("save/PascalVOC20Dataset/v20_acdc_matched/deyo_mlmp_composite_cc0.60_rst0.005_gm4"),
                      gs=rounds("save/PascalVOC20Dataset/v20_acdc_matched/deyo_mlmp_gradslope_dz0.008_rst0.01")),
        "ACDC": dict(noadapt=23.34, episodic=29.84, divgate=31.8,
                     comp=rounds(f"{A}/deyo_mlmp_composite_cc0.70_rst0.02_gm4"),
                     gs=rounds(f"{A}/deyo_mlmp_gradslope_dz0.008_rst0.01")),
        "Cityscapes": dict(noadapt=20.6, episodic=20.0, divgate=23.6,
                           comp=rounds("save/CityscapesDataset/deyo_mlmp_composite_cc0.66_rst0.005_gm4"),
                           gs=rounds("save/CityscapesDataset/deyo_mlmp_gradslope_dz0.008_rst0.01")),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (ds, d) in zip(axes, data.items()):
        cm = sum(d["comp"])/len(d["comp"]) if d["comp"] else float("nan")
        gm = sum(d["gs"])/len(d["gs"]) if d["gs"] else float("nan")
        names = ["No-Adapt", "Episodic", "GradSlope", "DivGate", "Composite\n(calibrated)"]
        vals = [d["noadapt"], d["episodic"], gm, d["divgate"], cm]
        colors = ["#bbbbbb", "#7aa6c2", "#f0a868", "#9b9b6f", "#3b7a57"]
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(ds, fontweight="bold"); ax.set_ylabel("mean mIoU (150R)")
        lo = min(v for v in vals if v == v)
        ax.set_ylim(lo - (max(vals)-lo)*0.6, max(vals)+(max(vals)-lo)*0.25)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x()+bb.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7.5)
        ax.axhline(d["divgate"], ls="--", color="#9b9b6f", lw=1, alpha=0.7)
    fig.suptitle("Calibrated Composite gate vs baselines (150R mean)", fontweight="bold")
    fig.tight_layout(); p = os.path.join(OUT, "cal_headline.png")
    fig.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    fig_acdc_fix(); fig_conf_lever(); fig_headline()
    print("done")
