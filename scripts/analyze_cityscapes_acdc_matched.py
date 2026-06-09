#!/usr/bin/env python
"""
Collect + analyze + plot the Cityscapes ACDC-matched CTTA results (404 imgs/round
= 101 x {snow,frost,fog,contrast}, 150 rounds). Reads each method's
results_all_rounds.txt, prints a summary table, and writes a Round-vs-Mean-mIoU
trajectory figure.

Usage:  python scripts/analyze_cityscapes_acdc_matched.py
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "save/CityscapesDataset"

# method display name -> save subdir (None-dir baselines handled separately)
METHODS = [
    ("No-Adapt",                 "No_Adaptation_weather",                          "baseline"),
    ("TENT-continual",           "tent_continual_acdc_matched",                    "tent"),
    ("TENT-DivGate",             "tent_divgate_continual_weather_threshold_2.0",   "tent"),
    ("TENT-DivGate-SmoothAnchor","tent_divgate_smooth_anchor_ceil2.0_floor1.4_weather", "tent"),
    ("CoTTA",                    "cotta",                                          "baseline"),
    ("SAR-continual",            "sar_continual_weather",                          "sar"),
    ("SAR-DivGate",              "sar_divgate_continual_weather",                  "sar"),
    ("MLMP-continual",           "mlmp_continual",                                 "mlmp"),
    ("MLMP-DivGate",             "mlmp_divgate_continual_weather_threshold_1.6",   "mlmp"),
    ("SAR-MLMP-SmoothAnchor*",   "sar_mlmp_smooth_anchor_continual_weather",       "ours"),
]

COLORS = {"baseline": "#888888", "tent": "#1f77b4", "sar": "#2ca02c",
          "mlmp": "#ff7f0e", "ours": "#d62728"}


def read_traj(subdir):
    f = os.path.join(ROOT, subdir, "results_all_rounds.txt")
    if not os.path.isfile(f):
        return [], []
    rounds, means = [], []
    with open(f) as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip().startswith("Round "):
                continue
            try:
                rn = int(row[0].split()[1])
                mean = float(row[-1])
            except (ValueError, IndexError):
                continue
            rounds.append(rn); means.append(mean)
    return rounds, means


def episodic_mean():
    f = os.path.join(ROOT, "mlmp_episodic_weather", "results.txt")
    if not os.path.isfile(f):
        return None
    vals = []
    with open(f) as fh:
        for line in fh:
            p = line.split(",")
            if len(p) >= 2 and p[0].strip() in ("snow", "frost", "fog", "contrast"):
                try:
                    vals.append(float(p[1].split("+/-")[0]))
                except ValueError:
                    pass
    return sum(vals) / len(vals) if vals else None


def main():
    epi = episodic_mean()
    na_last = None

    print(f"\n{'Method':<28}{'Rounds':>7}{'R1':>8}{'Peak':>8}{'@R':>5}{'Last':>8}{'MeanAll':>9}{'Verdict':>14}")
    print("-" * 90)
    rows = []
    for name, subdir, fam in METHODS:
        r, m = read_traj(subdir)
        if not r:
            print(f"{name:<28}{'-- no data --':>40}")
            continue
        r1 = m[0]
        peak = max(m); peak_r = r[m.index(peak)]
        last = m[-1]; nr = len(r)
        mean_all = sum(m) / len(m)
        if name.startswith("No-Adapt"):
            na_last = last
        verdict = ("collapsed" if last < 0.5 * peak else
                   "declining" if last < peak - 1.5 else
                   "stable")
        star = "  <= best" if fam == "ours" else ""
        print(f"{name:<28}{nr:>7}{r1:>8.2f}{peak:>8.2f}{peak_r:>5}{last:>8.2f}{mean_all:>9.2f}{verdict:>14}{star}")
        rows.append((name, subdir, fam, r, m, nr))

    print("-" * 90)
    print(f"MLMP-episodic (upper bound, per-sample reset): mean mIoU = {epi:.2f}" if epi else "episodic: n/a")
    print(f"No-Adapt (lower bound): {na_last:.2f}" if na_last else "")

    # ---------------- plot (2 panels: full + zoom on survivors) ----------------
    fig, (axf, axz) = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, (lo, hi, ttl) in zip((axf, axz),
                                 [(0, 22, "Full range (collapses visible)"),
                                  (17.5, 21, "Zoom: stable / surviving methods")]):
        for name, subdir, fam, r, m, nr in rows:
            is_ours = fam == "ours"
            ax.plot(r, m, color=COLORS[fam],
                    lw=3.0 if is_ours else 1.6,
                    alpha=1.0 if is_ours else 0.75,
                    ls="-" if fam != "baseline" else "--",
                    label=(f"{name}" + (f" (R{nr})" if nr < 150 else "")) if ax is axf else None,
                    zorder=5 if is_ours else 3)
        if epi:
            ax.axhline(epi, color="black", ls=":", lw=1.4, zorder=2,
                       label="MLMP-episodic upper bound" if ax is axf else None)
        ax.set_xlabel("Round (404 imgs/round: 101 x snow/frost/fog/contrast)")
        ax.set_ylabel("Mean mIoU")
        ax.set_ylim(lo, hi)
        ax.set_title(ttl)
        ax.grid(alpha=0.25)
    axf.legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("Cityscapes ACDC-matched CTTA — 150 rounds (404 imgs/round)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "svg"):
        out = os.path.join(ROOT, f"cityscapes_acdc_matched_round_miou.{ext}")
        fig.savefig(out, dpi=140)
        print(f"figure -> {out}")


if __name__ == "__main__":
    main()
