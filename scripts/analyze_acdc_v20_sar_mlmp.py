#!/usr/bin/env python
"""
Analyze + plot SAR-MLMP-SmoothAnchor against baselines on ACDC and V20
(ACDC-matched). For each dataset: print a summary table and write a 2-panel
Round-vs-Mean-mIoU figure (full range + zoom on survivors).

Usage:  python scripts/analyze_acdc_v20_sar_mlmp.py
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"baseline": "#888888", "tent": "#1f77b4", "sar": "#2ca02c",
          "mlmp": "#ff7f0e", "ours": "#d62728"}

# ---------------- dataset configs ----------------
ACDC = dict(
    root="save/ACDCDataset",
    title="ACDC CTTA (fog/night/rain/snow, 150 rounds)",
    out="acdc_sar_mlmp_round_miou",
    full=(0, 35), zoom=(22, 34),
    episodic=30.6, episodic_lbl="MLMP-episodic upper bound (30.6)",
    methods=[
        ("TENT-continual",            "tent_continual_Round150_lr_0.00001",        "tent"),
        ("MLMP-continual",            "mlmp_continual_round_150_step_1",            "mlmp"),
        ("SAR-continual",             "sar_continual_weather",                     "sar"),
        ("TENT-DivGate (best, 1.6)",  "tent_divgate_continual_cau_threshold_1.6",  "tent"),
        ("No-Adapt",                  "No_Adaptation",                             "baseline"),
        ("SAR-MLMP-SmoothAnchor*",    "sar_mlmp_smooth_anchor_continual",          "ours"),
    ],
)
V20 = dict(
    root="save/PascalVOC20Dataset/v20_acdc_matched",
    title="V20 ACDC-matched CTTA (snow/frost/fog/contrast, 150 rounds)",
    out="v20_acdc_matched_sar_mlmp_round_miou",
    full=(0, 82), zoom=(55, 80),
    episodic=76.21, episodic_lbl="MLMP-episodic upper bound (76.2)",
    methods=[
        ("No-Adapt",                 "No_Adaptation",                       "baseline"),
        ("TENT-continual",           "tent_continual",                      "tent"),
        ("MLMP-continual",           "mlmp_continual",                      "mlmp"),
        ("CoTTA",                    "cotta",                               "baseline"),
        ("TENT-DivGate (2.0)",       "tent_divgate_continual_threshold_2.0","tent"),
        ("MLMP-DivGate (1.6)",       "mlmp_divgate_continual_threshold_1.6","mlmp"),
        ("SAR-continual",            "sar_continual",                       "sar"),
        ("SAR-MLMP-SmoothAnchor*",   "sar_mlmp_smooth_anchor_continual",    "ours"),
    ],
)


def read_traj(root, subdir):
    f = os.path.join(root, subdir, "results_all_rounds.txt")
    if not os.path.isfile(f):
        return [], []
    r, m = [], []
    with open(f) as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip().startswith("Round "):
                continue
            try:
                r.append(int(row[0].split()[1])); m.append(float(row[-1]))
            except (ValueError, IndexError):
                continue
    return r, m


def analyze(cfg):
    print(f"\n===== {cfg['title']} =====")
    print(f"{'Method':<28}{'Rds':>5}{'R1':>8}{'Peak':>8}{'@R':>5}{'Last':>8}{'MeanAll':>9}  Verdict")
    print("-" * 82)
    rows = []
    for name, subdir, fam in cfg["methods"]:
        r, m = read_traj(cfg["root"], subdir)
        if not r:
            print(f"{name:<28}  -- no data --"); continue
        peak = max(m); peak_r = r[m.index(peak)]
        mean_all = sum(m) / len(m)
        verdict = ("collapsed" if m[-1] < 0.5 * peak else
                   "declining" if m[-1] < peak - 1.5 else "stable")
        star = "  <= ours" if fam == "ours" else ""
        print(f"{name:<28}{len(r):>5}{m[0]:>8.2f}{peak:>8.2f}{peak_r:>5}{m[-1]:>8.2f}{mean_all:>9.2f}  {verdict}{star}")
        rows.append((name, fam, r, m))
    print("-" * 82)
    print(f"{cfg['episodic_lbl']}")

    fig, (axf, axz) = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, (lo, hi, sub) in zip((axf, axz),
                                 [(*cfg["full"], "Full range"),
                                  (*cfg["zoom"], "Zoom: survivors")]):
        for name, fam, r, m in rows:
            ours = fam == "ours"
            ax.plot(r, m, color=COLORS[fam], lw=3.0 if ours else 1.6,
                    alpha=1.0 if ours else 0.8,
                    ls="-" if fam != "baseline" else "--",
                    label=name if ax is axf else None, zorder=5 if ours else 3)
        ax.axhline(cfg["episodic"], color="black", ls=":", lw=1.4,
                   label=cfg["episodic_lbl"] if ax is axf else None, zorder=2)
        ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU")
        ax.set_ylim(lo, hi); ax.set_title(sub); ax.grid(alpha=0.25)
    axf.legend(fontsize=8, loc="lower left")
    fig.suptitle(cfg["title"], fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "svg"):
        out = os.path.join(cfg["root"], f"{cfg['out']}.{ext}")
        fig.savefig(out, dpi=140); print(f"figure -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    analyze(ACDC)
    analyze(V20)
