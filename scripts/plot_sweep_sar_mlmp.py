#!/usr/bin/env python
"""
Plot the 16-config ceil/floor/lag/rst sweep of sar_mlmp_smooth_anchor_continual.
Two figures (ACDC, V20-weather-subset), each overlaying all 8 sweep trajectories
+ the original baseline config + No-Adapt + the MLMP-episodic upper bound. Best
config (by mean-all) is highlighted bold red.

Usage:  python scripts/plot_sweep_sar_mlmp.py
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACDC = dict(
    root="save/ACDCDataset",
    title="ACDC sweep — SAR-MLMP-SmoothAnchor (fog/night/rain/snow, 150R)",
    out="acdc_sweep_sar_mlmp",
    episodic=30.6, full=(22, 31), zoom=(29.0, 30.8),
    baseline="sar_mlmp_smooth_anchor_continual",
    configs=[
        "smlp_c2.4_f2.2_l150_r0.005", "smlp_c2.5_f2.2_l150_r0.005",
        "smlp_c2.9_f2.2_l150_r0.002", "smlp_c2.9_f2.0_l100_r0.005",
        "smlp_c2.9_f1.9_l150_r0.005", "smlp_c2.9_f2.4_l150_r0.005",
        "smlp_c2.6_f2.4_l150_r0.008", "smlp_c2.4_f1.9_l100_r0.003",
    ],
)
V20 = dict(
    root="save/PascalVOC20Dataset/v20_acdc_matched",
    title="V20 weather sweep — SAR-MLMP-SmoothAnchor (snow/frost/fog/contrast, 150R)",
    out="v20_sweep_sar_mlmp",
    episodic=76.21, full=(71, 79), zoom=(74.5, 78.5),
    baseline="sar_mlmp_smooth_anchor_continual",
    configs=[
        "smlp_c3.2_f2.2_l150_r0.010", "smlp_c3.4_f2.2_l300_r0.005",
        "smlp_c3.4_f2.6_l150_r0.010", "smlp_c3.2_f2.4_l150_r0.005",
        "smlp_c3.2_f2.5_l100_r0.005", "smlp_c2.9_f2.5_l150_r0.005",
        "smlp_c3.2_f2.2_l150_r0.005", "smlp_c3.4_f2.2_l150_r0.005",
    ],
)


def read_traj(root, sub):
    f = os.path.join(root, sub, "results_all_rounds.txt")
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


def plot(cfg):
    series = []
    for sub in cfg["configs"]:
        r, m = read_traj(cfg["root"], sub)
        if r:
            series.append((sub, r, m, sum(m) / len(m)))
    br, bm = read_traj(cfg["root"], cfg["baseline"])
    nr, nm = read_traj(cfg["root"], "No_Adaptation")

    series.sort(key=lambda s: s[3], reverse=True)
    best = series[0][0] if series else None

    print(f"\n===== {cfg['title']} =====")
    print(f"  episodic upper bound = {cfg['episodic']}")
    for sub, r, m, ma in series:
        flag = "  <== BEST" if sub == best else ""
        win = "OVER" if ma > cfg["episodic"] else "under"
        print(f"  {sub:<30} meanAll={ma:6.2f}  peak={max(m):6.2f}  last={m[-1]:6.2f}  [{win} episodic]{flag}")
    if br:
        bma = sum(bm) / len(bm)
        print(f"  {'(orig baseline) ' + cfg['baseline']:<30} meanAll={bma:6.2f}  peak={max(bm):6.2f}  last={bm[-1]:6.2f}")

    fig, (axf, axz) = plt.subplots(1, 2, figsize=(15, 6.2))
    cmap = plt.cm.viridis
    for ax, (lo, hi, ttl) in zip((axf, axz),
                                 [(*cfg["full"], "Full range"),
                                  (*cfg["zoom"], "Zoom: near episodic")]):
        for i, (sub, r, m, ma) in enumerate(series):
            is_best = sub == best
            ax.plot(r, m, lw=3.2 if is_best else 1.4,
                    color="#d62728" if is_best else cmap(i / max(1, len(series) - 1)),
                    alpha=1.0 if is_best else 0.7, zorder=6 if is_best else 3,
                    label=(f"{sub.replace('smlp_','')} ({ma:.2f})"
                           + ("  BEST" if is_best else "")) if ax is axf else None)
        if br:
            ax.plot(br, bm, lw=1.6, color="#888888", ls="--", zorder=2,
                    label=(f"orig baseline ({sum(bm)/len(bm):.2f})") if ax is axf else None)
        if nr:
            ax.plot(nr, nm, lw=1.4, color="black", ls="-.", alpha=0.6, zorder=1,
                    label="No-Adapt" if ax is axf else None)
        ax.axhline(cfg["episodic"], color="black", ls=":", lw=1.6, zorder=2,
                   label=f"MLMP-episodic ({cfg['episodic']})" if ax is axf else None)
        ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU")
        ax.set_ylim(lo, hi); ax.set_title(ttl); ax.grid(alpha=0.25)
    axf.legend(fontsize=7.5, loc="lower right", ncol=1)
    fig.suptitle(cfg["title"], fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "svg"):
        out = os.path.join(cfg["root"], f"{cfg['out']}.{ext}")
        fig.savefig(out, dpi=140); print(f"  figure -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot(ACDC)
    plot(V20)
