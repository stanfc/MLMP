#!/usr/bin/env python
"""
plot_smooth_report.py — report-ready figures for the smooth-anchor study.

Outputs (save/_compare/report_*):
  A_headline_bars        smooth-anchor (tuned) beats source-reset on ACDC
  B_trajectories         D holds vs matched decays vs source-reset
  C_hmargin_violin       H_margin distribution + ceil/floor regions
  D_hmargin_vs_miou      H_margin median vs mean mIoU (mechanism)
  E_gate_regions         stacked %time in no-restore / recent-anchor / source
  F_lag_curve            lag(H) mapping for matched vs D

Run from repo root:  python plot_smooth_report.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = "save/_compare"
A = "save/ACDCDataset"

# label : dir : (h_ceil, h_floor, lag_scale, max_lag)  (None geom = source-reset)
RUNS = {
    "source-reset":        (f"{A}/tent_divgate_continual_cau_rst_0.01", None),
    "matched (1.6/1.4)":   (f"{A}/smooth_anchor_ceil1.6_floor1.4",      (1.6, 1.4, 90, 3000)),
    "A: ceil2.0":          (f"{A}/smooth_sweep_A_ceil2.0",              (2.0, 1.4, 90, 3000)),
    "B: lag1000":          (f"{A}/smooth_sweep_B_lag1000",              (1.6, 1.4, 1000, 3000)),
    "C: ceil2.0+lag1000":  (f"{A}/smooth_sweep_C_ceil2.0_lag1000",      (2.0, 1.4, 1000, 3000)),
    "D: ceil1.8+floor1.55":(f"{A}/smooth_sweep_D_ceil1.8_floor1.55",    (1.8, 1.55, 90, 3000)),
    "win: mon25":          (f"{A}/smooth_sweep_win_mon25",              (1.6, 1.4, 90, 3000)),
    "decoup: buf50/mon10": (f"{A}/smooth_sweep_decoup_buf50_mon10",     (1.6, 1.4, 90, 3000)),
}
# sar_mlmp runs (different H scale) — for the gate-region "100% active" illustration
SARMLMP = {
    "sar_mlmp (Cityscapes)": ("save/CityscapesDataset/sar_mlmp_smooth_anchor_continual_weather", (2.9, 2.2, 150, 3000)),
    "sar_mlmp (ACDC)":       ("save/ACDCDataset/sar_mlmp_smooth_anchor_continual",               (2.9, 2.2, 150, 3000)),
}


def load_means(d):
    p = os.path.join(d, "results_all_rounds.txt")
    if not os.path.isfile(p): return []
    out, first = [], True
    for line in open(p):
        line = line.strip()
        if not line: continue
        if first: first = False; continue
        try: out.append(float(line.split(",")[-1]))
        except ValueError: pass
    return out


def load_hmargin(d):
    ep = os.path.join(d, "entropy_log.csv")
    if os.path.isfile(ep):
        v = []
        with open(ep, newline='') as f:
            for row in csv.DictReader(f):
                try: v.append(float(row['h_margin']))
                except (KeyError, ValueError): pass
        if v: return np.array(v)
    dl = os.path.join(d, "divgate_log.txt")
    if os.path.isfile(dl):
        v = []
        for line in open(dl):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('total'): continue
            try: v.append(float(line.split(',')[1]))
            except (IndexError, ValueError): pass
        if v: return np.array(v)
    return np.array([])


def regions(h, geom):
    """% time in (no_restore, recent_anchor, source) given smooth geom."""
    ceil, floor, lag_scale, max_lag = geom
    src_cut = floor + lag_scale / max_lag          # lag>max_lag -> source
    n = len(h)
    no_restore = 100 * np.mean(h >= ceil)
    source = 100 * np.mean(h < src_cut)
    recent = 100 - no_restore - source
    return no_restore, max(recent, 0.0), source


def main():
    os.makedirs(OUT, exist_ok=True)
    means = {k: load_means(v[0]) for k, (v, *_ ) in [(k, RUNS[k]) for k in RUNS]}
    means = {k: load_means(RUNS[k][0]) for k in RUNS}
    src_mean = np.mean(means["source-reset"])

    # ---------- A: headline bars ----------
    order = ["matched (1.6/1.4)", "A: ceil2.0", "B: lag1000", "C: ceil2.0+lag1000", "D: ceil1.8+floor1.55"]
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [np.mean(means[k]) for k in order]
    cols = ["#d62728" if v < src_mean else "#2ca02c" for v in vals]
    bars = ax.bar(order, vals, color=cols)
    ax.axhline(src_mean, color="#333", ls="--", lw=1.5, label=f"source-reset = {src_mean:.2f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("All-round mean mIoU (ACDC, 150R)")
    ax.set_title("Tuned smooth-anchor (D) beats source-reset; naive (matched) loses", fontweight="bold")
    ax.set_ylim(28, 33); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
    fig.tight_layout(); _save(fig, "report_A_headline_bars")

    # ---------- B: trajectories ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    style = {"source-reset": ("#333", "-"), "matched (1.6/1.4)": ("#d62728", "--"),
             "D: ceil1.8+floor1.55": ("#2ca02c", "-")}
    for k, (c, ls) in style.items():
        m = means[k]
        ax.plot(range(1, len(m)+1), m, color=c, ls=ls, lw=1.8, label=f"{k} (mean {np.mean(m):.2f})")
    ax.set_xlabel("Round"); ax.set_ylabel("Mean mIoU")
    ax.set_title("ACDC: D retreats-to-source sooner → holds; matched decays", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); _save(fig, "report_B_trajectories")

    # ---------- C: H_margin violin ----------
    keys = ["source-reset", "matched (1.6/1.4)", "D: ceil1.8+floor1.55", "C: ceil2.0+lag1000"]
    data = [load_hmargin(RUNS[k][0]) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    parts = ax.violinplot([d for d in data], showmedians=True, widths=0.8)
    for i, k in enumerate(keys):
        geom = RUNS[k][1]
        if geom:
            ax.hlines(geom[1], i+1-0.4, i+1+0.4, color="red", lw=1.5)     # floor
            ax.hlines(geom[0], i+1-0.4, i+1+0.4, color="blue", lw=1.2, ls=":")  # ceil
    ax.hlines([], [], [], color="red", label="h_floor (→source below)")
    ax.hlines([], [], [], color="blue", ls=":", label="h_ceil (no-restore above)")
    ax.set_xticks(range(1, len(keys)+1)); ax.set_xticklabels(keys, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("H_margin (per-batch marginal entropy)")
    ax.set_title("H_margin distribution: lower = more collapsed. matched sags lowest.", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "report_C_hmargin_violin")

    # ---------- D: H_margin median vs mean mIoU ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in RUNS:
        m = means[k]; h = load_hmargin(RUNS[k][0])
        if len(m) < 150 or len(h) == 0:    # skip partial
            continue
        x, y = np.median(h), np.mean(m)
        col = "#333" if k == "source-reset" else ("#2ca02c" if y >= src_mean else "#d62728")
        ax.scatter(x, y, s=80, color=col, zorder=3)
        ax.annotate(k, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(src_mean, color="#333", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("H_margin median (model health)"); ax.set_ylabel("All-round mean mIoU")
    ax.set_title("Preventing H_margin collapse lifts mIoU; D (hybrid) is the sweet spot", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "report_D_hmargin_vs_miou")

    # ---------- E: gate-region stacked bars ----------
    gr_keys = ["matched (1.6/1.4)", "D: ceil1.8+floor1.55", "C: ceil2.0+lag1000"]
    gr_keys_sar = list(SARMLMP.keys())
    allk = gr_keys + gr_keys_sar
    geoms = [RUNS[k][1] for k in gr_keys] + [SARMLMP[k][1] for k in gr_keys_sar]
    dirs = [RUNS[k][0] for k in gr_keys] + [SARMLMP[k][0] for k in gr_keys_sar]
    nr, ra, sr = [], [], []
    for d, g in zip(dirs, geoms):
        h = load_hmargin(d)
        a, b, c = regions(h, g) if len(h) else (0, 0, 0)
        nr.append(a); ra.append(b); sr.append(c)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(allk))
    ax.bar(x, nr, label="no-restore (H≥ceil)", color="#bbbbbb")
    ax.bar(x, ra, bottom=nr, label="recent-anchor (lag(H))", color="#ff7f0e")
    ax.bar(x, sr, bottom=[a+b for a, b in zip(nr, ra)], label="source-retreat", color="#2ca02c")
    for i in range(len(allk)):
        if sr[i] > 5: ax.text(i, nr[i]+ra[i]+sr[i]/2, f"{sr[i]:.0f}%", ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(allk, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("% of batches"); ax.set_ylim(0, 100)
    ax.set_title("Gate-region occupancy. sar_mlmp: ceil>band → 100% restore-active.", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); _save(fig, "report_E_gate_regions")

    # ---------- F: lag(H) curve ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    H = np.linspace(1.41, 2.0, 400)
    for label, (ceil, floor, ls, ml), col in [
        ("matched (floor1.4, lag90)", (1.6, 1.4, 90, 3000), "#d62728"),
        ("D (floor1.55, lag90)", (1.8, 1.55, 90, 3000), "#2ca02c")]:
        lag = np.where(H > floor, ls / np.maximum(H - floor, 1e-6), np.inf)
        lag = np.where((H >= ceil) | (lag > ml), np.nan, lag)   # no-restore or source
        ax.plot(H, lag, color=col, lw=2, label=label)
        ax.axvline(floor, color=col, ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("H_margin"); ax.set_ylabel("anchor lag (batches back)")
    ax.set_title("lag(H) = lag_scale/(H−floor): worse health → deeper anchor; H≤floor → source", fontweight="bold")
    ax.set_ylim(0, 3000); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "report_F_lag_curve")

    print("All report figures written to", OUT)


def _save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


if __name__ == "__main__":
    main()
