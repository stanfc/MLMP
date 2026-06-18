"""
Two-facet signal analysis for gate design.

For each signal × dataset, score it on TWO independent questions:

  Facet 1 — TREND match: does the signal move with mIoU across rounds?
            metric = |Spearman(signal, mIoU)|  in [0, 1]
            (high = tracks the *magnitude* of degradation; e.g. grad_norm)

  Facet 2 — THRESHOLD usability: is the signal MONOTONE in time, so a single
            threshold is crossed exactly once and gives a clean one-shot trigger?
            metric = |Spearman(signal, round)|  in [0, 1]
            (high = monotone drift -> a level crossing is an unambiguous trigger;
             e.g. h_margin. A U-shaped signal like grad_norm crosses any level
             twice, so it tracks magnitude well but is a poor *threshold*.)
            NOTE: pre/post-peak AUC was rejected here — it is ~1 for ANY
            monotone-in-time signal regardless of mIoU, so it does not
            discriminate. Monotonicity is the honest threshold-usability score.

A gate ideally wants Facet 2 (a usable trigger level). Facet 1 tells you how
hard the gate should brake. They rank signals differently — that's the point.

Reads the *_monitor signals_log.csv + results_all_rounds.txt for all 3 datasets.
Output:
  figures/signal_two_facets.png      (side-by-side heatmaps: trend | threshold)
  prints a combined per-dataset table.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from plot_signal_correlation import (
    DATASETS, SAVE, read_miou_per_round, read_signals_per_round, spearman,
)


def monotonicity_in_time(sig, rounds):
    """|Spearman(signal, round)| — how monotone the signal is over time.
    Monotone -> a threshold is crossed once -> clean one-shot gate trigger."""
    return abs(spearman(sig, rounds.astype(float)))


def main():
    trend = {}      # dataset -> {signal: |spearman(sig, mIoU)|}
    thresh = {}     # dataset -> {signal: |spearman(sig, round)| monotonicity}
    peak_val = {}   # dataset -> {signal: signal value at the peak round}
    peaks = {}
    all_signals = None

    for name, sub in DATASETS.items():
        sig_path = f"{SAVE}/{sub}/signals_log.csv"
        miou_path = f"{SAVE}/{sub}/results_all_rounds.txt"
        if not (os.path.exists(sig_path) and os.path.exists(miou_path)):
            print(f"[skip] {name}: missing logs")
            continue
        sr, sig = read_signals_per_round(sig_path)
        mr, miou = read_miou_per_round(miou_path)
        common = np.intersect1d(sr, mr)
        sig = {k: v[np.searchsorted(sr, common)] for k, v in sig.items()}
        miou_c = miou[np.searchsorted(mr, common)]
        peak_idx = int(np.argmax(miou_c))
        peaks[name] = (int(common[peak_idx]), float(miou_c[peak_idx]))
        all_signals = list(sig.keys()) if all_signals is None else all_signals
        trend[name], thresh[name], peak_val[name] = {}, {}, {}
        for k, v in sig.items():
            sp = spearman(v, miou_c)
            trend[name][k] = abs(sp) if np.isfinite(sp) else np.nan
            thresh[name][k] = monotonicity_in_time(v, common)
            peak_val[name][k] = float(v[peak_idx])

    if not trend:
        print("No monitor runs found.")
        return

    dsets = list(trend.keys())
    for name in dsets:
        pr, pm = peaks[name]
        print(f"\n=== {name} (peak mIoU {pm:.1f} @R{pr}) "
              f"| TREND |sp(sig,mIoU)|  &  THRESHOLD |sp(sig,round)| monotonicity ===")
        order = sorted(all_signals,
                       key=lambda s: -(trend[name].get(s, 0) or 0))
        for s in order:
            print(f"  {s:18s}  trend={trend[name].get(s, np.nan):.2f}   "
                  f"monotone={thresh[name].get(s, np.nan):.2f}   "
                  f"val@peak={peak_val[name].get(s, np.nan):+.3f}")

    # Cross-dataset threshold consistency for the leading candidates: if a
    # signal's value-at-peak is similar across datasets, ONE fixed threshold works.
    print("\n=== value@peak across datasets (consistency => one universal threshold) ===")
    for s in ["h_margin", "grad_norm", "ln_param_drift", "mean_conf", "pred_hist_drift"]:
        vals = [f"{name}={peak_val[name].get(s, np.nan):+.2f}" for name in dsets]
        print(f"  {s:18s}  " + "  ".join(vals))

    # ---- side-by-side heatmaps ----
    def matrix(d):
        return np.array([[d[ds].get(s, np.nan) for ds in dsets] for s in all_signals])

    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(dsets) + 7, 0.45 * len(all_signals) + 2))
    for ax, mat, title, cmap in [
        (axes[0], matrix(trend), "Facet 1: TREND  |Spearman(sig, mIoU)|", "Greens"),
        (axes[1], matrix(thresh), "Facet 2: THRESHOLD usability  |Spearman(sig, round)| (monotone)", "Purples"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(dsets)))
        ax.set_xticklabels(dsets)
        ax.set_yticks(range(len(all_signals)))
        ax.set_yticklabels(all_signals, fontsize=8)
        for i in range(len(all_signals)):
            for j in range(len(dsets)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if v < 0.6 else "white")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Signal scoring for gate design — trend vs threshold "
                 "(want a column strong on ALL 3, esp. VOC20)", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/signal_two_facets.png", dpi=130, bbox_inches="tight")
    print("\nsaved -> figures/signal_two_facets.png")


if __name__ == "__main__":
    main()
