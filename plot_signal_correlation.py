"""
Correlate collapse/degradation signals with mIoU across rounds, 3 datasets.

Reads, per dataset:
  - signals_log.csv     (per-batch, from --log_signals; utils/collapse_signals.py)
  - results_all_rounds.txt  (per-round mIoU)

Aggregates each signal to a per-round mean, aligns with per-round mIoU, and:
  1. prints a Pearson + Spearman correlation table per dataset (sorted by |Spearman|)
  2. saves a correlation heatmap (signals × datasets, Spearman vs mIoU)
  3. saves per-dataset signal-vs-round overlay panels (each signal twin-axed with mIoU)

Goal: find signals that track mIoU degradation on ALL THREE datasets — in particular
on VOC20 (uniform degradation) where h_margin is flat/blind.

Output:
  figures/signal_corr_heatmap.png
  figures/signal_trends_<dataset>.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE = "save"

DATASETS = {
    "ACDC": "ACDCDataset/deyo_mlmp_continual_monitor",
    "Cityscapes": "CityscapesDataset/deyo_mlmp_continual_monitor_5corr_sub100",
    "VOC20": "PascalVOC20Dataset/deyo_mlmp_continual_monitor_5corr_sub100",
}


def read_miou_per_round(path):
    rounds, miou = [], []
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line.startswith("Round "):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                rounds.append(int(parts[0].split()[1]))
                miou.append(float(parts[-1]))
            except (IndexError, ValueError):
                continue
    return np.array(rounds), np.array(miou)


def read_signals_per_round(path):
    """Return (round_array, {signal: per-round mean array})."""
    import csv
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        names = [c for c in r.fieldnames
                 if c not in ("total_batch", "round", "condition", "batch_idx")]
        for row in r:
            rows.append(row)
    by_round = {}
    for row in rows:
        rd = int(row["round"])
        by_round.setdefault(rd, []).append(row)
    rounds = sorted(by_round)
    sig = {n: [] for n in names}
    for rd in rounds:
        for n in names:
            vals = [float(x[n]) for x in by_round[rd]
                    if x[n] not in ("", "nan")]
            sig[n].append(np.mean(vals) if vals else np.nan)
    return np.array(rounds), {n: np.array(v) for n, v in sig.items()}


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ar = np.argsort(np.argsort(a[m]))
    br = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ar, br)[0, 1])


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    results = {}      # dataset -> {signal: (pearson, spearman)}
    sig_round = {}    # dataset -> (rounds, {signal: arr})
    miou_round = {}   # dataset -> (rounds, miou)
    all_signals = None

    for name, sub in DATASETS.items():
        sig_path = f"{SAVE}/{sub}/signals_log.csv"
        miou_path = f"{SAVE}/{sub}/results_all_rounds.txt"
        if not (os.path.exists(sig_path) and os.path.exists(miou_path)):
            print(f"[skip] {name}: missing {sig_path} or {miou_path}")
            continue
        sr, sig = read_signals_per_round(sig_path)
        mr, miou = read_miou_per_round(miou_path)
        # align on common rounds
        common = np.intersect1d(sr, mr)
        sidx = np.searchsorted(sr, common)
        midx = np.searchsorted(mr, common)
        sig = {k: v[sidx] for k, v in sig.items()}
        miou_c = miou[midx]
        sig_round[name] = (common, sig)
        miou_round[name] = (common, miou_c)
        all_signals = list(sig.keys()) if all_signals is None else all_signals
        results[name] = {k: (pearson(v, miou_c), spearman(v, miou_c))
                         for k, v in sig.items()}

    if not results:
        print("No monitor runs found yet. Run the *_monitor scripts first.")
        return

    # ---- correlation table ----
    for name, d in results.items():
        print(f"\n=== {name}: signal vs mIoU correlation (sorted by |Spearman|) ===")
        for sig_name, (pc, sc) in sorted(d.items(),
                                         key=lambda kv: -abs(kv[1][1] if np.isfinite(kv[1][1]) else 0)):
            print(f"  {sig_name:18s}  pearson={pc:+.3f}  spearman={sc:+.3f}")

    # ---- heatmap (signals × datasets, Spearman) ----
    dsets = list(results.keys())
    mat = np.array([[results[ds].get(s, (np.nan, np.nan))[1] for ds in dsets]
                    for s in all_signals])
    fig, ax = plt.subplots(figsize=(1.6 * len(dsets) + 3, 0.45 * len(all_signals) + 2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(dsets)))
    ax.set_xticklabels(dsets)
    ax.set_yticks(range(len(all_signals)))
    ax.set_yticklabels(all_signals)
    for i in range(len(all_signals)):
        for j in range(len(dsets)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.6 else "white")
    ax.set_title("Spearman(signal, mIoU) per round\n(want |corr|≈1 on ALL three, esp. VOC20)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="Spearman")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/signal_corr_heatmap.png", dpi=130, bbox_inches="tight")
    print("\nsaved -> figures/signal_corr_heatmap.png")

    # ---- per-dataset signal trend panels ----
    for name in results:
        rounds, sig = sig_round[name]
        mr, miou = miou_round[name]
        n = len(sig)
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
        axes = np.array(axes).reshape(-1)
        for ax, (sname, sval) in zip(axes, sig.items()):
            ax.plot(rounds, sval, color="tab:blue", lw=1.3)
            ax.set_ylabel(sname, color="tab:blue", fontsize=8)
            ax.tick_params(labelsize=7)
            axb = ax.twinx()
            axb.plot(mr, miou, color="tab:red", lw=1.0, alpha=0.6)
            axb.tick_params(labelsize=7, colors="tab:red")
            sc = results[name][sname][1]
            ax.set_title(f"{sname}  (ρ={sc:+.2f})", fontsize=8)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"{name}: signals (blue) vs mIoU (red) per round", fontsize=12)
        fig.tight_layout()
        out = f"figures/signal_trends_{name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
