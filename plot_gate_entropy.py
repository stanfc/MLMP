"""
Plot per-batch H_margin trajectory + rst overlay from gate_log.csv.

Usage:
    python plot_gate_entropy.py --run save/ACDCDataset/<run_dir>
    python plot_gate_entropy.py --run <run_dir> --out figures/<run_dir>/gate_entropy.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_gate_log(csv_path):
    batches, h_margins, rsts = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            batches.append(int(row["total_batch"]))
            h_margins.append(float(row["h_margin"]))
            rsts.append(float(row["rst"]))
    return batches, h_margins, rsts


def parse_run_meta(run_dir):
    """Extract h_high / h_low / max_rst from configurations.txt if present."""
    cfg = Path(run_dir) / "configurations.txt"
    meta = {"h_high": None, "h_low": None, "max_rst": None}
    if cfg.exists():
        with open(cfg) as f:
            for line in f:
                line = line.strip()
                for k in meta:
                    prefix = f"{k}:"
                    if line.startswith(prefix) or line.startswith(k):
                        try:
                            meta[k] = float(line.split(":", 1)[1].strip().rstrip(","))
                        except (IndexError, ValueError):
                            pass
    return meta


def plot_gate_entropy(run_dir, out_path):
    run_dir = Path(run_dir)
    csv_path = run_dir / "gate_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist")
    batches, h_margins, rsts = load_gate_log(csv_path)
    if not batches:
        raise ValueError(f"{csv_path} is empty")

    meta = parse_run_meta(run_dir)
    title = (f"Gate trajectory — {run_dir.name}\n"
             f"H_HIGH={meta['h_high']}  H_LOW={meta['h_low']}  "
             f"MAX_RST={meta['max_rst']}")

    max_rst_observed = max(rsts) if rsts else 0.0
    max_rst_ref = meta["max_rst"] if meta["max_rst"] is not None else max_rst_observed

    fig, ax1 = plt.subplots(figsize=(14, 6))
    sc = ax1.scatter(batches, h_margins, c=rsts, cmap="plasma",
                     s=2, alpha=0.6,
                     vmin=0.0, vmax=max(max_rst_ref, 1e-8))
    ax1.set_xlabel("total batch")
    ax1.set_ylabel("H_margin", color="black")
    ax1.grid(True, alpha=0.3)

    if meta["h_high"] is not None:
        ax1.axhline(meta["h_high"], color="C0", linestyle="-",
                    linewidth=1.5, alpha=0.6,
                    label=f"H_HIGH={meta['h_high']}")
    if meta["h_low"] is not None:
        ax1.axhline(meta["h_low"], color="purple", linestyle="--",
                    linewidth=1.5, alpha=0.6,
                    label=f"H_LOW={meta['h_low']}")
    if meta["h_high"] is not None or meta["h_low"] is not None:
        ax1.legend(loc="upper right")

    cbar = plt.colorbar(sc, ax=ax1, pad=0.01)
    cbar.set_label("rst (restoration rate)")

    ax2 = ax1.twinx()
    ax2.plot(batches, rsts, color="red", linewidth=0.6, alpha=0.4,
             label="rst")
    ax2.set_ylabel("rst", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, max(max_rst_ref * 1.1, 1e-6))

    n = len(batches)
    h_min = min(h_margins)
    h_mean = sum(h_margins) / n
    rst_mean = sum(rsts) / n
    summary = (f"batches={n}  H_min={h_min:.3f}  H_mean={h_mean:.3f}  "
               f"rst_mean={rst_mean:.5f}  rst_max={max_rst_observed:.5f}")
    fig.suptitle(title + "\n" + summary, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Path to run dir containing gate_log.csv")
    ap.add_argument("--out", default=None,
                    help="Output PNG path; defaults to figures/<run>/gate_entropy.png")
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_path = Path(args.out) if args.out else Path("figures") / run_dir.name / "gate_entropy.png"
    plot_gate_entropy(run_dir, out_path)


if __name__ == "__main__":
    main()
