"""
Plot mIoU trajectories from results_all_rounds.txt files.

Usage:
    python plot_results.py
    python plot_results.py --runs save/ACDCDataset/tent_divgate_continual_step_1
    python plot_results.py --runs path/to/run1 path/to/run2 --labels "TENT-DivGate" "TENT"
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CONDITIONS = ["fog", "night", "rain", "snow"]


def load_run(run_dir):
    """Read results_all_rounds.txt -> dict of column -> list."""
    path = Path(run_dir) / "results_all_rounds.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    rows = {"round": [], "fog": [], "night": [], "rain": [], "snow": [], "mean": []}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            row = [c.strip() for c in row]
            if not row or not row[0].lower().startswith("round"):
                continue
            rows["round"].append(int(row[0].split()[1]))
            rows["fog"].append(float(row[1]))
            rows["night"].append(float(row[2]))
            rows["rain"].append(float(row[3]))
            rows["snow"].append(float(row[4]))
            rows["mean"].append(float(row[5]))
    return rows


def plot_mean_trajectory(runs, labels, save_to):
    """Plot Mean_mIoU vs Round, one line per run."""
    plt.figure(figsize=(10, 5))
    for run, label in zip(runs, labels):
        plt.plot(run["round"], run["mean"], label=label, linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Mean mIoU")
    plt.title("Mean mIoU across continual rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, dpi=150)
    print(f"saved {save_to}")
    plt.close()


def plot_per_condition(run, label, save_to):
    """Plot one line per ACDC condition for a single run."""
    plt.figure(figsize=(10, 5))
    for cond in CONDITIONS:
        plt.plot(run["round"], run[cond], label=cond, linewidth=2)
    plt.plot(run["round"], run["mean"], label="mean", linewidth=2.5,
             color="black", linestyle="--")
    plt.xlabel("Round")
    plt.ylabel("mIoU")
    plt.title(f"Per-condition mIoU — {label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to, dpi=150)
    print(f"saved {save_to}")
    plt.close()


def print_summary(runs, labels):
    print(f"\n{'Run':<35} {'Rounds':>7} {'R1':>7} {'Peak':>10} {'R150':>7} {'Mean':>7}")
    print("-" * 80)
    for run, label in zip(runs, labels):
        n = len(run["round"])
        peak_idx = max(range(n), key=lambda i: run["mean"][i])
        peak_str = f"{run['mean'][peak_idx]:.2f}@R{run['round'][peak_idx]}"
        r150 = run["mean"][-1] if run["round"][-1] >= 150 else float("nan")
        mean_all = sum(run["mean"]) / n
        print(f"{label:<35} {n:>7} {run['mean'][0]:>7.2f} {peak_str:>10} "
              f"{r150:>7.2f} {mean_all:>7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+",
        default=["save/ACDCDataset/tent_divgate_continual_step_1"],
        help="paths to run directories containing results_all_rounds.txt",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="labels for each run (default: directory basename)",
    )
    parser.add_argument(
        "--out_dir", default="figures",
        help="where to save the figures",
    )
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [Path(p).name for p in args.runs]
    if len(args.labels) != len(args.runs):
        raise ValueError("--labels must have the same length as --runs")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    runs = [load_run(p) for p in args.runs]

    # Figure 1: mean trajectory (all runs overlaid)
    plot_mean_trajectory(
        runs, args.labels,
        save_to=Path(args.out_dir) / "mean_trajectory.png",
    )

    # Figure 2: per-condition plot (one PNG per run)
    for run, label, run_path in zip(runs, args.labels, args.runs):
        safe = label.replace(" ", "_").replace("/", "_")
        plot_per_condition(
            run, label,
            save_to=Path(args.out_dir) / f"per_condition_{safe}.png",
        )

    print_summary(runs, args.labels)


if __name__ == "__main__":
    main()
