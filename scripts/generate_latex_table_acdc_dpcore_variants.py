import os
import csv
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "results_acdc_dpcore_variants.csv")
OUTPUT_TEX = os.path.join(RESULTS_DIR, "results_acdc_dpcore_variants_table.tex")

CONDITIONS = ["fog", "night", "rain", "snow"]
CONDITION_DISPLAY = {"fog": "Fog", "night": "Night", "rain": "Rain", "snow": "Snow"}

ROUNDS_SHOW = [1, 4, 7, 10]
TOTAL_ROUNDS = 10

METHOD_ORDER = [
    "No_Adaptation",
    "dpcore_batch_16_LR_1e-7",
    "dpcore_batch_16_LR_5e-6",
    "dpcore_batch_16_LR_1e-5",
]

METHOD_DISPLAY = {
    "No_Adaptation":           "No Adaptation",
    "dpcore_batch_16_LR_1e-7": "DPCore (LR=1e-7)",
    "dpcore_batch_16_LR_5e-6": "DPCore (LR=5e-6)",
    "dpcore_batch_16_LR_1e-5": "DPCore (LR=1e-5)",
}


def load_data(path):
    """data[method][round][condition] = {mIoU, mDice, mAcc}; durations[method] = seconds."""
    data = defaultdict(lambda: defaultdict(dict))
    durations = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            rnd = int(row["round"])
            cond = row["condition"]
            data[method][rnd][cond] = {
                "mIoU":  float(row["mIoU"]),
                "mDice": float(row["mDice"]),
                "mAcc":  float(row["mAcc"]),
            }
            if row.get("duration_s") and method not in durations:
                try:
                    durations[method] = float(row["duration_s"])
                except ValueError:
                    pass
    return data, durations


def rank_methods(scores_by_method):
    """scores_by_method: {method: value}. Returns {method: rank} with 1=best, 2=second."""
    items = sorted(scores_by_method.items(), key=lambda x: x[1], reverse=True)
    ranks = {}
    if len(items) >= 1:
        ranks[items[0][0]] = 1
    if len(items) >= 2:
        ranks[items[1][0]] = 2
    return ranks


def fmt(val, rank=None):
    s = f"{val:.2f}"
    if rank == 1:
        return f"\\textbf{{{s}}}"
    if rank == 2:
        return f"\\underline{{{s}}}"
    return s


def fmt_time(seconds, rank=None):
    if seconds is None:
        return "---"
    tpr = seconds / TOTAL_ROUNDS
    s = f"{tpr:.0f}s"
    if rank == 1:
        return f"\\textbf{{{s}}}"
    if rank == 2:
        return f"\\underline{{{s}}}"
    return s


def method_mean(data, method, metric):
    """Mean across all rounds × conditions for given method/metric."""
    vals = []
    for rnd in data.get(method, {}):
        for cond in CONDITIONS:
            if cond in data[method][rnd]:
                vals.append(data[method][rnd][cond][metric])
    return sum(vals) / len(vals) if vals else float("nan")


def build_metric_table(data, durations, metric):
    methods = [m for m in METHOD_ORDER if m in data]
    n_conds = len(CONDITIONS)
    n_rounds = len(ROUNDS_SHOW)
    n_data = n_conds * n_rounds

    # col spec: method | round1(4) | round4(4) | round7(4) | round10(4) | mean | time
    col_spec = "l|" + "|".join(["cccc"] * n_rounds) + "|c|r"

    # Per-cell rank (bold best / underline 2nd across methods for this (round, cond))
    cell_ranks = {}
    for rnd in ROUNDS_SHOW:
        for cond in CONDITIONS:
            scores = {}
            for m in methods:
                if rnd in data[m] and cond in data[m][rnd]:
                    scores[m] = data[m][rnd][cond][metric]
            cell_ranks[(rnd, cond)] = rank_methods(scores)

    # Mean rank across methods
    mean_scores = {m: method_mean(data, m, metric) for m in methods}
    mean_ranks = rank_methods(mean_scores)

    # Time rank (lower = better)
    time_scores = {m: durations[m] for m in methods if m in durations}
    time_ranks_raw = sorted(time_scores.items(), key=lambda x: x[1])
    time_ranks = {}
    if len(time_ranks_raw) >= 1:
        time_ranks[time_ranks_raw[0][0]] = 1
    if len(time_ranks_raw) >= 2:
        time_ranks[time_ranks_raw[1][0]] = 2

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        f"\\caption{{ACDC DPCore Hyperparameter Comparison --- {metric}~(\\%). "
        f"Continual TTA, 10 rounds. Mean is averaged over all 10 rounds $\\times$ 4 conditions.}}"
    )
    lines.append(f"\\label{{tab:acdc_dpcore_variants_{metric.lower()}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Row 1: time arrow spanning all data columns
    lines.append(
        r"Time & \multicolumn{" + str(n_data) + r"}{c}{$t$"
        r" \hspace{0.5em}\leaders\hbox{$\relbar$}\hfill\rightarrow}"
        r" & & \\"
    )

    # Row 2: Round group headers
    round_cells = " & ".join(
        r"\multicolumn{4}{c|}{Round " + str(r) + "}" for r in ROUNDS_SHOW
    )
    lines.append(r"Round & " + round_cells + r" & Mean & Time/Round \\")
    lines.append(r"\hline")

    # Row 3: Condition names
    cond_cells = " & ".join([CONDITION_DISPLAY[c] for c in CONDITIONS] * n_rounds)
    lines.append(r"Method & " + cond_cells + r" & & \\")
    lines.append(r"\hline")

    for method in methods:
        row = [METHOD_DISPLAY.get(method, method)]
        for rnd in ROUNDS_SHOW:
            for cond in CONDITIONS:
                if rnd in data[method] and cond in data[method][rnd]:
                    val = data[method][rnd][cond][metric]
                    rank = cell_ranks[(rnd, cond)].get(method)
                    row.append(fmt(val, rank))
                else:
                    row.append("---")
        # Mean
        mean_val = mean_scores.get(method, float("nan"))
        row.append(fmt(mean_val, mean_ranks.get(method)) if mean_val == mean_val else "---")
        # Time/Round
        row.append(fmt_time(durations.get(method), time_ranks.get(method)))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


PREAMBLE = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""

if __name__ == "__main__":
    data, durations = load_data(INPUT_CSV)
    tables = []
    for metric in ["mIoU", "mDice", "mAcc"]:
        try:
            tables.append(build_metric_table(data, durations, metric))
        except Exception as e:
            print(f"Error building table for {metric}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)
    with open(OUTPUT_TEX, "w") as f:
        f.write(PREAMBLE + "\n\n".join(tables) + POSTAMBLE)
    print(f"Written {len(tables)} tables to {OUTPUT_TEX}")
