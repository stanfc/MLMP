"""
parse_acdc_results.py
Reads ACDC experiment results from save/ACDCDataset/ and generates a LaTeX
table in the style of CoTTA Table 5.

Usage:  python parse_acdc_results.py
"""

import os
import re
import math

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SAVE_ROOT    = "save/ACDCDataset"
CONDITIONS   = ["fog", "night", "rain", "snow"]
ROUNDS_SHOW  = [1, 4, 7, 10]
TOTAL_ROUNDS = 10          # continual experiments always run 10 rounds

# (display_name, save_dir, is_episodic)
METHODS = [
    ("No Adaptation",   "No_Adaptation",         False),
    ("TENT-continual",  "tent_continual_lr_0.00001",         False),
    ("MLMP-continual",  "mlmp_continual_batch_1_LR_0.00001", False),
    ("CoTTA",           "cotta_batch_1",          False),
    ("CMA-continual",   "cma_continual_step_1",  False),
    ("CMA-Proto-continual", "cma_proto_continual_step_1", False),
    ("CMA-Layered-continual", "cma_layered_continual_step_1", False),
    ("CMA-DivGate-continual", "cma_divgate_continual_step_1", False),
    ("MLMP (episodic)", "mlmp_batch_1",           True),
]

# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_results_all_rounds(path):
    """Parse results_all_rounds.txt → {round: {cond: miou}}"""
    results = {}
    with open(path) as f:
        lines = f.readlines()
    header = [h.strip() for h in lines[0].split(",")]
    cond_cols = {c: header.index(c) for c in CONDITIONS if c in header}
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"Round\s+(\d+)", line)
        if not m:
            continue
        rnd   = int(m.group(1))
        parts = [p.strip() for p in line.split(",")]
        results[rnd] = {c: float(parts[col]) for c, col in cond_cols.items()}
    return results


def parse_episodic(save_dir):
    """Parse per-condition results from main.py (episodic) → {cond: miou}"""
    cond_miou = {}
    for entry in os.listdir(save_dir):
        for cond in CONDITIONS:
            if cond in entry.lower():
                rfile = os.path.join(save_dir, entry, "results.txt")
                if os.path.isfile(rfile):
                    with open(rfile) as f:
                        val = float(f.readlines()[1].split("+/-")[0].strip().split(",")[0])
                    cond_miou[cond] = val
    return cond_miou


def parse_duration(results_txt_path):
    """Extract Total Duration (s) from results.txt → float seconds, or None."""
    if not os.path.isfile(results_txt_path):
        return None
    with open(results_txt_path) as f:
        for line in f:
            m = re.search(r"Total Duration \(s\):\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def load_method(save_dir, is_episodic):
    """Returns ({round: {cond: miou}}, time_per_round_min or None)"""
    all_rounds_path = os.path.join(save_dir, "results_all_rounds.txt")
    results_path    = os.path.join(save_dir, "results.txt")

    if is_episodic or not os.path.isfile(all_rounds_path):
        cond_miou = parse_episodic(save_dir)
        rounds    = {r: dict(cond_miou) for r in ROUNDS_SHOW}
        duration  = parse_duration(results_path)
        # episodic: total duration covers 1 full pass (≈ 1 round)
        time_per_round = duration if duration else None
    else:
        all_data  = parse_results_all_rounds(all_rounds_path)
        rounds    = {r: all_data[r] for r in ROUNDS_SHOW if r in all_data}
        duration  = parse_duration(results_path)
        time_per_round = duration / TOTAL_ROUNDS if duration else None

    return rounds, time_per_round


# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────

data      = {}   # name → {round → {cond → miou}}
durations = {}   # name → minutes per round (float or None)

for name, rel_dir, is_episodic in METHODS:
    full_dir = os.path.join(SAVE_ROOT, rel_dir)
    if not os.path.isdir(full_dir):
        print(f"[WARN] not found, skipping: {full_dir}")
        continue
    try:
        rounds, tpr = load_method(full_dir, is_episodic)
        data[name]      = rounds
        durations[name] = tpr
        tpr_str = f"{tpr:.0f}s" if tpr is not None else "N/A"
        print(f"[OK]  {name:25s}  time/round={tpr_str}")
    except Exception as e:
        print(f"[ERR] {name}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Find best per column (for bold)
# ─────────────────────────────────────────────────────────────────────────────

columns = [(r, c) for r in ROUNDS_SHOW for c in CONDITIONS]

def mean_all(name):
    vals = [data[name][r][c]
            for r in ROUNDS_SHOW for c in CONDITIONS
            if r in data[name] and c in data.get(name, {}).get(r, {})]
    return sum(vals) / len(vals) if vals else float("nan")

best_col  = {}
for col in columns:
    r, c = col
    candidates = {m: data[m][r][c] for m in data if r in data[m] and c in data[m][r]}
    best_col[col] = max(candidates, key=candidates.get) if candidates else None

all_means = {m: mean_all(m) for m in data}
best_mean = max(all_means, key=all_means.get)

# Best time = shortest (min) — lower is better
time_vals = {m: durations[m] for m in durations if durations[m] is not None}
best_time = min(time_vals, key=time_vals.get) if time_vals else None

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX generation
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, bold):
    s = f"{val:.1f}"
    return r"\textbf{" + s + "}" if bold else s

def fmt_time(val, bold):
    if val is None:
        return "--"
    s = f"{val:.0f}s"
    return r"\textbf{" + s + "}" if bold else s

n_conds  = len(CONDITIONS)   # 4
n_rounds = len(ROUNDS_SHOW)  # 4
n_data   = n_conds * n_rounds  # 16
# Columns: 1 (method) + 16 (data) + 1 (mean) + 1 (time) = 19
n_total  = 1 + n_data + 1 + 1

# column spec: method | 4×4 data | mean | time
col_spec = "l|" + "|".join(["cccc"] * n_rounds) + "|c|r"

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\footnotesize")   # smaller font fixes oversized caption
lines.append(r"\caption{Continual TTA on ACDC (mIoU~\%). "
             r"Mean is averaged over all 10 rounds.}")
lines.append(r"\label{tab:acdc_continual}")
lines.append(r"\resizebox{\textwidth}{!}{")
lines.append(r"\begin{tabular}{" + col_spec + "}")
lines.append(r"\toprule")

# ── Row 1: time arrow ──────────────────────────────────────────────────────
arrow_span = n_data + n_rounds - 1   # spans data cols + separators inside groups
lines.append(
    r"Time & \multicolumn{" + str(n_data) + r"}{c}{$t$"
    r" \hspace{0.5em}\leaders\hbox{$\relbar$}\hfill\rightarrow}"
    r" & & \\"
)

# ── Row 2: Round numbers ───────────────────────────────────────────────────
round_cells = " & ".join(
    r"\multicolumn{4}{c|}{Round " + str(r) + "}" for r in ROUNDS_SHOW
)
lines.append(r"Round & " + round_cells + r" & Mean & Time/Round \\")
lines.append(r"\hline")   # line under round row

# ── Row 3: Condition names ─────────────────────────────────────────────────
cond_cap   = [c.capitalize() for c in CONDITIONS]
cond_cells = " & ".join(cond_cap * n_rounds)
lines.append(r"Method & " + cond_cells + r" & & \\")
lines.append(r"\hline")   # line under condition row

# ── Data rows ──────────────────────────────────────────────────────────────
for name, _, _ in METHODS:
    if name not in data:
        continue
    row = [name]
    for r in ROUNDS_SHOW:
        for c in CONDITIONS:
            if r in data[name] and c in data[name][r]:
                val  = data[name][r][c]
                bold = (best_col.get((r, c)) == name)
                row.append(fmt(val, bold))
            else:
                row.append("--")
    # Mean
    am   = all_means.get(name, float("nan"))
    bold = (best_mean == name)
    row.append(fmt(am, bold))
    # Time/Round
    row.append(fmt_time(durations.get(name), best_time == name))
    lines.append(" & ".join(row) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"}")   # resizebox
lines.append(r"\end{table}")

latex = "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Save & print
# ─────────────────────────────────────────────────────────────────────────────

out_path = os.path.join(SAVE_ROOT, "acdc_table.tex")
with open(out_path, "w") as f:
    f.write(latex + "\n")

print(f"\nLaTeX table saved → {out_path}")
print("\n" + "=" * 72)
print(latex)

# ─────────────────────────────────────────────────────────────────────────────
# Delta table  (round-over-round Δ mIoU)
# Round 1 → 0 baseline; Round N → data[N] − data[prev shown round]
# ─────────────────────────────────────────────────────────────────────────────

# Build delta_data: {name: {round: {cond: delta}}}
delta_data = {}
for name in data:
    delta_data[name] = {}
    prev = None
    for r in ROUNDS_SHOW:
        if r not in data[name]:
            continue
        if prev is None:
            delta_data[name][r] = {c: 0.0 for c in CONDITIONS}
        else:
            delta_data[name][r] = {
                c: data[name][r].get(c, float("nan")) - data[name][prev].get(c, float("nan"))
                for c in CONDITIONS
            }
        prev = r


def fmt_delta(val):
    """Color-coded delta: green for gain, red for loss, gray for zero."""
    if math.isnan(val):
        return "--"
    if abs(val) < 0.05:
        return r"\textcolor{gray}{0.0}"
    s = f"{val:+.1f}"
    if val > 0:
        return r"\textcolor{ForestGreen}{" + s + "}"
    else:
        return r"\textcolor{red}{" + s + "}"


def delta_row_mean(name):
    """Mean of all non-zero deltas (rounds 4/7/10 × all conditions)."""
    vals = [
        delta_data[name][r][c]
        for r in ROUNDS_SHOW[1:]          # skip round 1 (all zeros)
        for c in CONDITIONS
        if r in delta_data.get(name, {}) and c in delta_data[name][r]
        and not math.isnan(delta_data[name][r][c])
    ]
    return sum(vals) / len(vals) if vals else float("nan")


delta_lines = []
delta_lines.append(r"% Requires: \usepackage[dvipsnames]{xcolor}")
delta_lines.append(r"\begin{table}[t]")
delta_lines.append(r"\centering")
delta_lines.append(r"\footnotesize")
delta_lines.append(
    r"\caption{Continual TTA on ACDC --- round-over-round $\Delta$mIoU~(\%). "
    r"Round~1 is the baseline (0); each subsequent column shows the change "
    r"from the previous shown round. "
    r"\textcolor{ForestGreen}{Green}~= improvement, \textcolor{red}{red}~= degradation.}"
)
delta_lines.append(r"\label{tab:acdc_continual_delta}")
delta_lines.append(r"\resizebox{\textwidth}{!}{")
delta_lines.append(r"\begin{tabular}{" + col_spec + "}")
delta_lines.append(r"\toprule")

# Row 1: time arrow (reuse same string as main table)
delta_lines.append(
    r"Time & \multicolumn{" + str(n_data) + r"}{c}{$t$"
    r" \hspace{0.5em}\leaders\hbox{$\relbar$}\hfill\rightarrow}"
    r" & & \\"
)

# Row 2: round numbers
delta_lines.append(r"Round & " + round_cells + r" & Mean $\Delta$ & Time/Round \\")
delta_lines.append(r"\hline")

# Row 3: condition names
delta_lines.append(r"Method & " + cond_cells + r" & & \\")
delta_lines.append(r"\hline")

# Data rows
for name, _, _ in METHODS:
    if name not in delta_data:
        continue
    row = [name]
    for r in ROUNDS_SHOW:
        for c in CONDITIONS:
            if r in delta_data[name] and c in delta_data[name][r]:
                row.append(fmt_delta(delta_data[name][r][c]))
            else:
                row.append("--")
    # Mean Δ (average of rounds 4/7/10 deltas across conditions)
    dm = delta_row_mean(name)
    row.append(fmt_delta(dm) if not math.isnan(dm) else "--")
    # Time/Round (same as main table)
    row.append(fmt_time(durations.get(name), best_time == name))
    delta_lines.append(" & ".join(row) + r" \\")

delta_lines.append(r"\bottomrule")
delta_lines.append(r"\end{tabular}")
delta_lines.append(r"}")   # resizebox
delta_lines.append(r"\end{table}")

delta_latex = "\n".join(delta_lines)

delta_out = os.path.join(SAVE_ROOT, "acdc_table_delta.tex")
with open(delta_out, "w") as f:
    f.write(delta_latex + "\n")

print(f"\nDelta LaTeX table saved → {delta_out}")
print("\n" + "=" * 72)
print(delta_latex)
