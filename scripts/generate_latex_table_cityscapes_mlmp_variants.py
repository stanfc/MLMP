import os
import csv
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "results_cityscapes_mlmp_variants.csv")
OUTPUT_TEX = os.path.join(RESULTS_DIR, "results_cityscapes_mlmp_variants_table.tex")

CORRUPTION_ORDER = [
    "original",
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CORRUPTION_DISPLAY = {
    "original": "Original",
    "gaussian_noise":    "Gaussian Noise",
    "shot_noise":        "Shot Noise",
    "impulse_noise":     "Impulse Noise",
    "defocus_blur":      "Defocus Blur",
    "glass_blur":        "Glass Blur",
    "motion_blur":       "Motion Blur",
    "zoom_blur":         "Zoom Blur",
    "snow":              "Snow",
    "frost":             "Frost",
    "fog":               "Fog",
    "brightness":        "Brightness",
    "contrast":          "Contrast",
    "elastic_transform": "Elastic Transform",
    "pixelate":          "Pixelate",
    "jpeg_compression":  "JPEG Compression",
}

METHOD_DISPLAY = {
    "mlmp_batch1_steps10_trail_1": "MLMP (batch=1, 10 steps)",
    "mlmp": "MLMP",
    "mlmp_continual": "MLMP-C",
}

METHOD_ORDER = ["mlmp_batch1_steps10_trail_1", "mlmp", "mlmp_continual"]


def fmt(mean, std, rank=None):
    base = f"{mean:.2f}{{\\scriptsize $\\pm${std:.2f}}}"
    if rank == 1:
        return f"\\textbf{{{base}}}"
    elif rank == 2:
        return f"\\underline{{{base}}}"
    return base


def load_data(path):
    data = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            corr = row["corruption"]
            data[corr][method] = {
                "mIoU_mean": float(row["mIoU_mean"]),
                "mIoU_std":  float(row["mIoU_std"]),
                "mDice_mean": float(row["mDice_mean"]),
                "mDice_std":  float(row["mDice_std"]),
                "mAcc_mean": float(row["mAcc_mean"]),
                "mAcc_std":  float(row["mAcc_std"]),
            }
    return data


def rank_methods(row_data, methods, metric_type):
    scores = []
    for m in methods:
        if m in row_data:
            scores.append((m, row_data[m][f"{metric_type}_mean"]))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranks = {}
    if len(scores) >= 1:
        ranks[scores[0][0]] = 1
    if len(scores) >= 2:
        ranks[scores[1][0]] = 2
    return ranks


def build_metric_table(data, metric_type="mIoU"):
    lines = []

    metric_display = {"mIoU": "mIoU", "mDice": "mDice", "mAcc": "mAcc"}

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Cityscapes MLMP Variants Comparison - {metric_display[metric_type]}.}}")
    lines.append(f"\\label{{tab:cityscapes_mlmp_variants_{metric_type.lower()}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    all_methods = set(m for corr_data in data.values() for m in corr_data.keys())
    methods = [m for m in METHOD_ORDER if m in all_methods]
    method_labels = [METHOD_DISPLAY.get(m, m) for m in methods]

    n_methods = len(methods)
    col_spec = "l" + "c" * n_methods
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = "Corruption"
    for label in method_labels:
        header += f" & {label}"
    header += " \\\\"
    lines.append(header)
    lines.append(r"\midrule")

    for corr in CORRUPTION_ORDER:
        if corr not in data:
            continue

        disp = CORRUPTION_DISPLAY.get(corr, corr)
        ranks = rank_methods(data[corr], methods, metric_type)
        row = disp
        for method in methods:
            if method in data[corr]:
                d = data[corr][method]
                cell = fmt(d[f"{metric_type}_mean"], d[f"{metric_type}_std"],
                           rank=ranks.get(method))
            else:
                cell = "---"
            row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    lines.append(r"\midrule")
    avg_row = "\\textit{Average (C)}"
    corruption_only = [c for c in CORRUPTION_ORDER if c != "original" and c in data]
    avg_data = {}
    for method in methods:
        method_vals = [data[c][method] for c in corruption_only if method in data[c]]
        if method_vals:
            avg_data[method] = {
                f"{metric_type}_mean": sum(m[f"{metric_type}_mean"] for m in method_vals) / len(method_vals),
                f"{metric_type}_std": sum(m[f"{metric_type}_std"] for m in method_vals) / len(method_vals),
            }
    avg_ranks = rank_methods(avg_data, methods, metric_type)
    for method in methods:
        if method in avg_data:
            cell = fmt(avg_data[method][f"{metric_type}_mean"],
                       avg_data[method][f"{metric_type}_std"],
                       rank=avg_ranks.get(method))
        else:
            cell = "---"
        avg_row += f" & {cell}"
    avg_row += " \\\\"
    lines.append(avg_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}% end resizebox")
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
    data = load_data(INPUT_CSV)
    tables = []

    for metric in ["mIoU", "mDice", "mAcc"]:
        try:
            table = build_metric_table(data, metric)
            tables.append(table)
        except Exception as e:
            print(f"Error building table for {metric}: {e}")

    with open(OUTPUT_TEX, "w") as f:
        f.write(PREAMBLE + "\n\n".join(tables) + POSTAMBLE)
    print(f"Written {len(tables)} tables to {OUTPUT_TEX}")
