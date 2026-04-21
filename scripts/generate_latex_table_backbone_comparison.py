import os
import csv
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "results_backbone_comparison.csv")
OUTPUT_TEX = os.path.join(RESULTS_DIR, "results_backbone_comparison_table.tex")

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

# Column keys: (backbone, dataset)
COLUMN_ORDER = [
    ("NACLIP", "V20"),
    ("CAT-Seg", "V20"),
    ("NACLIP", "Cityscapes"),
    ("CAT-Seg", "Cityscapes"),
]

COLUMN_DISPLAY = {
    ("NACLIP", "V20"): "NACLIP",
    ("CAT-Seg", "V20"): "CAT-Seg",
    ("NACLIP", "Cityscapes"): "NACLIP",
    ("CAT-Seg", "Cityscapes"): "CAT-Seg",
}


def fmt(mean, std, rank=None):
    """Format metric as mean +/- std for LaTeX. rank=1 for bold, rank=2 for underline."""
    base = f"{mean:.2f}{{\\scriptsize $\\pm${std:.2f}}}"
    if rank == 1:
        return f"\\textbf{{{base}}}"
    elif rank == 2:
        return f"\\underline{{{base}}}"
    return base


def load_data(path):
    """Load CSV and organize as data[corruption][(backbone, dataset)] = {metrics}"""
    data = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["backbone"], row["dataset"])
            corr = row["corruption"]
            data[corr][key] = {
                "mIoU_mean": float(row["mIoU_mean"]),
                "mIoU_std":  float(row["mIoU_std"]),
                "mDice_mean": float(row["mDice_mean"]),
                "mDice_std":  float(row["mDice_std"]),
                "mAcc_mean": float(row["mAcc_mean"]),
                "mAcc_std":  float(row["mAcc_std"]),
            }
    return data


def rank_columns(row_data, columns, metric_type, dataset):
    """Rank columns within a dataset group. rank=1 for best, rank=2 for second."""
    group = [c for c in columns if c[1] == dataset]
    scores = []
    for c in group:
        if c in row_data:
            scores.append((c, row_data[c][f"{metric_type}_mean"]))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranks = {}
    if len(scores) >= 1:
        ranks[scores[0][0]] = 1
    if len(scores) >= 2:
        ranks[scores[1][0]] = 2
    return ranks


def build_metric_table(data, metric_type="mIoU"):
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Backbone Comparison (No Adaptation) - {metric_type}.}}")
    lines.append(f"\\label{{tab:backbone_comparison_{metric_type.lower()}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # l | cc | cc
    lines.append(r"\begin{tabular}{l|cc|cc}")
    lines.append(r"\toprule")
    # Two-level header
    lines.append(r" & \multicolumn{2}{c|}{Pascal VOC 2012} & \multicolumn{2}{c}{Cityscapes} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    header = "Corruption"
    for col in COLUMN_ORDER:
        header += f" & {COLUMN_DISPLAY[col]}"
    header += " \\\\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for corr in CORRUPTION_ORDER:
        if corr not in data:
            continue

        disp = CORRUPTION_DISPLAY.get(corr, corr)
        # Rank within each dataset group
        v20_ranks = rank_columns(data[corr], COLUMN_ORDER, metric_type, "V20")
        city_ranks = rank_columns(data[corr], COLUMN_ORDER, metric_type, "Cityscapes")
        all_ranks = {**v20_ranks, **city_ranks}

        row = disp
        for col in COLUMN_ORDER:
            if col in data[corr]:
                d = data[corr][col]
                cell = fmt(d[f"{metric_type}_mean"], d[f"{metric_type}_std"],
                           rank=all_ranks.get(col))
            else:
                cell = "---"
            row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    # Average row (excluding original)
    lines.append(r"\midrule")
    avg_row = "\\textit{Average (C)}"
    corruption_only = [c for c in CORRUPTION_ORDER if c != "original" and c in data]
    avg_data = {}
    for col in COLUMN_ORDER:
        vals = [data[c][col] for c in corruption_only if col in data[c]]
        if vals:
            avg_data[col] = {
                f"{metric_type}_mean": sum(v[f"{metric_type}_mean"] for v in vals) / len(vals),
                f"{metric_type}_std": sum(v[f"{metric_type}_std"] for v in vals) / len(vals),
            }
    v20_ranks = rank_columns(avg_data, COLUMN_ORDER, metric_type, "V20")
    city_ranks = rank_columns(avg_data, COLUMN_ORDER, metric_type, "Cityscapes")
    all_ranks = {**v20_ranks, **city_ranks}
    for col in COLUMN_ORDER:
        if col in avg_data:
            cell = fmt(avg_data[col][f"{metric_type}_mean"],
                       avg_data[col][f"{metric_type}_std"],
                       rank=all_ranks.get(col))
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
