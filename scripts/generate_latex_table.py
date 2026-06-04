import os
import csv
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "results_summary.csv")
OUTPUT_TEX = os.path.join(RESULTS_DIR, "results_table.tex")

CORRUPTION_ORDER = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CORRUPTION_DISPLAY = {
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

# Dataset display config: (short_name, label_for_original, label_for_c_average, multirow_label)
DATASET_CONFIG = {
    "PascalVOC20Dataset":        ("V20",        "V20 (Original)",       "V20-C Average",       None),
    "PascalVOC21Dataset":        ("V21",        "V21 (Original)",       "V21-C Average",       None),
    "PascalContext59Dataset":    ("P59",        "P59 (Original)",       "P59-C Average",       None),
    "PascalContext60Dataset":    ("P60",        "P60 (Original)",       "P60-C Average",       None),
    "CityscapesDataset":         ("CityScapes", "CityScapes (Original)", "CityScapes-C Average", None),
    "COCOObjectDataset":         ("COCOObj",    "COCOObject (Original)", "COCOObject-C Average", None),
    "COCOStuffDataset":          ("COCOStuff",  "COCOStuff (Original)",  "COCOStuff-C Average",  None),
}

# Datasets that have per-corruption rows shown (with multirow grouping)
MULTIROW_DATASETS = list(DATASET_CONFIG.keys())

# Display order
DATASET_ORDER = [
    "PascalVOC20Dataset",
    "PascalVOC21Dataset",
    "PascalContext59Dataset",
    "PascalContext60Dataset",
    "CityscapesDataset",
    "COCOObjectDataset",
    "COCOStuffDataset",
]


def fmt(mean, std):
    return f"{mean:.2f} {{\\scriptsize $\\pm${std:.2f}}}"


def load_data(path):
    data = defaultdict(dict)  # data[dataset][corruption] = {mIoU_mean, ...}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            corr = row["corruption"]
            data[ds][corr] = {
                "mIoU_mean": float(row["mIoU_mean"]),
                "mIoU_std":  float(row["mIoU_std"]),
                "mDice_mean": float(row["mDice_mean"]),
                "mDice_std":  float(row["mDice_std"]),
                "mAcc_mean": float(row["mAcc_mean"]),
                "mAcc_std":  float(row["mAcc_std"]),
            }
    return data


def compute_average(rows):
    """Average over a list of metric dicts."""
    n = len(rows)
    if n == 0:
        return None
    avg = {}
    for key in ["mIoU_mean", "mIoU_std", "mDice_mean", "mDice_std", "mAcc_mean", "mAcc_std"]:
        avg[key] = sum(r[key] for r in rows) / n
    return avg


def metric_cells(d):
    """Return three LaTeX cells for mIoU, mDice, mAcc."""
    return (
        fmt(d["mIoU_mean"], d["mIoU_std"]),
        fmt(d["mDice_mean"], d["mDice_std"]),
        fmt(d["mAcc_mean"], d["mAcc_std"]),
    )


def build_single_table(ds, ds_data, short, orig_label, avg_label):
    lines = []

    caption_ds = orig_label if orig_label else avg_label.replace(" (Average)", "")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Full results on {caption_ds.replace(' (Original)', '')}.}}")
    lines.append(f"\\label{{tab:results_{short.lower().replace('-', '')}}}")
    lines.append(r"\begin{tabular}{ll|ccc}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{3}{c}{MLMP} \\")
    lines.append(r"\cmidrule(lr){3-5}")
    lines.append(r"Dataset & Corruption & mIoU & mDice & mAcc \\")
    lines.append(r"\midrule")

    # Original row (if exists)
    if orig_label and "original" in ds_data:
        orig = ds_data["original"]
        c1, c2, c3 = metric_cells(orig)
        lines.append(f"\\multicolumn{{2}}{{l}}{{{orig_label}}} & {c1} & {c2} & {c3} \\\\")

    # Per-corruption rows
    corr_present = [c for c in CORRUPTION_ORDER if c in ds_data]
    n = len(corr_present)
    for i, corr in enumerate(corr_present):
        disp = CORRUPTION_DISPLAY[corr]
        c1, c2, c3 = metric_cells(ds_data[corr])
        if i == 0:
            lines.append(f"\\multirow{{{n+1}}}{{*}}{{{short}-C}} & {disp} & {c1} & {c2} & {c3} \\\\")
        else:
            lines.append(f" & {disp} & {c1} & {c2} & {c3} \\\\")
    lines.append(r"\cmidrule(l){2-5}")

    # Average row
    avg = compute_average([ds_data[c] for c in corr_present])
    if avg:
        c1, c2, c3 = metric_cells(avg)
        lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{avg_label}}}}} & {c1} & {c2} & {c3} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
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
    for ds in DATASET_ORDER:
        if ds not in data:
            continue
        short, orig_label, avg_label, _ = DATASET_CONFIG[ds]
        tables.append(build_single_table(ds, data[ds], short, orig_label, avg_label))

    with open(OUTPUT_TEX, "w") as f:
        f.write(PREAMBLE + "\n\n".join(tables) + POSTAMBLE)
    print(f"Written {len(tables)} tables to {OUTPUT_TEX}")
