import os
import re
import csv

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
SAVE_DIR = os.path.join(REPO_DIR, ".save")
OUTPUT_CSV = os.path.join(REPO_DIR, "results", "results_backbone_comparison.csv")

# (backbone_label, dataset_label, save_dir_name)
SOURCES = [
    ("NACLIP", "V20", "PascalVOC20Dataset"),
    ("CAT-Seg", "V20", "PascalVOC20Dataset_catseg"),
    ("NACLIP", "Cityscapes", "CityscapesDataset"),
    ("CAT-Seg", "Cityscapes", "CityscapesDataset_catseg"),
]

METHOD = "No_Adaptation"


def parse_metric(s):
    """Parse a metric string like '83.81 +/- 0.03' into (mean, std)"""
    s = s.strip()
    m = re.match(r"([\d.]+)\s*\+/-\s*([\d.]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def get_corruption_name(entry):
    """Extract corruption name from directory entry like '01_gaussian_noise'"""
    parts = entry.split("_", 1)
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    return parts[1]


rows = []

for backbone, dataset, dir_name in SOURCES:
    method_dir = os.path.join(SAVE_DIR, dir_name, METHOD)
    if not os.path.isdir(method_dir):
        print(f"Warning: directory not found: {method_dir}")
        continue

    for entry in sorted(os.listdir(method_dir)):
        sub_dir = os.path.join(method_dir, entry)
        if not os.path.isdir(sub_dir):
            continue

        corruption = get_corruption_name(entry)
        if corruption is None:
            continue

        results_path = os.path.join(sub_dir, "results.txt")
        if not os.path.isfile(results_path):
            print(f"Warning: results.txt not found in {sub_dir}")
            continue

        with open(results_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("mIoU"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

            miou_mean, miou_std = parse_metric(parts[0])
            mdice_mean, mdice_std = parse_metric(parts[1])
            macc_mean, macc_std = parse_metric(parts[2])

            if miou_mean is None:
                continue

            rows.append({
                "backbone": backbone,
                "dataset": dataset,
                "corruption": corruption,
                "mIoU_mean": miou_mean,
                "mIoU_std": miou_std,
                "mDice_mean": mdice_mean,
                "mDice_std": mdice_std,
                "mAcc_mean": macc_mean,
                "mAcc_std": macc_std,
            })
            break

fieldnames = ["backbone", "dataset", "corruption",
              "mIoU_mean", "mIoU_std", "mDice_mean", "mDice_std", "mAcc_mean", "mAcc_std"]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
