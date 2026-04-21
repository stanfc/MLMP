import os
import re
import csv

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
SAVE_DIR = os.path.join(REPO_DIR, ".save")
OUTPUT_CSV = os.path.join(REPO_DIR, "results", "results_summary.csv")


def parse_metric(s):
    s = s.strip()
    m = re.match(r"([\d.]+)\s*\+/-\s*([\d.]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


rows = []

for dataset in sorted(os.listdir(SAVE_DIR)):
    mlmp_dir = os.path.join(SAVE_DIR, dataset, "mlmp")
    if not os.path.isdir(mlmp_dir):
        continue

    # Read from per-corruption subdirectories (e.g. 00_original, 01_gaussian_noise, ...)
    for entry in sorted(os.listdir(mlmp_dir)):
        sub_dir = os.path.join(mlmp_dir, entry)
        if not os.path.isdir(sub_dir):
            continue
        # entry format: NN_corruption_name
        parts = entry.split("_", 1)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        corruption = parts[1]
        results_path = os.path.join(sub_dir, "results.txt")
        if not os.path.isfile(results_path):
            continue

        with open(results_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("mIoU") or line.startswith("GPU") or line.startswith("Total") or line.startswith("Mean"):
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
                "dataset": dataset,
                "corruption": corruption,
                "mIoU_mean": miou_mean,
                "mIoU_std": miou_std,
                "mDice_mean": mdice_mean,
                "mDice_std": mdice_std,
                "mAcc_mean": macc_mean,
                "mAcc_std": macc_std,
            })
            break  # only first data line per subdirectory

fieldnames = ["dataset", "corruption", "mIoU_mean", "mIoU_std", "mDice_mean", "mDice_std", "mAcc_mean", "mAcc_std"]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
