import os
import re
import csv

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
SAVE_DIR = os.path.join(REPO_DIR, ".save")
V20_DIR = os.path.join(SAVE_DIR, "PascalVOC20Dataset")
OUTPUT_CSV = os.path.join(REPO_DIR, "results", "results_v20_methods.csv")

# Methods to compare (excluding DPcore)
METHODS = ["No_Adaptation", "mlmp", "tent", "cotta", "mlmp_continual_32", "tent_continual_64", "dpcore","kff"]
# METHODS = ["No_Adaptation","mlmp_continual_2", "mlmp_continual_4", "mlmp_continual_8", "mlmp_continual_16", "mlmp_continual_32", "mlmp_continual_64"]


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
all_corruptions = set()

# Scan through each method
for method in METHODS:
    method_dir = os.path.join(V20_DIR, method)
    if not os.path.isdir(method_dir):
        print(f"Warning: Method directory not found: {method_dir}")
        continue

    # Scan through each corruption subdirectory
    for entry in sorted(os.listdir(method_dir)):
        sub_dir = os.path.join(method_dir, entry)
        if not os.path.isdir(sub_dir):
            continue

        corruption = get_corruption_name(entry)
        if corruption is None:
            continue

        all_corruptions.add(corruption)
        results_path = os.path.join(sub_dir, "results.txt")
        if not os.path.isfile(results_path):
            print(f"Warning: results.txt not found in {sub_dir}")
            continue

        with open(results_path, "r") as f:
            lines = f.readlines()

        # Skip header and parse data line
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
                "method": method,
                "corruption": corruption,
                "mIoU_mean": miou_mean,
                "mIoU_std": miou_std,
                "mDice_mean": mdice_mean,
                "mDice_std": mdice_std,
                "mAcc_mean": macc_mean,
                "mAcc_std": macc_std,
            })
            break  # only first data line per subdirectory

fieldnames = ["method", "corruption", "mIoU_mean", "mIoU_std", "mDice_mean", "mDice_std", "mAcc_mean", "mAcc_std"]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
print(f"Corruptions found: {sorted(all_corruptions)}")
