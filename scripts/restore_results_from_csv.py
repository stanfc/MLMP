"""Restore results.txt files from results_v20_methods.csv for a specific method."""
import os
import csv
import argparse

CORRUPTION_ORDER = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

def main():
    parser = argparse.ArgumentParser()
    default_csv = os.path.join(os.path.dirname(__file__), "..", "results", "results_v20_methods.csv")
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--method", required=True, help="Method name in CSV, e.g. mlmp_continual")
    parser.add_argument("--output_dir", required=True, help="Output directory, e.g. .save/PascalVOC20Dataset/mlmp_continual_2")
    args = parser.parse_args()

    # Read CSV and filter by method
    rows = {}
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] == args.method:
                rows[row["corruption"]] = row

    if not rows:
        print(f"No data found for method '{args.method}' in {args.csv}")
        return

    # Write results.txt for each corruption
    for idx, corr in enumerate(CORRUPTION_ORDER):
        if corr not in rows:
            continue
        r = rows[corr]
        subdir = os.path.join(args.output_dir, f"{idx:02d}_{corr}")
        os.makedirs(subdir, exist_ok=True)

        result_path = os.path.join(subdir, "results.txt")
        miou = f"{float(r['mIoU_mean']):.2f} +/- {float(r['mIoU_std']):.2f}"
        mdice = f"{float(r['mDice_mean']):.2f} +/- {float(r['mDice_std']):.2f}"
        macc = f"{float(r['mAcc_mean']):.2f} +/- {float(r['mAcc_std']):.2f}"

        with open(result_path, "w") as f:
            f.write("mIoU, mDice, mAcc\n")
            f.write(f"{miou}, {mdice}, {macc}\n")

        print(f"  Written: {result_path}")

    print(f"\nRestored {len([c for c in CORRUPTION_ORDER if c in rows])} corruptions to {args.output_dir}")


if __name__ == "__main__":
    main()
