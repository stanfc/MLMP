import os
import re
import csv

SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.join(SCRIPT_DIR, "..")
SAVE_DIR = os.path.join(REPO_DIR, ".save")
ACDC_DIR = os.path.join(SAVE_DIR, "ACDCDataset")
OUTPUT_CSV = os.path.join(REPO_DIR, "results", "results_acdc_dpcore_variants.csv")

METHODS = [
    "No_Adaptation",
    "dpcore_batch_16_LR_1e-7",
    "dpcore_batch_16_LR_5e-6",
    "dpcore_batch_16_LR_1e-5",
]

CONDITIONS = ["fog", "night", "rain", "snow"]

LINE_RE = re.compile(
    r"Round\s+(\d+)\s*/\s*(\w+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)"
)
DURATION_RE = re.compile(r"Total Duration \(s\):\s*([\d.]+)")


def parse_results_txt(path):
    rows = []
    duration = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if m:
                rnd = int(m.group(1))
                cond = m.group(2).lower()
                miou = float(m.group(3))
                mdice = float(m.group(4))
                macc = float(m.group(5))
                rows.append((rnd, cond, miou, mdice, macc))
                continue
            m = DURATION_RE.search(line)
            if m:
                duration = float(m.group(1))
    return rows, duration


all_rows = []
durations = {}

for method in METHODS:
    method_dir = os.path.join(ACDC_DIR, method)
    if not os.path.isdir(method_dir):
        print(f"Warning: method directory not found: {method_dir}")
        continue

    results_path = os.path.join(method_dir, "results.txt")
    if not os.path.isfile(results_path):
        print(f"Warning: results.txt not found in {method_dir}")
        continue

    parsed, duration = parse_results_txt(results_path)
    if not parsed:
        print(f"Warning: no parseable rows in {results_path}")
        continue

    if duration is not None:
        durations[method] = duration

    for rnd, cond, miou, mdice, macc in parsed:
        if cond not in CONDITIONS:
            continue
        all_rows.append({
            "method": method,
            "round": rnd,
            "condition": cond,
            "mIoU": miou,
            "mDice": mdice,
            "mAcc": macc,
            "duration_s": duration if duration is not None else "",
        })

fieldnames = ["method", "round", "condition", "mIoU", "mDice", "mAcc", "duration_s"]

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")
print(f"Methods found: {sorted({r['method'] for r in all_rows})}")
for m, d in durations.items():
    print(f"  {m}: total duration {d:.1f}s")
