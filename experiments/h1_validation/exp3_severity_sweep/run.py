"""Exp 3 — Source-only mIoU sweep across corruption severity 1-5.

For each cell, invokes main_continual.py with the corresponding flags
(no --adapt so it's source-only evaluation), then reads the 1-round
Mean_mIoU out of the resulting results_all_rounds.txt.

Usage:
    python -m experiments.h1_validation.exp3_severity_sweep.run \
        --dataset VOC20_matched --corruption snow --severity 3 \
        [--out experiments/h1_validation/results/exp3/all.csv] \
        [--save_dir experiments/h1_validation/results/exp3/raw/<ds>_<corr>_s<sev>/] \
        [--skip_done]
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys

from experiments.h1_validation.common import DATASET_REGISTRY, append_row, already_done


def parse_first_round_miou(results_path: str) -> float:
    """Read first 'Round NN, ..., Mean_mIoU' row of results_all_rounds.txt."""
    with open(results_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = [x.strip() for x in lines[0].split(",")]
    mean_idx = header.index("Mean_mIoU")
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        if re.match(r"Round\s+\d+", parts[0]):
            return float(parts[mean_idx])
    raise RuntimeError(f"no Round row in {results_path}")


def build_cmd(args, entry, save_dir):
    """Compose the main_continual.py invocation for a single cell."""
    cmd = [
        "python", "main_continual.py",
        "--dataset", entry["main_dataset_name"],
        "--data_dir", entry["data_dir"],
        "--method", "tent_continual",   # arbitrary; --adapt omitted → source-only
        "--ovss_type", "naclip",
        "--ovss_backbone", "ViT-L/14",
        "--corruptions_list", args.corruption,
        "--severity", str(args.severity),
        "--continual_rounds", "1",
        "--save_dir", save_dir,
    ]
    kw = entry["prepare_data_kwargs"]
    if "init_resize" in kw:
        cmd += ["--init_resize", str(kw["init_resize"][0]), str(kw["init_resize"][1])]
    if "patch_size" in kw:
        ps = kw["patch_size"]
        cmd += ["--patch_size", str(ps[0]), str(ps[1])]
    if "patch_stride" in kw:
        cmd += ["--patch_stride", str(kw["patch_stride"])]
    if "batch_size" in kw:
        cmd += ["--batch_size", str(kw["batch_size"])]
    if "num_workers" in kw:
        cmd += ["--workers", str(kw["num_workers"])]
    if kw.get("ann_file"):
        cmd += ["--ann_file", kw["ann_file"]]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=[k for k, v in DATASET_REGISTRY.items() if v["kind"] == "synthetic"])
    ap.add_argument("--corruption", required=True)
    ap.add_argument("--severity", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="experiments/h1_validation/results/exp3/all.csv")
    ap.add_argument("--save_dir", default=None,
                    help="default: experiments/h1_validation/results/exp3/raw/<ds>_<corr>_s<sev>/")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    entry = DATASET_REGISTRY[args.dataset]
    if args.corruption not in entry["conditions"]:
        sys.exit(f"corruption {args.corruption!r} not in {entry['conditions']}")

    key = {"dataset": args.dataset, "corruption": args.corruption, "severity": args.severity}
    if args.skip_done and already_done(args.out, key):
        print(f"[exp3] skip done {args.dataset}/{args.corruption}/sev={args.severity}", flush=True)
        return

    save_dir = args.save_dir or os.path.join(
        "experiments/h1_validation/results/exp3/raw",
        f"{args.dataset}_{args.corruption}_s{args.severity}",
    )
    os.makedirs(save_dir, exist_ok=True)
    cmd = build_cmd(args, entry, save_dir)
    print(f"[exp3] running: {' '.join(cmd)}", flush=True)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"main_continual.py failed (exit {e.returncode})")

    miou = parse_first_round_miou(os.path.join(save_dir, "results_all_rounds.txt"))
    append_row(args.out, {
        "dataset": args.dataset,
        "kind": entry["kind"],
        "corruption": args.corruption,
        "severity": args.severity,
        "miou": miou,
    })
    print(f"[exp3] DONE  {args.dataset}/{args.corruption}/sev={args.severity}  miou={miou:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
