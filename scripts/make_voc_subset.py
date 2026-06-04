#!/usr/bin/env python3
"""
Generate a deterministic VOC val subset split file.

Used by bash/v20_acdc_matched/ to match ACDC's ~406 images per round.
See docs/v20_acdc_matched_spec.md.

Usage:
    python scripts/make_voc_subset.py --n 101 --seed 0
      -> writes data/VOC/VOC2012/ImageSets/Segmentation/val_subset_101_seed0.txt

Determinism: random.Random(seed) + sorted() -> bit-identical output across
runs/machines for the same (n, seed) combination.
"""

import argparse
import os
import random

VOC_ROOT = "data/VOC/VOC2012"
SPLIT_DIR = os.path.join(VOC_ROOT, "ImageSets", "Segmentation")
FULL_VAL = os.path.join(SPLIT_DIR, "val.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True,
                   help="Number of images to sample from the source split")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (different seeds produce different subsets)")
    p.add_argument("--full_split", default=FULL_VAL,
                   help=f"Source split file (default: {FULL_VAL})")
    args = p.parse_args()

    with open(args.full_split) as f:
        ids = [l.strip() for l in f if l.strip()]
    if args.n > len(ids):
        raise ValueError(f"n={args.n} exceeds source size {len(ids)}")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(ids, args.n))

    out = os.path.join(SPLIT_DIR, f"val_subset_{args.n}_seed{args.seed}.txt")
    with open(out, "w") as f:
        f.write("\n".join(sampled) + "\n")
    print(f"Wrote {out} ({args.n} images, seed={args.seed})")


if __name__ == "__main__":
    main()
