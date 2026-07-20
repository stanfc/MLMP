#!/usr/bin/env python
"""Seed validation for the entropy-weighted-prompt gain.

For each dataset, compares the BEST config against its control (GDG-PA, adapt b0.0)
across seeds, as a PAIRED test (the image subset is pinned by --subset_seed 0, so
seeds vary only algorithmic stochasticity).

Reports per seed: control mR30+, best mR30+, and the paired delta. Then the mean +-
std of each and of the delta. The gain is credible when the paired delta is positive
on EVERY seed and its mean clearly exceeds the seed-to-seed spread of the control.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_prompt_sweep import parse_results, stats  # noqa

# dataset -> (root, control(side,beta), best(side,beta))
CASES = [
    ("ACDC",       "save/ACDCDataset",                         ("adapt", "0.0"), ("adapt", "0.5")),
    ("Cityscapes", "save/CityscapesDataset",                   ("adapt", "0.0"), ("eval",  "1.0")),
    ("VOC20",      "save/PascalVOC20Dataset/v20_acdc_matched", ("adapt", "0.0"), ("adapt", "0.5")),
]
SEEDS = [0, 1, 2]


def load(root, side, beta, seed):
    suf = "" if seed == 0 else f"_s{seed}"
    p = os.path.join(root, f"promptw_{side}_b{beta}{suf}", "results_all_rounds.txt")
    rounds, means = parse_results(p)
    if rounds is None:
        return None
    return stats(rounds, means)


def main():
    for name, root, ctrl, best in CASES:
        print(f"\n=== {name} ===")
        print(f"  control = {ctrl[0]} b{ctrl[1]} (GDG-PA)   best = {best[0]} b{best[1]}")
        print(f"  {'seed':>4} {'control':>9} {'best':>9} {'Δ':>8}  {'rounds(c/b)':>12}")
        cs, bs, ds = [], [], []
        for s in SEEDS:
            c = load(root, ctrl[0], ctrl[1], s)
            b = load(root, best[0], best[1], s)
            if c is None or b is None:
                miss = []
                if c is None: miss.append("control")
                if b is None: miss.append("best")
                print(f"  {s:>4} {'(pending: ' + ','.join(miss) + ')':>32}")
                continue
            d = b["mean_stable"] - c["mean_stable"]
            cs.append(c["mean_stable"]); bs.append(b["mean_stable"]); ds.append(d)
            print(f"  {s:>4} {c['mean_stable']:9.2f} {b['mean_stable']:9.2f} {d:+8.2f}"
                  f"  {c['n']:>5}/{b['n']:<5}")
        if len(ds) >= 2:
            cs, bs, ds = map(np.array, (cs, bs, ds))
            print(f"  {'mean':>4} {cs.mean():9.2f} {bs.mean():9.2f} {ds.mean():+8.2f}")
            print(f"  {'std':>4} {cs.std(ddof=1):9.2f} {bs.std(ddof=1):9.2f} {ds.std(ddof=1):8.2f}")
            allpos = bool((ds > 0).all())
            verdict = ("CREDIBLE — positive on every seed and mean Δ > control spread"
                       if allpos and ds.mean() > cs.std(ddof=1)
                       else ("consistent sign but within seed noise" if allpos
                             else "NOT credible — sign flips across seeds"))
            print(f"  -> {verdict}")
        elif len(ds) == 1:
            print("  (only one seed complete so far)")


if __name__ == "__main__":
    main()
