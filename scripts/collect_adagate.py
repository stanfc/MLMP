#!/usr/bin/env python3
"""Collect the AdaGate sweep against the GDG-PA (hmgate2) reference.

For each arm it reports the mIoU summary plus the two gate diagnostics the sweep
exists to move: the trigger firing rate (A) and the realised shallow-lag spread
(B). GDG-PA's lag histogram is binary {0, maxlag_shallow}; a working B fix shows
mass on the intermediate lags.

Usage:
    python scripts/collect_adagate.py                # both datasets
    python scripts/collect_adagate.py --dataset acdc
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import Counter

REF = {
    'acdc': ('save/ACDCDataset', 'hmgate2_prompt_S0_baseline', 'save/ACDCDataset/adagate_*'),
    'v20': ('save/PascalVOC20Dataset/v20_acdc_matched', 'hmgate2_prompt_S0_baseline',
            'save/PascalVOC20Dataset/v20_acdc_matched/adagate_*'),
    'cityscapes': ('save/CityscapesDataset', 'deyo_mlmp_hmgate2_continual_5corr_sub100',
                   'save/CityscapesDataset/adagate_*'),
}


def miou(run_dir):
    p = os.path.join(run_dir, 'results_all_rounds.txt')
    if not os.path.exists(p):
        return None
    vals = []
    for line in open(p):
        parts = [x.strip() for x in line.split(',')]
        if len(parts) < 2 or not parts[0].lower().startswith('round '):
            continue
        try:
            vals.append(float(parts[-1]))
        except ValueError:
            continue
    return vals or None


def gate(run_dir):
    p = os.path.join(run_dir, 'gate_log.csv')
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    n = len(rows)
    lags = [int(r['lag']) for r in rows]
    deep = [int(r['deep']) for r in rows]
    shallow = Counter(l for l, d in zip(lags, deep) if l > 0 and not d)
    return {
        'n': n,
        'fire': sum(1 for r in rows if float(r['rst']) > 0) / n,
        'deep': sum(deep) / n,
        'shallow': shallow,
        # fraction of shallow restores that landed strictly inside the budget,
        # i.e. the lag actually carried information rather than saturating
        'graded': (sum(v for k, v in shallow.items() if 1 < k < max(shallow, default=1))
                   / max(1, sum(shallow.values()))),
    }


def fmt(name, m, g, ref_mean):
    if m is None:
        return f"  {name:<26} (no results yet)"
    mean, peak, last = sum(m) / len(m), max(m), m[-1]
    tail = m[max(0, len(m) - 30):]
    delta = f"{mean - ref_mean:+.2f}" if ref_mean is not None else "  n/a"
    s = (f"  {name:<26} R{len(m):<3} mean={mean:6.2f} ({delta})  peak={peak:6.2f}@R{m.index(peak)+1:<3} "
         f"last={last:6.2f}  std(tail30)={st.pstdev(tail) if len(tail) > 1 else 0:.2f}")
    if g:
        top = ' '.join(f"{k}:{v}" for k, v in sorted(g['shallow'].items()))
        s += (f"\n  {'':<26} gate: fire={g['fire']:.3f} deep={g['deep']:.3f} "
              f"graded={g['graded']:.2f}  shallow-lag[{top}]")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None, choices=list(REF))
    args = ap.parse_args()

    for ds in ([args.dataset] if args.dataset else list(REF)):
        root, ref_name, pattern = REF[ds]
        ref_dir = os.path.join(root, ref_name)
        ref = miou(ref_dir)
        ref_mean = sum(ref) / len(ref) if ref else None
        print(f"\n===== {ds.upper()} =====")
        if ref:
            print(fmt(f"GDG-PA ({ref_name[:18]})", ref, gate(ref_dir), ref_mean))
        for d in sorted(glob.glob(pattern)):
            if not os.path.isdir(d):
                continue
            print(fmt(os.path.basename(d), miou(d), gate(d), ref_mean))


if __name__ == '__main__':
    main()
