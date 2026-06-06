#!/usr/bin/env python
"""
analyze_gate.py — quick read-out for DivGate / smooth-anchor CTTA runs.

For each run directory it reports:
  1. Performance   : all-round mean mIoU, R1, R_last, peak(@round), per-condition last round
  2. H_margin band : min / p5 / median / p95 / max  (the gate's health signal)
  3. Gate activity : how often restoration actually engages

Data sources (auto-detected, in priority order):
  - entropy_log.csv   (NEW runs, all methods): per-batch h_margin  -> dense, uniform
  - divgate_log.txt   (source-reset DivGate, pre-merge): per-window h_margin + mode
  - results_all_rounds.txt : per-round mIoU
  - configurations.txt     : method + gate thresholds (h_threshold/h_warning or h_ceil/h_floor)

Usage:
  python scripts/analyze_gate.py save/ACDCDataset/smooth_anchor_ceil1.6_floor1.4 \
                                 save/ACDCDataset/tent_divgate_continual_cau_rst_0.01
  python scripts/analyze_gate.py save/CityscapesDataset/*          # globs work via shell
"""
import os
import sys
import csv
import argparse
from statistics import median


# ----------------------------------------------------------------------
def _pctile(sorted_vals, q):
    if not sorted_vals:
        return float('nan')
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def read_config(run_dir):
    """Parse configurations.txt -> dict of str:str (best-effort)."""
    cfg = {}
    path = os.path.join(run_dir, 'configurations.txt')
    if not os.path.isfile(path):
        return cfg
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, _, v = line.partition(':')
                cfg[k.strip()] = v.strip()
    return cfg


def read_results(run_dir):
    """results_all_rounds.txt -> dict with mean/r1/rlast/peak/conditions."""
    path = os.path.join(run_dir, 'results_all_rounds.txt')
    if not os.path.isfile(path):
        return None
    rows, header = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = [c.strip() for c in line.split(',')]
                continue
            parts = [p.strip() for p in line.split(',')]
            try:
                mean_val = float(parts[-1])
            except (ValueError, IndexError):
                continue
            rows.append((parts[0], parts[1:-1], mean_val))
    if not rows:
        return None
    means = [r[2] for r in rows]
    peak_i = max(range(len(means)), key=lambda i: means[i])
    cond_names = header[1:-1] if header else []
    return {
        'n_rounds': len(rows),
        'mean_all': sum(means) / len(means),
        'r1': means[0],
        'rlast': means[-1],
        'rlast_label': rows[-1][0],
        'peak': means[peak_i],
        'peak_round': rows[peak_i][0],
        'cond_names': cond_names,
        'cond_last': rows[-1][1],
    }


def read_hmargin(run_dir):
    """Return (values:list[float], source:str, modes:dict|None)."""
    # Prefer entropy_log.csv (dense, every batch, all methods)
    ep = os.path.join(run_dir, 'entropy_log.csv')
    if os.path.isfile(ep):
        vals = []
        with open(ep, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    vals.append(float(row['h_margin']))
                except (KeyError, ValueError):
                    pass
        if vals:
            return vals, 'entropy_log.csv', None
    # Fallback: divgate_log.txt (has a mode column)
    dl = os.path.join(run_dir, 'divgate_log.txt')
    if os.path.isfile(dl):
        vals, modes = [], {}
        with open(dl) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('total_batches'):
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    continue
                if len(parts) >= 3:
                    modes[parts[2]] = modes.get(parts[2], 0) + 1
        if vals:
            return vals, 'divgate_log.txt', (modes or None)
    return [], None, None


def _fnum(cfg, *keys):
    for k in keys:
        if k in cfg:
            try:
                return float(cfg[k])
            except ValueError:
                pass
    return None


def analyze(run_dir):
    cfg = read_config(run_dir)
    res = read_results(run_dir)
    vals, src, modes = read_hmargin(run_dir)

    method = cfg.get('method', '?')
    # gate geometry: source-reset uses h_threshold/h_warning; smooth uses h_ceil/h_floor
    ceil = _fnum(cfg, 'h_ceil', 'h_threshold')
    floor = _fnum(cfg, 'h_floor', 'h_warning')

    print("=" * 74)
    print(f"RUN  {run_dir}")
    print(f"     method={method}  ceil/thr={ceil}  floor/warn={floor}")

    # ---- performance ----
    if res:
        cond = "  ".join(f"{n}={v}" for n, v in zip(res['cond_names'], res['cond_last']))
        print(f"  PERF  mean(all {res['n_rounds']}R)={res['mean_all']:.2f}   "
              f"R1={res['r1']:.2f}   {res['rlast_label'].replace(' ', '')}={res['rlast']:.2f}   "
              f"peak={res['peak']:.2f}@{res['peak_round'].replace(' ', '')}")
        if cond:
            print(f"        last-round per-condition: {cond}")
    else:
        print("  PERF  (no results_all_rounds.txt yet)")

    # ---- H_margin band ----
    if vals:
        sv = sorted(vals)
        print(f"  H_margin  n={len(sv)}  src={src}")
        print(f"        min={sv[0]:.3f}  p5={_pctile(sv,.05):.3f}  "
              f"median={median(sv):.3f}  p95={_pctile(sv,.95):.3f}  max={sv[-1]:.3f}")
        # ---- gate activity ----
        if modes:
            tot = sum(modes.values())
            md = "  ".join(f"{k}={100*v/tot:.1f}%" for k, v in sorted(modes.items()))
            print(f"  GATE  (from mode column)  {md}")
        elif ceil is not None:
            below_ceil = 100 * sum(v < ceil for v in sv) / len(sv)
            line = f"  GATE  restore-active (H<ceil={ceil}): {below_ceil:.1f}% of batches"
            if floor is not None:
                below_floor = 100 * sum(v < floor for v in sv) / len(sv)
                line += f"   |  source/brake region (H<floor={floor}): {below_floor:.1f}%"
            print(line)
            if below_ceil < 2:
                print("        ⚠️  gate almost never engages -> behaves ~like plain TENT here")
    else:
        print("  H_margin  (no entropy_log.csv / divgate_log.txt found)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='+', help='one or more run directories (containing '
                    'entropy_log.csv / divgate_log.txt / results_all_rounds.txt)')
    args = ap.parse_args()

    summary = []
    for d in args.run_dirs:
        d = d.rstrip('/')
        if not os.path.isdir(d):
            print(f"!! not a directory: {d}", file=sys.stderr)
            continue
        res = analyze(d)
        if res:
            summary.append((os.path.basename(d), res['mean_all'], res['rlast'], res['peak']))

    if len(summary) > 1:
        print("=" * 74)
        print("SUMMARY  (sorted by all-round mean)")
        print(f"  {'run':<46}{'mean':>7}{'Rlast':>8}{'peak':>8}")
        for name, m, rl, pk in sorted(summary, key=lambda x: -x[1]):
            print(f"  {name[:46]:<46}{m:>7.2f}{rl:>8.2f}{pk:>8.2f}")


if __name__ == '__main__':
    main()
