#!/usr/bin/env python
"""Collect the entropy-weighted-prompt sweep on GDG-PA.

Scans save/<DATASET>[/<SUBDIR>]/promptw_<side>_b<beta>/results_all_rounds.txt.
Reference (control) = adapt_b0.0 = bit-identical GDG-PA. Reports, per dataset x side,
beta vs peak / mean(R>=30) / R_last / collapse-onset, and Δ(mR30+) vs the control.
Also plots mean(R>=30) vs beta (adapt & eval curves) per dataset with the control line.
"""
import os, glob, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_prompt_sweep import parse_results, stats  # noqa

DATASETS = [
    ("ACDC",       "save/ACDCDataset"),
    ("VOC20",      "save/PascalVOC20Dataset/v20_acdc_matched"),
    ("Cityscapes", "save/CityscapesDataset"),
]


def gather(root):
    """{(side, beta): stats} for promptw_<side>_b<beta> dirs under root."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "promptw_*_b*"))):
        m = re.search(r"promptw_(adapt|eval)_b([0-9.]+)$", os.path.basename(d))
        if not m:
            continue
        side, beta = m.group(1), float(m.group(2))
        rounds, means = parse_results(os.path.join(d, "results_all_rounds.txt"))
        if rounds is None:
            continue
        out[(side, beta)] = stats(rounds, means)
    return out


def main():
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt; have = True
    except Exception:
        have = False

    for name, root in DATASETS:
        data = gather(root)
        print(f"\n=== {name}  ({root}) ===")
        if not data:
            print("  (no results yet)"); continue
        ctrl = data.get(("adapt", 0.0))
        base = ctrl["mean_stable"] if ctrl else None
        print(f"  {'side':6} {'beta':>5} {'peak':>6} {'peakR':>6} {'mR30+':>7} "
              f"{'Δst':>6} {'R_last':>7} {'coll@':>6} {'nR':>4}")
        for side in ("adapt", "eval"):
            ks = sorted([b for (s, b) in data if s == side])
            for b in ks:
                s = data[(side, b)]
                dst = "" if base is None else f"{s['mean_stable']-base:+.2f}"
                tag = "  <-- GDG-PA" if (side == "adapt" and b == 0.0) else ""
                print(f"  {side:6} {b:5.1f} {s['peak']:6.2f} @R{s['peak_r']:<4d} "
                      f"{s['mean_stable']:7.2f} {dst:>6} {s['r_last']:7.2f} "
                      f"{str(s['collapse_r']):>6} {s['n']:4d}{tag}")

        if have:
            plt.figure(figsize=(7.5, 5))
            for side, col in (("adapt", "#c0392b"), ("eval", "#2980b9")):
                pts = sorted([(b, data[(side, b)]["mean_stable"]) for (s, b) in data if s == side])
                if pts:
                    xs, ys = zip(*pts)
                    plt.plot(xs, ys, "o-", color=col, label=f"{side}-side", lw=1.8)
            if base is not None:
                plt.axhline(base, ls="--", c="#555", lw=1, label="GDG-PA (adapt β0)")
            plt.xlabel("β  (prompt entropy-weight sharpness)"); plt.ylabel("mean mIoU (R≥30)")
            plt.title(f"Entropy-weighted prompt aggregation — {name}")
            plt.legend(); plt.grid(alpha=.3)
            os.makedirs("save/_compare", exist_ok=True)
            out = f"save/_compare/promptw_{name}.png"
            plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
            print(f"  -> figure: {out}")


if __name__ == "__main__":
    main()
