#!/usr/bin/env python
"""Collect the text-residual + orthogonality sweep on GDG-PA.

Scans save/<DATASET>[/<SUBDIR>]/textres_<arm>/results_all_rounds.txt where arm is
"ctrl" (= --text_res_lr 0 = bit-identical GDG-PA, the control) or a lambda_orth
value ("0.0" = residual on / regularizer off).

Two outputs per dataset:
  1. table: arm vs peak / mean(R>=30) / R_last / collapse-onset, Δ vs ctrl.
  2. figure: round-vs-mIoU trajectories (left) and, when text_log.csv exists, the
     mean off-diagonal cosine of the class text vectors over the stream (right).
     The second panel is the one that answers "does text-side degradation happen
     at all?" -- a RISING cosine on the lambda=0 arm is that degradation.
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
# ctrl first, then ascending lambda
def arm_key(a):
    return (0, -1.0) if a == "ctrl" else (1, float(a))


def read_text_log(path):
    """-> (batches, mean_offdiag_cos, res_norm_mean) or None."""
    if not os.path.exists(path):
        return None
    b, c, n = [], [], []
    with open(path) as f:
        next(f, None)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                b.append(int(p[0])); n.append(float(p[1])); c.append(float(p[3]))
            except ValueError:
                continue
    return (np.array(b), np.array(c), np.array(n)) if b else None


def gather(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "textres_*"))):
        m = re.match(r"textres_(ctrl|[0-9.]+)$", os.path.basename(d))
        if not m:
            continue
        arm = m.group(1)
        rounds, means = parse_results(os.path.join(d, "results_all_rounds.txt"))
        if rounds is None:
            continue
        s = stats(rounds, means)
        s["rounds"], s["means"] = rounds, means
        s["text"] = read_text_log(os.path.join(d, "text_log.csv"))
        out[arm] = s
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
        ctrl = data.get("ctrl")
        base = ctrl["mean_stable"] if ctrl else None
        print(f"  {'arm':>8} {'peak':>6} {'peakR':>6} {'mR30+':>7} {'Δst':>6} "
              f"{'R_last':>7} {'coll@':>6} {'nR':>4}  {'offdiag_cos':>12}")
        for arm in sorted(data, key=arm_key):
            s = data[arm]
            dst = "" if base is None else f"{s['mean_stable']-base:+.2f}"
            cos = ""
            if s["text"] is not None:
                cos = f"{s['text'][1][0]:.3f}->{s['text'][1][-1]:.3f}"
            tag = "  <-- GDG-PA control" if arm == "ctrl" else (
                  "  <-- residual, NO reg" if arm == "0.0" else "")
            print(f"  {arm:>8} {s['peak']:6.2f} @R{s['peak_r']:<4d} "
                  f"{s['mean_stable']:7.2f} {dst:>6} {s['r_last']:7.2f} "
                  f"{str(s['collapse_r']):>6} {s['n']:4d}  {cos:>12}{tag}")

        if not have:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        cmap = plt.get_cmap("viridis")
        arms = sorted(data, key=arm_key)
        lam = [a for a in arms if a != "ctrl"]
        for arm in arms:
            s = data[arm]
            if arm == "ctrl":
                col, lw, ls, lab = "#000000", 2.0, "--", "GDG-PA (no residual)"
            else:
                col = cmap(lam.index(arm) / max(1, len(lam) - 1) * 0.85)
                lw, ls = 1.6, "-"
                lab = f"residual, λ_orth={arm}" + (" (no reg)" if arm == "0.0" else "")
            ax[0].plot(s["rounds"], s["means"], color=col, lw=lw, ls=ls, label=lab)
            if s["text"] is not None:
                ax[1].plot(s["text"][0], s["text"][1], color=col, lw=lw, ls=ls, label=lab)
        ax[0].set_xlabel("round"); ax[0].set_ylabel("mean mIoU")
        ax[0].set_title(f"Text residual + orthogonality — {name}")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("adapt batches"); ax[1].set_ylabel("mean |off-diagonal cosine|")
        ax[1].set_title("class text-vector collapse (higher = more collapsed)")
        ax[1].grid(alpha=.3)
        os.makedirs("save/_compare", exist_ok=True)
        out = f"save/_compare/textres_{name}.png"
        plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
        print(f"  -> figure: {out}")


if __name__ == "__main__":
    main()
