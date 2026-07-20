#!/usr/bin/env python
"""Collect the GDG-PA prompt-set sweep.

Scans save/<DATASET>[/<SUBDIR>]/hmgate2_prompt_<ID>/results_all_rounds.txt for
every prompt set, and reports per (dataset, prompt set):
  peak         : max round-mean mIoU
  peak@R       : round of that peak
  mean(R30+)   : mean of round-mean mIoU over rounds >= 30 (stable-phase quality)
  R_last       : most recent round's mean mIoU (=R150 when complete)
  collapse@R   : first round after the peak whose mean < 0.6*peak ('-' = no collapse)
  n_rounds     : rounds completed so far (runs may still be in flight)

Also renders a round-vs-mIoU trajectory overlay per dataset to save/_compare/.
Δ columns are versus S0_baseline within the same dataset.
"""
import os, glob, re
import numpy as np

_ROUND_RE = re.compile(r"(\d+)")

DATASETS = [
    ("ACDC",  "save/ACDCDataset"),
    ("VOC20", "save/PascalVOC20Dataset/v20_acdc_matched"),
]
COLLAPSE_FRAC = 0.6
STABLE_FROM = 30


def parse_results(path):
    """Return (rounds, means) arrays from a results_all_rounds.txt, or (None, None)."""
    if not os.path.isfile(path):
        return None, None
    rounds, means = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # data rows look like "Round 01, 33.54, ..., 29.21"; header is
            # "Round, fog, ..., Mean_mIoU" (parts[0] has no digit -> skipped).
            m = _ROUND_RE.search(parts[0])
            if m is None:
                continue
            try:
                rounds.append(int(m.group(1)))
                means.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
    if not rounds:
        return None, None
    return np.array(rounds), np.array(means)


def stats(rounds, means):
    peak_i = int(np.argmax(means))
    peak, peak_r = float(means[peak_i]), int(rounds[peak_i])
    stable = means[rounds >= STABLE_FROM]
    mean_stable = float(stable.mean()) if stable.size else float("nan")
    r_last = float(means[-1])
    collapse_r = "-"
    thr = COLLAPSE_FRAC * peak
    for r, m in zip(rounds[peak_i + 1:], means[peak_i + 1:]):
        if m < thr:
            collapse_r = int(r)
            break
    return dict(peak=peak, peak_r=peak_r, mean_stable=mean_stable,
                r_last=r_last, collapse_r=collapse_r, n=int(rounds[-1]))


def collect_dataset(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "hmgate2_prompt_*"))):
        pid = os.path.basename(d).replace("hmgate2_prompt_", "")
        rounds, means = parse_results(os.path.join(d, "results_all_rounds.txt"))
        if rounds is None:
            continue
        out[pid] = (rounds, means, stats(rounds, means))
    return out


def fmt_row(pid, s, base_peak, base_stable):
    dp = "" if base_peak is None else f"{s['peak']-base_peak:+5.2f}"
    dsb = "" if base_stable is None else f"{s['mean_stable']-base_stable:+5.2f}"
    return (f"  {pid:18s} {s['peak']:6.2f} {dp:>6}  @R{s['peak_r']:<3d} "
            f"{s['mean_stable']:7.2f} {dsb:>6}  {s['r_last']:6.2f}  "
            f"{str(s['collapse_r']):>6}  {s['n']:4d}")


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_plt = True
    except Exception:
        have_plt = False

    for name, root in DATASETS:
        data = collect_dataset(root)
        print(f"\n=== {name}  ({root}) ===")
        if not data:
            print("  (no results yet)")
            continue
        base = data.get("S0_baseline")
        base_peak = base[2]["peak"] if base else None
        base_stable = base[2]["mean_stable"] if base else None
        print(f"  {'prompt_set':18s} {'peak':>6} {'Δpk':>6}  {'peakR':<5} "
              f"{'mR30+':>7} {'Δst':>6}  {'R_last':>6}  {'coll@':>6}  {'nR':>4}")
        # rank by mean(R30+) desc
        for pid in sorted(data, key=lambda k: -data[k][2]["mean_stable"]):
            print(fmt_row(pid, data[pid][2], base_peak, base_stable))

        if have_plt:
            plt.figure(figsize=(9, 5))
            for pid in sorted(data):
                r, m, _ = data[pid]
                lw = 2.4 if pid == "S0_baseline" else 1.3
                z = 5 if pid == "S0_baseline" else 2
                plt.plot(r, m, label=pid, linewidth=lw, zorder=z)
            plt.xlabel("Round"); plt.ylabel("Mean mIoU")
            plt.title(f"GDG-PA prompt-set sweep — {name}")
            plt.legend(fontsize=7, ncol=2); plt.grid(alpha=0.3)
            os.makedirs("save/_compare", exist_ok=True)
            out = f"save/_compare/prompt_sweep_{name}.png"
            plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
            print(f"  -> figure: {out}")


if __name__ == "__main__":
    main()
