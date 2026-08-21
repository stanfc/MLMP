"""Batch-size ablation figures: does our method hold up at batch = 1 / 8 / 64?

Run:  python plot_batch_ablation.py            # -> figures/batch_ablation/
      python plot_batch_ablation.py --outdir X --format pdf

Emits 4 figures -- ACDC and Cityscapes at batch 1 and 8:
    acdc_b1  acdc_b8  cityscapes_b1  cityscapes_b8
VOC20 is deliberately NOT covered here: 學長 already ran all three V20 batch
sizes separately. batch=64 is impossible on ACDC/Cityscapes anyway -- at
1120x560 with patch 224 / stride 112 each image is 36 patches, so b64 = 2304
patches and the bilinear upsample tensor [2304,19,224,224] = 2.20e9 elements
exceeds INT_MAX (it would also need ~630GB; b8 alone peaks at 79.2GB).

Each panel carries four series:
    gradnorm_scaled  ours   - adagate, shallow_cap_mode=growing_scaled, base_rst=0.01
    deyo_mlmp        ablat. - IDENTICAL but base_rst=0.0 (gate never restores)
    mlmp_episodic    ref    - resets per sample -> flat line
    no_adapt         ref    - model never updates -> flat line

Palette is CVD-validated (OKLab dE, Machado protan/deutan, all pairs >= 9.3;
normal-vision >= 18.7; contrast >= 3.0 on white). Colour is never the only cue:
the two adapting methods are solid lines, the two flat references are
dashed/dotted, and each series is direct-labelled at its right edge.

Partial runs are fine -- whatever exists is plotted and a coverage report is
printed, so this can be run repeatedly while the sweep is still going.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "save"
OUTDIR_DEFAULT = "figures/batch_ablation"

# CVD-validated palette (see module docstring). Order is fixed, never cycled.
C_OURS = "#d62728"   # gradnorm_scaled - ours
C_DEYO = "#0072B2"   # deyo_mlmp       - ablation
C_EPI  = "#009E73"   # mlmp_episodic   - reference
C_NOAD = "#7b3f6b"   # no_adapt        - reference

INK, INK_MUTED, GRIDC = "#1a1a1a", "#5c5c5c", "#d9d9d9"

DATASETS = {
    "acdc":       dict(dir="ACDCDataset",        title="ACDC",       note="full, 406 img/round"),
    "cityscapes": dict(dir="CityscapesDataset",  title="Cityscapes", note="subset 100/corruption"),
}
# (dataset key, batch size) -> one figure
PANELS = [("acdc", 1), ("acdc", 8),
          ("cityscapes", 1), ("cityscapes", 8)]


def _dir(ds, method, bs):
    return os.path.join(ROOT, DATASETS[ds]["dir"], "batch_ablation", f"{method}_b{bs}")


def rounds_series(ds, method, bs):
    """Mean_mIoU per round from results_all_rounds.txt; empty array if absent."""
    f = os.path.join(_dir(ds, method, bs), "results_all_rounds.txt")
    if not os.path.exists(f):
        return np.array([])
    vals = []
    for line in open(f):
        if line.startswith("Round "):
            try:
                vals.append(float(line.strip().split(",")[-1]))
            except ValueError:
                pass
    return np.array(vals)


def episodic_value(ds, bs):
    """Mean mIoU over corruptions from an episodic results.txt (a flat line)."""
    f = os.path.join(_dir(ds, "mlmp_episodic", bs), "results.txt")
    if not os.path.exists(f):
        return None
    vals = []
    for line in open(f):
        parts = line.split(",")
        if len(parts) >= 2 and not line.startswith("mIoU") and "+/-" in parts[1]:
            try:
                vals.append(float(parts[1].split("+/-")[0]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else None


def noadapt_value(ds):
    """no_adapt never updates the model, so it is constant across rounds and
    across batch sizes -- run once (batch=1) and reused in every panel.
    Verified empirically on VOC20: batch 1 vs 8 agree to 0.01-0.02 mIoU, i.e.
    fp16 kernel-selection noise, not a batch-size effect."""
    v = rounds_series(ds, "no_adapt", 1)
    return float(np.mean(v)) if v.size else None


def draw(ax, ds, bs):
    """Returns a dict of what was actually found, for the coverage report."""
    found = {}
    ours = rounds_series(ds, "gradnorm_scaled", bs)
    deyo = rounds_series(ds, "deyo_mlmp", bs)
    epi = episodic_value(ds, bs)
    noad = noadapt_value(ds)

    xmax = max(len(ours), len(deyo), 1)
    labels = []   # (y, text, colour) for right-edge direct labels

    if noad is not None:
        ax.axhline(noad, color=C_NOAD, ls=":", lw=2.2, zorder=2)
        labels.append((noad, "no-adapt", C_NOAD)); found["no_adapt"] = 1
    if epi is not None:
        ax.axhline(epi, color=C_EPI, ls="--", lw=2.2, zorder=2)
        labels.append((epi, "MLMP-episodic", C_EPI)); found["mlmp_episodic"] = 1
    if deyo.size:
        ax.plot(np.arange(1, deyo.size + 1), deyo, color=C_DEYO, lw=2.0, zorder=3)
        labels.append((deyo[-1], "DeYO+MLMP", C_DEYO)); found["deyo_mlmp"] = deyo.size
    if ours.size:
        ax.plot(np.arange(1, ours.size + 1), ours, color=C_OURS, lw=2.8, zorder=4)
        labels.append((ours[-1], "ours", C_OURS)); found["gradnorm_scaled"] = ours.size

    # Guard against the auto-scale magnifying a 0.02-mIoU early stretch into what
    # looks like large structure: enforce a floor on the visible y-range.
    MIN_SPAN = 2.0
    y0, y1 = ax.get_ylim()
    if (y1 - y0) < MIN_SPAN:
        mid = 0.5 * (y0 + y1)
        ax.set_ylim(mid - MIN_SPAN / 2, mid + MIN_SPAN / 2)

    meta = DATASETS[ds]
    ax.set_title(f"{meta['title']}  ·  batch = {bs}", fontsize=17, color=INK, pad=10)
    ax.set_xlabel("round", fontsize=13, color=INK_MUTED)
    ax.set_ylabel("mean mIoU", fontsize=13, color=INK_MUTED)
    ax.set_xlim(0, max(xmax, 10) * 1.02)
    ax.grid(True, color=GRIDC, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_MUTED, labelsize=11)

    # selective direct labels, drawn OUTSIDE the axes in the reserved right
    # margin (clip_on=False) so a flat reference line never strikes through its
    # own label, and de-collided vertically against each other.
    labels.sort()
    if labels:
        span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        minsep, prev = span * 0.075, None
        for y, txt, col in labels:
            yy = y if prev is None else max(y, prev + minsep)
            ax.text(1.015, yy, txt, transform=ax.get_yaxis_transform(),
                    color=col, fontsize=11, va="center", ha="left",
                    fontweight="bold", clip_on=False)
            prev = yy
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"batch-size ablation figures -> {args.outdir}/\n")
    report = []
    for ds, bs in PANELS:
        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        found = draw(ax, ds, bs)

        # legend is always present for >=2 series; line style repeats the
        # solid/flat distinction so identity is never carried by colour alone
        spec = [("gradnorm_scaled", C_OURS, "-", 2.8, "ours (gradnorm-scaled)"),
                ("deyo_mlmp",       C_DEYO, "-", 2.0, "DeYO+MLMP (no gate/restore)"),
                ("mlmp_episodic",   C_EPI, "--", 2.2, "MLMP-episodic"),
                ("no_adapt",        C_NOAD, ":", 2.2, "no adaptation")]
        handles = [plt.Line2D([], [], color=c, lw=w, ls=st, label=lab)
                   for k, c, st, w, lab in spec if k in found]
        ax.legend(handles=handles, fontsize=10, frameon=False,
                  loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=2,
                  handlelength=2.6, columnspacing=2.2, labelcolor=INK)
        fig.subplots_adjust(left=0.125, right=0.775, top=0.905, bottom=0.30)
        meta = DATASETS[ds]
        note = meta["note"] + "   ·   no-adapt and MLMP-episodic are flat references"
        fig.text(0.125, 0.035, note, fontsize=9.5, color=INK_MUTED, ha="left")

        name = f"{ds}_b{bs}.{args.format}"
        fig.savefig(os.path.join(args.outdir, name), dpi=args.dpi,
                    facecolor="white")
        plt.close(fig)

        got = ", ".join(f"{k}={v}" for k, v in found.items()) or "NO DATA YET"
        report.append((name, len(found), got))
        print(f"  {name:24s} [{len(found)}/4 series]  {got}")

    done = sum(1 for _, n, _ in report if n == 4)
    print(f"\n{done}/{len(report)} figures have all 4 series; "
          f"re-run this script as more runs finish.")


if __name__ == "__main__":
    main()
