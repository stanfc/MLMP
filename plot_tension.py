"""The core tension figure: methods that improve collapse; methods that never
collapse never improve.

ACDC, 150 continual rounds, batch = 1, full 406 img/round.
Every series was verified protocol-compatible before plotting (no --subset_size,
same corruption order fog/night/rain/snow, same batch size, same 150-round
budget). ACDC is currently the ONLY dataset where the classic baselines and the
proposed method were run under one protocol -- the Cityscapes and VOC20
baselines sit in a different subset/corruption group and must not be overlaid.

"Ours w/o gate" is the single-flag ablation (--base_rst 0.0): identical command,
gate computed and logged but never restoring. It carries the highest peak in the
figure (33.50 @ R34) and then dies -- which is the point.

Palette is the repo's CVD-validated set extended with two Okabe-Ito hues; all
pairs were checked under deuteranopia / protanopia / tritanopia simulation
(worst CVD dE = 8.5, normal-vision floor 18.1). The ablation shares Ours' hue and
is separated by dash pattern, so no new hue enters the categorical set.

Run:  python plot_tension.py      # -> figures/tension/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "figures/tension"
ROOT = "save/ACDCDataset"
C_OURS = "#d62728"

# (label, path, colour, linewidth, linestyle, zorder)
SERIES = [
    ("TENT",           f"{ROOT}/tent_continual_Round150_lr_0.00001", "#0072B2", 2.0, "-",  3),
    ("MLMP-continual", f"{ROOT}/mlmp_continual_round_150_step_1",    "#E69F00", 2.0, "-",  3),
    ("SAR",            f"{ROOT}/sar_continual_weather",              "#CC79A7", 2.0, "-",  3),
    ("EATA",           f"{ROOT}/eata_continual",                     "#009E73", 2.0, "-",  3),
    ("Ours w/o gate",  f"{ROOT}/batch_ablation/deyo_mlmp_b1",        C_OURS,    2.0, (0, (5, 2)), 4),
    ("Ours",           f"{ROOT}/batch_ablation/gradnorm_scaled_b1",  C_OURS,    2.9, "-",  5),
]
NO_ADAPT = 23.34
C_NOAD = "#7b3f6b"
INK, INK_MUTED, GRIDC = "#1a1a1a", "#5c5c5c", "#d9d9d9"


def load(run_dir):
    """results_all_rounds.txt -> mean-mIoU array indexed by round."""
    out = []
    with open(os.path.join(run_dir, "results_all_rounds.txt")) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("Round ") or "," not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                out.append((int(parts[0].split()[1]), float(parts[-1])))
            except (ValueError, IndexError):
                continue
    out.sort()
    return np.array([v for _, v in out])


def draw(ax, data, ylim, title):
    # no-adaptation drawn ON TOP so its dots stay visible where EATA (23.35) sits
    # on the same value (23.34) -- that coincidence is the finding, not a collision
    ax.axhline(NO_ADAPT, color=C_NOAD, ls=":", lw=2.4, zorder=6)
    for (name, _, col, lw, ls, z), y in zip(SERIES, data):
        ax.plot(np.arange(1, y.size + 1), y, color=col, lw=lw, ls=ls, zorder=z,
                solid_capstyle="round")
    ax.set_xlim(1, 150)
    ax.set_ylim(*ylim)
    ax.set_xlabel("continual round", fontsize=12, color=INK_MUTED)
    ax.set_title(title, fontsize=12.5, color=INK, pad=8)
    ax.grid(True, color=GRIDC, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_MUTED, labelsize=10.5)


def main():
    os.makedirs(SAVE, exist_ok=True)
    data = [load(p) for _, p, _, _, _, _ in SERIES]
    peaks = {n: (int(np.argmax(y)) + 1, float(y.max())) for (n, *_), y in zip(SERIES, data)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.3))
    draw(axL, data, (0, 36), "(a) full range — who survives 150 rounds")
    draw(axR, data, (22.5, 34.5), "(b) zoom on the top band — who keeps the gains")
    axL.set_ylabel("mean mIoU (ACDC, 4 conditions)", fontsize=12, color=INK_MUTED)

    # (a): the two halves of the tension
    axL.annotate("improve early,\nthen collapse", xy=(97, 17.0), xytext=(118, 11.0),
                 color=INK_MUTED, fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.1))
    axL.annotate("never collapse,\nnever improve", xy=(52, 23.34), xytext=(38, 16.5),
                 color=INK_MUTED, fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.1))
    r, v = peaks["Ours w/o gate"]
    axL.annotate(f"highest peak in the figure\n({v:.1f} @ R{r}) — and it dies",
                 xy=(r, v), xytext=(78, 33.0), color=C_OURS, fontsize=10.5, ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_OURS, lw=1.1))

    # (b): retention separates the methods, so label the line ends directly --
    # SAR oscillates across most of this band, leaving no room for leader lines
    axR.text(75, 34.05, "every other method ends below its own peak",
             color=INK_MUTED, fontsize=10.5, ha="center", va="center")
    for name, y, txt, col, bold in [
            ("Ours", 31.92, "Ours  31.9", C_OURS, True),
            ("SAR", 25.31, "SAR  25.3", "#CC79A7", False)]:
        axR.annotate(txt, xy=(150, y), xytext=(153, y), textcoords="data",
                     color=col, fontsize=10.5, va="center", ha="left",
                     annotation_clip=False,
                     fontweight="bold" if bold else "normal")

    handles = [plt.Line2D([], [], color=C_NOAD, lw=2.2, ls=":", label="No adaptation")]
    handles += [plt.Line2D([], [], color=c, lw=w, ls=ls, label=n)
                for n, _, c, w, ls, _ in SERIES]
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=11,
               ncol=7, handlelength=2.3, columnspacing=1.7, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Methods that improve collapse; methods that never collapse never improve",
                 fontsize=15, color=INK, y=0.975)
    fig.text(0.5, 0.075,
             "ACDC · 150 continual rounds · batch = 1 · evaluate-before-adapt · no reset · seed 0"
             "   ·   “Ours w/o gate” = identical command with --base_rst 0.0",
             fontsize=9.5, color=INK_MUTED, ha="center")
    fig.subplots_adjust(left=0.062, right=0.925, top=0.86, bottom=0.175, wspace=0.14)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{SAVE}/acdc_tension.{ext}", dpi=200, facecolor="white")
    print(f"wrote {SAVE}/acdc_tension.{{png,pdf,svg}}")
    for n, (r, v) in peaks.items():
        print(f"  peak  {n:16s} {v:6.2f} @ R{r}")


if __name__ == "__main__":
    main()
