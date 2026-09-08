"""Single-panel version of the plug-and-play figure for TENT at 74 trainable params.

Same style as plot_plugplay.py, but this pair is run at the flagship's parameter
setting (`--top_block_exclude 6`: ln_post and resblocks 18-23 frozen, 74 of the
100 visual LayerNorm params adapt) instead of the published baselines' full 100.

Because the published `tent_continual` trains all 100, it is NOT a valid control
here -- so this panel uses the matched internal control instead: the identical
command with `--base_rst 0.0`, i.e. the gate is computed and logged but never
restores. Verified matched pair: both 74 params, LR 1e-5, single-level forward,
150 rounds, seed 0; `--base_rst` is the only difference.

Run:  python plot_tent74.py      # -> figures/plugplay/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "figures/plugplay"
R = "save/ACDCDataset/plugplay"
NO_ADAPT = 23.34
C_GATE, C_NOGATE, C_NOAD = "#d62728", "#8c8c8c", "#7b3f6b"
INK, INK_MUTED, GRIDC = "#1a1a1a", "#5c5c5c", "#d9d9d9"


def load(run_dir):
    out = []
    for line in open(os.path.join(run_dir, "results_all_rounds.txt")):
        line = line.strip()
        if not line.startswith("Round ") or "," not in line:
            continue
        parts = [q.strip() for q in line.split(",")]
        try:
            out.append((int(parts[0].split()[1]), float(parts[-1])))
        except (ValueError, IndexError):
            continue
    out.sort()
    return np.array([v for _, v in out])


def main():
    os.makedirs(SAVE, exist_ok=True)
    y_no = load(f"{R}/tent_adagate_nogate_b1")
    y_ga = load(f"{R}/tent_adagate_gate_b1")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhline(NO_ADAPT, color=C_NOAD, ls=":", lw=2.2, zorder=2)
    ax.plot(np.arange(1, y_no.size + 1), y_no, color=C_NOGATE, lw=2.1,
            ls=(0, (5, 2)), zorder=3, solid_capstyle="round")
    ax.plot(np.arange(1, y_ga.size + 1), y_ga, color=C_GATE, lw=2.8,
            zorder=4, solid_capstyle="round")

    ax.set_xlim(1, 150)
    ax.set_ylim(0, 36)
    ax.set_xlabel("continual round", fontsize=12, color=INK_MUTED)
    ax.set_ylabel("mean mIoU", fontsize=12, color=INK_MUTED)
    ax.set_title("TENT (ICLR'21) with 74 of 100 LayerNorm params adapting",
                 fontsize=13.5, color=INK, pad=10)
    ax.grid(True, color=GRIDC, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_MUTED, labelsize=11)

    # right-margin direct labels
    for y, txt, col, bold in [(y_ga[-1], f"+ AdaGate   {y_ga[-1]:.1f}", C_GATE, True),
                              (NO_ADAPT, "no adaptation  23.3", C_NOAD, False),
                              (y_no[-1], f"no gate   {y_no[-1]:.1f}", C_NOGATE, False)]:
        ax.annotate(txt, xy=(150, y), xytext=(153, y), textcoords="data",
                    color=col, fontsize=11, va="center", ha="left",
                    annotation_clip=False, fontweight="bold" if bold else "normal")

    # when the ungated arm gives up
    coll = next((i + 1 for i, v in enumerate(y_no) if v < 15), None)
    if coll:
        ax.annotate(f"no-gate arm falls below 15 mIoU at R{coll}",
                    xy=(coll, 15), xytext=(coll + 14, 20.5), color=INK_MUTED,
                    fontsize=10.5, ha="left",
                    arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.1))

    # placed in the empty band above the gated line -- the lower half is crossed
    # by the collapsing arm, so nothing legible fits there
    ax.text(0.5, 0.93, f"R150:  {y_no[-1]:.1f}  →  {y_ga[-1]:.1f}   "
                       f"({y_ga[-1] - y_no[-1]:+.1f})",
            transform=ax.transAxes, fontsize=12.5, color=C_GATE,
            fontweight="bold", va="center", ha="center")

    handles = [
        plt.Line2D([], [], color=C_NOGATE, lw=2.1, ls=(0, (5, 2)), label="no gate (--base_rst 0)"),
        plt.Line2D([], [], color=C_GATE, lw=2.8, label="+ AdaGate (--base_rst 0.01)"),
        plt.Line2D([], [], color=C_NOAD, lw=2.2, ls=":", label="no adaptation"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.06),
              frameon=False, fontsize=10.5, handlelength=2.3, labelcolor=INK)

    fig.text(0.5, 0.045,
             "ACDC · 150 rounds · batch = 1 · LR 1e-5 · seed 0",
             fontsize=8.8, color=INK_MUTED, ha="center")
    fig.text(0.5, 0.016,
             "top_block_exclude = 6 (ln_post + resblocks 18-23 frozen) · matched pair, "
             "only --base_rst differs",
             fontsize=8.8, color=INK_MUTED, ha="center")
    fig.subplots_adjust(left=0.095, right=0.775, top=0.90, bottom=0.205)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{SAVE}/acdc_tent74.{ext}", dpi=200, facecolor="white")
    print(f"wrote {SAVE}/acdc_tent74.{{png,pdf,svg}}")
    print(f"  no gate : R1={y_no[0]:.2f} peak={y_no.max():.2f}@R{int(np.argmax(y_no))+1} "
          f"R150={y_no[-1]:.2f} mean={y_no.mean():.2f}")
    print(f"  +AdaGate: R1={y_ga[0]:.2f} peak={y_ga.max():.2f}@R{int(np.argmax(y_ga))+1} "
          f"R150={y_ga[-1]:.2f} mean={y_ga.mean():.2f}")


if __name__ == "__main__":
    main()
