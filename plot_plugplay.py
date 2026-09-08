"""Plug-and-play figure: the same gate, four published base objectives.

2x2 small multiples, one panel per base objective, all on a shared y-axis so the
collapses are directly comparable. Two lines per panel:

    no gate   (grey, dashed) -- the published method, already in save/
    + AdaGate (red,  solid)  -- identical base objective, our gate added

Every gated run uses `--top_block_exclude 0`, i.e. all 100 visual LayerNorm
params, exactly what the published baselines train -- so the published run is a
matched no-gate control and the only difference is the gate.

⚠️ SAR is the one arm that REPLACES rather than ADDS: SAR ships its own hard
recovery (Filter C), which is disabled so the two controllers do not fight. The
panel is labelled accordingly; do not describe it as a pure addition.

Palette: the repo's "Ours" red plus a neutral grey; all three pairs verified
under deuteranopia / protanopia / tritanopia simulation (worst CVD dE = 11.4,
normal-vision floor 19.5).

Runs still in progress are drawn as far as they got and flagged in the panel.

Run:  python plot_plugplay.py      # -> figures/plugplay/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "figures/plugplay"
ROOT = "save/ACDCDataset"
NO_ADAPT = 23.34

C_GATE, C_NOGATE, C_NOAD = "#d62728", "#8c8c8c", "#7b3f6b"
INK, INK_MUTED, GRIDC = "#1a1a1a", "#5c5c5c", "#d9d9d9"

# (panel title, no-gate run, gated run, footnote marker)
PANELS = [
    ("TENT  (ICLR'21)",
     f"{ROOT}/tent_continual_Round150_lr_0.00001",
     f"{ROOT}/plugplay/tent_adagate_b1", ""),
    ("MLMP-continual  (NeurIPS'25)",
     f"{ROOT}/mlmp_continual_round_150_step_1",
     f"{ROOT}/plugplay/mlmp_adagate_b1", ""),
    ("DELTA  (ICLR'23)",
     f"{ROOT}/delta_continual_alpha_1.0_mom_0.9",
     f"{ROOT}/plugplay/delta_adagate_b1", ""),
    ("SAR  (ICLR'23)",
     f"{ROOT}/sar_continual_weather",
     f"{ROOT}/plugplay/sar_adagate_b1", "*"),
]


def load(run_dir):
    p = os.path.join(run_dir, "results_all_rounds.txt")
    if not os.path.exists(p):
        return np.array([])
    out = []
    for line in open(p):
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
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True, sharey=True)

    for ax, (title, p_no, p_gate, mark) in zip(axes.ravel(), PANELS):
        y_no, y_ga = load(p_no), load(p_gate)

        ax.axhline(NO_ADAPT, color=C_NOAD, ls=":", lw=2.0, zorder=2)
        if y_no.size:
            ax.plot(np.arange(1, y_no.size + 1), y_no, color=C_NOGATE, lw=2.0,
                    ls=(0, (5, 2)), zorder=3, solid_capstyle="round")
        if y_ga.size:
            ax.plot(np.arange(1, y_ga.size + 1), y_ga, color=C_GATE, lw=2.7,
                    zorder=4, solid_capstyle="round")

        ax.set_title(title + mark, fontsize=12.5, color=INK, pad=7)
        ax.grid(True, color=GRIDC, lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRIDC)
        ax.tick_params(colors=INK_MUTED, labelsize=10.5)

        # the headline number of each panel: what the last round became
        if y_no.size and y_ga.size:
            note = f"R{y_ga.size}:  {y_no[min(y_ga.size, y_no.size) - 1]:.1f}  →  {y_ga[-1]:.1f}"
            if y_ga.size < 150:
                note += f"\n(gated run still at round {y_ga.size}/150)"
            ax.text(0.035, 0.06, note, transform=ax.transAxes, fontsize=11.5,
                    color=C_GATE, fontweight="bold", va="bottom", ha="left")

    axes[0, 0].set_xlim(1, 150)
    axes[0, 0].set_ylim(0, 36)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean mIoU", fontsize=12, color=INK_MUTED)
    for ax in axes[1, :]:
        ax.set_xlabel("continual round", fontsize=12, color=INK_MUTED)

    handles = [
        plt.Line2D([], [], color=C_NOGATE, lw=2.0, ls=(0, (5, 2)), label="published method (no gate)"),
        plt.Line2D([], [], color=C_GATE, lw=2.7, label="+ AdaGate (ours)"),
        plt.Line2D([], [], color=C_NOAD, lw=2.0, ls=":", label="no adaptation (23.34)"),
    ]
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=11.5,
               ncol=3, handlelength=2.4, columnspacing=2.2, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.088))

    fig.suptitle("One gate, four published objectives: the base method decides how high, "
                 "the gate decides whether it falls",
                 fontsize=14.5, color=INK, y=0.975)
    fig.text(0.5, 0.048,
             "ACDC · 150 continual rounds · batch = 1 · all 100 visual LayerNorm params trainable "
             "(matching each published baseline) · LR 1e-5 · seed 0",
             fontsize=9.5, color=INK_MUTED, ha="center")
    fig.text(0.5, 0.020,
             "* SAR replaces rather than adds: SAR's own hard recovery (Filter C) is disabled so "
             "AdaGate's graded restore takes its place.",
             fontsize=9.5, color=INK_MUTED, ha="center")
    fig.subplots_adjust(left=0.068, right=0.985, top=0.895, bottom=0.185,
                        wspace=0.09, hspace=0.20)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{SAVE}/acdc_plugplay.{ext}", dpi=200, facecolor="white")
    print(f"wrote {SAVE}/acdc_plugplay.{{png,pdf,svg}}")

    print(f"\n{'base objective':28s} {'no gate':>10s} {'+AdaGate':>10s} {'delta':>8s}  rounds")
    for title, p_no, p_gate, _ in PANELS:
        y_no, y_ga = load(p_no), load(p_gate)
        if y_no.size and y_ga.size:
            a, b = y_no[-1], y_ga[-1]
            print(f"{title:28s} {a:10.2f} {b:10.2f} {b - a:+8.2f}  {y_ga.size}/150")
        else:
            print(f"{title:28s} {'--':>10s} {'--':>10s} {'--':>8s}  "
                  f"{y_ga.size if y_ga.size else 0}/150")


if __name__ == "__main__":
    main()
