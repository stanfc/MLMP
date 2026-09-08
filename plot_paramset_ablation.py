"""Ablation: does freezing the top transformer blocks matter?

`--top_block_exclude 6` freezes ln_post + resblocks 18-23, so only 74 of the 100
visual LayerNorm params adapt. It is the default of every AdaGate-family run in
this repo (241 of them) and had never been ablated, while every published
baseline trains all 100.

2x2 small multiples. Rows = base objective, columns = what you measure:

    row 1  Ours (DeYO+MLMP+AdaGate, LR 5e-6)   -- 74 vs 100 params
    row 2  TENT+AdaGate       (LR 1e-5)        -- 74 vs 100 params
    col 1  mean mIoU          -- the outcome
    col 2  H_margin           -- the internal state that explains the outcome

The point of the figure: the same intervention lands in opposite places. For Ours
the parameter set is irrelevant (31.92 vs 31.94) even though the model drifts 3x
harder at 100 params -- the gate absorbs it. For TENT the same jump costs 5.7 mIoU
because its marginal collapses (-55%) beyond what the gate can hold.

H_margin is logged per gate window (every `monitor_interval` batches); windows are
mapped to rounds by `ceil(total_batches / 406)` (ACDC = 406 img/round at batch 1)
and averaged within each round.

Run:  python plot_paramset_ablation.py   # -> figures/ablation/
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE = "figures/ablation"
R = "save/ACDCDataset"
BPR = 406                      # ACDC images per round at batch 1
NO_ADAPT = 23.34

C_74, C_100, C_NOAD = "#d62728", "#0072B2", "#7b3f6b"
INK, INK_MUTED, GRIDC = "#1a1a1a", "#5c5c5c", "#d9d9d9"

ROWS = [
    ("Ours  (DeYO+MLMP + AdaGate, LR 5e-6)",
     f"{R}/batch_ablation/gradnorm_scaled_b1",     # 74 params
     f"{R}/ablation/adagate_allln_b1",             # 100 params
     (28.5, 33.0)),
    ("TENT + AdaGate  (LR 1e-5)",
     f"{R}/plugplay/tent_adagate_gate_b1",         # 74 params
     f"{R}/plugplay/tent_adagate_b1",              # 100 params
     (18.0, 32.0)),
]


def miou(run_dir):
    out = []
    for line in open(os.path.join(run_dir, "results_all_rounds.txt")):
        line = line.strip()
        if not line.startswith("Round ") or "," not in line:
            continue
        f = [q.strip() for q in line.split(",")]
        try:
            out.append((int(f[0].split()[1]), float(f[-1])))
        except (ValueError, IndexError):
            continue
    out.sort()
    return np.array([v for _, v in out])


def hmargin_per_round(run_dir, n_rounds):
    """gate_log windows -> per-round mean H_margin."""
    rows = list(csv.DictReader(open(os.path.join(run_dir, "gate_log.csv"))))
    tb = np.array([float(r["total_batches"]) for r in rows])
    hm = np.array([float(r["h_margin"]) for r in rows])
    idx = np.ceil(tb / BPR).astype(int)
    out = np.full(n_rounds, np.nan)
    for r in range(1, n_rounds + 1):
        m = idx == r
        if m.any():
            out[r - 1] = hm[m].mean()
    return out


def style(ax, ylab, title):
    ax.set_xlim(1, 150)
    ax.set_title(title, fontsize=12, color=INK, pad=7)
    ax.set_ylabel(ylab, fontsize=11.5, color=INK_MUTED)
    ax.grid(True, color=GRIDC, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_MUTED, labelsize=10)


def main():
    os.makedirs(SAVE, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0))

    for row, (name, p74, p100, ylim) in enumerate(ROWS):
        y74, y100 = miou(p74), miou(p100)
        h74 = hmargin_per_round(p74, y74.size)
        h100 = hmargin_per_round(p100, y100.size)

        # --- column 1: the outcome ---
        ax = axes[row, 0]
        if ylim[0] < NO_ADAPT:
            ax.axhline(NO_ADAPT, color=C_NOAD, ls=":", lw=2.0, zorder=2)
        ax.plot(np.arange(1, y74.size + 1), y74, color=C_74, lw=2.6, zorder=4)
        ax.plot(np.arange(1, y100.size + 1), y100, color=C_100, lw=2.0,
                ls=(0, (5, 2)), zorder=3)
        ax.set_ylim(*ylim)
        style(ax, "mean mIoU", f"{name}  —  outcome")
        d = y100[-1] - y74[-1]
        ax.text(0.035, 0.06,
                f"R150   74p {y74[-1]:.2f}   ·   100p {y100[-1]:.2f}   ({d:+.2f})",
                transform=ax.transAxes, fontsize=11, color=INK,
                fontweight="bold", va="bottom", ha="left")

        # --- column 2: the internal state ---
        ax = axes[row, 1]
        ax.plot(np.arange(1, h74.size + 1), h74, color=C_74, lw=2.6, zorder=4)
        ax.plot(np.arange(1, h100.size + 1), h100, color=C_100, lw=2.0,
                ls=(0, (5, 2)), zorder=3)
        ax.set_ylim(0.8, 2.6)
        style(ax, "H_margin (class diversity)", f"{name}  —  internal state")
        f74 = 100 * (1 - np.nanmin(h74[-10:]) / h74[0])
        f100 = 100 * (1 - np.nanmin(h100[-10:]) / h100[0])
        ax.text(0.035, 0.06,
                f"drop from start   74p {f74:.1f}%   ·   100p {f100:.1f}%",
                transform=ax.transAxes, fontsize=11, color=INK,
                fontweight="bold", va="bottom", ha="left")

    for ax in axes[1, :]:
        ax.set_xlabel("continual round", fontsize=11.5, color=INK_MUTED)

    handles = [
        plt.Line2D([], [], color=C_74, lw=2.6, label="74 params  (top_block_exclude = 6)"),
        plt.Line2D([], [], color=C_100, lw=2.0, ls=(0, (5, 2)),
                   label="100 params  (top_block_exclude = 0, all LayerNorms)"),
        plt.Line2D([], [], color=C_NOAD, lw=2.0, ls=":", label="no adaptation (23.34)"),
    ]
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=11,
               ncol=3, handlelength=2.4, columnspacing=2.2, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.008))

    fig.suptitle("Freezing the top blocks is irrelevant for Ours and load-bearing for TENT",
                 fontsize=14.5, color=INK, y=0.975)
    fig.subplots_adjust(left=0.062, right=0.985, top=0.885, bottom=0.125,
                        wspace=0.17, hspace=0.33)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{SAVE}/acdc_paramset_ablation.{ext}", dpi=200, facecolor="white")
    print(f"wrote {SAVE}/acdc_paramset_ablation.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
