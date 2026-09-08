"""Backbone-transfer figures: does the AdaGate threshold transfer across the OVSS backbone?

The gate's trigger is unitless (robust z of the grad-norm slope vs its own recent
spread) and its restore depth is a rank (ECDF), so nothing in it is expressed in
units that belong to one backbone.  That predicts the SAME trend_thr should give
the SAME duty cycle on a different visual encoder.  Two transfer axes are already
published -- datasets (0.29/0.32/0.30) and objectives (0.292-0.310) -- and these
figures add the third.

Figures written to figures/backbone/:
  1. backbone_floor.*         matched zero-shot floors; which backbones are fair
                              adaptation targets and which are not (arm selection)
  2. backbone_trajectories.*  150 rounds, gate vs no-gate vs floor, one panel per
                              backbone
  3. backbone_transfer.*      firing rate on all three transfer axes
Tables written to figures/backbone/:
  backbone_table.md / backbone_table.tex

Palette: the repo's established trajectory colours (same as plot_plugplay.py) so
this figure sits beside the plug-and-play one.  Verified with the dataviz
six-checks validator: the chromatic trio #d62728/#2a78d6/#7b3f6b passes every
check (worst CVD dE 12.6, normal-vision floor 19.5); #8c8c8c is a deliberate
NEUTRAL for the no-gate reference series, not a categorical hue, so the chroma
floor does not apply to it -- its separations pass (CVD dE 12.3).

Partial runs are drawn as far as they got and flagged in the panel.

Run:  python plot_backbone.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "save/ACDCDataset"
ABL = f"{ROOT}/backbone_ablation"
SMOKE = f"{ROOT}/backbone_smoke"
OUT = "figures/backbone"

C_GATE, C_NOGATE, C_FLOOR, C_ALT = "#d62728", "#8c8c8c", "#7b3f6b", "#2a78d6"
INK, INK_MUTED, GRIDC = "#1a1a1a", "#5c5c5c", "#d9d9d9"

TOTAL_ROUNDS = 150

# panel title, no-gate dir, gated dir, matched no-adapt floor dir
PANELS = [
    ("NA-CLIP  ViT-L/14   (reference)", f"{ROOT}/batch_ablation/deyo_mlmp_b1",
     f"{ROOT}/batch_ablation/gradnorm_scaled_b1", f"{SMOKE}/noadapt_naclip-ViT-L_14"),
    ("ClearCLIP k-k  ViT-L/14", f"{ABL}/clearclip_L14_nogate",
     f"{ABL}/clearclip_L14_gate", f"{SMOKE}/noadapt_clearclip-ViT-L_14"),
    ("ClearCLIP q-q  ViT-L/14", f"{ABL}/clearqq_L14_nogate",
     f"{ABL}/clearqq_L14_gate", f"{SMOKE}/noadapt_clearclip_qq-ViT-L_14"),
    ("MaskCLIP  ViT-L/14", f"{ABL}/maskclip_L14_nogate",
     f"{ABL}/maskclip_L14_gate", f"{SMOKE}/noadapt_maskclip-ViT-L_14"),
    ("v-v attention  ViT-L/14", f"{ABL}/vvclip_L14_nogate",
     f"{ABL}/vvclip_L14_gate", f"{SMOKE}/noadapt_vvclip-ViT-L_14"),
    ("NA-CLIP Gaussian-only  ViT-L/14", f"{ABL}/nonly_L14_nogate",
     f"{ABL}/nonly_L14_gate", f"{SMOKE}/noadapt_naclip_nonly-ViT-L_14"),
    ("NA-CLIP  ViT-B/16", f"{ABL}/naclip_B16_nogate",
     f"{ABL}/naclip_B16_gate", f"{SMOKE}/noadapt_naclip-ViT-B_16"),
    ("ClearCLIP k-k  ViT-B/16", f"{ABL}/clearclip_B16_nogate",
     f"{ABL}/clearclip_B16_gate", f"{SMOKE}/noadapt_clearclip-ViT-B_16"),
    ("NA-CLIP  ViT-B/32", f"{ABL}/naclip_B32_nogate",
     f"{ABL}/naclip_B32_gate", f"{SMOKE}/noadapt_naclip-ViT-B_32"),
]

# a panel is drawn only once its GATED arm has produced at least one round, so
# the figure grows as the pool completes arms instead of showing empty cells
def _live_panels():
    live = [pn for pn in PANELS if os.path.exists(os.path.join(pn[2], "results_all_rounds.txt"))]
    return live or PANELS[:1]


# label, dir, state  ('arm' = used, 'out' = excluded, 'artifact' = protocol artifact)
FLOORS = [
    ("NA-CLIP      ViT-L/14", f"{SMOKE}/noadapt_naclip-ViT-L_14", "arm"),
    ("ClearCLIP    ViT-L/14", f"{SMOKE}/noadapt_clearclip-ViT-L_14", "arm"),
    ("NA-CLIP      ViT-B/16", f"{SMOKE}/noadapt_naclip-ViT-B_16", "arm"),
    ("ClearCLIP    ViT-B/16", f"{SMOKE}/noadapt_clearclip-ViT-B_16", "arm"),
    ("NA-CLIP      ViT-B/32", f"{SMOKE}/noadapt_naclip-ViT-B_32", "arm"),
    ("ClearCLIP qq ViT-L/14", f"{SMOKE}/noadapt_clearclip_qq-ViT-L_14", "arm"),
    ("NACLIP gauss ViT-L/14", f"{SMOKE}/noadapt_naclip_nonly-ViT-L_14", "arm"),
    ("v-v atten    ViT-L/14", f"{SMOKE}/noadapt_vvclip-ViT-L_14", "arm"),
    ("MaskCLIP     ViT-L/14", f"{SMOKE}/noadapt_maskclip-ViT-L_14", "arm"),
    ("SCLIP        ViT-B/16", f"{SMOKE}/noadapt_sclip-ViT-B_16", "arm"),
    ("SCLIP        ViT-L/14", f"{SMOKE}/noadapt_sclip-ViT-L_14", "out"),
    ("vanilla CLIP ViT-B/16", f"{SMOKE}/noadapt_clip-ViT-B_16", "out"),
    ("vanilla CLIP ViT-L/14", f"{SMOKE}/noadapt_clip-ViT-L_14", "out"),
    ("NA-CLIP ViT-L/14  (1 prompt)", f"{SMOKE}/noadapt_naclip-ViT-L_14_1prompt", "artifact"),
]

# published transfer evidence (EXPERIMENT_STATUS §19 and §20 T-1)
FIRING_DATASET = [("ACDC", 0.292), ("VOC20", 0.315), ("Cityscapes", 0.299)]
FIRING_OBJECTIVE = [("DELTA", 0.292), ("Ours", 0.298), ("SAR", 0.299),
                    ("TENT", 0.308), ("MLMP", 0.310)]


def load_miou(run_dir):
    """Mean-mIoU per round from results_all_rounds.txt (empty array if absent)."""
    p = os.path.join(run_dir, "results_all_rounds.txt")
    if not os.path.exists(p):
        return np.array([])
    out = []
    for line in open(p):
        parts = [q.strip() for q in line.strip().split(",")]
        if len(parts) < 2 or not parts[0].lower().startswith("round "):
            continue
        try:
            out.append((int(parts[0].split()[1]), float(parts[-1])))
        except (ValueError, IndexError):
            continue
    out.sort()
    return np.array([v for _, v in out])


def firing_rate(run_dir):
    """Fraction of gate windows in which a restore fired -- the canonical
    definition used by scripts/collect_adagate.py (rst > 0)."""
    p = os.path.join(run_dir, "gate_log.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    return sum(1 for r in rows if float(r["rst"]) > 0) / len(rows)


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRIDC, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def fig_floor():
    """Which backbones are fair adaptation targets?  Magnitude + selection state,
    so this is a sorted bar chart with a status colour, not a categorical one."""
    rows = [(lab, load_miou(d), st) for lab, d, st in FLOORS]
    rows = [(lab, y[0], st) for lab, y, st in rows if y.size]
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    _style(ax)
    ypos = np.arange(len(rows))
    colors = {"arm": C_GATE, "out": "#c8c8c8", "artifact": C_ALT}
    for i, (lab, val, st) in enumerate(rows):
        ax.barh(i, val, height=0.62, color=colors[st],
                edgecolor="white", linewidth=2, zorder=3)
        ax.text(val + 0.45, i, f"{val:.2f}", va="center", ha="left",
                fontsize=9.5, color=INK, fontweight="bold" if st == "arm" else "normal")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK,
                       fontfamily="monospace")
    ax.set_xlabel("zero-shot mIoU on ACDC, 7-prompt ensemble  (no adaptation)",
                  fontsize=10, color=INK_MUTED)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.16)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("arm", "out", "artifact")]
    ax.legend(handles,
              ["run as an ablation arm",
               "measured, not run as an arm",
               "protocol artifact -- see note"],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK_MUTED)

    ax.set_title("Which OVSS backbones can our method actually be tested on?",
                 fontsize=13, color=INK, pad=12, loc="left", fontweight="bold")
    fig.text(0.008, 0.015,
             "SCLIP@ViT-L/14 (15.51) and vanilla CLIP (3-5) are far below every other configuration: adapting from there measures the\n"
             "backbone's brokenness, not the gate, so neither is an arm. SCLIP is additionally incompatible with MLMP's multi-level\n"
             "UAML average (arch='vanilla' exposes the full residual stream, not the text-aligned attention branch).\n"
             "Note: every ADAPTED run in this project uses the 7-prompt ensemble (--prompt_dir prompts.yaml), but the historical\n"
             "no-adapt baseline of 23.34 was measured with a SINGLE prompt (--prompt_dir ''). Re-measured under the matched\n"
             "7-prompt protocol the NA-CLIP ViT-L/14 floor is 28.21, not 23.34. The 1-prompt bar is shown only to document this.",
             fontsize=8, color=INK_MUTED, va="bottom")
    fig.tight_layout(rect=[0, 0.19, 1, 1])
    _save(fig, "backbone_floor")


def fig_trajectories():
    panels = _live_panels()
    ncol = 2 if len(panels) <= 4 else 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 3.9 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    drew_partial = False
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    for ax, (title, p_no, p_ga, p_fl) in zip(axes.ravel(), panels):
        _style(ax)
        y_no, y_ga = load_miou(p_no), load_miou(p_ga)
        y_fl = load_miou(p_fl)
        floor = y_fl[0] if y_fl.size else None

        if floor is not None:
            ax.axhline(floor, color=C_FLOOR, ls=":", lw=2.0, zorder=2)
            ax.text(TOTAL_ROUNDS * 0.985, floor + 0.9, f"no-adapt {floor:.2f}",
                    ha="right", va="bottom", fontsize=8.5, color=C_FLOOR)
        if y_no.size:
            ax.plot(np.arange(1, len(y_no) + 1), y_no, color=C_NOGATE, ls="--",
                    lw=2.0, zorder=3, label="no gate")
        if y_ga.size:
            ax.plot(np.arange(1, len(y_ga) + 1), y_ga, color=C_GATE, ls="-",
                    lw=2.0, zorder=4, label="+ AdaGate (ours)")

        done = min(len(y_no) if y_no.size else 0, len(y_ga) if y_ga.size else 0)
        note = ""
        if 0 < done < TOTAL_ROUNDS:
            note = f"   (in progress: R{done}/{TOTAL_ROUNDS})"
            drew_partial = True
        elif done == 0:
            note = "   (not started)"
            drew_partial = True
        ax.set_title(title + note, fontsize=11, color=INK, loc="left", pad=7,
                     fontweight="bold")

        if y_ga.size and y_no.size:
            n = min(len(y_ga), len(y_no))
            ax.annotate(f"$\\Delta$ = {y_ga[n-1] - y_no[n-1]:+.2f} @R{n}",
                        xy=(0.975, 0.06), xycoords="axes fraction",
                        ha="right", fontsize=9.5, color=INK)

    for ax in axes.ravel()[:len(panels)]:
        if ax.get_subplotspec().is_last_row() or ax is axes.ravel()[len(panels) - 1]:
            ax.set_xlabel("continual round", fontsize=10, color=INK_MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean mIoU", fontsize=10, color=INK_MUTED)
    axes[0][0].set_xlim(1, TOTAL_ROUNDS)

    axes[0][0].legend(loc="lower left", frameon=False, fontsize=9.5,
                      labelcolor=INK_MUTED)
    fig.suptitle(f"The same gate, at the same unitless threshold, on {len(panels)} OVSS backbones",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.985,
                 fontweight="bold")
    sub = ("Nothing is re-tuned. top_block_exclude and UAML depth are held at the same FRACTION of the encoder "
           "(24 blocks -> 6 / 18 layers; 12 blocks -> 3 / 9).")
    if drew_partial:
        sub += "  Dashed panels marked 'in progress' are still running."
    fig.text(0.008, 0.944, sub, fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    _save(fig, "backbone_trajectories")


def fig_transfer():
    """One quantity (firing rate) on three independent transfer axes.

    Horizontal so every run carries its own row label -- with 11+ labelled runs a
    vertical dot plot collides the series names into the group names.
    """
    backbone = []
    for title, p_no, p_ga, _ in _live_panels():
        f = firing_rate(p_ga)
        if f is None:
            continue
        n = len(load_miou(p_ga))
        backbone.append((title.replace("   (reference)", " (ref)").strip(), f,
                         n >= TOTAL_ROUNDS, n))

    groups = [
        ("across DATASETS   (published, \u00a719)",
         [(n, v, True, None) for n, v in FIRING_DATASET]),
        ("across OBJECTIVES   (published, \u00a720 T-1)",
         [(n, v, True, None) for n, v in FIRING_OBJECTIVE]),
        ("across BACKBONES   (this campaign)", backbone),
    ]

    # the published band is the reference the new axis is read against; an
    # unfinished run must not be allowed to widen it
    pub = [v for _, g in groups[:2] for _, v, _, _ in g]

    rows = []
    for gi, (gname, items) in enumerate(groups):
        rows.append(("__header__", gname, None, None, None))
        if not items:
            rows.append(("__note__", "still running -- no gate windows logged yet",
                         None, None, None))
        for nm, v, done, n in items:
            rows.append(("row", nm, v, gi, (done, n)))

    fig, ax = plt.subplots(figsize=(10.9, 0.40 * len(rows) + 2.4))
    _style(ax)
    ax.axvspan(min(pub), max(pub), color=C_ALT, alpha=0.09, zorder=1)
    ax.axvline(min(pub), color=C_ALT, lw=1.0, alpha=0.5, zorder=2)
    ax.axvline(max(pub), color=C_ALT, lw=1.0, alpha=0.5, zorder=2)

    yticks, ylabels = [], []
    y = len(rows) - 1
    for kind, nm, v, gi, meta in rows:
        if kind == "__header__":
            ax.text(0.196, y, nm, fontsize=10.5, color=INK, fontweight="bold",
                    va="center", ha="left")
        elif kind == "__note__":
            ax.text(0.235, y, nm, fontsize=9, color=INK_MUTED, va="center",
                    ha="left", style="italic")
        else:
            done, n = meta
            col = C_GATE if gi == 2 else C_ALT
            ax.plot([v], [y], "o", ms=11, color=col if done else "white",
                    markeredgecolor=col, markeredgewidth=2.2, zorder=4)
            tag = f"{v:.3f}" + ("" if done else f"   (in progress, R{n})")
            ax.text(v + 0.004, y, tag, va="center", ha="left", fontsize=9.5,
                    color=INK if done else INK_MUTED)
            yticks.append(y); ylabels.append("   " + nm)
        y -= 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5, color=INK_MUTED)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(0.19, 0.40)
    ax.set_xlabel("gate firing rate  (fraction of gate windows in which a restore fired, rst > 0)",
                  fontsize=10, color=INK_MUTED)
    ax.grid(axis="y", visible=False)
    ax.text(np.mean([min(pub), max(pub)]), len(rows) - 0.55,
            f"published band  {min(pub):.3f}-{max(pub):.3f}  ({max(pub)/min(pub):.2f}\u00d7)",
            ha="center", va="top", fontsize=9, color=C_ALT, fontweight="bold")

    ax.set_title("One unitless threshold (trend_thr = 0.5), three independent transfer axes",
                 fontsize=13, color=INK, pad=26, loc="left", fontweight="bold")
    fig.text(0.008, 0.005,
             "Hollow markers are runs that have not reached R150; their firing rate is computed over very few gate windows and will move.\n"
             "The shaded band is the PUBLISHED range only (datasets + objectives) -- unfinished runs are never allowed to widen it.",
             fontsize=8, color=INK_MUTED, va="bottom")
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    _save(fig, "backbone_transfer")


def tables():
    md = ["| backbone | zero-shot floor | no gate R_last | + AdaGate R_last | Δ | firing rate |",
          "|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Backbone & No-adapt & No gate & +AdaGate & $\Delta$ & Firing \\", r"\midrule"]
    for title, p_no, p_ga, p_fl in _live_panels():
        fl, no, ga = load_miou(p_fl), load_miou(p_no), load_miou(p_ga)
        f = firing_rate(p_ga)
        n = min(len(no) if no.size else 0, len(ga) if ga.size else 0)
        name = title.replace("   (reference)", " (ref)").strip()
        if n == 0:
            md.append(f"| {name} | {fl[0]:.2f} | — | — | — | — |" if fl.size else
                      f"| {name} | — | — | — | — | — |")
            continue
        d = ga[n - 1] - no[n - 1]
        tag = "" if n == TOTAL_ROUNDS else f" (R{n})"
        md.append(f"| {name} | {fl[0]:.2f} | {no[n-1]:.2f} | {ga[n-1]:.2f} | "
                  f"{d:+.2f}{tag} | {f:.3f} |" if f is not None else
                  f"| {name} | {fl[0]:.2f} | {no[n-1]:.2f} | {ga[n-1]:.2f} | {d:+.2f}{tag} | — |")
        tex.append(f"{name} & {fl[0]:.2f} & {no[n-1]:.2f} & {ga[n-1]:.2f} & "
                   f"{d:+.2f} & {f:.3f} \\\\" if f is not None else
                   f"{name} & {fl[0]:.2f} & {no[n-1]:.2f} & {ga[n-1]:.2f} & {d:+.2f} & -- \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/backbone_table.md", "w").write("\n".join(md) + "\n")
    open(f"{OUT}/backbone_table.tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(md))


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=160, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT}/{name}.png / .svg")


if __name__ == "__main__":
    fig_floor()
    fig_trajectories()
    fig_transfer()
    tables()
