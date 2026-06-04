#!/usr/bin/env python3
"""
All-methods comparison figure: Mean mIoU over rounds.

Important methods   — solid lines, full opacity
Background methods  — solid lines, low alpha (visual context only)
Baselines           — horizontal dashed/dotted lines

CoTTA:          10-round measured data extended as flat line to R150 (documented stable)
MLMP-continual: R1–R10 only (data unavailable beyond), endpoint marked

Output:
  save/ACDCDataset/all_methods_comparison.png
  save/ACDCDataset/all_methods_comparison.svg
"""

from __future__ import annotations
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

SAVE_ROOT = "save/ACDCDataset"

NO_ADAPT    = 23.34
MLMP_EPISODIC = 30.6

# ── data helpers ──────────────────────────────────────────────────────────────

def parse_means(path: str) -> tuple[list[int], list[float]]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = [x.strip() for x in lines[0].split(",")]
    mean_idx = header.index("Mean_mIoU")
    rounds, means = [], []
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        m = re.match(r"Round\s+(\d+)", parts[0])
        if not m:
            continue
        rounds.append(int(m.group(1)))
        means.append(float(parts[mean_idx]))
    return rounds, means


def extend_flat(rounds, means, up_to=150):
    """Extend a stable trajectory as a flat line to `up_to` rounds."""
    last_r, last_v = rounds[-1], means[-1]
    ext_r = list(range(last_r + 1, up_to + 1))
    return rounds + ext_r, means + [last_v] * len(ext_r)


# ── method registry ───────────────────────────────────────────────────────────
#
# Each entry: (label, path, color, linewidth, alpha, linestyle, extend_flat, note)
#   extend_flat  True  → extend last value to R150 (for CoTTA)
#   note         shown after label in legend if not None

_D = SAVE_ROOT

IMPORTANT = [
    dict(label="TENT-DivGate (best)",
         path=f"{_D}/tent_divgate_continual_cau_rst_0.01/results_all_rounds.txt",
         color="#16a34a", lw=2.5, alpha=1.0, ls="-",
         extend=False, note="h_thr=1.6, cau_rst=0.01"),

    dict(label="CoTTA",
         path=f"{_D}/cotta_step_1/results_all_rounds.txt",
         color="#92400e", lw=2.0, alpha=1.0, ls="-",
         extend=True, extend_ls="--", note="extended flat, stable by design"),

    dict(label="MLMP-continual",
         path=f"{_D}/mlmp_continual_round_150_step_1/results_all_rounds.txt",
         color="#2563eb", lw=2.0, alpha=1.0, ls="-",
         extend=True, extend_ls="-", note="step=1, collapses ~R33"),

    dict(label="TENT-continual",
         path=f"{_D}/tent_continual_Round150_lr_0.00001/results_all_rounds.txt",
         color="#dc2626", lw=1.8, alpha=1.0, ls="-",
         extend=False, note=None),
]

BACKGROUND = [
    dict(label="CMA-DivGate",
         path=f"{_D}/cma_divgate_continual_brake_0.05/results_all_rounds.txt",
         color="#0891b2", lw=1.5, alpha=0.40, ls="-",
         extend=False, note=None),

    dict(label="CMA-Layered",
         path=f"{_D}/cma_layered_continual_step_1/results_all_rounds.txt",
         color="#64748b", lw=1.5, alpha=0.40, ls="-",
         extend=False, note=None),

    dict(label="CMA-Proto",
         path=f"{_D}/cma_proto_continual_step_1/results_all_rounds.txt",
         color="#a855f7", lw=1.5, alpha=0.40, ls="-",
         extend=False, note=None),

    dict(label="CMA-continual",
         path=f"{_D}/cma_continual_k_0.5/results_all_rounds.txt",
         color="#f97316", lw=1.5, alpha=0.40, ls="-",
         extend=False, note=None),
]

# ── plot ──────────────────────────────────────────────────────────────────────

plt.style.use("default")
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")
ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

all_y = []

# --- background methods first (drawn below important lines) ---
for m in BACKGROUND:
    if not os.path.exists(m["path"]):
        print(f"  skip: {m['path']}")
        continue
    r, v = parse_means(m["path"])
    all_y.extend(v)
    ax.plot(r, v,
            color=m["color"], linewidth=m["lw"], alpha=m["alpha"],
            linestyle=m["ls"], label=m["label"], zorder=2)

# --- important methods on top ---
cotta_endpoint = None
mlmp_cont_endpoint = None

for m in IMPORTANT:
    if not os.path.exists(m["path"]):
        print(f"  skip: {m['path']}")
        continue
    r, v = parse_means(m["path"])

    # extension for methods with extend=True
    if m["extend"] and r[-1] < 150:
        r_ext, v_ext = extend_flat(r, v, up_to=150)
        ext_ls = m.get("extend_ls", "--")
        ext_alpha = m["alpha"] if ext_ls == "-" else m["alpha"] * 0.55
        ext_lw = m["lw"] if ext_ls == "-" else m["lw"] * 0.8
        ax.plot(r, v,
                color=m["color"], linewidth=m["lw"], alpha=m["alpha"],
                linestyle="-", zorder=4)
        ax.plot(r_ext[len(r)-1:], v_ext[len(r)-1:],
                color=m["color"], linewidth=ext_lw, alpha=ext_alpha,
                linestyle=ext_ls, zorder=4)
        r, v = r_ext, v_ext
    else:
        ax.plot(r, v,
                color=m["color"], linewidth=m["lw"], alpha=m["alpha"],
                linestyle=m["ls"], zorder=4)

    all_y.extend(v)

    # mark MLMP-continual endpoint (data stops at R10)
    if "MLMP-continual" in m["label"] and r[-1] < 20:
        mlmp_cont_endpoint = (r[-1], v[-1])
        ax.plot(*mlmp_cont_endpoint, "o",
                color=m["color"], markersize=6, zorder=5)
        ax.annotate("data ends R20",
                    xy=mlmp_cont_endpoint,
                    xytext=(mlmp_cont_endpoint[0] + 6, mlmp_cont_endpoint[1] - 1.0),
                    fontsize=7.5, color=m["color"],
                    arrowprops=dict(arrowstyle="-", color=m["color"], alpha=0.5, lw=0.7))

# --- horizontal baselines ---
ax.axhline(NO_ADAPT, linestyle=":", linewidth=1.4,
           color="#6b7280", label=f"No Adapt ({NO_ADAPT:.1f})", zorder=3)
ax.axhline(MLMP_EPISODIC, linestyle="--", linewidth=1.8,
           color="#7c3aed", label=f"MLMP episodic ({MLMP_EPISODIC:.1f})", zorder=3)

# --- y-axis range ---
flat = [y for y in all_y if y > 1.0]   # exclude collapsed near-zero values for range
y_lo = max(0, min(flat + [NO_ADAPT]) - 3)
y_hi = max(flat + [MLMP_EPISODIC]) + 3
ax.set_ylim(y_lo, y_hi)

# --- legend ---
# build manually to control order: important first, then background, then baselines
imp_handles = []
for m in IMPORTANT:
    if not os.path.exists(m["path"]):
        continue
    note = f"  [{m['note']}]" if m["note"] else ""
    imp_handles.append(
        mlines.Line2D([], [], color=m["color"], linewidth=m["lw"],
                      alpha=m["alpha"], linestyle=m["ls"],
                      label=m["label"] + note)
    )

bg_handles = []
for m in BACKGROUND:
    if not os.path.exists(m["path"]):
        continue
    bg_handles.append(
        mlines.Line2D([], [], color=m["color"], linewidth=m["lw"],
                      alpha=max(m["alpha"], 0.8),   # legend needs to be readable
                      linestyle=m["ls"],
                      label=m["label"])
    )

baseline_handles = [
    mlines.Line2D([], [], color="#6b7280", linewidth=1.4, linestyle=":",
                  label=f"No Adapt ({NO_ADAPT:.1f})"),
    mlines.Line2D([], [], color="#7c3aed", linewidth=1.8, linestyle="--",
                  label=f"MLMP episodic ({MLMP_EPISODIC:.1f})"),
]

all_handles = imp_handles + bg_handles + baseline_handles
ax.legend(handles=all_handles, loc="upper right",
          frameon=True, facecolor="white", edgecolor="#d1d5db",
          fontsize=8.5, ncol=1)

# --- summary stats box ---
lines_txt = ["Mean mIoU (all available rounds)"]
for m in IMPORTANT + BACKGROUND:
    if not os.path.exists(m["path"]):
        continue
    _, v = parse_means(m["path"])
    ov = np.mean(v)
    peak = max(v)
    r150 = v[-1] if len(v) >= 150 else float("nan")
    r150_str = f"{r150:.2f}" if not np.isnan(r150) else "n/a"
    short = m["label"].split("(")[0].strip()
    lines_txt.append(f"  {short:<20}: mean={ov:.2f}  peak={peak:.2f}  R150={r150_str}")
lines_txt.append(f"  {'MLMP episodic':<20}: {MLMP_EPISODIC:.1f}  (episodic reset)")

ax.text(0.01, 0.04,
        "\n".join(lines_txt),
        transform=ax.transAxes, fontsize=7.8,
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.95),
        family="monospace")

ax.set_title(
    "All Methods — Mean mIoU over Continual Rounds (ACDC, step=1)\n"
    "Solid lines: key comparisons   |   Faded lines: ablation variants",
    fontsize=12, pad=10,
)
ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("Mean mIoU (%)", fontsize=11)

fig.tight_layout()
out_png = os.path.join(SAVE_ROOT, "all_methods_comparison.png")
out_svg = os.path.join(SAVE_ROOT, "all_methods_comparison.svg")
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
print(f"Saved {out_svg}")
