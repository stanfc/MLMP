#!/usr/bin/env python3
"""
PascalVOC20 4-method comparison (weather subset, 5 corruptions):
  No Adapt, MLMP-episodic, TENT-DivGate (best), SAR-continual (post-fix).

Mirrors plot_acdc_4methods.py. Round vs Mean mIoU; episodic baselines as
horizontal reference lines.

Output:
  save/PascalVOC20Dataset/voc20_4methods_comparison.png
  save/PascalVOC20Dataset/voc20_4methods_comparison.svg
"""

from __future__ import annotations
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SAVE_ROOT = "save/PascalVOC20Dataset"


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


def parse_episodic_mean(path: str) -> float:
    """Average mIoU across condition rows in an episodic results.txt."""
    vals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("mIoU") or line.startswith("GPU") \
               or line.startswith("Total") or line.startswith("Mean Duration"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                vals.append(float(parts[1].split("+/-")[0].strip()))
            except ValueError:
                continue
    return float(np.mean(vals)) if vals else float("nan")


# No-Adapt baseline (mean over its rounds, near-constant)
NA_PATH = f"{SAVE_ROOT}/No_Adaptation_weather/results_all_rounds.txt"
no_adapt = float(np.mean(parse_means(NA_PATH)[1])) if os.path.exists(NA_PATH) else float("nan")

# MLMP-episodic baseline (per-condition results.txt)
EP_PATH = f"{SAVE_ROOT}/mlmp_episodic_weather/results.txt"
mlmp_episodic = parse_episodic_mean(EP_PATH) if os.path.exists(EP_PATH) else float("nan")

METHODS = [
    dict(label="TENT-DivGate (best)",
         path=f"{SAVE_ROOT}/tent_divgate_continual_threshold_1.6_weather/results_all_rounds.txt",
         color="#16a34a", lw=2.5, ls="-",
         note="h_thr=1.6"),
    dict(label="SAR-continual",
         path=f"{SAVE_ROOT}/sar_continual_weather/results_all_rounds.txt",
         color="#dc2626", lw=2.2, ls="-",
         note="sam_rho=0.05, recovery"),
]

plt.style.use("default")
fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

all_y = []
summary_rows = []

for m in METHODS:
    if not os.path.exists(m["path"]):
        print(f"  skip (missing): {m['path']}")
        continue
    r, v = parse_means(m["path"])
    all_y.extend(v)
    ax.plot(r, v, color=m["color"], linewidth=m["lw"],
            linestyle=m["ls"], zorder=4)
    last_r, last_v = r[-1], v[-1]
    ax.plot(last_r, last_v, "o", color=m["color"], markersize=5, zorder=5)
    short = m["label"].split("(")[0].strip()
    summary_rows.append((short, np.mean(v), max(v), last_r, last_v))

if not np.isnan(no_adapt):
    ax.axhline(no_adapt, linestyle=":", linewidth=1.6,
               color="#6b7280", zorder=3)
if not np.isnan(mlmp_episodic):
    ax.axhline(mlmp_episodic, linestyle="--", linewidth=1.8,
               color="#7c3aed", zorder=3)

flat = [y for y in all_y if y > 1.0]
baselines = [v for v in (no_adapt, mlmp_episodic) if not np.isnan(v)]
y_lo = max(0, min(flat + baselines) - 3)
y_hi = max(flat + baselines) + 3
if any(y < 1.0 for y in all_y):
    y_lo = -1.0
ax.set_ylim(y_lo, y_hi)

max_r = max((max(parse_means(m["path"])[0]) for m in METHODS
             if os.path.exists(m["path"])), default=150)
ax.set_xlim(0, max_r + 3)

handles = []
for m in METHODS:
    if not os.path.exists(m["path"]):
        continue
    note = f"  [{m['note']}]" if m["note"] else ""
    handles.append(mlines.Line2D([], [], color=m["color"], linewidth=m["lw"],
                                  linestyle=m["ls"], label=m["label"] + note))
if not np.isnan(mlmp_episodic):
    handles.append(mlines.Line2D([], [], color="#7c3aed", linewidth=1.8, linestyle="--",
                                  label=f"MLMP episodic ({mlmp_episodic:.2f})  [requires reset]"))
if not np.isnan(no_adapt):
    handles.append(mlines.Line2D([], [], color="#6b7280", linewidth=1.6, linestyle=":",
                                  label=f"No Adapt ({no_adapt:.2f})"))
ax.legend(handles=handles, loc="lower left",
          frameon=True, facecolor="white", edgecolor="#d1d5db", fontsize=9)

lines_txt = ["Summary (all available rounds)"]
for short, mean, peak, last_r, last_v in summary_rows:
    lines_txt.append(
        f"  {short:<22}: mean={mean:.2f}  peak={peak:.2f}  R{last_r}={last_v:.2f}"
    )
if not np.isnan(mlmp_episodic):
    lines_txt.append(f"  {'MLMP episodic':<22}: {mlmp_episodic:.2f}  (episodic reset)")
if not np.isnan(no_adapt):
    lines_txt.append(f"  {'No Adapt':<22}: {no_adapt:.2f}")

ax.text(0.01, 0.04, "\n".join(lines_txt),
        transform=ax.transAxes, fontsize=8.2,
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.95),
        family="monospace")

ax.set_title("PascalVOC20 — Mean mIoU over Continual Rounds (weather subset, step=1)",
             fontsize=13, pad=10)
ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("Mean mIoU (%)", fontsize=11)

fig.tight_layout()
out_png = os.path.join(SAVE_ROOT, "voc20_4methods_comparison.png")
out_svg = os.path.join(SAVE_ROOT, "voc20_4methods_comparison.svg")
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
print(f"Saved {out_svg}")
