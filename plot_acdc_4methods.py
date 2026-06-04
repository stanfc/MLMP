#!/usr/bin/env python3
"""
ACDC 4-method comparison: No Adapt, MLMP-episodic, TENT-DivGate (best), SAR.

Round vs Mean mIoU (1 line per method). Episodic baselines drawn as
horizontal reference lines since they require per-sample reset and have
no round trajectory.

Output:
  save/ACDCDataset/acdc_4methods_comparison.png
  save/ACDCDataset/acdc_4methods_comparison.svg
"""

from __future__ import annotations
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SAVE_ROOT = "save/ACDCDataset"

NO_ADAPT      = 23.34
MLMP_EPISODIC = 30.6


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


METHODS = [
    dict(label="TENT-DivGate (best)",
         path=f"{SAVE_ROOT}/tent_divgate_continual_cau_rst_0.01/results_all_rounds.txt",
         color="#16a34a", lw=2.5, ls="-",
         note="h_thr=1.6, cau_rst=0.01"),
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
    ax.plot(r, v,
            color=m["color"], linewidth=m["lw"], linestyle=m["ls"],
            zorder=4)
    short = m["label"].split("(")[0].strip()
    summary_rows.append(
        (short, np.mean(v), max(v), v[-1] if len(v) >= 150 else float("nan"))
    )

ax.axhline(NO_ADAPT, linestyle=":", linewidth=1.6,
           color="#6b7280", zorder=3)
ax.axhline(MLMP_EPISODIC, linestyle="--", linewidth=1.8,
           color="#7c3aed", zorder=3)

# y-axis: leave headroom for both baselines and trajectories
flat = [y for y in all_y if y > 1.0]
y_lo = max(0, min(flat + [NO_ADAPT]) - 2)
y_hi = max(flat + [MLMP_EPISODIC]) + 2
ax.set_ylim(y_lo, y_hi)

# x-axis
max_r = max((max(parse_means(m["path"])[0]) for m in METHODS
             if os.path.exists(m["path"])), default=150)
ax.set_xlim(0, max_r + 2)

# legend
handles = []
for m in METHODS:
    if not os.path.exists(m["path"]):
        continue
    note = f"  [{m['note']}]" if m["note"] else ""
    handles.append(mlines.Line2D([], [], color=m["color"], linewidth=m["lw"],
                                  linestyle=m["ls"], label=m["label"] + note))
handles.append(mlines.Line2D([], [], color="#7c3aed", linewidth=1.8, linestyle="--",
                              label=f"MLMP episodic ({MLMP_EPISODIC:.1f})  [requires reset]"))
handles.append(mlines.Line2D([], [], color="#6b7280", linewidth=1.6, linestyle=":",
                              label=f"No Adapt ({NO_ADAPT:.2f})"))
ax.legend(handles=handles, loc="upper right",
          frameon=True, facecolor="white", edgecolor="#d1d5db",
          fontsize=9)

# summary stats box
lines_txt = ["Summary (all available rounds)"]
for short, mean, peak, r150 in summary_rows:
    r150_str = f"{r150:.2f}" if not np.isnan(r150) else "n/a"
    lines_txt.append(f"  {short:<22}: mean={mean:.2f}  peak={peak:.2f}  R150={r150_str}")
lines_txt.append(f"  {'MLMP episodic':<22}: {MLMP_EPISODIC:.2f}  (episodic reset)")
lines_txt.append(f"  {'No Adapt':<22}: {NO_ADAPT:.2f}")

ax.text(0.01, 0.04,
        "\n".join(lines_txt),
        transform=ax.transAxes, fontsize=8.2,
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#d1d5db", alpha=0.95),
        family="monospace")

ax.set_title("ACDC — Mean mIoU over Continual Rounds (step=1)",
             fontsize=13, pad=10)
ax.set_xlabel("Round", fontsize=11)
ax.set_ylabel("Mean mIoU (%)", fontsize=11)

fig.tight_layout()
out_png = os.path.join(SAVE_ROOT, "acdc_4methods_comparison.png")
out_svg = os.path.join(SAVE_ROOT, "acdc_4methods_comparison.svg")
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_png}")
print(f"Saved {out_svg}")
