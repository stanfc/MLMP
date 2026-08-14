"""Mechanism diagram (not experimental data): how far can SHALLOW restore reach
back, as a function of "windows since grad_norm's own minimum" (windows_since_min),
under the three cap modes actually still in play (fixed / uncapped / scaled).

  fixed (original flagship): cap = min(6, windows_since_min)        -- hard ceiling at 6
  uncapped (gradnorm_uncapped, A): cap = windows_since_min           -- no ceiling at all
  scaled (gradnorm_scaled, P1):    cap = windows_since_min * u       -- u = severity in [0,1],
                                    so it's bounded between 0 and the uncapped line, not a
                                    single curve -- shown as a shaded band + its average (u=0.5).

Output: figures/cap_mode_explainer.{png,svg}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 19, "axes.labelsize": 23, "axes.titlesize": 19,
                     "xtick.labelsize": 17, "ytick.labelsize": 17,
                     "axes.linewidth": 1.4, "font.family": "DejaVu Sans"})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

x = np.linspace(0, 100, 400)  # windows_since_min (time since grad_norm's own best point)

fixed_cap = np.minimum(6, x)
uncapped_cap = x
scaled_avg = 0.5 * x       # E[u] = 0.5
scaled_lo = 0.1 * x        # illustrative low-severity band edge
scaled_hi = 0.9 * x        # illustrative high-severity band edge

fig, ax = plt.subplots(figsize=(13, 9.5))

ax.fill_between(x, scaled_lo, scaled_hi, color="#1f77b4", alpha=0.15, zorder=1,
                label=None)
ax.plot(x, scaled_avg, color="#1f77b4", lw=2.8, zorder=4)
ax.plot(x, uncapped_cap, color="#ff7f0e", lw=2.8, zorder=3, ls="--")
ax.plot(x, fixed_cap, color="#7f7f7f", lw=3.2, zorder=5)

# annotate where fixed caps out
ax.annotate("hard ceiling at 6\n(never grows past this)",
            xy=(45, 6), xytext=(30, 28),
            fontsize=15, color="#555555", ha="center",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3))
ax.annotate("no ceiling --\nreach = full time\nsince best point",
            xy=(93, 93), xytext=(65, 75),
            fontsize=15, color="#a85200", ha="left",
            arrowprops=dict(arrowstyle="->", color="#a85200", lw=1.3))
ax.annotate("bounded by severity u in [0,1]:\nshaded band = possible range,\nline = average (u=0.5)",
            xy=(70, 35), xytext=(38, 52),
            fontsize=15, color="#1f4e79", ha="left",
            arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.3))

legend_handles = [
    Line2D([], [], color="#7f7f7f", lw=3.2, label="fixed (original flagship): cap = min(6, t)"),
    Line2D([], [], color="#ff7f0e", lw=2.8, ls="--", label="uncapped (A): cap = t"),
    Line2D([], [], color="#1f77b4", lw=2.8, label="scaled (P1): cap = t x u,  u = severity rank"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=15, framealpha=0.93)

ax.set_xlabel("t = windows since grad_norm's own best point (windows_since_min)")
ax.set_ylabel("SHALLOW restore reach ceiling (cap, in windows)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.25, lw=0.9)
ax.set_title("How far back can SHALLOW restore reach, as drift persists?",
             pad=12)

fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/cap_mode_explainer.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote cap_mode_explainer.png")
