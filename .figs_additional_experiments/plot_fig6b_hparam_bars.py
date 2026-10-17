"""Figure 6(b), standalone: hyperparameter sensitivity as four small bar charts
(trend threshold, anchor h ratio, monitor interval, lr), 2x2 grid. R150 mIoU,
y-axis zoomed to the narrow band the runs actually occupy (same framing as the
earlier dot-plot version) -- called out explicitly in the caption/axis label."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})
FS_TICK, FS_LABEL, FS_TITLE, FS_VAL = 12, 14, 14, 10
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")

PANELS = [
    ("trend threshold", [("0.25", 30.97), ("0.5", 30.95), ("1.0", 31.10)], 1),
    ("anchor $h$ ratio", [("0.90", 30.96), ("0.95", 30.95), ("0.98", 30.80)], 1),
    ("monitor interval", [("25", 31.02), ("50", 30.95), ("100", 31.23)], 1),
    ("lr", [("2.5e-6", 30.92), ("5e-6", 30.95), ("1e-5", 31.03)], 1),
]
YLO, YHI = 20, 37
DEFAULT_C, OTHER_C = '#1f4e9c', '#aecad9'

fig, axes = plt.subplots(2, 2, figsize=(4.0, 3.3))
for ax, (name, pts, default_idx) in zip(axes.flat, PANELS):
    xs = range(len(pts))
    colors = [DEFAULT_C if i == default_idx else OTHER_C for i in xs]
    vals = [v for _, v in pts]
    bars = ax.bar(xs, [v - YLO for v in vals], bottom=YLO, width=0.55, color=colors,
                   edgecolor='#1f4e9c', linewidth=0.6, zorder=3)
    for i, v in zip(xs, vals):
        ax.text(i, v + 0.6, f"{v:.2f}", ha='center', va='bottom', fontsize=FS_VAL, color='#333333')
    ax.set_xticks(list(xs)); ax.set_xticklabels([lab for lab, _ in pts], fontsize=FS_TICK)
    ax.set_xlim(-0.65, len(pts) - 0.35)
    ax.set_ylim(YLO, YHI)
    ax.set_yticks([20, 25, 30, 35])
    ax.tick_params(axis='y', labelsize=FS_TICK, pad=2.5)
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_title(name, fontsize=FS_TITLE, pad=12)
    ax.grid(axis='y', color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')

fig.text(0.01, 0.55, 'R150 mIoU (%)', fontsize=FS_LABEL, rotation=90, va='center')
fig.tight_layout(rect=(0.06, 0, 1, 1), pad=0.6, h_pad=1.3, w_pad=1.6)

out = ".figs_additional_experiments/fig6b_hparam_sensitivity_bars"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
print("saved", out, fig.get_size_inches())
