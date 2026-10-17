"""Figure 7(a): fixed vs shuffled domain order, grouped bars (R150 mIoU), appendix wide ratio (5.5 x 2.2 in)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
FS_TICK, FS_LABEL, FS_LEG, FS_VAL = 9.5, 10.5, 9, 9
BLUE, GREY = '#3a78c4', '#c4c4c4'
def last(p):
    o = []
    for l in open(f"save/{p}/results_all_rounds.txt"):
        if l.startswith('Round '):
            try: o.append(float(l.split(',')[-1]))
            except ValueError: pass
    return o[-1]
G = ["ACDC", "VOC20", "Cityscapes"]
fixed = [last("ACDCDataset/adagate_hgate"), last("PascalVOC20Dataset/adagate_hgate_5corr_sub100"), last("CityscapesDataset/adagate_hgate_5corr_sub100")]
shuf = [last("ACDCDataset/adagate_hgate_shuffledomain"), last("PascalVOC20Dataset/adagate_hgate_shuffledomain_5corr_sub100"), last("CityscapesDataset/adagate_hgate_shuffledomain_5corr_sub100")]
fig, ax = plt.subplots(figsize=(5.5, 2.2)); w = 0.3
for j, (lab, col, vals) in enumerate([("Fixed order", GREY, fixed), ("Shuffled order", BLUE, shuf)]):
    xs = [i + (j - 0.5) * w for i in range(3)]
    ax.bar(xs, vals, width=w * 0.95, color=col, label=lab, zorder=3)
    for x, v in zip(xs, vals): ax.text(x, v + 1, f"{v:.2f}", ha='center', va='bottom', fontsize=FS_VAL, color='#333333')
ax.set_xticks(range(3)); ax.set_xticklabels(G); ax.set_ylim(0, 95); ax.yaxis.set_major_locator(MultipleLocator(25))
ax.set_ylabel('mIoU (%)', fontsize=FS_LABEL, labelpad=2); ax.tick_params(axis='y', labelsize=FS_TICK, pad=2); ax.tick_params(axis='x', labelsize=FS_TICK, length=0, pad=3)
ax.grid(axis='y', color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=FS_LEG, handlelength=1.4, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.1)
fig.tight_layout(pad=0.5); out = ".figs_additional_experiments/fig7a_domain_order_delta"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight'); print("saved", out, fixed, shuf)
