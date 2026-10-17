"""fig10: SegFormer-B5 (ACDC, batch 1, 150R), single wide panel, y 40-52. TRIBE (35-37) is off this scale and omitted."""
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
SEG = "/home/stanfc/TTA-on-OVSS/DeYO/output/acdc_seg_b5"
def seg(m):
    R, M = [], []
    for r in csv.DictReader(open(f"{SEG}/{m}/results_all_rounds.csv")):
        R.append(int(r['round'])); M.append(float(r['mean_miou']))
    return R, M
STABLE = {'SPA': ('spa', '#2a78a8', (0, (6, 2)), 1.6), 'MGP': ('mgp', '#cf4f3e', (0, (1, 1.4)), 1.9),
          'TENT': ('tent', '#e8b13c', (0, (3, 1.5, 1, 1.5)), 1.6), 'SAR': ('sar', '#7d5ba6', (0, (8, 2, 1, 2)), 1.6),
          'DAF': ('daf', '#3f9e9a', (0, (2, 2)), 1.6)}
DEGRADE = {'DeYO': ('deyo', '#e37a35'), 'COME': ('come', '#8c6d46'), 'ViDA': ('vida', '#8e2f2f'), 'PTTA': ('ptta', '#e07fa8')}
LW = 1.8   # every curve, and the legend handles, share one line width
fig, ax = plt.subplots(figsize=(5.5, 2.6))
h = {}
R, M = seg('deyo_adagate_hgate'); h['CAGA'], = ax.plot(R, M, color='#3a78c4', lw=LW, alpha=1.0, zorder=9, solid_capstyle='round')
R, M = seg('no_adapt'); h['No-Adapt'], = ax.plot(R, M, color='#555555', lw=LW, ls=(0, (4, 2)), zorder=7)
for lab, (m, c, ls, lw) in STABLE.items():
    R, M = seg(m); h[lab], = ax.plot(R, M, color=c, ls=ls, lw=LW, zorder=6)
for lab, (m, c) in DEGRADE.items():
    R, M = seg(m); h[lab], = ax.plot(R, M, color=c, lw=LW, zorder=4)
ax.set_xlim(0, 150); ax.set_ylim(40, 52)
ax.xaxis.set_major_locator(MultipleLocator(50)); ax.yaxis.set_major_locator(MultipleLocator(5))
ax.set_xlabel('Round', fontsize=13, labelpad=2); ax.set_ylabel('mIoU (%)', fontsize=13, labelpad=2)
ax.tick_params(labelsize=12, pad=2); ax.grid(True, color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')
order = ['CAGA', 'No-Adapt', 'SPA', 'MGP', 'TENT', 'SAR', 'DAF', 'DeYO', 'COME', 'ViDA', 'PTTA']
ax.legend([h[k] for k in order], order, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False,
          fontsize=10.5, handlelength=2.0, handletextpad=0.4, labelspacing=0.3, columnspacing=0.9)
fig.tight_layout(pad=0.5)
out = ".figs_additional_experiments/fig10_segformer_all_methods"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
