"""Appendix D: long-horizon trajectories on the 5 additional benchmarks (V21-C, P59-C, P60-C, COCO-Stuff-C, COCO-Object-C).
2x3 grid = 5 dataset panels + a legend cell. Main-figure palette; CAGA is the hero line."""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
def load(p):
    R, M = [], []
    for l in open(p, errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            q = l.strip().split(','); R.append(int(q[0].split()[1])); M.append(float(q[-1]))
    return R, M
def epi(p):
    v = []
    for l in list(open(p, errors='ignore'))[1:]:
        if not l.strip() or 'GPU' in l or 'Duration' in l or 'Mean' in l: continue
        try: v.append(float(l.split(',')[1].split('+/-')[0]))
        except (ValueError, IndexError): pass
    return sum(v) / len(v)
SETS = [("V21-C", "PascalVOC21Dataset"), ("P59-C", "PascalContext59Dataset"), ("P60-C", "PascalContext60Dataset"),
        ("COCO-Stuff-C", "COCOStuffDataset"), ("COCO-Object-C", "COCOObjectDataset")]
M = {"TENT": "tent_continual", "SAR": "sar_continual", "SPA": "spa_continual", "TRIBE": "tribe_continual", "DeYO": "deyo_continual",
     "ViDA": "vida_continual", "PTTA": "ptta_continual", "COME": "come_continual", "MGP": "mgp_deyo_mlmp_continual", "DAF": "daf_mlmp_continual"}
COL = {'CAGA': '#3a78c4', 'TRIBE': '#aecad9', 'SPA': '#2a78a8', 'SAR': '#7d5ba6', 'DAF': '#3f9e9a', 'TENT': '#e8b13c', 'COME': '#c2a17a',
       'PTTA': '#e07fa8', 'DeYO': '#e37a35', 'MGP': '#cf4f3e', 'ViDA': '#8e2f2f'}
fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.9), gridspec_kw=dict(wspace=0.42, hspace=0.55))
h = {}
for ax, (title, folder) in zip(axes.flat, SETS):
    root = f"save/{folder}"
    na = load(f"{root}/no_adapt_15corr_sub100/results_all_rounds.txt")[1][-1]
    ep = epi(f"{root}/mlmp_episodic_15corr_sub100/results.txt")
    h['No-Adapt'], = ax.plot([0, 150], [na] * 2, color='#a8a8a8', lw=1.0, ls=(0, (4, 2)), zorder=1)
    h['MLMP-episodic'], = ax.plot([0, 150], [ep] * 2, color='#5f5f5f', lw=1.0, ls=(0, (1.4, 1.4)), zorder=1)
    for lab, d in M.items():
        R, Y = load(f"{root}/{d}_15corr_sub100/results_all_rounds.txt")
        h[lab], = ax.plot(R, Y, color=COL[lab], lw=0.9, zorder=2)
    R, Y = load(f"{root}/adagate_hgate_15corr_sub100/results_all_rounds.txt")
    h['CAGA'], = ax.plot(R, Y, color=COL['CAGA'], lw=2.0, zorder=6)
    ax.set_xlim(0, 150); ax.set_ylim(0, None); ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.set_title(title, fontsize=9, pad=3); ax.tick_params(labelsize=7.5, pad=1.5)
    ax.grid(True, color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')
for ax in axes[1, :2]: ax.set_xlabel('Round', fontsize=8.5, labelpad=1)
axes[1, 1].set_xlabel('Round', fontsize=8.5, labelpad=1)
for ax in (axes[0, 0], axes[1, 0]): ax.set_ylabel('mIoU (%)', fontsize=8.5, labelpad=1)
axes[1, 2].axis('off')
order = ['CAGA', 'No-Adapt', 'MLMP-episodic', 'TENT', 'SAR', 'SPA', 'TRIBE', 'DeYO', 'ViDA', 'PTTA', 'COME', 'MGP', 'DAF']
axes[1, 2].legend([h[k] for k in order], order, loc='center', frameon=False, fontsize=7.5, ncol=1,
                  handlelength=2.0, labelspacing=0.28, borderaxespad=0)
out = ".figs_additional_experiments/appendixD_additional_benchmarks_5panel"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
print("saved", out)
