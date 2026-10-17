"""Appendix figure: resample-100-images-every-round (V20 5corr sub100).
Raw per-round mIoU (faint), 10-round moving average (bold), fixed-subset control
overlay, No-Adapt reference. Square half-column figure, no title, legend inside
the axes in the empty band below the No-Adapt line (mirrors appendix_ood_generalization)."""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})
FS_TICK, FS_LABEL, FS_LEG = 13, 15, 12

def load(d):
    R, M = [], []
    for l in open(f"save/{d}/results_all_rounds.txt", errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            p = l.strip().split(','); R.append(int(p[0].split()[1])); M.append(float(p[-1]))
    return R, M

def moving_avg(M, k=10):
    out = []
    for i in range(len(M)):
        lo = max(0, i - k + 1)
        out.append(sum(M[lo:i+1]) / (i - lo + 1))
    return out

V = "PascalVOC20Dataset/adagate_hgate_5corr_sub100"
Rr, Mr = load(V + "_resample")
Rc, Mc = load(V)
Mr_ma = moving_avg(Mr, 10)

fig, ax = plt.subplots(figsize=(3.2, 2.7))
l1, = ax.plot(Rr, Mr, color='#e37a35', lw=0.6, alpha=0.4, zorder=2, label='Resample raw')
l2, = ax.plot(Rr, Mr_ma, color='#e37a35', lw=1.6, zorder=4, label='Resample avg')
l3, = ax.plot(Rc, Mc, color='#1f4e9c', lw=1.6, zorder=5, label='Fixed order')
l4 = ax.axhline(68.51, color='#a8a8a8', lw=0.7, ls=(0, (4, 2)), zorder=1, label='No-Adapt')

ax.set_xlim(0, 150); ax.set_ylim(64, 78)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.set_xlabel('Round', fontsize=FS_LABEL, labelpad=2); ax.set_ylabel('mIoU (%)', fontsize=FS_LABEL, labelpad=2)
ax.tick_params(labelsize=FS_TICK, pad=2)
ax.grid(True, color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')
ax.legend(handles=[l1, l2, l3, l4], loc='lower center', bbox_to_anchor=(0.44, 1.02), ncol=2, frameon=False, fontsize=FS_LEG,
          handlelength=1.4, handletextpad=0.4, labelspacing=0.3, columnspacing=1.0, borderaxespad=0.1)

fig.tight_layout(pad=0.5)

out = ".figs_additional_experiments/appendix_resample_robustness"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
print("saved", out, fig.get_size_inches())
