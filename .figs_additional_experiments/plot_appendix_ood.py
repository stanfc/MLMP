"""Appendix figure: does the adaptation gain transfer to unseen corruptions?
V20 5corr sub100, OOD-probe run: in-domain (trained) trajectory vs held-out
(5 different ImageNet-C corruptions, frozen-weight eval every round, never
adapted on) trajectory, both against their own No-Adapt reference.
Sized for a half-column (square) slot: no title, extra bottom margin."""
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

def load(f):
    R, M = [], []
    for l in open(f, errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            p = l.strip().split(','); R.append(int(p[0].split()[1])); M.append(float(p[-1]))
    return R, M

D = "PascalVOC20Dataset/adagate_hgate_5corr_sub100_oodprobe"
Ri, Mi = load(f"save/{D}/results_all_rounds.txt")
Ro, Mo = load(f"save/{D}/results_ood.txt")
na_line = next(l for l in open("save/PascalVOC20Dataset/no_adapt_sub100_15corr/results_all_rounds.txt") if l.startswith("Round 01"))
na_vals = [float(x) for x in na_line.strip().split(',')[1:6]]
na_ood = sum(na_vals) / len(na_vals)

fig, ax = plt.subplots(figsize=(3.2, 2.7))
l1, = ax.plot(Ri, Mi, color='#1f4e9c', lw=1.6, zorder=5, label='In-domain')
l2, = ax.plot(Ro, Mo, color='#cf4f3e', lw=1.6, zorder=4, label='Held-out')
l3 = ax.axhline(68.51, color='#1f4e9c', lw=0.7, ls=(0, (4, 2)), alpha=0.6, zorder=1, label='No-Adapt (ID)')
l4 = ax.axhline(na_ood, color='#cf4f3e', lw=0.7, ls=(0, (4, 2)), alpha=0.6, zorder=1, label='No-Adapt (Held-out)')

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

out = ".figs_additional_experiments/appendix_ood_generalization"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
print("saved", out, fig.get_size_inches())
