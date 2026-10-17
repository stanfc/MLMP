"""Figure 7(b), standalone: abrupt domain shift (V20 5corr sub100), fixed-domain
control vs domain switch @R76."""
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
FS_TICK, FS_LABEL, FS_LEG = 9.5, 10.5, 9

def load(d):
    R, M = [], []
    for l in open(f"save/{d}/results_all_rounds.txt", errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            p = l.strip().split(','); R.append(int(p[0].split()[1])); M.append(float(p[-1]))
    return R, M

V = "PascalVOC20Dataset/adagate_hgate_5corr_sub100"
Rc, Mc = load(V)
Rs, Ms = load(V + "_phaseswitch")

fig, ax = plt.subplots(figsize=(5.5, 2.2))
ax.plot(Rc, Mc, color='#1f4e9c', lw=1.8, label='Fixed-domain control', zorder=4)
ax.plot(Rs, Ms, color='#c0392b', lw=1.8, label='Domain switch @R76', zorder=5)
# No-Adapt references: easy5 (R1-75, and the fixed control throughout) vs hard5 (R76-150 of the switch run)
ax.plot([0, 76], [69.60, 69.60], color='#a8a8a8', lw=0.8, ls=(0, (2, 1.5)), zorder=1, label='No-Adapt (easy5)')
ax.plot([76, 150], [64.48, 64.48], color='#e0a8a8', lw=0.8, ls=(0, (2, 1.5)), zorder=1, label='No-Adapt (hard5)')
ax.plot([76, 76], [69.60, 64.48], color='#c9a3a3', lw=0.8, ls=(0, (2, 1.5)), zorder=1)

ax.set_xlim(0, 150); ax.set_ylim(63, 78)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.set_xlabel('Round', fontsize=FS_LABEL, labelpad=2); ax.set_ylabel('mIoU (%)', fontsize=FS_LABEL, labelpad=2)
ax.tick_params(labelsize=FS_TICK, pad=2)
ax.grid(True, color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=FS_LEG,
          handlelength=1.8, handletextpad=0.4, columnspacing=1.2, labelspacing=0.3, borderaxespad=0.1)

fig.tight_layout(pad=0.5)
out = ".figs_additional_experiments/fig7b_abrupt_domain_shift"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
print("saved", out, fig.get_size_inches())
