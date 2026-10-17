"""Figure 6(a), standalone: stochastic restoration is necessary (ACDC, rho sweep trajectory)."""
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

A = "ACDCDataset/adagate_hgate"
# (label, dir, color, linestyle, linewidth, zorder) -- the three nonzero-rho
# curves sit almost exactly on top of each other (that IS the finding: CAGA
# isn't sensitive to rho), so they need different dash patterns and to be
# drawn thin-on-top, or the thick default line completely occludes the others.
RHO = [
    ("0", A + "_abl_baserst_0", "#c0392b", '-', 2.6, 4),
    ("0.01 (default)", A, "#1f4e9c", '-', 2.2, 3),
    ("0.005", A + "_abl_baserst_0.005", "#e8b13c", (0, (5, 2)), 1.6, 6),
    ("0.02", A + "_abl_baserst_0.02", "#3f9e9a", (0, (1, 1.3)), 1.8, 5),
]

fig, ax = plt.subplots(figsize=(3.2, 2.7))
for lab, d, c, ls, lw, z in RHO:
    R, M = load(d)
    hero = 'default' in lab
    ax.plot(R, M, color=c, ls=ls, lw=lw, zorder=z,
            label=fr"$\rho={lab.split(' ')[0]}$" + (" (default)" if hero else ""))

ax.set_xlim(0, 150); ax.set_ylim(5, 35)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.set_xlabel('Round', fontsize=FS_LABEL, labelpad=2); ax.set_ylabel('mIoU (%)', fontsize=FS_LABEL, labelpad=2)
ax.tick_params(labelsize=FS_TICK, pad=2)
ax.grid(True, color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')
ax.legend(loc='lower center', bbox_to_anchor=(0.44, 1.02), ncol=2, frameon=False, fontsize=FS_LEG,
          handlelength=1.4, handletextpad=0.4, labelspacing=0.3, columnspacing=1.0, borderaxespad=0.1)

fig.tight_layout(pad=0.5)
out = ".figs_additional_experiments/fig6a_stochastic_restoration"
fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
print("saved", out, fig.get_size_inches())
