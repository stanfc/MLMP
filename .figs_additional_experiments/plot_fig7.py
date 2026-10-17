"""Figure 7: (a) domain-order robustness (paired dots, one small panel per dataset,
independent zoomed x-range per panel since absolute mIoU scales differ),
(b) abrupt domain shift (V20 5corr sub100 trajectory, control vs switch@R76)."""
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
FS_TICK, FS_LABEL, FS_PANEL, FS_LEG = 7.5, 8.5, 9, 7.5

def load(d):
    R, M = [], []
    for l in open(f"save/{d}/results_all_rounds.txt", errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            p = l.strip().split(','); R.append(int(p[0].split()[1])); M.append(float(p[-1]))
    return R, M

W = 5.5
fig = plt.figure(figsize=(W, 2.5))
gsL = fig.add_gridspec(3, 2, left=0.135, right=0.485, top=0.76, bottom=0.17, hspace=0.9, width_ratios=[1, 1])
gsR = fig.add_gridspec(1, 1, left=0.60, right=0.985, top=0.86, bottom=0.17)

# ---------------- (a) domain-order robustness: 3 stacked mini dumbbells ----------------
DATA = [("ACDC", 30.95, 31.07), ("VOC20", 75.86, 75.78), ("Cityscapes", 23.92, 23.80)]
axes_a = []
for i, (name, fixed, shuf) in enumerate(DATA):
    ax = fig.add_subplot(gsL[i, :])
    axes_a.append(ax)
    lo, hi = min(fixed, shuf), max(fixed, shuf)
    pad = max(0.35, (hi - lo) * 1.6)
    ax.plot([fixed, shuf], [0, 0], color='#bbbbbb', lw=1.2, zorder=1)
    ax.scatter([fixed], [0], s=32, facecolor='#1f4e9c', edgecolor='#1f4e9c', zorder=5, label='Fixed' if i == 0 else None)
    ax.scatter([shuf], [0], s=32, facecolor='white', edgecolor='#c0392b', linewidth=1.3, zorder=5, label='Shuffled' if i == 0 else None)
    d = shuf - fixed
    ax.annotate(f"{d:+.2f}", xy=((fixed+shuf)/2, 0), xytext=(0, 9), textcoords='offset points',
                ha='center', fontsize=6.8, color='#444444')
    ax.set_xlim((lo+hi)/2 - pad, (lo+hi)/2 + pad)
    ax.set_ylim(-1, 1.3)
    ax.set_yticks([]); ax.set_ylabel(name, fontsize=7.3, rotation=0, ha='right', va='center', labelpad=2)
    ax.tick_params(axis='x', labelsize=6.3, pad=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5) if pad < 1 else MultipleLocator(1))
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#999999')
axes_a[0].set_title('(a) Domain-order robustness', fontsize=FS_PANEL, pad=6, x=0.62)
fig.text(0.31, 0.085, 'mIoU (%)', fontsize=FS_LABEL, ha='center')
h_f = plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#1f4e9c', markeredgecolor='#1f4e9c', markersize=5)
h_s = plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#c0392b', markersize=5, markeredgewidth=1.3)
fig.legend(handles=[h_f, h_s], labels=['Fixed', 'Shuffled'], loc='upper center', bbox_to_anchor=(0.31, 0.995),
           ncol=2, frameon=False, fontsize=FS_LEG, handletextpad=0.3, columnspacing=1.0)

# ---------------- (b) abrupt domain shift ----------------
axB = fig.add_subplot(gsR[0, 0])
V = "PascalVOC20Dataset/adagate_hgate_5corr_sub100"
Rc, Mc = load(V)
Rs, Ms = load(V + "_phaseswitch")
axB.plot(Rc, Mc, color='#1f4e9c', lw=1.6, label='Fixed-domain control', zorder=4)
axB.plot(Rs, Ms, color='#c0392b', lw=1.6, label='Domain switch @R76', zorder=5)
axB.axvline(76, color='#666666', lw=0.9, ls=(0, (4, 2)), zorder=2)
axB.text(78, 71.5, 'Domain\nswitch', fontsize=6.6, color='#555555', va='top')
axB.set_xlim(0, 150); axB.set_ylim(67, 78)
axB.xaxis.set_major_locator(MultipleLocator(50))
axB.set_xlabel('Round', fontsize=FS_LABEL, labelpad=2); axB.set_ylabel('mIoU (%)', fontsize=FS_LABEL, labelpad=2)
axB.tick_params(labelsize=FS_TICK, pad=2)
axB.grid(True, color='#ececec', lw=0.5, zorder=0); axB.set_axisbelow(True)
for s in ('top', 'right'): axB.spines[s].set_visible(False)
for s in ('left', 'bottom'): axB.spines[s].set_color('#666666')
axB.legend(loc='upper left', frameon=False, fontsize=FS_LEG, handlelength=1.6, labelspacing=0.3, borderaxespad=0.1)
axB.set_title('(b) Abrupt domain shift', fontsize=FS_PANEL, pad=4)

out = ".figs_additional_experiments/fig7_domain_order_and_abrupt_shift"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("saved", out)
