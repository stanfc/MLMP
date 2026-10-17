"""Figure 6: (a) stochastic restoration is necessary (ACDC, rho sweep trajectory),
(b) hyperparameter sensitivity (compact dot plot, R150 mIoU). ICLR single-column width."""
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
fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, 1]))

# ---------------- (a) stochastic restoration ----------------
A = "ACDCDataset/adagate_hgate"
RHO = [("0", A+"_abl_baserst_0", "#c0392b"), ("0.005", A+"_abl_baserst_0.005", "#7fb3d5"),
       ("0.01 (default)", A, "#1f4e9c"), ("0.02", A+"_abl_baserst_0.02", "#3f9e9a")]
for lab, d, c in RHO:
    R, M = load(d)
    hero = 'default' in lab
    axA.plot(R, M, color=c, lw=2.0 if (hero or lab == "0") else 1.3, zorder=6 if hero else (5 if lab=="0" else 3),
              label=fr"$\rho={lab.split(' ')[0]}$" + (" (default)" if hero else ""))
# annotate the two key numbers on the rho=0 curve
R0, M0 = load(A+"_abl_baserst_0")
i_peak = max(range(len(M0)), key=lambda i: M0[i])
axA.annotate(f"{M0[i_peak]:.2f}@R{R0[i_peak]}", xy=(R0[i_peak], M0[i_peak]), xytext=(R0[i_peak]+8, M0[i_peak]+1.3),
             fontsize=7, color="#c0392b", arrowprops=dict(arrowstyle='-', color="#c0392b", lw=0.6))
axA.annotate(f"{M0[-1]:.2f}@R150", xy=(150, M0[-1]), xytext=(96, M0[-1]-5.5),
             fontsize=7, color="#c0392b", arrowprops=dict(arrowstyle='-', color="#c0392b", lw=0.6))
axA.set_xlim(0, 150); axA.set_ylim(5, 35)
axA.xaxis.set_major_locator(MultipleLocator(50))
axA.set_xlabel('Round', fontsize=FS_LABEL, labelpad=2); axA.set_ylabel('mIoU (%)', fontsize=FS_LABEL, labelpad=2)
axA.tick_params(labelsize=FS_TICK, pad=2)
axA.grid(True, color='#ececec', lw=0.5, zorder=0); axA.set_axisbelow(True)
for s in ('top', 'right'): axA.spines[s].set_visible(False)
for s in ('left', 'bottom'): axA.spines[s].set_color('#666666')
axA.legend(loc='lower left', frameon=False, fontsize=FS_LEG, handlelength=1.6, labelspacing=0.3, borderaxespad=0.1)
axA.set_title('(a) Stochastic restoration', fontsize=FS_PANEL, pad=4)

# ---------------- (b) hyperparameter sensitivity ----------------
HP = [
    ("lr", [("2.5e-6", 30.92), ("5e-6", 30.95), ("1e-5", 31.03)], 1),
    ("monitor interval", [("25", 31.02), ("50", 30.95), ("100", 31.23)], 1),
    ("anchor $h$ ratio", [("0.90", 30.96), ("0.95", 30.95), ("0.98", 30.80)], 1),
    ("trend threshold", [("0.25", 30.97), ("0.5", 30.95), ("1.0", 31.10)], 1),
]
for row_i, (name, pts, default_idx) in enumerate(HP):
    y = (len(HP) - row_i) * 1.35
    xs = [v for _, v in pts]
    axB.plot(xs, [y]*len(xs), color='#bbbbbb', lw=0.9, zorder=1)
    order = sorted(range(len(pts)), key=lambda j: pts[j][1])  # left-to-right by x
    for rank, j in enumerate(order):
        lab, v = pts[j]
        is_default = (j == 1)
        axB.scatter([v], [y], s=34 if is_default else 26,
                    facecolor='#1f4e9c' if is_default else 'white',
                    edgecolor='#1f4e9c', linewidth=1.1, zorder=5)
        above = (rank % 2 == 0)
        axB.annotate(lab, (v, y), xytext=(0, 6.5 if above else -9), textcoords='offset points',
                     ha='center', va='bottom' if above else 'top', fontsize=6.3, color='#444444')
axB.set_yticks([(len(HP) - i) * 1.35 for i in range(len(HP))])
axB.set_yticklabels([n for n, _, _ in HP], fontsize=FS_TICK)
axB.set_xlim(30.55, 31.45)
axB.set_ylim(0.4, (len(HP) + 0.75) * 1.35)
axB.set_xlabel('R150 mIoU (%)', fontsize=FS_LABEL, labelpad=2)
axB.tick_params(axis='x', labelsize=FS_TICK, pad=2)
axB.tick_params(axis='y', length=0)
axB.grid(axis='x', color='#ececec', lw=0.5, zorder=0); axB.set_axisbelow(True)
for s in ('top', 'right', 'left'): axB.spines[s].set_visible(False)
axB.spines['bottom'].set_color('#666666')
# legend for filled=default, hollow=other
h_def = plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#1f4e9c', markeredgecolor='#1f4e9c', markersize=5.2, label='default')
h_oth = plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#1f4e9c', markersize=5.2, label='other')
axB.legend(handles=[h_def, h_oth], loc='upper right', frameon=False, fontsize=FS_LEG, handletextpad=0.3, borderaxespad=0.1)
axB.set_title('(b) Hyperparameter sensitivity', fontsize=FS_PANEL, pad=4)

fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.16)
out = ".figs_additional_experiments/fig6_stochastic_restoration_and_sensitivity"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
print("saved", out)
