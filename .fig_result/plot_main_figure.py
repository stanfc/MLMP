"""Main empirical figure: 3-panel trajectory plot (ACDC / Cityscapes-C / VOC20-C), one shared legend.
Sized for the ICLR single-column template (text width 5.5in). Run from MLMP/ root.
Outputs: .fig_result/main_figure_trajectories.{pdf,png}"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',   # Times-like, matches ICLR body
    'pdf.fonttype': 42, 'ps.fonttype': 42,                                               # embed TrueType
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})
FS_TICK, FS_LABEL, FS_PANEL, FS_LEG = 7, 8, 8.5, 7

COLORS = {'CAGA': '#3a78c4', 'No-Adapt': '#a8a8a8', 'MLMP-episodic': '#333333',
          'TRIBE': '#aecad9', 'SPA': '#7d5ba6', 'SAR': '#7d5ba6', 'DAF': '#3f9e9a', 'TENT': '#e8b13c', 'COME': '#c2a17a',
          'PTTA': '#e07fa8', 'DeYO': '#e37a35', 'MGP': '#cf4f3e', 'ViDA': '#8e2f2f'}
ORDER = ['CAGA', 'No-Adapt', 'MLMP-episodic', 'SPA', 'DAF', 'TENT', 'DeYO', 'MGP']   # methods shown in the main figure

def load(path):
    R, M = [], []
    for l in open(path, errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            p = l.strip().split(',')
            try: R.append(int(p[0].split()[1])); M.append(float(p[-1]))
            except (ValueError, IndexError): pass
    return R, M
def episodic_mean(path):
    v = []
    for l in list(open(path, errors='ignore'))[1:]:
        if not l.strip() or 'GPU' in l or 'Duration' in l or 'Mean' in l: continue
        try: v.append(float(l.split(',')[1].split('+/-')[0]))
        except (ValueError, IndexError): pass
    return sum(v) / len(v) if v else None
def std(sfx):
    return {"DAF": f"daf_mlmp_continual{sfx}", "TENT": f"tent_continual{sfx}", "SAR": f"sar_continual{sfx}", "SPA": f"spa_continual{sfx}",
            "TRIBE": f"tribe_continual{sfx}", "DeYO": f"deyo_continual{sfx}", "ViDA": f"vida_continual{sfx}", "PTTA": f"ptta_continual{sfx}",
            "COME": f"come_continual{sfx}", "MGP": f"mgp_deyo_mlmp_continual{sfx}"}

panels = [
  ("(a) ACDC", "save/ACDCDataset",
   {"CAGA": "adagate_hgate", "DAF": "daf_mlmp_continual", "TENT": "naclip_tent_continual", "SAR": "sar_continual", "SPA": "spa_continual",
    "TRIBE": "tribe_continual", "DeYO": "deyo_continual", "ViDA": "vida_continual", "PTTA": "ptta_continual", "COME": "come_continual",
    "MGP": "mgp_deyo_mlmp_continual"}, "naclip_no_adapt", "naclip_mlmp_episodic", 36),
  ("(b) Cityscapes-C", "save/CityscapesDataset", {"CAGA": "adagate_hgate_sub100_15corr", **std("_sub100_15corr")},
   "no_adapt_sub100_15corr", "mlmp_episodic_sub100_15corr", 26.5),
  ("(c) VOC20-C", "save/PascalVOC20Dataset", {"CAGA": "adagate_hgate_sub100_15corr", **std("_sub100_15corr")},
   "no_adapt_sub100_15corr", "mlmp_episodic_sub100_15corr", 83),
]

W = 5.5                                                   # ICLR \textwidth
fig, axes = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw=dict(wspace=0.36))
handles = {}
for ax, (title, root, d, na, ep, ymax) in zip(axes, panels):
    curves = {}
    for lab, sub in d.items():
        p = f"{root}/{sub}/results_all_rounds.txt"
        if os.path.exists(p):
            R, M = load(p)
            if R: curves[lab] = (R, M)
    refs = {'No-Adapt': load(f"{root}/{na}/results_all_rounds.txt")[1][-1], 'MLMP-episodic': episodic_mean(f"{root}/{ep}/results.txt")}
    for lab in ORDER:
        if lab == 'No-Adapt':
            h, = ax.plot([0, 150], [refs[lab]] * 2, color=COLORS[lab], lw=0.7, ls=(0, (4, 2)), zorder=10)
        elif lab == 'MLMP-episodic':
            h, = ax.plot([0, 150], [refs[lab]] * 2, color=COLORS[lab], lw=0.8, ls=(0, (1.4, 1.4)), zorder=10)
        elif lab in curves:
            R, M = curves[lab]; hero = lab == 'CAGA'
            h, = ax.plot(R, M, color=COLORS[lab], lw=1.25 if hero else 0.65, zorder=6 if hero else 2, solid_capstyle='round')
        else:
            continue
        handles.setdefault(lab, h)
    ax.set_xlim(0, 150); ax.set_ylim(0, ymax)
    ax.xaxis.set_major_locator(MultipleLocator(50)); ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(labelsize=FS_TICK, pad=2, which='both'); ax.tick_params(which='minor', length=1.4, width=0.4)
    ax.set_xlabel('Round', fontsize=FS_LABEL, labelpad=1.5)
    ax.set_title(title, fontsize=FS_PANEL, pad=3)
    ax.grid(True, which='major', color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')
axes[0].set_ylabel('mIoU (%)', fontsize=FS_LABEL, labelpad=2)

# legend rows (matplotlib fills columns first, so interleave the two rows)
ROW1 = ['CAGA', 'No-Adapt', 'MLMP-episodic', 'TENT']
ROW2 = ['DeYO', 'SPA', 'MGP', 'DAF']
labs = [l for l in ROW1 + ROW2 if l in handles]     # single row, left to right
fig.legend([handles[l] for l in labs], labs, loc='lower center', ncol=len(labs), frameon=False, fontsize=FS_LEG,
           handlelength=1.5, handletextpad=0.35, columnspacing=0.75, labelspacing=0.3, bbox_to_anchor=(0.5, -0.075))
fig.subplots_adjust(left=0.075, right=0.975, top=0.905, bottom=0.235)

out = "/home/stanfc/TTA-on-OVSS/MLMP/.fig_result/main_figure_trajectories"
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved", out, fig.get_size_inches())
