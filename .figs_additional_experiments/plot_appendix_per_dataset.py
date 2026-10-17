"""Appendix: one large full-width figure per dataset (8), all baselines + CAGA hero. Legend above axes."""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.7, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3})
def load(p):
    R, M = [], []
    for l in open(p, errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            q = l.strip().split(',')
            try: R.append(int(q[0].split()[1])); M.append(float(q[-1]))
            except (ValueError, IndexError): pass
    return R, M
def epi(p):
    v = []
    for l in list(open(p, errors='ignore'))[1:]:
        if not l.strip() or 'GPU' in l or 'Duration' in l or 'Mean' in l: continue
        try: v.append(float(l.split(',')[1].split('+/-')[0]))
        except (ValueError, IndexError): pass
    return sum(v) / len(v)
M = {"TENT": "tent_continual", "SAR": "sar_continual", "SPA": "spa_continual", "TRIBE": "tribe_continual", "DeYO": "deyo_continual",
     "ViDA": "vida_continual", "PTTA": "ptta_continual", "COME": "come_continual", "MGP": "mgp_deyo_mlmp_continual", "DAF": "daf_mlmp_continual"}
COL = {'CAGA': '#3a78c4', 'TRIBE': '#aecad9', 'SPA': '#2a78a8', 'SAR': '#7d5ba6', 'DAF': '#3f9e9a', 'TENT': '#e8b13c', 'COME': '#c2a17a',
       'PTTA': '#e07fa8', 'DeYO': '#e37a35', 'MGP': '#cf4f3e', 'ViDA': '#8e2f2f'}
def find(root, base):
    for s in (f"{base}_15corr_sub100", f"{base}_sub100_15corr"):
        p = f"{root}/{s}"
        if os.path.exists(p + "/results_all_rounds.txt") or os.path.exists(p + "/results.txt"): return p
    return None
SETS = [("ACDC", "ACDCDataset"), ("Cityscapes-C", "CityscapesDataset"), ("VOC20-C", "PascalVOC20Dataset"), ("VOC21-C", "PascalVOC21Dataset"),
        ("PC59-C", "PascalContext59Dataset"), ("PC60-C", "PascalContext60Dataset"), ("COCO-Stuff-C", "COCOStuffDataset"), ("COCO-Object-C", "COCOObjectDataset")]
ORDER = ['CAGA', 'No-Adapt', 'MLMP-episodic', 'TENT', 'SAR', 'SPA', 'TRIBE', 'DeYO', 'ViDA', 'PTTA', 'COME', 'MGP', 'DAF']
for title, folder in SETS:
    root = f"save/{folder}"; acdc = folder == "ACDCDataset"
    fig, ax = plt.subplots(figsize=(5.5, 2.2)); h = {}
    if acdc:
        na = load(f"{root}/naclip_no_adapt/results_all_rounds.txt")[1][-1]; ep = epi(f"{root}/naclip_mlmp_episodic/results.txt")
    else:
        na = load(find(root, "no_adapt") + "/results_all_rounds.txt")[1][-1]; ep = epi(find(root, "mlmp_episodic") + "/results.txt")
    h['No-Adapt'], = ax.plot([0, 150], [na] * 2, color='#a8a8a8', lw=1.2, ls=(0, (4, 2)), zorder=1)
    h['MLMP-episodic'], = ax.plot([0, 150], [ep] * 2, color='#5f5f5f', lw=1.2, ls=(0, (1.4, 1.4)), zorder=1)
    for lab, d in M.items():
        p = f"{root}/{d}" if acdc else find(root, d)
        if lab == "TENT" and acdc: p = f"{root}/naclip_tent_continual"
        if p is None or not os.path.exists(p + "/results_all_rounds.txt"): print("missing", title, lab); continue
        R, Y = load(p + "/results_all_rounds.txt"); h[lab], = ax.plot(R, Y, color=COL[lab], lw=1.1, zorder=2)
    p = f"{root}/adagate_hgate" if acdc else find(root, "adagate_hgate")
    R, Y = load(p + "/results_all_rounds.txt"); h['CAGA'], = ax.plot(R, Y, color=COL['CAGA'], lw=2.2, zorder=6)
    ax.set_xlim(0, 150); ax.set_ylim(0, None); ax.xaxis.set_major_locator(MultipleLocator(25)); ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Round', fontsize=9.5); ax.set_ylabel('mIoU (%)', fontsize=9.5); ax.tick_params(labelsize=8.5, which='both'); ax.tick_params(which='minor', length=1.6, width=0.4)
    ax.grid(True, color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    out = f".figs_additional_experiments/appendix_per_dataset/{title.replace('-','_')}_all_methods"
    fig.savefig(out + ".pdf", bbox_inches='tight'); fig.savefig(out + ".png", dpi=250, bbox_inches='tight'); plt.close(fig); print(out)

# separate legend file (shared by all 8 figures)
from matplotlib.lines import Line2D
sty = {'CAGA': dict(color=COL['CAGA'], lw=2.2), 'No-Adapt': dict(color='#a8a8a8', lw=1.2, ls=(0, (4, 2))),
       'MLMP-episodic': dict(color='#5f5f5f', lw=1.2, ls=(0, (1.4, 1.4)))}
hs = [Line2D([], [], **sty.get(k, dict(color=COL.get(k), lw=1.1))) for k in ORDER]
lf = plt.figure(figsize=(5.5, 0.4))
lf.legend(hs, ORDER, loc='center', ncol=7, frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.2)
lf.savefig(".figs_additional_experiments/appendix_per_dataset/legend_all_methods.pdf", bbox_inches='tight')
lf.savefig(".figs_additional_experiments/appendix_per_dataset/legend_all_methods.png", dpi=250, bbox_inches='tight')
