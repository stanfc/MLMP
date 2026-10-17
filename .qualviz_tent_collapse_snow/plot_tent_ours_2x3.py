"""Qualitative collapse figure (ACDC snow): legend | input / GT | R0 source / R150 Tent | R150 Ours. Vector PDF."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D
D = '/home/stanfc/TTA-on-OVSS/MLMP/.qualviz_tent_collapse_snow/'
N = '/home/stanfc/TTA-on-OVSS/MLMP/.qualviz_acdc_snow_all_methods/'   # final-method predictions (same sample, img_idx 0)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42, 'ps.fonttype': 42})
TITLE, CAP, LEG = 27, 19, 20
st = json.load(open(D + 'stats.json')); pal = np.array(st['palette'], dtype=np.uint8)
new = json.load(open(N + 'stats.json'))['stats']
def colorize(a):
    out = np.full(a.shape + (3,), 255, np.uint8); m = a != 255; out[m] = pal[a[m]]; return out
def stats(a):
    u, c = np.unique(a, return_counts=True); return len(u), c.max() / c.sum()
COLORS = {'CAGA': '#1f4e9c', 'No Adapt': '#b3b3b3', 'MLMP-episodic': '#7d7d7d', 'CoTTA': '#5b8fb5', 'TRIBE': '#aecad9', 'SAR': '#7d5ba6',
          'Tent': '#e8b13c', 'DeYO': '#e37a35', 'MGP': '#cf4f3e', 'ViDA': '#8e2f2f'}
SPEC = [('CAGA', 3.0, '-'), ('No Adapt', 1.4, '--'), ('MLMP-episodic', 1.6, ':'), ('CoTTA', 1.6, '-'), ('Tent', 1.6, '-'),
        ('SAR', 1.6, '-'), ('DeYO', 1.6, '-'), ('MGP', 1.6, '-'), ('TRIBE', 1.6, '-'), ('ViDA', 1.6, '-')]
gt = np.load(D + 'gt.npy'); src = np.load(N + 'pd_source_R0.npy'); ck = np.load(N + 'pd_TENT_R150.npy'); ours = np.load(N + 'pd_CAGA_R150.npy')
img = imread(D + 'image.png')
panels = {(0, 1): ("Input, ACDC snow", img, None), (1, 0): ("Ground Truth", colorize(gt), None),
          (1, 1): ("Round 0, source", colorize(src), stats(src)), (2, 0): ("Round 150, Tent", colorize(ck), stats(ck)),
          (2, 1): ("Round 150, CAGA", colorize(ours), stats(ours))}
fig, axes = plt.subplots(3, 2, figsize=(9.0, 10.2), gridspec_kw=dict(hspace=0.58, wspace=0.08, left=0.03, right=0.98, top=0.95, bottom=0.07))
for ax in axes.flat: ax.axis('off')
for (r, c), (title, im, s) in panels.items():
    ax = axes[r, c]; ax.imshow(im); ax.set_title(title, fontsize=TITLE, fontweight='bold', pad=8)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor('#444444'); sp.set_linewidth(1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis('on')
    if s: ax.set_xlabel(f"{s[0]} active classes\ntop-1 {s[1]*100:.0f}%,  conf {int(round({'Round 0, source': new['source_R0']['conf'], 'Round 150, Tent': new['TENT_R150']['conf'], 'Round 150, CAGA': new['CAGA_R150']['conf']}[title]*100))}%",
                        fontsize=CAP, labelpad=8)
axes[0, 0].axis('off')
hs = [Line2D([0], [0], color=COLORS[n], lw=lw, ls=ls) for n, lw, ls in SPEC]
axes[0, 0].legend(hs, [n for n, *_ in SPEC], loc='center', ncol=2, frameon=False, fontsize=LEG, columnspacing=0.6, handlelength=1.2, labelspacing=0.7, handletextpad=0.5, borderaxespad=0.1)
fig.savefig(D + 'tent_ours_2x3_snow.pdf'); fig.savefig('/tmp/tent_ours_preview.png', dpi=120); print('ok')
