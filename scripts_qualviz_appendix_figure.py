"""Appendix qualitative figure (ACDC snow): input, GT, source R0, 11 methods at R150, class legend. Reads .qualviz_acdc_snow_all_methods/."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Rectangle
D = '/home/stanfc/TTA-on-OVSS/MLMP/.qualviz_acdc_snow_all_methods/'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42, 'ps.fonttype': 42})
ref = json.load(open('/home/stanfc/TTA-on-OVSS/MLMP/.qualviz_tent_collapse_snow/stats.json'))
classes, pal = ref['org_classes'], ref['palette']
st = json.load(open(D + 'stats.json'))['stats']
TITLE, CAP = 12, 10.5
cells = [('image', 'Input, ACDC snow', None), ('gt_mask', 'Ground Truth', None), ('pred_source_R0', 'Round 0, source', 'source_R0')]
for m in ['TENT', 'SAR', 'SPA', 'TRIBE', 'DeYO', 'ViDA', 'PTTA', 'COME', 'MGP', 'DAF', 'CAGA']:
    cells.append((f'pred_{m}_R150', f'Round 150, {m}', f'{m}_R150'))
fig, axes = plt.subplots(5, 3, figsize=(8.0, 9.0), gridspec_kw=dict(hspace=0.42, wspace=0.06, left=0.01, right=0.99, top=0.965, bottom=0.035))
for ax in axes.flat: ax.axis('off')
for ax, (f, title, k) in zip(axes.flat, cells):
    ax.imshow(imread(D + f + '.png')); ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
    hero = k == 'CAGA_R150'
    for sp in ax.spines.values(): sp.set_edgecolor('#1f4e9c' if hero else '#444444'); sp.set_linewidth(2.4 if hero else 0.9)
    ax.set_title(title, fontsize=TITLE, fontweight='bold', pad=5, color='#1f4e9c' if hero else 'black')
    if k: ax.set_xlabel(f"{st[k]['n_active']} classes,  top-1 {st[k]['top1']*100:.0f}%", fontsize=CAP, labelpad=4)
# class legend in the last cell
lg = axes[4, 2]; lg.axis('off'); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
n = len(classes); rows = (n + 1) // 2
for i, (c, p) in enumerate(zip(classes, pal)):
    col, r = divmod(i, rows); x = 0.02 + col * 0.52; y = 0.97 - (r + 0.5) * (0.94 / rows)
    lg.add_patch(Rectangle((x, y - 0.03), 0.06, 0.06, color=np.array(p) / 255, transform=lg.transAxes))
    lg.text(x + 0.08, y, c, fontsize=8.5, va='center', transform=lg.transAxes)
for e in ('pdf', 'png'): fig.savefig(D + f'appendix_qualitative_acdc_snow.{e}', dpi=200)
print('saved')
