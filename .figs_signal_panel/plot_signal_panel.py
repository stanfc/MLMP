"""mIoU vs. label-free collapse signal, phase view (scatter, coloured by round).
One standalone figure PER (signal, dataset) -- IDENTICAL style to the earlier
per-dataset scatters (figsize 6.4x5.4, no title, Pearson+Spearman box at
bottom-left, colorbar ticks every 30, marker s=28).

Grounded in the no-gate observation runs (plain deyo_mlmp_continual + --log_signals,
signals_log.csv) so the correlation is not circular. Reads real logs only.

Outputs (each .png + .svg, plus a `_nobox` twin):
  .figs_gradnorm_miou/miou_gradnorm_scatter_<ds>
  .figs_hmargin_miou/miou_hmargin_scatter_<ds>
for <ds> in ACDC, Cityscapes, VOC20.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr, pearsonr

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../MLMP

RUNS = {
    'ACDC':       'save/ACDCDataset/deyo_mlmp_continual_monitor',
    'Cityscapes': 'save/CityscapesDataset/deyo_mlmp_continual_monitor_5corr_sub100',
    'VOC20':      'save/PascalVOC20Dataset/deyo_mlmp_continual_monitor_5corr_sub100',
}
SIGNALS = [
    ('grad_norm', 'Gradient norm',          '.figs_gradnorm_miou', 'miou_gradnorm_scatter'),
    ('h_margin',  'Class-marginal entropy', '.figs_hmargin_miou',  'miou_hmargin_scatter'),
]


def per_round(D, col):
    sl = pd.read_csv(f'{ROOT}/{D}/signals_log.csv')
    res = pd.read_csv(f'{ROOT}/{D}/results_all_rounds.txt', skipinitialspace=True)
    res['Round'] = res['Round'].astype(str).str.extract(r'(\d+)').astype(int)
    res = res.sort_values('Round')
    s = sl.groupby('round')[col].mean()
    return res.set_index('Round')['Mean_mIoU'].to_frame().join(s.rename(col)).dropna()


def scatter(col, sig_label, D, out_stem, with_box):
    m = per_round(D, col)
    r, p = pearsonr(m[col], m['Mean_mIoU'])
    rho, ps = spearmanr(m[col], m['Mean_mIoU'])

    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    sc = ax.scatter(m[col], m['Mean_mIoU'], c=m.index, cmap='viridis',
                    s=30, edgecolors='none', linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Round', fontsize=20)
    cb.ax.tick_params(labelsize=17)
    cb.ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.set_xlabel(sig_label, fontsize=20)
    ax.set_ylabel('mIoU (%)', fontsize=20)
    ax.tick_params(axis='both', labelsize=17)
    if with_box:
        # compact box (r = Pearson, rho = Spearman) placed in whichever corner hides the fewest points
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        fx = (m[col].values - x0) / (x1 - x0); fy = (m['Mean_mIoU'].values - y0) / (y1 - y0)
        W, H = 0.42, 0.27
        corners = {'bl': (0.03, 0.04, 'left', 'bottom', (0.0, W + 0.03, 0.0, H + 0.04)),
                   'br': (0.97, 0.04, 'right', 'bottom', (0.97 - W, 1.0, 0.0, H + 0.04)),
                   'tl': (0.03, 0.96, 'left', 'top', (0.0, W + 0.03, 0.96 - H, 1.0)),
                   'tr': (0.97, 0.96, 'right', 'top', (0.97 - W, 1.0, 0.96 - H, 1.0))}
        def hidden(k):
            xa, xb, ya, yb = corners[k][4]
            return int(((fx > xa) & (fx < xb) & (fy > ya) & (fy < yb)).sum())
        k = min(corners, key=hidden)
        cx, cy, ha, va, _ = corners[k]
        ax.text(cx, cy, f'r = {r:.2f}\n' rf'$\rho$ = {rho:.2f}', transform=ax.transAxes, ha=ha, va=va,
                fontsize=16, zorder=10, bbox=dict(boxstyle='round', facecolor='white', edgecolor='#999999', alpha=0.9))
        print('   box corner', k, 'hidden pts', hidden(k))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', f'{out_stem}.png   (r={r:+.2f}, rho={rho:+.2f})')


for col, sig_label, folder, prefix in SIGNALS:
    d = os.path.join(ROOT, folder)
    os.makedirs(d, exist_ok=True)
    for ds, D in RUNS.items():
        scatter(col, sig_label, D, os.path.join(d, f'{prefix}_{ds}'), with_box=True)
        scatter(col, sig_label, D, os.path.join(d, f'{prefix}_{ds}_nobox'), with_box=False)
