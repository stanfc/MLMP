"""mIoU vs. label-free collapse signal, trajectory view (twin y-axis vs round).
One standalone figure PER (signal, dataset). Colours: mIoU = #61A2DA, signal =
#D75725. Style matches the earlier twin-axis figures (figsize 8.2x5.2, axes not
inverted, faint grid, x ticks every 30).

Outputs (each .png + .svg):
  .figs_gradnorm_miou/miou_gradnorm_twinaxis_<ds>
  .figs_hmargin_miou/miou_hmargin_twinaxis_<ds>
for <ds> in ACDC, Cityscapes, VOC20.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIOU_C = '#61A2DA'
SIG_C = '#D75725'

RUNS = {
    'ACDC':       'save/ACDCDataset/deyo_mlmp_continual_monitor',
    'Cityscapes': 'save/CityscapesDataset/deyo_mlmp_continual_monitor_5corr_sub100',
    'VOC20':      'save/PascalVOC20Dataset/deyo_mlmp_continual_monitor_5corr_sub100',
}
SIGNALS = [
    ('grad_norm', 'Gradient norm',          '.figs_gradnorm_miou', 'miou_gradnorm_twinaxis'),
    ('h_margin',  'Class-marginal entropy', '.figs_hmargin_miou',  'miou_hmargin_twinaxis'),
]


def per_round(D, col):
    sl = pd.read_csv(f'{ROOT}/{D}/signals_log.csv')
    res = pd.read_csv(f'{ROOT}/{D}/results_all_rounds.txt', skipinitialspace=True)
    res['Round'] = res['Round'].astype(str).str.extract(r'(\d+)').astype(int)
    res = res.sort_values('Round')
    s = sl.groupby('round')[col].mean()
    m = res.set_index('Round')['Mean_mIoU'].to_frame().join(s.rename(col)).dropna()
    return m, res['Round'].max()


def twinaxis(col, sig_label, D, out_stem):
    m, n_rounds = per_round(D, col)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(m.index, m['Mean_mIoU'], color=MIOU_C, linewidth=1.5)
    ax.set_xlabel('Round', fontsize=20)
    ax.set_ylabel('mIoU (%)', color=MIOU_C, fontsize=20)
    ax.tick_params(axis='y', labelcolor=MIOU_C, labelsize=17)
    ax.tick_params(axis='x', labelsize=17)
    ax.set_xlim(1, n_rounds)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.grid(axis='both', color='#eeeeee', linewidth=0.5)
    ax.spines['top'].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(m.index, m[col], color=SIG_C, linewidth=1.5)
    ax2.set_ylabel(sig_label.replace('Class-marginal ', 'Class-marginal\n'), color=SIG_C, fontsize=20)
    ax2.tick_params(axis='y', labelcolor=SIG_C, labelsize=17)
    ax2.spines['top'].set_visible(False)

    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    rho, _ = spearmanr(m[col], m['Mean_mIoU'])
    print('saved', f'{out_stem}.png   (rho={rho:+.2f})')


for col, sig_label, folder, prefix in SIGNALS:
    d = os.path.join(ROOT, folder)
    os.makedirs(d, exist_ok=True)
    for ds, D in RUNS.items():
        twinaxis(col, sig_label, D, os.path.join(d, f'{prefix}_{ds}'))
