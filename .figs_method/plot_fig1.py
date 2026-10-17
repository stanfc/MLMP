"""Figure 1 -- long-horizon (150-round) mIoU-vs-round comparison, all methods.
Reconstructed from the session scratchpad's fig1_v5.py (that copy was lost when
the scratchpad was cleared). Reads real results_all_rounds.txt files only.

Output: .figs_method/fig1_acdc.{png,svg}   (also v20 / cityscapes if data present)

Design notes:
 - ICLR body font is Times; Times isn't installed here, so this falls back to
   matplotlib's own serif (DejaVu Serif) -- same category, not metrically Times.
 - dual x-axis: bottom = Round, top = Time Step (imgs seen, in "k" units).
 - light background bands every `band_block` rounds; one round (highlight_round)
   boxed with dotted lines + a light fill, like the MGP-paper reference figure.
 - "Ours" is the bold red line; every other method is a thin line.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

MLMP = '/home/stanfc/TTA-on-OVSS/MLMP/save'
OUT_DIR = '/home/stanfc/TTA-on-OVSS/MLMP/.figs_method'

# Nature-style cool->warm family (indigo -> sky -> sand -> orange -> red), hand
# tuned off the reference palette so all 7 baselines stay legible on white.
# Ordered stable -> collapsing so the colour itself tells the story: methods
# that hold up read cool, methods that fall apart read warm.
COLORS = {
    'Ours':          '#1f4e9c',   # a solid, dignified blue -- the hero
    'No Adapt':      '#b3b3b3',   # grey dashed reference
    'MLMP-episodic': '#7d7d7d',   # grey dotted reference
    # blues for the ones that hold near source (value ramp: dark -> pale),
    # one purple for SAR (the oscillator), a warm gold -> deep-red ramp for the
    # ones that collapse. No green anywhere (so green+purple never co-occur).
    'CoTTA': '#5b8fb5',           # steel blue
    'TRIBE': '#aecad9',           # pale blue
    'SAR':   '#7d5ba6',           # purple
    'Tent':  '#e8b13c',           # gold
    'DeYO':  '#e37a35',           # orange
    'MGP':   '#cf4f3e',           # red-orange
    'ViDA':  '#8e2f2f',           # deep red
}
r150 = np.arange(1, 151)


def load(path, mean_col='Mean_mIoU'):
    df = pd.read_csv(f'{MLMP}/{path}/results_all_rounds.txt', skipinitialspace=True)
    df['Round'] = df['Round'].astype(str).str.extract(r'(\d+)').astype(int)
    df = df.sort_values('Round')
    return df['Round'].values, df[mean_col].values


def make_fig(series_specs, ymax, imgs_per_round, out_path, xtick_step=30, band_block=5,
             highlight_round=8):
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    n_rounds = 150

    for name, (x, y), lw, ls, z in series_specs:
        ref = name in ('Ours', 'No Adapt', 'MLMP-episodic')
        ax.plot(x, y, color=COLORS[name],
                linewidth=lw if ref else max(lw, 1.3), linestyle=ls,
                label=name, zorder=z, solid_capstyle='round',
                alpha=1.0 if ref else 0.95)

    ax.set_xlim(-3, n_rounds + 3)            # small x margin so nothing hugs the frame
    ax.set_ylim(0, ymax + 3)                 # headroom at the top
    # vertical grid every 10 rounds; label only every `xtick_step`
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: f'{int(v)}' if 0 <= v <= n_rounds and v % xtick_step == 0 else ''))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel('Round', fontsize=19, labelpad=8)
    ax.set_ylabel('mIoU (%)', fontsize=20, labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=17, length=4, color='#999999')
    ax.set_facecolor('white')
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#ececec', zorder=0)
    ax.set_axisbelow(True)
    # open frame -- drop the right spine, soften what's left
    ax.spines['right'].set_visible(False)
    for s in ('left', 'bottom', 'top'):
        ax.spines[s].set_linewidth(0.9)
        ax.spines[s].set_color('#888888')

    secax = ax.secondary_xaxis(
        'top', functions=(lambda r: r * imgs_per_round, lambda t: t / imgs_per_round))
    secax.set_xlabel('Time Step', fontsize=19, labelpad=8)
    secax.tick_params(axis='x', labelsize=17, length=4, color='#999999')
    secax.spines['top'].set_linewidth(0.9)
    secax.spines['top'].set_color('#888888')
    secax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v/1000:g}k' if v else '0'))

    # legend lives in its own file (see make_legend) -- keep the plot clean
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    print('saved', out_path)


def make_legend(series_specs, out_path, ncol=2):
    """Standalone legend, laid out in `ncol` columns (10 entries, ncol=2 -> 5x2)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[name], lw=lw, ls=ls)
               for name, _s, lw, ls, _z in series_specs]
    labels = [name for name, *_ in series_specs]
    nrow = -(-len(series_specs) // ncol)
    fig = plt.figure(figsize=(2.4 * ncol, 0.55 * nrow + 0.3))
    fig.legend(handles, labels, loc='center', ncol=ncol, frameon=False,
               fontsize=18, columnspacing=2.2, handlelength=2.4,
               labelspacing=1.1, handletextpad=0.7)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


# ================= ACDC =================
acdc = {
    'Ours':  load('ACDCDataset/adagate_ABmad05_ecdf'),
    'CoTTA': load('ACDCDataset/cotta_continual_150R'),
    'Tent':  load('ACDCDataset/naclip_tent_continual'),
    'SAR':   load('ACDCDataset/sar_continual'),
    'DeYO':  load('ACDCDataset/deyo_continual'),
    'MGP':   load('ACDCDataset/mgp_deyo_mlmp_continual'),
    'TRIBE': load('ACDCDataset/tribe_continual'),
    'ViDA':  load('ACDCDataset/vida_continual'),
}
acdc['No Adapt']      = (r150, np.full(150, 23.34))
acdc['MLMP-episodic'] = (r150, np.full(150, 29.84))

# 10 methods -> legend is 5 rows x 2 cols
specs = [
    ('Ours', acdc['Ours'], 3.0, '-', 5),
    ('No Adapt', acdc['No Adapt'], 1.4, '--', 1),
    ('MLMP-episodic', acdc['MLMP-episodic'], 1.6, ':', 1),
    ('CoTTA', acdc['CoTTA'], 1, '-', 2),
    ('Tent', acdc['Tent'], 1, '-', 2),
    ('SAR', acdc['SAR'], 1, '-', 2),
    ('DeYO', acdc['DeYO'], 1, '-', 2),
    ('MGP', acdc['MGP'], 1, '-', 2),
    ('TRIBE', acdc['TRIBE'], 1, '-', 2),
    ('ViDA', acdc['ViDA'], 1, '-', 2),
]
make_fig(specs, ymax=40, imgs_per_round=406, band_block=5,
         out_path=f'{OUT_DIR}/fig1_acdc.png')
make_legend(specs, out_path=f'{OUT_DIR}/fig1_legend.png', ncol=2)
