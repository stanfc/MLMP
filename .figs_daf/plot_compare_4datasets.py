"""DAF vs AdaGate (ours) vs MLMP-episodic vs No-Adapt -- one figure per
sub100 dataset (V21, P59, P60, COCO-Stuff). no_adapt and mlmp_episodic have
no round trajectory (single pass / continual_rounds=1), so they're drawn as
flat reference lines.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = '/home/stanfc/TTA-on-OVSS/MLMP'

DATASETS = {
    'V21':        'PascalVOC21Dataset',
    'P59':        'PascalContext59Dataset',
    'P60':        'PascalContext60Dataset',
    'COCO-Stuff': 'COCOStuffDataset',
}


def load_rounds(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df['Round'] = df['Round'].astype(str).str.extract(r'(\d+)').astype(int)
    return df.sort_values('Round')


def episodic_mean(path):
    vals = []
    with open(path) as f:
        next(f)  # header: "mIoU, mDice, mAcc"
        for line in f:
            line = line.strip()
            if not line or 'GPU' in line or 'Duration' in line:
                continue
            fields = line.split(',')
            if len(fields) < 2:
                continue
            miou_str = fields[1].strip().split('+/-')[0].strip()
            vals.append(float(miou_str))
    return sum(vals) / len(vals)


for short, dataset in DATASETS.items():
    daf = load_rounds(f'{ROOT}/save/{dataset}/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt')
    adagate = load_rounds(f'{ROOT}/save/{dataset}/adagate_15corr_sub100/results_all_rounds.txt')
    no_adapt = load_rounds(f'{ROOT}/save/{dataset}/no_adapt_15corr_sub100/results_all_rounds.txt')
    mlmp_ep = episodic_mean(f'{ROOT}/save/{dataset}/mlmp_episodic_15corr_sub100/results.txt')
    no_adapt_val = no_adapt['Mean_mIoU'].iloc[0]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(adagate['Round'], adagate['Mean_mIoU'], color='#1f4e9c', linewidth=2.2,
            label=f'AdaGate (ours, R{adagate["Round"].max()})')
    ax.plot(daf['Round'], daf['Mean_mIoU'], color='#cf4f3e', linewidth=2.2,
            label=f'DAF (R{daf["Round"].max()})')
    ax.axhline(mlmp_ep, color='#7d7d7d', linestyle=':', linewidth=1.8,
               label=f'MLMP-episodic ({mlmp_ep:.1f})')
    ax.axhline(no_adapt_val, color='#b3b3b3', linestyle='--', linewidth=1.8,
               label=f'No-Adapt ({no_adapt_val:.1f})')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Mean mIoU (%)', fontsize=12)
    ax.set_title(f'{short}: AdaGate vs DAF vs MLMP-episodic vs No-Adapt', fontsize=12)
    ax.set_xlim(0, 155)
    ax.grid(True, linestyle='-', linewidth=0.5, color='#ececec')
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc='best')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    out = f'{ROOT}/.figs_daf/compare_{short.lower().replace("-", "_")}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
