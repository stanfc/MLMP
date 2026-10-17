"""DAF (SGD, paper-faithful) -- round-vs-mIoU trajectories across all 7 runs,
current progress. Two panels: sub100 datasets (batch=8, exactly matches
DAF's paper config) and full-scale datasets (batch=1, LR scaled by 1/8,
Filter/SAFS inert there).
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = '/home/stanfc/TTA-on-OVSS/MLMP'

SUB100 = {
    'V21':        f'{ROOT}/save/PascalVOC21Dataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
    'P59':        f'{ROOT}/save/PascalContext59Dataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
    'P60':        f'{ROOT}/save/PascalContext60Dataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
    'COCO-Stuff': f'{ROOT}/save/COCOStuffDataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
}
FULL = {
    'ACDC':            f'{ROOT}/save/ACDCDataset/daf_mlmp_continual/results_all_rounds.txt',
    'V20-full':        f'{ROOT}/save/PascalVOC20Dataset/daf_mlmp_continual_full_15corr/results_all_rounds.txt',
    'Cityscapes-full': f'{ROOT}/save/CityscapesDataset/daf_mlmp_continual_full_15corr/results_all_rounds.txt',
}
COLORS_SUB100 = {'V21': '#1f4e9c', 'P59': '#cf4f3e', 'P60': '#e8b13c', 'COCO-Stuff': '#5b8fb5'}
COLORS_FULL = {'ACDC': '#7d5ba6', 'V20-full': '#2e8b57', 'Cityscapes-full': '#c76b2e'}


def load(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df['Round'] = df['Round'].astype(str).str.extract(r'(\d+)').astype(int)
    return df.sort_values('Round')


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
for name, path in SUB100.items():
    df = load(path)
    ax.plot(df['Round'], df['Mean_mIoU'], color=COLORS_SUB100[name], linewidth=2,
            marker='o', markersize=2.5, label=f'{name} (R{df["Round"].max()})')
ax.set_title('sub100 (batch=8, paper-exact SGD+lr=1e-3)', fontsize=11)
ax.set_xlabel('Round', fontsize=12)
ax.set_ylabel('Mean mIoU (%)', fontsize=12)
ax.set_xlim(0, 155)
ax.set_ylim(0, 50)
ax.xaxis.set_major_locator(MultipleLocator(30))
ax.grid(True, linestyle='-', linewidth=0.5, color='#ececec')
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc='upper right')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

ax = axes[1]
for name, path in FULL.items():
    df = load(path)
    ax.plot(df['Round'], df['Mean_mIoU'], color=COLORS_FULL[name], linewidth=2,
            marker='o', markersize=3, label=f'{name} (R{df["Round"].max()})')
ax.set_title('full-scale (batch=1, SGD+lr=1.25e-4, Filter inert)', fontsize=11)
ax.set_xlabel('Round', fontsize=12)
ax.set_xlim(0, max(load(p)['Round'].max() for p in FULL.values()) + 5)
ax.set_ylim(0, 60)
ax.grid(True, linestyle='-', linewidth=0.5, color='#ececec')
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc='upper right')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

fig.suptitle('DAF (Diversify, Anchor, Filter) -- paper-faithful reruns, current progress', fontsize=13)
fig.tight_layout()
fig.savefig(f'{ROOT}/.figs_daf/all_daf_sgd_progress.png', dpi=200, bbox_inches='tight')
print('saved .figs_daf/all_daf_sgd_progress.png')
