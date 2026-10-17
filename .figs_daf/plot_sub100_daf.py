"""DAF (Adam + lr=1e-3, batch=8) on the 4 sub100 datasets -- round-vs-mIoU
trajectories, current progress snapshot.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = '/home/stanfc/TTA-on-OVSS/MLMP'
RUNS = {
    'V21':        f'{ROOT}/save/PascalVOC21Dataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
    'P59':        f'{ROOT}/save/PascalContext59Dataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
    'P60':        f'{ROOT}/save/PascalContext60Dataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
    'COCO-Stuff': f'{ROOT}/save/COCOStuffDataset/daf_mlmp_continual_15corr_sub100/results_all_rounds.txt',
}
COLORS = {'V21': '#1f4e9c', 'P59': '#cf4f3e', 'P60': '#e8b13c', 'COCO-Stuff': '#5b8fb5'}

fig, ax = plt.subplots(figsize=(8, 5.2))
for name, path in RUNS.items():
    df = pd.read_csv(path, skipinitialspace=True)
    df['Round'] = df['Round'].astype(str).str.extract(r'(\d+)').astype(int)
    df = df.sort_values('Round')
    ax.plot(df['Round'], df['Mean_mIoU'], color=COLORS[name], linewidth=2,
            marker='o', markersize=3, label=f'{name} (R{df["Round"].max()})')

ax.set_xlabel('Round', fontsize=12)
ax.set_ylabel('Mean mIoU (%)', fontsize=12)
ax.set_title('DAF sub100 (Adam, lr=1e-3, batch=8) -- current progress', fontsize=12)
ax.set_xlim(0, 95)
ax.set_ylim(0, 50)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.grid(True, linestyle='-', linewidth=0.5, color='#ececec')
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10, loc='upper right')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

fig.savefig(f'{ROOT}/.figs_daf/sub100_daf_progress.png', dpi=200, bbox_inches='tight')
print('saved .figs_daf/sub100_daf_progress.png')
