"""DAF (Diversify, Anchor, Filter) on ACDC -- round-vs-mIoU trajectory,
showing the collapse to a degenerate dead state.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = '/home/stanfc/TTA-on-OVSS/MLMP'
df = pd.read_csv(f'{ROOT}/save/ACDCDataset/daf_mlmp_continual/results_all_rounds.txt',
                  skipinitialspace=True)
df['Round'] = df['Round'].astype(str).str.extract(r'(\d+)').astype(int)
df = df.sort_values('Round')

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(df['Round'], df['Mean_mIoU'], color='#cf4f3e', linewidth=2.2, marker='o', markersize=4)
ax.axvline(4, color='#999999', linestyle='--', linewidth=1)
ax.text(4.3, df['Mean_mIoU'].max() * 0.9, 'collapse @ R4', fontsize=10, color='#666666')

ax.set_xlabel('Round', fontsize=12)
ax.set_ylabel('Mean mIoU (%)', fontsize=12)
ax.set_title('DAF (Diversify+Anchor+Filter) on ACDC -- LR=1e-3 (paper default)', fontsize=12)
ax.set_xlim(0, df['Round'].max() + 1)
ax.set_ylim(0, df['Mean_mIoU'].max() + 3)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.grid(True, linestyle='-', linewidth=0.5, color='#ececec')
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

fig.savefig(f'{ROOT}/.figs_daf/acdc_daf_trajectory.png', dpi=200, bbox_inches='tight')
print('saved .figs_daf/acdc_daf_trajectory.png')
print(df[['Round', 'Mean_mIoU']].to_string(index=False))
