"""Three figures for the newest result sets:
 fig8  monitoring-signal ablation (5 arms, ACDC, 10-round trailing mean; running arms end early)
 fig9  15-backbone CAGA(hgate) ablation, R150 mIoU (ACDC)
 fig10 SegFormer-B5 traditional segmentation, all methods (ACDC)
Style matches the other .figs_additional_experiments figures (serif, no titles, big fonts)."""
import os, re, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})
OUT = ".figs_additional_experiments"

def load(d):
    R, M = [], []
    for l in open(f"save/ACDCDataset/{d}/results_all_rounds.txt", errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            p = l.strip().split(','); R.append(int(p[0].split()[1])); M.append(float(p[-1]))
    return R, M
def trailing(M, k=10):
    return [sum(M[max(0, i-k+1):i+1]) / (i - max(0, i-k+1) + 1) for i in range(len(M))]
def style(ax, fs_tick, fs_label, xl, yl):
    ax.set_xlabel(xl, fontsize=fs_label, labelpad=2); ax.set_ylabel(yl, fontsize=fs_label, labelpad=2)
    ax.tick_params(labelsize=fs_tick, pad=2)
    ax.grid(True, color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#666666')

# ---------------- fig8: monitoring-signal ablation ----------------
ARMS = [("CAGA (default)", "adagate_hgate", '#3a78c4', '-', 2.6, 6),
        ("Always restore", "adagate_hgate_alwaysrestore", '#c0392b', '-', 1.8, 4),
        ("Gradient norm", "adagate_gradnormonly", '#3f9e9a', '-', 1.8, 4),
        ("Class-marginal entropy", "adagate_hmarginonly", '#7d5ba6', '-', 1.8, 4)]
fig, ax = plt.subplots(figsize=(3.4, 2.8))
for lab, d, c, ls, lw, z in ARMS:
    R, M = load(d)
    ax.plot(R, trailing(M), color=c, ls=ls, lw=lw, zorder=z, label=lab)
ax.set_xlim(0, 150); ax.set_ylim(29.5, 32.5)
ax.xaxis.set_major_locator(MultipleLocator(50)); ax.yaxis.set_major_locator(MultipleLocator(1))
style(ax, 13, 15, 'Round', 'mIoU (%)')
ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.02), ncol=2, frameon=False, fontsize=12,
          handlelength=1.4, handletextpad=0.4, labelspacing=0.3, columnspacing=1.0, borderaxespad=0.1)
fig.tight_layout(pad=0.5)
fig.savefig(f"{OUT}/fig8_monitoring_signal_ablation.pdf", bbox_inches='tight')
fig.savefig(f"{OUT}/fig8_monitoring_signal_ablation.png", dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- fig9: backbone ablation ----------------
BB = [("NA-CLIP", "adagate_hgate", "L14"), ("ClearCLIP (k-k)", "adagate_hgate_backbone_clearclip_kk_L14", "L14"),
      ("ClearCLIP (q-q)", "adagate_hgate_backbone_clearclip_qq_L14", "L14"), ("SCLIP-reduced", "adagate_hgate_backbone_sclip_reduced_L14", "L14"),
      ("v-v attention", "adagate_hgate_backbone_vv_attention_L14", "L14"),
      ("MaskCLIP", "adagate_hgate_backbone_maskclip_L14", "L14"), ("NA-CLIP @336px", "adagate_hgate_backbone_naclip_L14_336", "L14"),
      ("ITACLIP", "adagate_hgate_backbone_itaclip_L14", "L14"),
      ("NA-CLIP", "adagate_hgate_backbone_naclip_B16", "B16"), ("ClearCLIP (k-k)", "adagate_hgate_backbone_clearclip_kk_B16", "B16"),
      ("ClearCLIP (q-q)", "adagate_hgate_backbone_clearclip_qq_B16", "B16"), ("SCLIP-reduced", "adagate_hgate_backbone_sclip_reduced_B16", "B16"),
      ("MaskCLIP", "adagate_hgate_backbone_maskclip_B16", "B16"),
      ("NA-CLIP", "adagate_hgate_backbone_naclip_B32", "B32")]
COL = {"L14": '#1f4e9c', "B16": '#3f9e9a', "B32": '#e37a35'}
NAME = {"L14": "ViT-L/14", "B16": "ViT-B/16", "B32": "ViT-B/32"}
EPI = {"adagate_hgate": "naclip_L14"}
def epi_mean(d):
    n = EPI.get(d, d.replace("adagate_hgate_backbone_", ""))
    v = []
    for l in list(open(f"save/ACDCDataset/mlmp_episodic_backbone_ablation/{n}/results.txt"))[1:5]:
        v.append(float(l.split(',')[1].split('+/-')[0]))
    return sum(v) / len(v)
rows = [(lab, load(d)[1][-1], g, epi_mean(d)) for lab, d, g in BB]
order = []
for g in ("L14", "B16", "B32"):
    order += sorted([r for r in rows if r[2] == g], key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(5.0, 4.1))       # same aspect ratio as fig6a (~1.23)
BLUE, GREY = '#3a78c4', '#c4c4c4'
x = 0; xpos = []; heads = []; prev = None; seps = []
for lab, v, g, e in order:
    if g != prev:
        if prev is not None: x += 0.7; seps.append(x - 0.65)
        heads.append([g, x, x])
    xpos.append(x); heads[-1][2] = x; x += 1; prev = g
W = 0.36
SHORT = {'SCLIP-reduced': 'SCLIP-red.', 'ClearCLIP (q-q)': 'ClearCLIP q-q', 'ClearCLIP (k-k)': 'ClearCLIP k-k', 'v-v attention': 'v-v attn', 'NA-CLIP @336px': 'NA-CLIP 336'}
for (lab, v, g, e), xx in zip(order, xpos):
    ax.bar(xx - W / 2, e, width=W, color=GREY, zorder=3)
    ax.bar(xx + W / 2, v, width=W, color=BLUE, zorder=3)
ax.set_xticks(xpos); ax.set_xticklabels([SHORT.get(r[0], r[0]) for r in order], rotation=40, ha='right', rotation_mode='anchor', fontsize=12.5)
ax.set_ylim(20, 36); ax.set_xlim(-0.7, x - 0.3)
ax.yaxis.set_major_locator(MultipleLocator(5))
for sx in seps: ax.axvline(sx, color='#d0d0d0', lw=0.8, ls=(0, (3, 3)), zorder=1)
for g, a, b in heads:
    ax.text((a + b) / 2, 34.6, NAME[g], ha='center', va='center', fontsize=13, fontweight='bold', color='#333333')
ax.set_ylabel('mIoU (%)', fontsize=21, labelpad=2)
ax.tick_params(axis='y', labelsize=14, pad=2); ax.tick_params(axis='x', length=0, pad=2)
ax.grid(axis='y', color='#ececec', lw=0.5, zorder=0); ax.set_axisbelow(True)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
for s_ in ('left', 'bottom'): ax.spines[s_].set_color('#666666')
hs = [plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
ax.legend(hs, ["MLMP-episodic", "CAGA"], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=19,
          handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
fig.tight_layout(pad=0.5)
fig.savefig(f"{OUT}/fig9_backbone_ablation.pdf", bbox_inches='tight')
fig.savefig(f"{OUT}/fig9_backbone_ablation.png", dpi=300, bbox_inches='tight'); plt.close(fig)

print("done (fig10 is produced by plot_fig10_segformer.py)")
