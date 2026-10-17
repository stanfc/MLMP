"""Section 5.3 Ablation study figure: components (base_rst 0%/1%), hyperparameter
sweeps (trend_thr, anchor_h_min_ratio, monitor_interval, lr), and the V20 5corr
sub100 analysis runs (in-domain vs OOD-probe, resample-100-per-round, mid-stream
domain switch, shuffle domain order). All runs are CAGA (hgate, locked config),
150 rounds, complete. Run from MLMP/."""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral']})

def load(d):
    f = f"save/{d}/results_all_rounds.txt"
    R, M = [], []
    if os.path.exists(f):
        for l in open(f, errors='ignore'):
            if re.match(r'\s*Round\s+\d+', l):
                p = l.strip().split(','); R.append(int(p[0].split()[1])); M.append(float(p[-1]))
    return R, M

A = "ACDCDataset/adagate_hgate"
groups = {
 "ACDC: base_rst (0% vs 1%)": [("0.01 (default)", A), ("0", A+"_abl_baserst_0"), ("0.005", A+"_abl_baserst_0.005"), ("0.02", A+"_abl_baserst_0.02")],
 "ACDC: trend_thr": [("0.5 (default)", A), ("0.25", A+"_abl_trendthr_0.25"), ("1.0", A+"_abl_trendthr_1.0")],
 "ACDC: anchor_h_min_ratio": [("0.95 (default)", A), ("0.9", A+"_abl_anchorh_0.9"), ("0.98", A+"_abl_anchorh_0.98")],
 "ACDC: monitor_interval": [("50 (default)", A), ("25", A+"_abl_monint_25"), ("100", A+"_abl_monint_100")],
 "ACDC: lr": [("5e-6 (default)", A), ("2.5e-6", A+"_abl_lr_2.5e-6"), ("1e-5", A+"_abl_lr_1e-5")],
}
V = "PascalVOC20Dataset/adagate_hgate_5corr_sub100"
v20 = [("control", V), ("OOD-probe (in-domain)", V+"_oodprobe"), ("resample 100 imgs/round", V+"_resample"),
       ("domain switch @R76", V+"_phaseswitch"), ("shuffle domain order", "PascalVOC20Dataset/adagate_hgate_shuffledomain_5corr_sub100")]

fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
cols = ['#1f4e9c', '#e37a35', '#3f9e9a', '#cf4f3e', '#7d5ba6']
for ax, (name, runs) in zip(axes.flat, list(groups.items()) + [("V20 5corr sub100: analysis", v20)]):
    for c, (lab, d) in zip(cols, runs):
        R, M = load(d)
        if R: ax.plot(R, M, color=c, lw=2.2 if ('default' in lab or lab == 'control') else 1.5, label=f"{lab}  (R{R[-1]}: {M[-1]:.2f})")
    ax.set_title(name, fontsize=12); ax.set_xlabel('Round'); ax.set_ylabel('mIoU (%)')
    ax.grid(color='#ececec'); ax.legend(fontsize=7.5, frameon=False)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

fig.suptitle("Ablation study (Sec. 5.3) -- all runs complete, 150 rounds", fontsize=12)
fig.tight_layout()
fig.savefig(".figs_additional_experiments/ablation_study.png", dpi=150)
fig.savefig(".figs_additional_experiments/ablation_study.svg")
print("saved .figs_additional_experiments/ablation_study.png")
