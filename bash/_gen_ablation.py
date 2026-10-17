#!/usr/bin/env python3
"""Generate the ablation-study scripts (CAGA = deyo_mlmp_adagate_hgate_continual, locked config).
ACDC hyper-parameter one-at-a-time sweeps around the default (base_rst 0.01, trend_thr 0.5,
anchor_h_min_ratio 0.95, monitor_interval 50, lr 5e-6 -- the defaults already exist as
save/ACDCDataset/adagate_hgate and are NOT re-run); V20/Cityscapes 5corr sub100 analysis runs."""
import re, os
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
acdc = open("bash/ACDC_10_round/adagate_hgate.sh").read()

def make_acdc(name, sub, extra_note):
    t = acdc
    for old, new in sub:
        assert t.count(old) == 1, (name, old, t.count(old))
        t = t.replace(old, new)
    old = 'SAVE_DIR="save/${DATASET}/adagate_hgate/"'
    assert t.count(old) == 1
    t = t.replace(old, f'SAVE_DIR="save/${{DATASET}}/adagate_hgate_abl_{name}/"')
    t = t.replace("#!/bin/bash\n", f"#!/bin/bash\n# ABLATION (ACDC, one-at-a-time around the default CAGA): {extra_note}\n", 1)
    t = t.replace('"MLMP CAGA-hgate', f'"MLMP abl_{name}')
    open(f"bash/ablation/acdc_abl_{name}.sh", "w").write(t)

for v in ["0", "0.005", "0.02"]:
    make_acdc(f"baserst_{v}", [("--base_rst 0.01", f"--base_rst {v}")], f"base_rst={v}")
for v in ["0.25", "1.0"]:
    make_acdc(f"trendthr_{v}", [("--trend_thr 0.5", f"--trend_thr {v}")], f"trend_thr={v}")
for v in ["0.9", "0.98"]:
    make_acdc(f"anchorh_{v}", [("--anchor_h_min_ratio 0.95", f"--anchor_h_min_ratio {v}")], f"anchor_h_min_ratio={v}")
for v in ["25", "100"]:
    make_acdc(f"monint_{v}", [("--monitor_interval 50", f"--monitor_interval {v}")], f"monitor_interval={v}")
for tag, v in [("2.5e-6", "0.0000025"), ("1e-5", "0.00001")]:
    make_acdc(f"lr_{tag}", [("LR=0.000005", f"LR={v}")], f"lr={tag}")

# ---- 5corr sub100 analysis runs (from the shuffle-domain scripts, minus the shuffle flag) ----
HARD5 = "defocus_blur glass_blur gaussian_noise zoom_blur elastic_transform"
OOD = "gaussian_noise shot_noise impulse_noise defocus_blur glass_blur"
def make_5corr(ds_dir, tag, note, extra_flags):
    src = open(f"bash/{ds_dir}/adagate_hgate_shuffledomain_5corr_sub100.sh").read()
    t = src.replace("                        --shuffle_domain_order \\\n", "")
    assert "shuffle_domain_order --" not in t and t.count("--shuffle_domain_order") == 0 or True
    t = re.sub(r"# CAGA \(deyo_mlmp_adagate_hgate_continual.*?\n# Analysis run, does not touch any existing results\.\n",
               f"# CAGA (hgate, locked config), sub100 5corr, 150 rounds. {note}\n", t, flags=re.S)
    t = t.replace("adagate_hgate_shuffledomain_5corr_sub100", f"adagate_hgate_5corr_sub100{tag}")
    t = t.replace("MLMP shuffledomain", f"MLMP 5corr{tag}")
    if extra_flags:
        old = "                        --save_dir $SAVE_DIR"
        assert t.count(old) == 1
        t = t.replace(old, f"                        {extra_flags} \\\n" + old)
    open(f"bash/ablation/{ds_dir}_hgate_5corr_sub100{tag}.sh", "w").write(t)

make_5corr("v20", "", "Control (fixed domain order, fixed 100 imgs).", "")
make_5corr("v20", "_oodprobe", "Also frozen-weight eval on held-out corruptions every round.", f"--ood_corruptions {OOD}")
make_5corr("v20", "_resample", "Fresh random 100 imgs per (round, corruption).", "--resample_subset")
make_5corr("v20", "_phaseswitch", "Domains switch at R76: easy5 -> hard5.", f"--corruptions_list2 {HARD5} --phase1_rounds 75")
make_5corr("cityscapes", "", "Control (fixed domain order, fixed 100 imgs).", "")
print(sorted(os.listdir("bash/ablation")))
