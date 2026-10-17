#!/usr/bin/env python3
"""Generate CAGA(hgate)-on-ACDC scripts for the 16-backbone ablation table.
naclip_L14 is the existing flagship run (save/ACDCDataset/adagate_hgate, R150=30.95) -- skipped,
not regenerated. The other 15 vary --ovss_type / --ovss_backbone / --vision_outputs only; every
other flag is identical to bash/ACDC_10_round/adagate_hgate.sh (locked CAGA config)."""
import os
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
base = open("bash/ACDC_10_round/adagate_hgate.sh").read()

FULL_VO = "-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
# name -> (ovss_type, ovss_backbone, vision_outputs)
CONFIGS = {
    "clearclip_kk_L14":  ("clearclip_kk",  "ViT-L/14", FULL_VO),
    "clearclip_qq_L14":  ("clearclip_qq",  "ViT-L/14", FULL_VO),
    "sclip_reduced_L14": ("sclip_reduced", "ViT-L/14", FULL_VO),
    "naclip_gaussonly_L14": ("naclip_gaussonly", "ViT-L/14", FULL_VO),
    "vv_attention_L14":  ("vv_attention",  "ViT-L/14", FULL_VO),
    "maskclip_L14":      ("maskclip",      "ViT-L/14", FULL_VO),
    "naclip_L14_336":    ("naclip",        "ViT-L/14@336px", FULL_VO),
    "itaclip_L14":       ("itaclip",       "ViT-L/14", FULL_VO),
    "naclip_B16":        ("naclip",        "ViT-B/16", FULL_VO),
    "clearclip_kk_B16":  ("clearclip_kk",  "ViT-B/16", FULL_VO),
    "clearclip_qq_B16":  ("clearclip_qq",  "ViT-B/16", FULL_VO),
    "sclip_reduced_B16": ("sclip_reduced", "ViT-B/16", FULL_VO),
    "sclip_uaml1_B16":   ("sclip",         "ViT-B/16", "-1"),
    "maskclip_B16":      ("maskclip",      "ViT-B/16", FULL_VO),
    "naclip_B32":        ("naclip",        "ViT-B/32", FULL_VO),
}

for name, (ovss_type, backbone, vo) in CONFIGS.items():
    t = base
    t = t.replace('OVSS_TYPE="naclip"', f'OVSS_TYPE="{ovss_type}"')
    t = t.replace('OVSS_BACKBONE="ViT-L/14"', f'OVSS_BACKBONE="{backbone}"')
    old_vo = f"--vision_outputs {FULL_VO} \\"
    assert t.count(old_vo) == 1
    t = t.replace(old_vo, f"--vision_outputs {vo} \\")
    old_save = 'SAVE_DIR="save/${DATASET}/adagate_hgate/"'
    assert t.count(old_save) == 1
    t = t.replace(old_save, f'SAVE_DIR="save/${{DATASET}}/adagate_hgate_backbone_{name}/"')
    t = t.replace(
        "# CAGA-hgate: deyo_mlmp_adagate_hgate_continual -- same locked CAGA config\n",
        f"# CAGA-hgate: deyo_mlmp_adagate_hgate_continual, backbone ablation arm '{name}'\n"
        f"# (ovss_type={ovss_type}, ovss_backbone={backbone}). Same locked CAGA config otherwise.\n")
    t = t.replace('"MLMP CAGA-hgate', f'"MLMP backbone_{name}')
    open(f"bash/backbone_ablation/adagate_hgate_backbone_{name}.sh", "w").write(t)

print(sorted(CONFIGS))
print(len(CONFIGS), "scripts written")
