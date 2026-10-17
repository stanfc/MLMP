"""Generate full-dataset + 15-corruption continual bash scripts for the 13
requested baselines, across 4 datasets (COCOObject, COCOStuff, PascalContext59,
PascalContext60). Hyperparameters copied from the canonical ACDC/Cityscapes/VOC20
scripts already validated this session (see bash/ACDC_10_round/, bash/cityscapes/,
bash/v20/). GPU_ID is left overridable via env var so a queue runner can assign
GPUs dynamically. Each script ends with the standard ntfy.sh notify block.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))  # MLMP/bash
CONDITIONS = ("gaussian_noise shot_noise impulse_noise defocus_blur glass_blur "
              "motion_blur zoom_blur snow frost fog brightness contrast "
              "elastic_transform pixelate jpeg_compression")
VISION18 = "-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

DATASETS = {
    "coco_obj":    ("COCOObjectDataset",      ".data/coco/"),  # NOT .data/coco_object/ -- that path doesn't exist; the real val2017 images+annotations live under .data/coco/ (confirmed by smoke test)
    "coco_stuff":  ("COCOStuffDataset",       ".data/coco_stuff164k/"),
    "p59":         ("PascalContext59Dataset", ".data/VOC2010/"),
    "p60":         ("PascalContext60Dataset", ".data/VOC2010/"),
}

NOTIFY = '''
# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP full15corr ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP full15corr ❌"
fi
'''

HEADER = """#!/bin/bash
# {comment}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${{GPU_ID:-0}}

DATASET={dataset}
DATA_DIR="{data_dir}"
INIT_RESIZE="224 224"
CONDITIONS="{conditions}"
WORKERS=0
"""


def no_adapt(dataset, data_dir):
    return HEADER.format(comment="No-Adaptation baseline, full dataset, 15 corruptions.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + '''
METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=1

SAVE_DIR="save/${DATASET}/no_adapt_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --prompt_dir prompts.yaml \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def mlmp_episodic(dataset, data_dir):
    return HEADER.format(comment="MLMP-episodic upper bound, full dataset, 15 corruptions (main.py, episodic).",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + f'''
METHOD="mlmp"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

BATCH_SIZE=1
LR=0.001
STEPS=1
TRIALS=1

SAVE_DIR="save/${{DATASET}}/mlmp_episodic_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --vision_outputs {VISION18} \\
                        --prompt_dir $PROMPT_DIR \\
                        --alpha_cls $ALPHA_CLS \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch-size $BATCH_SIZE --trials $TRIALS \\
                        --seed 0 \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def adagate(dataset, data_dir):
    return HEADER.format(comment="Ours: deyo_mlmp_adagate_continual, ABmad05_ecdf config (current best), full+15corr.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + f'''
METHOD="deyo_mlmp_adagate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${{DATASET}}/adagate_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --prompt_dir $PROMPT_DIR \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --vision_outputs {VISION18} \\
                        --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \\
                        --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \\
                        --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \\
                        --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 --monitor_interval 50 \\
                        --trend_stat mad --trend_thr 0.5 --trend_hist 50 --lag_mode ecdf --lag_sat 1.5 \\
                        --shallow_cap_mode fixed \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def mgp(dataset, data_dir):
    return HEADER.format(comment="MGP (gradient-projection anti-saturation) on DeYO+MLMP, full+15corr.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + f'''
METHOD="mgp_deyo_mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

MGP_DISTILL_FREQ=100
MGP_BUFFER_SIZE=32
MGP_MAX_RANK=32
MGP_RESIDUAL_THR=0.75
MGP_COLLECT_FREQ=40

SAVE_DIR="save/${{DATASET}}/mgp_deyo_mlmp_continual_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --prompt_dir $PROMPT_DIR \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --vision_outputs {VISION18} \\
                        --mgp_distill_freq $MGP_DISTILL_FREQ --mgp_buffer_size $MGP_BUFFER_SIZE \\
                        --mgp_max_rank $MGP_MAX_RANK --mgp_residual_thr $MGP_RESIDUAL_THR \\
                        --mgp_collect_freq $MGP_COLLECT_FREQ \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def cotta(dataset, data_dir):
    return HEADER.format(comment="CoTTA, full+15corr.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + '''
METHOD="cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

MT=0.999
RST=0.01
AP=0.92
AUG_N=32
OUT_VISION="-1"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/cotta_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --vision_outputs $OUT_VISION \\
                        --mt $MT --rst $RST --ap $AP --aug_n $AUG_N \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def tent(dataset, data_dir):
    return HEADER.format(comment="TENT-continual, full+15corr.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + '''
METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/tent_continual_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def sar(dataset, data_dir):
    return HEADER.format(comment="SAR-continual, full+15corr.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + '''
METHOD="sar_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
E_MARGIN=1.8
SAM_RHO=0.05
E_0=0.1
EMA_FACTOR=0.9
RECOVERY_WARMUP=50
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/sar_continual_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --e_margin $E_MARGIN --sam_rho $SAM_RHO --e_0 $E_0 \\
                        --ema_factor $EMA_FACTOR --recovery_warmup $RECOVERY_WARMUP \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def deyo(dataset, data_dir):
    return HEADER.format(comment="DeYO-continual (plain, single-prompt), full+15corr.",
                         dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + '''
METHOD="deyo_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/deyo_continual_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY


def simple_baseline(method, folder_name):
    """ViDA / SPA / PTTA / COME / TRIBE all share the same plain single-prompt shape."""
    def gen(dataset, data_dir):
        return HEADER.format(comment=f"{method}, full+15corr.",
                             dataset=dataset, data_dir=data_dir, conditions=CONDITIONS) + f'''
METHOD="{method}"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${{DATASET}}/{folder_name}_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \\
                        --adapt \\
                        --method $METHOD \\
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \\
                        --dataset $DATASET --data_dir $DATA_DIR \\
                        --init_resize $INIT_RESIZE \\
                        --patch_size 224 224 --patch_stride 112 \\
                        --corruptions_list $CONDITIONS --workers $WORKERS \\
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \\
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \\
                        --save_dir $SAVE_DIR \\
                        --class_extensions
''' + NOTIFY
    return gen


METHODS = [
    ("no_adapt",             no_adapt),
    ("mlmp_episodic",        mlmp_episodic),
    ("adagate",              adagate),
    ("cotta",                cotta),
    ("tent_continual",       tent),
    ("sar_continual",        sar),
    ("deyo_continual",       deyo),
    ("mgp_deyo_mlmp_continual", mgp),
    ("vida_continual",       simple_baseline("vida_continual", "vida_continual")),
    ("spa_continual",        simple_baseline("spa_continual", "spa_continual")),
    ("ptta_continual",       simple_baseline("ptta_continual", "ptta_continual")),
    ("come_continual",       simple_baseline("come_continual", "come_continual")),
    ("tribe_continual",      simple_baseline("tribe_continual", "tribe_continual")),
]

if __name__ == "__main__":
    n = 0
    for folder, (dataset, data_dir) in DATASETS.items():
        outdir = os.path.join(ROOT, folder)
        os.makedirs(outdir, exist_ok=True)
        for name, fn in METHODS:
            body = fn(dataset, data_dir)
            path = os.path.join(outdir, f"{name}_full_15corr.sh")
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, 0o755)
            n += 1
    print(f"wrote {n} scripts across {len(DATASETS)} datasets x {len(METHODS)} methods")
