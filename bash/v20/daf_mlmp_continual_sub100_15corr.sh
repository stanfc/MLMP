#!/bin/bash
# DAF (Diversify, Anchor, Filter) on MLMP-continual, PascalVOC20Dataset --
# FULL val set (1449 imgs), ALL 15 ImageNet-C corruptions, 50 rounds
# (matches the established full+15corr precedent for V20/Cityscapes in this
# folder, e.g. deyo_mlmp_continual_full_15corr.sh).
#
# batch_size=1 here, matching every other full+15corr script in this repo
# (the sub100 DAF runs use batch_size=8, DAF's own paper setting, since 224x224
# single-patch images there tolerate it -- kept consistent with the rest of
# this folder's convention instead). At batch_size=1 the Filter/SAFS module
# is a no-op (its keep-threshold is a batch statistic), so this exercises
# Diversify + Anchor only, not the full three-part mechanism.
#
# LR + optimizer fix (2nd attempt): DAF's own README uses --optimizer sgd,
# not Adam -- Adam's per-parameter adaptive scaling makes the same nominal LR
# much more aggressive. Also, lr=1e-3 was tuned for batch_size=8, so per the
# linear scaling rule (Goyal et al. 2017) batch_size=1 should use lr/8 =
# 1.25e-4. Both fixed here (the first ACDC attempt at lr=1e-3+Adam collapsed
# to a dead state by Round 4).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-0}

DATASET=PascalVOC20Dataset
DATA_DIR=".data/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=0

METHOD="daf_mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0
LAMB_ENT=1.0
LAMB_DIV=2.0
LAMB_CMAC=0.5
ALPHA_SAFS=0.5

BATCH_SIZE=1
LR=0.000125
OPTIMIZER="sgd"
STEPS=1
CONTINUAL_ROUNDS=150
SUBSET_SIZE=100
SUBSET_SEED=0

SAVE_DIR="save/${DATASET}/daf_mlmp_continual_sub100_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
                        --lamb_ent $LAMB_ENT --lamb_div $LAMB_DIV --lamb_cmac $LAMB_CMAC \
                        --alpha_safs $ALPHA_SAFS \
                        --optimizer $OPTIMIZER \
                        --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
                        --dataset $DATASET --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 --patch_stride 112 \
                        --corruptions_list $CONDITIONS --workers $WORKERS \
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \
                        --subset_size $SUBSET_SIZE --subset_seed $SUBSET_SEED \
                        --save_dir $SAVE_DIR \
                        --class_extensions

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP full15corr ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP full15corr ❌"
fi
