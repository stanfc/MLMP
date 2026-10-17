#!/bin/bash
# DAF (Diversify, Anchor, Filter) on MLMP-continual, sub100+15corr.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-0}

DATASET=PascalContext60Dataset
DATA_DIR=".data/VOC2010/"
INIT_RESIZE="224 224"
CONDITIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=0

SUBSET_SIZE=100
SUBSET_SEED=0

METHOD="daf_mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0
LAMB_ENT=1.0
LAMB_DIV=2.0
LAMB_CMAC=0.5
ALPHA_SAFS=0.5
OPTIMIZER="sgd"

BATCH_SIZE=8
LR=0.001
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/daf_mlmp_continual_15corr_sub100/"

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
                        --subset_size $SUBSET_SIZE --subset_seed $SUBSET_SEED \
                        --lr $LR --steps $STEPS --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS --seed 0 \
                        --save_dir $SAVE_DIR \
                        --class_extensions

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP full15corr ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP full15corr ❌"
fi
