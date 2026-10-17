#!/bin/bash
# No-Adaptation baseline, full dataset, 15 corruptions.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-0}

DATASET=COCOObjectDataset
DATA_DIR=".data/coco/"
INIT_RESIZE="224 224"
CONDITIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=0

METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=1

SAVE_DIR="save/${DATASET}/no_adapt_full_15corr/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir prompts.yaml \
                        --dataset $DATASET --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 --patch_stride 112 \
                        --corruptions_list $CONDITIONS --workers $WORKERS \
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
