#!/bin/bash
# DPCore on ACDC, 150-round CTTA, NA-CLIP ViT-L/14.
# Visual-prompt coreset adaptation (DPCore paper):
#   - Source stats from clean Cityscapes (ACDC has no clean source)
#   - Per-batch ID/OOD decision via loss_new vs loss_raw * thr_rho
#   - ID path: load nearest coreset prompt + 1 fine-tune step + EMA update
#   - OOD path: reset to source, learn from scratch with E_OOD=$STEPS, append new entry

# CPU thread limits — match other ACDC 150-round runs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="224 224"
CONDITIONS="fog night rain snow"
WORKERS=0

# DPCore source statistics: clean Cityscapes
SRC_DATASET=CityscapesDataset
SRC_DATA_DIR=".data/cityscapes/"

# Method Configuration
METHOD="dpcore"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1"
PROMPT_DIR="prompts.yaml"

# DPCore Hyperparameters (matches the 10-round baseline that already produced
# usable results in .save/ACDCDataset/dpcore_batch_16_LR_1e-5/).
BATCH_SIZE=64
LR=1e-5
STEPS=50            # OOD inner-loop steps (E_OOD)
TEMP_TAU=1.0        # softmax temperature for coreset weighting
EMA_ALPHA=0.999     # EMA momentum for coreset stats
THR_RHO=0.9         # loss-ratio threshold for ID/OOD
PROMPT_NUM=8        # # of visual prompt tokens
VERBOSE=false

# CTTA
CONTINUAL_ROUNDS=150

# Output
SAVE_DIR="save/${DATASET}/dpcore_naclip_batch${BATCH_SIZE}_lr${LR}_steps${STEPS}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        \
                        --batch_size $BATCH_SIZE \
                        --lr $LR \
                        --steps $STEPS \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --temp_tau $TEMP_TAU \
                        --ema_alpha $EMA_ALPHA \
                        --thr_rho $THR_RHO \
                        --prompt_num $PROMPT_NUM \
                        $( [ "$VERBOSE" = "true" ] && echo "--verbose_dpcore" ) \
                        \
                        --src_dataset $SRC_DATASET \
                        --src_data_dir $SRC_DATA_DIR \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
