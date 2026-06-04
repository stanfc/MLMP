#!/bin/bash
# TENT-Early-Continual on ACDC (CTTA, 150 rounds).
# Pure TENT pixel-wise entropy, but only the LayerNorm parameters of
# blocks [0, EARLY_CUTOFF) plus ln_pre are trained. Every other LN
# (mid + late blocks + ln_post) is hard-frozen.
#
# No DivGate, no stochastic restoration — strict ablation of the
# "drift comes from late layers" hypothesis.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="tent_early_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (match tent_continual.sh / tent_divgate_continual.sh)
BATCH_SIZE=1
LR=0.00001
STEPS=1

# Layered freeze (24-block ViT-L/14)
EARLY_CUTOFF=8        # block < 8 + ln_pre -> trainable; everything else FROZEN

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_cutoff_${EARLY_CUTOFF}/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --early_cutoff $EARLY_CUTOFF \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions \
&& python plot_results.py \
        --runs $SAVE_DIR \
              "save/${DATASET}/tent_divgate_continual_step_1" \
        --labels "tent_early_${EARLY_CUTOFF}" "tent_divgate (full LN)" \
        --out_dir "figures/${METHOD}_cutoff_${EARLY_CUTOFF}"
