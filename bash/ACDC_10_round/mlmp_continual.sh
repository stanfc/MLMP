#!/bin/bash
# MLMP-Continual on ACDC (10-round CTTA).
# Naive continual version of MLMP: identical adaptation signal (multi-prompt
# entropy + ILE) but model state is NEVER reset between samples.
#
# Expected behaviour: performance may degrade across rounds due to error
# accumulation and catastrophic forgetting — no anti-forgetting mechanism.
# Use this as an ablation baseline against mlmp_cotta.sh to isolate the
# contribution of CoTTA's EMA teacher + stochastic restoration.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# MLMP multi-level: use last 18 layers of ViT-L/14
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

# Hyperparameters
BATCH_SIZE=1
LR=0.00001      # lower than episodic TTA — updates accumulate over 10 rounds
STEPS=10        # online: 1 step per sample

# Experiment
CONTINUAL_ROUNDS=10

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
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
                        --save_dir $SAVE_DIR \
                        --class_extensions
