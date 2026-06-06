#!/bin/bash
# SAR-Continual on Nighttime Driving (CTTA, 150 rounds).
# SAM optimizer + reliable sample filter + model recovery (ICLR 2023).
# See docs/2026-05-17-sar-eata-v20-design.md.

# GPU Configuration
GPU_ID=${GPU_ID:-3}

# Dataset Configuration
DATASET=NighttimeDrivingDataset
DATA_DIR="data/NighttimeDrivingTest/"
INIT_RESIZE="1120 560"
CONDITIONS="night"
WORKERS=1

# Method Configuration
METHOD="sar_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters
BATCH_SIZE=1
LR=0.00001
STEPS=1

# SAR-specific (paper defaults rescaled for 19-class)
E_MARGIN=1.8          # sample-level mean-pixel entropy threshold
SAM_RHO=0.05
E_0=0.1               # recovery threshold (rescaled from 0.2 for 1000-class)
EMA_FACTOR=0.9
RECOVERY_WARMUP=50

CONTINUAL_ROUNDS=1200
SAVE_DIR="${SAVE_DIR:-save/${DATASET}/${METHOD}/}"

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
                        --e_margin $E_MARGIN \
                        --sam_rho $SAM_RHO \
                        --e_0 $E_0 \
                        --ema_factor $EMA_FACTOR \
                        --recovery_warmup $RECOVERY_WARMUP \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
