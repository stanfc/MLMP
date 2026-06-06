#!/bin/bash
# TENT-Continual on ACDC + synthetic corruption overlay (one-to-one pairing).
# Each ACDC condition gets its own corruption overlaid on top:
#   fog   x gaussian_noise
#   night x shot_noise
#   rain  x defocus_blur
#   snow  x jpeg_compression
# Severity 5 (matches cityscapes/V20 default).
#
# Goal: probe whether adding synthetic corruption to ACDC's real weather
# condition kills the TENT peak, isolating "real condition shift" vs
# "synthetic corruption" as the cause of ACDC's adaptation advantage.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=0

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
OVERLAYS="gaussian_noise shot_noise defocus_blur jpeg_compression"
SEVERITY=5
WORKERS=1

METHOD="tent_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_overlay_sev${SEVERITY}/"

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
                        --acdc_overlay_corruptions $OVERLAYS \
                        --corruption_severity $SEVERITY \
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
