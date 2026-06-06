#!/bin/bash
# TENT-DivGate-Continual on ACDC with 19 classes merged into 3 super-classes
# (flat / vertical / movable). Same hyperparameters as the 19-class baseline
# script, only DATASET and SAVE_DIR differ.
#
# Class extensions file: utils/class_extensions/acdc_3class.txt
# Each line lists the original Cityscapes class names that map to one super
# class — TextEncoder still produces per-prompt embeddings, then logits are
# max-pooled back to 3 super-class logits via the dataset's
# extentions_to_real_class_idx (see main_continual.py).
#
# Note: log(3) ≈ 1.10, so DivGate H_THRESHOLD/H_WARNING (designed for log(19)
# range) are now larger than the theoretical max H_margin. Brake mode will
# fire constantly. This is expected; we want to see what happens.

GPU_ID=0

DATASET=ACDCMerged3Dataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="tent_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

BATCH_SIZE=1
LR=0.00001
STEPS=1

H_THRESHOLD=1.8
H_WARNING=1.5
MONITOR_INTERVAL=50
CAUTIOUS_RST=0.005
BRAKE_RST=0.02

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_caut_0.005_brake_0.02/"

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
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
