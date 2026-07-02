#!/bin/bash
# MLMP-episodic on PascalVOC20-C (5 corruptions).
# Episodic: reset -> adapt -> evaluate per sample. Constant across rounds
# (re-instantiates per corruption), so 1 trial through main.py is enough.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=1

DATASET=COCOObjectDataset
DATA_DIR=".data/coco_object/"
INIT_RESIZE="224 224"
ALL_CORRUPTIONS="snow frost fog brightness contrast"
WORKERS=0

METHOD="mlmp"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# MLMP multi-level: last 18 layers of ViT-L/14
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0

BATCH_SIZE=1
LR=0.001
STEPS=1
TRIALS=1

SAVE_DIR="save/${DATASET}/mlmp_episodic_5corr_sub100/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
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
                        --corruptions_list $ALL_CORRUPTIONS \
                        --subset_size 100 \
                        --subset_seed 0 \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch-size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
