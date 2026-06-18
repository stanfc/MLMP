#!/bin/bash
# deyo_mlmp_composite_gate_continual on CityscapesDataset (5corr sub100).
# Composite gate: mean_conf TRIGGER + grad_norm DEPTH.
# conf_ceil from observed mean_conf at the mIoU peak (Cityscapes peak conf 0.707 @R15).
# See docs/2026-06-18-contribution.md §7.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=0

DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1120 560"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=0

METHOD="deyo_mlmp_composite_gate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

# conf_ceil on the GATE-INTERNAL conf scale. Prev 0.72 triggered ~R24 (near peak
# R15, 8 windows) — kept as is; it fired sensibly last run.
CONF_CEIL=0.72
BASE_RST=0.005
GRAD_MULT_MAX=4.0
MONITOR_INTERVAL=50

SAVE_DIR="save/${DATASET}/${METHOD}_5corr_sub100/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        --subset_size 100 \
                        --subset_seed 0 \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --conf_ceil $CONF_CEIL \
                        --base_rst $BASE_RST \
                        --grad_mult_max $GRAD_MULT_MAX \
                        --monitor_interval $MONITOR_INTERVAL \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
