#!/bin/bash
# deyo_mlmp_hmgate2_repel_continual on ACDCDataset.
# GDG-PA base + REPULSION text-align: push visual features AWAY from the top-k
# most-confusable WRONG-class frozen text anchors on confident pixels. Uses the
# inter-class text geometry (non-redundant with entropy, unlike attraction).
# Compare against GDG-PA base (last 31.8) at the same LR 5e-6.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=1

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=0

METHOD="deyo_mlmp_hmgate2_repel_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

SLOPE_WINDOW=10
SLOPE_DEADZONE=0.002
LAG_GAIN=1500
BASE_RST=0.01
H_DROP_RATIO=0.9
MAXLAG_SHALLOW=6
MONITOR_INTERVAL=50

LAMBDA_REPEL=0.1
TOP_K_ALIGN=0.2
REPEL_K=3

SAVE_DIR="save/${DATASET}/${METHOD}/"

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
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --slope_window $SLOPE_WINDOW \
                        --slope_deadzone $SLOPE_DEADZONE \
                        --lag_gain $LAG_GAIN \
                        --base_rst $BASE_RST \
                        --h_drop_ratio $H_DROP_RATIO \
                        --maxlag_shallow $MAXLAG_SHALLOW \
                        --monitor_interval $MONITOR_INTERVAL \
                        --lambda_repel $LAMBDA_REPEL \
                        --top_k_align $TOP_K_ALIGN \
                        --repel_k $REPEL_K \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP ❌"
fi
