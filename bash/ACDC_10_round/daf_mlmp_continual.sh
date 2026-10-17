#!/bin/bash
# DAF (Diversify, Anchor, Filter) on MLMP-continual, ACDC (real 4-condition
# weather stream, 150 rounds -- matches adagate_ABmad05_ecdf's protocol).
#
# batch_size=1 here: batch_size=8 (DAF's own README setting) needs ~37GB by
# itself on ACDC's 1120x560/36-patch-per-image config (dual-model forward,
# 18 vision outputs) -- OOM'd twice even on the least-loaded GPU under the
# current 44-job sub100 batch's contention. Falling back to batch_size=1 with
# lr scaled down by 8x (linear scaling rule, Goyal et al. 2017) to
# approximate the same per-step update magnitude. optimizer=sgd still matches
# the paper. NOTE: at batch_size=1 the Filter/SAFS module is a no-op (its
# keep-threshold is a batch statistic), so this exercises Diversify + Anchor
# only, not the full three-part mechanism.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-0}

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="daf_mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0
LAMB_ENT=1.0
LAMB_DIV=2.0
LAMB_CMAC=0.5
ALPHA_SAFS=0.5

BATCH_SIZE=1
LR=0.000125
OPTIMIZER="sgd"
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/daf_mlmp_continual/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        --alpha_cls $ALPHA_CLS \
                        --lamb_ent $LAMB_ENT --lamb_div $LAMB_DIV --lamb_cmac $LAMB_CMAC \
                        --alpha_safs $ALPHA_SAFS \
                        --optimizer $OPTIMIZER \
                        --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
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
