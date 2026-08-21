#!/bin/bash
# CoTTA on ACDC — 150-round CTTA, to sit alongside the other 150-round baselines
# (TENT / DeYO / EATA / SAR / RoTTA / CLIPArTT) in the intro figure.
#
# Differences from bash/ACDC_10_round/cotta.sh:
#   * CONTINUAL_ROUNDS 10 -> 150
#   * STEPS 10 -> 1        (all other 150R baselines use step=1; the existing
#                           10-round result we plot is cotta_step_1 = 23.41 flat)
#   * DATA_DIR "data/ACDC/" -> ".data/ACDC/"   <-- the original is MISSING the
#                           leading dot and would crash with FileNotFoundError
#
# CoTTA three mechanisms:
#   1. EMA teacher (mt=0.999) for stable pseudo-labels
#   2. Augmentation-averaged pseudo-labels when anchor confidence < ap
#   3. Stochastic restoration (rst=0.01) to prevent catastrophic forgetting
#
# NOTE: aug_n=32 means up to 32 extra augmented teacher forwards per sample —
# this run is SLOW. Lower AUG_N (e.g. 8) if 150 rounds is too expensive.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

# CoTTA keeps THREE full ViT-L/14 copies in memory (student + EMA teacher +
# frozen anchor) plus backward activations for all 36 patches -> it needs a lot
# of free VRAM. Do NOT put it on a GPU that already has a big run on it.
GPU_ID=0

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=0

METHOD="cotta"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# CoTTA hyperparameters (original paper values)
MT=0.999        # EMA smoothing factor for teacher
RST=0.01        # stochastic restoration probability
AP=0.92         # anchor confidence threshold
AUG_N=32        # number of augmented teacher views

OUT_VISION="-1"   # last layer only (CoTTA spirit — no multi-level fusion)

BATCH_SIZE=1
LR=0.00001
STEPS=1
CONTINUAL_ROUNDS=150

SAVE_DIR="save/${DATASET}/cotta_continual_150R/"

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --vision_outputs $OUT_VISION \
                        --mt $MT \
                        --rst $RST \
                        --ap $AP \
                        --aug_n $AUG_N \
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

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP ❌"
fi
