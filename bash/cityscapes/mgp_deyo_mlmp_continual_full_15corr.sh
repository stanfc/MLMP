#!/bin/bash
# MGP on Cityscapes 5corr sub100 (GDG-PA's full_15corr there only reached R50)
# MGP (anonymous tta-373C, methods/MGP/proposal.py) ported onto DeYO+MLMP:
# after backward, each LN gradient is projected onto the orthogonal complement of
# a subspace distilled by SVD from buffered past gradients (anti-saturation).
# Same base loss as our GDG-PA runs -> the only difference is the gradient projection.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=3

DATASET=CityscapesDataset
DATA_DIR=".data/cityscapes/"
INIT_RESIZE="1120 560"
CONDITIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=0

METHOD="mgp_deyo_mlmp_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=150

# MGP hyperparameters (upstream conf.py defaults)
MGP_DISTILL_FREQ=100
MGP_BUFFER_SIZE=32
MGP_MAX_RANK=32
MGP_RESIDUAL_THR=0.75
MGP_COLLECT_FREQ=40

SAVE_DIR="save/${DATASET}/${METHOD}_full_15corr/"

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
                        --mgp_distill_freq $MGP_DISTILL_FREQ \
                        --mgp_buffer_size $MGP_BUFFER_SIZE \
                        --mgp_max_rank $MGP_MAX_RANK \
                        --mgp_residual_thr $MGP_RESIDUAL_THR \
                        --mgp_collect_freq $MGP_COLLECT_FREQ \
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
