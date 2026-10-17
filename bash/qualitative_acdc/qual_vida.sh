#!/bin/bash
# Qualitative-figure re-run of ViDA on ACDC (config = this method's existing ACDC run; only save_dir differs).
# main_continual.py keeps ckpt.pt (atomic, after every (round,condition)) -- final one = 150R weights. DO NOT delete it.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
GPU_ID=${GPU_ID:-0}
SAVE_DIR="save/ACDCDataset/qual_vida/"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
    --save_dir save/ACDCDataset/qual_vida/ \
    --data_dir .data/ACDC/ \
    --dataset ACDCDataset \
    --workers 4 \
    --init_resize 1120 560 \
    --patch_size 224 224 --patch_stride 112 \
    --corruptions_list fog night rain snow \
    --class_extensions --split val --subset_seed 0 --corruption_severity 5 \
    --ovss_type naclip --ovss_backbone ViT-L/14 \
    --adapt \
    --method vida_continual \
    --batch_size 1 \
    --lr 2e-07 \
    --steps 1 \
    --r2 16 \
    --ema 0.99 \
    --ema_vida 0.8 \
    --unc_n 10 \
    --unc_thr 0.2 \
    --continual_rounds 150 --seed 0

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP qual ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP qual ❌"
fi
