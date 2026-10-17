#!/bin/bash
# RDumb (TENT + full reset to source every 1000 images), V20. Config copied from this dataset's TENT run.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
GPU_ID=${GPU_ID:-0}
SAVE_DIR="save/PascalVOC20Dataset/rdumb_continual/"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
    --save_dir save/PascalVOC20Dataset/rdumb_continual/ \
    --data_dir .data/VOC2012/ \
    --dataset PascalVOC20Dataset \
    --workers 0 \
    --init_resize 224 224 \
    --patch_size 224 224 --patch_stride 112 \
    --corruptions_list gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression \
    --class_extensions \
    --split val \
    --subset_size 100 \
    --subset_seed 0 \
    --corruption_severity 5 \
    --ovss_type naclip --ovss_backbone ViT-L/14 \
    --adapt \
    --method rdumb_continual \
    --batch_size 1 \
    --lr 1e-05 \
    --steps 1 \
    --reset_interval 1000 \
    --continual_rounds 150 \
    --seed 0

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP rdumb ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP rdumb ❌"
fi
