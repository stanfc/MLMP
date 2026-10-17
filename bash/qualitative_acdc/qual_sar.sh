#!/bin/bash
# Qualitative-figure re-run of SAR on ACDC. NOTE: the existing ACDC SAR run used an older arg set (margin_e0_factor/reset_constant_em/sar_rho) the current code no longer accepts; this uses the current SAR args (same as the other 7 datasets), so its curve may differ from the old ACDC SAR results.
# main_continual.py keeps ckpt.pt (atomic, after every (round,condition)) -- final one = 150R weights. DO NOT delete it.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
GPU_ID=${GPU_ID:-1}
SAVE_DIR="save/ACDCDataset/qual_sar/"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
    --save_dir save/ACDCDataset/qual_sar/ \
    --data_dir .data/ACDC/ \
    --dataset ACDCDataset \
    --workers 4 \
    --init_resize 1120 560 \
    --patch_size 224 224 --patch_stride 112 \
    --corruptions_list fog night rain snow \
    --class_extensions --split val --subset_seed 0 --corruption_severity 5 \
    --ovss_type naclip --ovss_backbone ViT-L/14 \
    --adapt \
    --method sar_continual \
    --batch_size 1 \
    --lr 1e-05 \
    --steps 1 \
    --sam_rho 0.05 \
    --e_margin 1.8 \
    --e_0 0.1 \
    --ema_factor 0.9 \
    --recovery_warmup 50 \
    --continual_rounds 150 --seed 0

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP qual ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP qual ❌"
fi
