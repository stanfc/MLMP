#!/bin/bash
# Qualitative-figure re-run of DAF on ACDC (config = this method's existing ACDC run; only save_dir differs).
# main_continual.py keeps ckpt.pt (atomic, after every (round,condition)) -- final one = 150R weights. DO NOT delete it.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
GPU_ID=${GPU_ID:-2}
SAVE_DIR="save/ACDCDataset/qual_daf/"
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
    --save_dir save/ACDCDataset/qual_daf/ \
    --data_dir .data/ACDC/ \
    --dataset ACDCDataset \
    --workers 1 \
    --init_resize 1120 560 \
    --patch_size 224 224 --patch_stride 112 \
    --corruptions_list fog night rain snow \
    --prompt_dir prompts.yaml \
    --class_extensions --split val --subset_seed 0 --corruption_severity 5 \
    --ovss_type naclip --ovss_backbone ViT-L/14 \
    --adapt \
    --method daf_mlmp_continual \
    --batch_size 1 \
    --lr 0.000125 \
    --steps 1 \
    --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
    --prompt_integration loss \
    --alpha_cls 1.0 \
    --lamb_ent 1.0 \
    --lamb_div 2.0 \
    --lamb_cmac 0.5 \
    --alpha_safs 0.5 \
    --optimizer sgd \
    --continual_rounds 150 --seed 0

# --- notify phone when finished (ntfy.sh) ---
STATUS=$?
if [ $STATUS -eq 0 ]; then
  bash notify.sh "✅ $(basename "$0") DONE | $(tail -1 "$SAVE_DIR/results_all_rounds.txt" 2>/dev/null)" "MLMP qual ✅"
else
  bash notify.sh "❌ $(basename "$0") FAILED (exit $STATUS)" "MLMP qual ❌"
fi
