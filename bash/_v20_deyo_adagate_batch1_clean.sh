#!/bin/bash
# Clean batch=1 arm for the gated OVSS batch ablation -- same trainval.txt,
# no subset cap, same flagship hyperparams as batch=8/64
# (bash/_v20_deyo_adagate_batch8_64.sh), so all three batch sizes are
# directly comparable (the previous "batch=1" line in
# figures/v20_deyo_adagate_batch_ablation.png was borrowed from a different
# experiment -- --split val, subset_size=100 -- not a clean match).
# Block length at batch=1 on full trainval.txt (2913 images) is ~2913
# steps/round/corruption -- much heavier than batch=8's ~364 or batch=64's
# ~45, so this will take substantially longer than either of those did.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="snow frost fog brightness contrast"
ANN_FILE="ImageSets/Segmentation/trainval.txt"

SAVE="save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch1_clean/"
LOG="save/_sweep_logs/v20_deyo_adagate_batch1_clean.log"
mkdir -p "$SAVE" "$(dirname $LOG)"
echo "[$(date +%H:%M)] START batch=1 (clean, trainval.txt) -> ${SAVE}"
CUDA_VISIBLE_DEVICES=1 conda run -n mlmp python main_continual.py \
  --adapt --method deyo_mlmp_adagate_continual \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
  --ann_file "$ANN_FILE" --subset_seed 0 \
  --workers 1 --lr 5e-06 --steps 1 --batch_size 1 --continual_rounds 150 --seed 0 \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled \
  --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
echo "[$(date +%H:%M)] DONE batch=1 clean (exit $?)"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "OVSS deyo_mlmp_adagate batch=1(乾淨版,trainval.txt)跑完了" "EXP-OVSS-GATE-B1-CLEAN" 2>/dev/null || true
