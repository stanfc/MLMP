#!/bin/bash
# Rerun of the no-gate V20 batch ablation (bash/_v20_batch_ablation_nogate.sh)
# WITHOUT the block-length-controlling subset cap (was subset_size=45*batch),
# using the full trainval.txt (2913 images) instead -- matching the gated
# arms' data config (bash/_v20_deyo_adagate_batch8_64.sh,
# bash/_v20_deyo_adagate_batch1_clean.sh) so the no-gate vs gated comparison
# is finally apples-to-apples at all three batch sizes. The old subset-capped
# version made batch=1 (subset=45) and batch=8 (subset=360) start from a very
# different R1 than their full-trainval gated counterparts; only batch=64
# (subset=2880 vs full 2913) happened to look aligned by coincidence.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="snow frost fog brightness contrast"
ANN_FILE="ImageSets/Segmentation/trainval.txt"

run_one() {
  local BS=$1 GPU=$2
  local SAVE="save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch${BS}_fulldata/"
  local LOG="save/_sweep_logs/v20_batch_ablation_nogate_fulldata_batch${BS}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  echo "[$(date +%H:%M)] START batch=${BS} gpu=${GPU} -> ${SAVE}"
  CUDA_VISIBLE_DEVICES=$GPU python main_continual.py \
    --adapt --method deyo_mlmp_adagate_continual \
    --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
    --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
    --ann_file "$ANN_FILE" --subset_seed 0 \
    --workers 1 --lr 0.000005 --steps 1 --batch_size $BS --continual_rounds 150 --seed 0 \
    --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
    --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 --base_rst 0.0 \
    --h_drop_ratio 0.9 --maxlag_shallow 6 --shallow_cap_mode fixed \
    --monitor_interval 5 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
    --lag_mode ecdf --lag_sat 1.5 \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE batch=${BS} (exit $?)"
}

run_one 1  1 &
sleep 10
run_one 8  1 &
sleep 10
run_one 64 1 &
wait
echo "[$(date +%H:%M)] v20_batch_ablation_nogate_fulldata ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "V20 no-gate batch ablation(全量trainval.txt重跑)全部完成" "EXP-V20-NOGATE-FULL" 2>/dev/null || true
