#!/bin/bash
# batch-size ablation on OVSS/V20, no-gate (base_rst=0), block length HELD CONSTANT
# at 45 steps/corruption across all three batch sizes (subset_size = 45 * batch_size),
# using VOC's largest available split (trainval.txt, 2913 images) via --ann_file.
# Mirrors the classification-side K3/K5 test, but here block=45 is inside the
# "safe" zone (established threshold ~110-125) rather than the risky zone --
# if batch=1 still collapses at a block length that's normally safe, that's a
# clean, conservative demonstration that batch size itself is a risk factor,
# independent of block length.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="snow frost fog brightness contrast"
ANN_FILE="ImageSets/Segmentation/trainval.txt"

run_one() {
  local BS=$1 SUBSET=$2 GPU=$3
  local SAVE="save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch${BS}/"
  local LOG="save/_sweep_logs/v20_batch_ablation_batch${BS}.log"
  echo "[$(date +%H:%M)] START batch=${BS} subset=${SUBSET} gpu=${GPU} -> ${SAVE}"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n mlmp python main_continual.py \
    --adapt --method deyo_mlmp_adagate_continual \
    --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
    --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
    --ann_file "$ANN_FILE" --subset_size $SUBSET --subset_seed 0 \
    --workers 1 --lr 0.000005 --steps 1 --batch_size $BS --continual_rounds 150 --seed 0 \
    --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
    --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500 --base_rst 0.0 \
    --h_drop_ratio 0.9 --maxlag_shallow 6 --shallow_cap_mode fixed \
    --monitor_interval 5 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
    --lag_mode ecdf --lag_sat 1.5 \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE batch=${BS} (exit $?)"
}

run_one 64 2880 2 &
sleep 10
run_one 8 360 5 &
sleep 10
run_one 1 45 0 &
wait
echo "[$(date +%H:%M)] v20_batch_ablation ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "V20 batch-size ablation(no-gate, block=45 constant)全部完成" "EXP-BA" 2>/dev/null || true
