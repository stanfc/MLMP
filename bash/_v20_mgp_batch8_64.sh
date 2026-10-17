#!/bin/bash
# MGP has never been run on V20/OVSS at batch=8 or batch=64 (only batch=1,
# full_15corr). Matches the established batch-ablation config (same as
# bash/_v20_deyo_adagate_batch8_64.sh): trainval.txt, no subset cap, 5
# weather corruptions, lr=5e-6, 150R. MGP-specific hyperparams copied from
# save/PascalVOC20Dataset/mgp_deyo_mlmp_continual_full_15corr/cmd.sh
# (mgp_distill_freq=100, buffer_size=32, max_rank=32, residual_thr=0.75,
# collect_freq=40 -- all upstream defaults).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="snow frost fog brightness contrast"
ANN_FILE="ImageSets/Segmentation/trainval.txt"

run_one() {
  local BS=$1 GPU=$2
  local SAVE="save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/mgp_batch${BS}/"
  local LOG="save/_sweep_logs/v20_mgp_batch${BS}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  echo "[$(date +%H:%M)] START mgp batch=${BS} gpu=${GPU} -> ${SAVE}"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n mlmp python main_continual.py \
    --adapt --method mgp_deyo_mlmp_continual \
    --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
    --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
    --ann_file "$ANN_FILE" --subset_seed 0 \
    --workers 1 --lr 5e-06 --steps 1 --batch_size $BS --continual_rounds 150 --seed 0 \
    --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
    --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
    --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \
    --mgp_distill_freq 100 --mgp_buffer_size 32 --mgp_max_rank 32 --mgp_residual_thr 0.75 \
    --mgp_collect_freq 40 \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE mgp batch=${BS} (exit $?)"
}

run_one 8  0 &
sleep 10
run_one 64 0 &
wait
echo "[$(date +%H:%M)] v20_mgp_batch8_64 ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "OVSS MGP batch=8/64 全部完成" "EXP-OVSS-MGP-BATCH" 2>/dev/null || true
