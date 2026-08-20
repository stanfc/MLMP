#!/bin/bash
# OVSS gated batch-size follow-up: does deyo_mlmp_adagate_continual
# (gradnorm_scaled = shallow_cap_mode=growing_scaled) hold up at batch=8/64
# the way the flagship batch=1 run does (ACDC R150=31.92, stable), given the
# no-gate observation run showed ALL THREE batch sizes collapse (negative
# tail slope) when ungated? Same V20 dataset/corruptions/hyperparams as the
# no-gate batch ablation (bash/_v20_batch_ablation_nogate.sh) for direct
# comparison, but with the flagship's real DeYO+gate config (copied from
# save/ACDCDataset/adagate_ABmad05_ecdf_grow_scaled/cmd.sh) instead of the
# no-gate diagnostic config. Block length NOT controlled (full trainval.txt,
# no subset_size cap) per standing instruction. LR fixed at 5e-6 for both
# batch sizes -- this codebase's OVSS convention (no batch-scaled LR rule
# exists here, unlike the traditional-classification line).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP
CONDS="snow frost fog brightness contrast"
ANN_FILE="ImageSets/Segmentation/trainval.txt"

run_one() {
  local BS=$1 GPU=$2
  local SAVE="save/PascalVOC20Dataset/v20_deyo_adagate_batch_ablation/batch${BS}/"
  local LOG="save/_sweep_logs/v20_deyo_adagate_batch${BS}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  echo "[$(date +%H:%M)] START batch=${BS} gpu=${GPU} -> ${SAVE}"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n mlmp python main_continual.py \
    --adapt --method deyo_mlmp_adagate_continual \
    --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
    --dataset PascalVOC20Dataset --data_dir .data/VOC2012/ --init_resize 224 224 \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS \
    --ann_file "$ANN_FILE" --subset_seed 0 \
    --workers 1 --lr 5e-06 --steps 1 --batch_size $BS --continual_rounds 150 --seed 0 \
    --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
    --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
    --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 6 \
    --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \
    --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 \
    --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
    --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE batch=${BS} (exit $?)"
}

run_one 8 3 &
sleep 10
run_one 64 0 &
wait
echo "[$(date +%H:%M)] v20_deyo_adagate_batch8_64 ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "OVSS deyo_mlmp_adagate(gradnorm_scaled) batch=8/64 全部完成" "EXP-OVSS-GATE-BATCH" 2>/dev/null || true
