#!/bin/bash
# MLMP episodic on ACDC across all 16 backbone rows (15-backbone table + ITACLIP).
# Run in batches of 4 (one per GPU 1-4, avoiding 0/5 which are near-full from
# other users/agents) since each episodic run is short (~4min).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlmp
cd /home/stanfc/TTA-on-OVSS/MLMP

DATASET=ACDCDataset
DATA_DIR=".data/ACDC/"
CONDITIONS="fog night rain snow"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"

run() {
  local NAME=$1 GPU=$2 OVSS_TYPE=$3 BACKBONE=$4 VISION=$5
  local SAVE="save/${DATASET}/mlmp_episodic_backbone_ablation/${NAME}/"
  local LOG="save/_sweep_logs/mlmp_episodic_${NAME}.log"
  mkdir -p "$SAVE" "$(dirname $LOG)"
  echo "[$(date +%H:%M)] START $NAME (ovss_type=$OVSS_TYPE backbone=$BACKBONE) gpu=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --adapt --method mlmp --ovss_type "$OVSS_TYPE" --ovss_backbone "$BACKBONE" \
    --vision_outputs $VISION --prompt_dir prompts.yaml --alpha_cls 1.0 \
    --dataset $DATASET --data_dir $DATA_DIR --init_resize 1120 560 \
    --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDITIONS \
    --workers 1 --lr 0.001 --steps 1 --batch-size 1 --trials 1 --seed 0 \
    --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $NAME (exit $?)"
}

batch() {
  run "$1" 1 "$2" "$3" "$OUT_VISION" &
  run "$4" 2 "$5" "$6" "$OUT_VISION" &
  run "$7" 3 "$8" "$9" "$OUT_VISION" &
  run "${10}" 4 "${11}" "${12}" "$OUT_VISION" &
  wait
}

# Batch 1
batch \
  naclip_L14         naclip            "ViT-L/14" \
  naclip_B16         naclip            "ViT-B/16" \
  naclip_B32         naclip            "ViT-B/32" \
  naclip_L14_336     naclip            "ViT-L/14@336px"

# Batch 2
batch \
  naclip_gaussonly_L14 naclip_gaussonly "ViT-L/14" \
  clearclip_kk_L14     clearclip_kk     "ViT-L/14" \
  clearclip_kk_B16     clearclip_kk     "ViT-B/16" \
  clearclip_qq_L14     clearclip_qq     "ViT-L/14"

# Batch 3
batch \
  clearclip_qq_B16   clearclip_qq      "ViT-B/16" \
  sclip_reduced_L14  sclip_reduced     "ViT-L/14" \
  sclip_reduced_B16  sclip_reduced     "ViT-B/16" \
  vv_attention_L14   vv_attention      "ViT-L/14"

# Batch 4 (maskclip_L14/B16, itaclip; sclip_uaml1 uses --vision_outputs -1 only, handled separately below)
run maskclip_L14 1 maskclip "ViT-L/14" "$OUT_VISION" &
run maskclip_B16 2 maskclip "ViT-B/16" "$OUT_VISION" &
run itaclip_L14  3 itaclip  "ViT-L/14" "$OUT_VISION" &
run sclip_uaml1_B16 4 sclip "ViT-B/16" "-1" &
wait

echo "[$(date +%H:%M)] MLMP-episodic 16-backbone ablation ALL DONE"
bash /home/stanfc/TTA-on-OVSS/MLMP/notify.sh "MLMP episodic 16個backbone全部跑完" "EXP-MLMP-EPISODIC-16BACKBONE" 2>/dev/null || true
