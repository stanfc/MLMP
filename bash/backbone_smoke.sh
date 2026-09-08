#!/bin/bash
# Step (ii) of the backbone campaign: NO-ADAPT smoke test on a new backbone.
#
# Purpose: before spending 22.3 h on a 150-round adapted arm, confirm that the
# candidate backbone (a) runs at all under our ACDC protocol and (b) has enough
# adaptation headroom to be worth adapting.  The ViT-L/14 reference for this
# exact command is save/ACDCDataset/No_Adaptation/ = 23.34 mIoU
# (fog 23.89 / night 22.09 / rain 23.85 / snow 23.53).
#
# Everything below is byte-identical to No_Adaptation/cmd.sh except
# --ovss_backbone, --continual_rounds (10 -> 1, results are constant anyway
# because --adapt is omitted) and --save_dir.
#
# usage: bash bash/backbone_smoke.sh <gpu> <backbone> [ovss_type] [tag]
#   e.g. bash bash/backbone_smoke.sh 4 ViT-B/16 naclip
set -u
GPU="${1:?gpu id}"
BACKBONE="${2:?backbone, e.g. ViT-B/16}"
OVSS_TYPE="${3:-naclip}"
TAG="${4:-$(echo "$OVSS_TYPE-$BACKBONE" | tr '/' '_')}"

PY=~/miniconda3/envs/MLMP/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

SAVE="save/ACDCDataset/backbone_smoke/noadapt_${TAG}/"
LOG="save/_backbone_logs/smoke_${TAG}.log"
mkdir -p "$SAVE" save/_backbone_logs

echo "[$(date '+%m-%d %H:%M')] START smoke ${OVSS_TYPE} ${BACKBONE} gpu=${GPU} -> ${SAVE}"
CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
  --save_dir "$SAVE" --data_dir data/ACDC/ --prompt_dir prompts.yaml \
  --dataset ACDCDataset --workers 4 --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 \
  --corruptions_list fog night rain snow --class_extensions \
  --ovss_type "$OVSS_TYPE" --ovss_backbone "$BACKBONE" \
  --method tent_continual \
  --batch_size 1 --lr 0.0001 --steps 1 --continual_rounds 1 --seed 0 \
  > "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE  smoke ${TAG} (exit $?)"
tail -2 "$SAVE/results_all_rounds.txt" 2>/dev/null
