#!/bin/bash
# Plug-and-play evidence: the SAME AdaGate on DIFFERENT base objectives.
#
# `--top_block_exclude 0` makes every visual LayerNorm trainable (100 params on
# NA-CLIP ViT-L/14), which is exactly what mlmp_continual / cma_continual /
# tent_continual do. So the ALREADY-PUBLISHED no-gate run of each base method is a
# valid control and no second arm has to be run:
#
#   base objective  no-gate control (existing, published method)      R150
#   MLMP-continual  save/ACDCDataset/mlmp_continual_round_150_step_1   1.53
#   DELTA           save/ACDCDataset/delta_continual_alpha_1.0_mom_0.9 12.24
#   SAR             save/ACDCDataset/sar_continual_weather             25.31 (oscillates)
#   TENT            save/ACDCDataset/tent_continual_Round150_lr_1e-5   7.90
#
# NOTE on SAR: unlike the others this REPLACES a mechanism rather than adding one --
# SAR's own hard recovery (Filter C) is disabled and AdaGate's graded restore takes
# its place, following adapt/sar_divgate_continual.py. Say so in the paper.
#
# (The earlier bash/plugplay_tent_adagate.sh used top_block_exclude=6 and therefore
# needed its own internal --base_rst 0.0 control; that pair is already finished.)
#
# LR 1e-5 = the convention of all three published continual baselines.
#
# usage: bash bash/plugplay_adagate.sh <mlmp|delta|sar|cma> <gpu> [rounds]
set -u
MK="${1:?base objective: tent|mlmp|delta|sar|cma}"
GPU="${2:?gpu id}"
ROUNDS="${3:-150}"
PY=~/miniconda3/envs/MLMP/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

# OUT_VISION: the published MLMP baseline fuses 18 layers; DELTA / SAR / CMA are
# single-level, exactly as their own implementations forward the model.
case "$MK" in
  mlmp)  METHOD=mlmp_adagate_continual
         OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18" ;;
  tent)  METHOD=tent_adagate_continual;  OUT_VISION="-1" ;;
  delta) METHOD=delta_adagate_continual; OUT_VISION="-1" ;;
  sar)   METHOD=sar_adagate_continual;   OUT_VISION="-1" ;;
  cma)   METHOD=cma_adagate_continual;   OUT_VISION="-1" ;;
  *) echo "bad base objective: $MK"; exit 1 ;;
esac

SAVE="save/ACDCDataset/plugplay/${MK}_adagate_b1/"
LOG="save/_plugplay_logs/acdc_${MK}_adagate.log"
mkdir -p "$SAVE" save/_plugplay_logs

echo "[$(date '+%m-%d %H:%M')] START $METHOD gpu=$GPU rounds=$ROUNDS -> $SAVE"
CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
  --adapt --method "$METHOD" \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 --top_block_exclude 0 \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst 0.01 \
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 \
  --monitor_interval 50 --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled \
  --lr 0.00001 --steps 1 \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset ACDCDataset --data_dir data/ACDC/ --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list fog night rain snow \
  --workers 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
  --vision_outputs $OUT_VISION \
  --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE  $METHOD (exit $?)"
