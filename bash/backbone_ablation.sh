#!/bin/bash
# ============================================================================
# Backbone-transfer ablation for the AdaGate gate  (campaign step (i))
# ============================================================================
# Question: the gate's threshold is unitless and self-calibrating.  It has
# already been shown to transfer across DATASETS (§19: firing 0.29/0.32/0.30)
# and across OBJECTIVES (§20 T-1: firing 0.292-0.310, 1.06x spread).  Does it
# also transfer across the OVSS BACKBONE?
#
# Every arm below is the flagship command
#   save/ACDCDataset/batch_ablation/gradnorm_scaled_b1/cmd.sh   (R150 = 31.92)
# with ONLY the backbone-dependent quantities changed:
#
#   --ovss_type / --ovss_backbone   the backbone itself
#   --top_block_exclude             kept at the SAME FRACTION of the encoder.
#                                   ViT-L/14 = 24 blocks -> 6 (74/100 params).
#                                   ViT-B/16 = 12 blocks -> 3 (38/50 params).
#                                   (`_is_excluded` is now depth-aware; it is
#                                    verified bit-identical on ViT-L/14.)
#   --vision_outputs                UAML depth kept at the SAME FRACTION.
#                                   24 blocks -> 18 layers (75%).
#                                   12 blocks ->  9 layers (75%).
#
# NOT changed, on purpose: lr, base_rst, trend_stat/thr, lag_mode, maxlag_shallow
# (lag is measured in WINDOWS, not blocks, so it is backbone-independent), and
# every DeYO/MLMP hyperparameter.  The whole point is that nothing is re-tuned.
#
# Each arm ships with its own no-gate control (--base_rst 0.0), because the
# published-baseline trick of §20 T-1 is not available on a new backbone.
#
# Reference row (already on disk, do not re-run):
#   NA-CLIP ViT-L/14  gate  -> save/ACDCDataset/batch_ablation/gradnorm_scaled_b1  R150 31.92
#   NA-CLIP ViT-L/14  none  -> save/ACDCDataset/batch_ablation/deyo_mlmp_b1        R150  6.04
#
# Matched no-adapt floors (7-prompt, bash/backbone_smoke.sh):
#   NA-CLIP    ViT-L/14  28.21     NA-CLIP    ViT-B/16  28.70
#   ClearCLIP  ViT-L/14  28.37     SCLIP      ViT-B/16  26.54
# (SCLIP @ ViT-L/14 scores only 15.51 zero-shot and vanilla CLIP only 3.36, so
#  neither is a fair adaptation target; both are excluded and the smoke numbers
#  are the documented reason.)
#
# SCLIP / vanilla-CLIP caveat (measured, not assumed).  MLMP's UAML averages the
# dense features of many depths.  arch='reduced' (NA-CLIP, ClearCLIP) exposes the
# ATTENTION BRANCH at each depth, which stays in the text-aligned space, so the
# average is meaningful.  arch='vanilla' (SCLIP, vanilla CLIP) exposes the FULL
# residual stream, and averaging that across 9 depths destroys the alignment:
# SCLIP@ViT-B/16 scores 6.61 mIoU with 9-layer UAML against a 26.54 zero-shot
# floor.  The `sclip_B16_uaml1_*` arms therefore use UAML depth 1 and are an
# OPTIONAL extra, not part of the main table -- they change the method, not only
# the backbone.  (Multi-layer output under arch='vanilla' used to raise
# UnboundLocalError outright; ovss/clip/model.py now defines `final_x` for
# intermediate layers too.)
#
# usage:  bash bash/backbone_ablation.sh <arm> <gpu> [rounds]
#         bash bash/backbone_ablation.sh list
# ----------------------------------------------------------------------------
set -u

# arm | ovss_type | backbone | top_block_exclude | uaml_depth | base_rst
ARMS="
clearclip_L14_gate    | clearclip | ViT-L/14 | 6 | 18 | 0.01
clearclip_L14_nogate  | clearclip | ViT-L/14 | 6 | 18 | 0.0
naclip_B16_gate       | naclip    | ViT-B/16 | 3 |  9 | 0.01
naclip_B16_nogate     | naclip    | ViT-B/16 | 3 |  9 | 0.0
clearclip_B16_gate    | clearclip | ViT-B/16 | 3 |  9 | 0.01
clearclip_B16_nogate  | clearclip | ViT-B/16 | 3 |  9 | 0.0
naclip_B32_gate       | naclip    | ViT-B/32 | 3 |  9 | 0.01
naclip_B32_nogate     | naclip    | ViT-B/32 | 3 |  9 | 0.0
maskclip_L14_gate     | maskclip  | ViT-L/14 | 6 | 18 | 0.01
maskclip_L14_nogate   | maskclip  | ViT-L/14 | 6 | 18 | 0.0
clearqq_L14_gate      | clearclip_qq | ViT-L/14 | 6 | 18 | 0.01
clearqq_L14_nogate    | clearclip_qq | ViT-L/14 | 6 | 18 | 0.0
vvclip_L14_gate       | vvclip    | ViT-L/14 | 6 | 18 | 0.01
vvclip_L14_nogate     | vvclip    | ViT-L/14 | 6 | 18 | 0.0
nonly_L14_gate        | naclip_nonly | ViT-L/14 | 6 | 18 | 0.01
nonly_L14_nogate      | naclip_nonly | ViT-L/14 | 6 | 18 | 0.0
sclipred_L14_gate     | sclip_reduced | ViT-L/14 | 6 | 18 | 0.01
sclipred_L14_allln_gate   | sclip_reduced | ViT-L/14 | 0 | 18 | 0.01 | deyo_mlmp_adagate_allln_continual
sclipred_L14_allln_nogate | sclip_reduced | ViT-L/14 | 0 | 18 | 0.0  | deyo_mlmp_adagate_allln_continual
sclipred_L14_nogate   | sclip_reduced | ViT-L/14 | 6 | 18 | 0.0
sclipred_B16_gate     | sclip_reduced | ViT-B/16 | 3 |  9 | 0.01
sclipred_B16_nogate   | sclip_reduced | ViT-B/16 | 3 |  9 | 0.0
maskclip_B16_gate     | maskclip  | ViT-B/16 | 3 |  9 | 0.01
maskclip_B16_nogate   | maskclip  | ViT-B/16 | 3 |  9 | 0.0
clearqq_B16_gate      | clearclip_qq | ViT-B/16 | 3 |  9 | 0.01
clearqq_B16_nogate    | clearclip_qq | ViT-B/16 | 3 |  9 | 0.0
naclip_L14336_gate    | naclip    | ViT-L/14@336px | 6 | 18 | 0.01
naclip_L14336_nogate  | naclip    | ViT-L/14@336px | 6 | 18 | 0.0
sclip_B16_uaml1_gate  | sclip     | ViT-B/16 | 3 |  1 | 0.01
sclip_B16_uaml1_nogate| sclip     | ViT-B/16 | 3 |  1 | 0.0
"

if [ "${1:-}" = "list" ]; then
  echo "$ARMS" | sed '/^\s*$/d'
  exit 0
fi

ARM="${1:?arm name (or 'list')}"
GPU="${2:?gpu id}"
ROUNDS="${3:-150}"

# NB: do not gsub on $1 here -- awk would rebuild $0 with OFS and destroy the
# '|' delimiters.  Trim into a temporary instead.
ROW=$(echo "$ARMS" | awk -F'|' -v a="$ARM" '{t=$1; gsub(/ /,"",t); if (t==a) print $0}')
[ -z "$ROW" ] && { echo "unknown arm '$ARM'; try: bash $0 list" >&2; exit 1; }

OVSS_TYPE=$(echo "$ROW" | awk -F'|' '{gsub(/ /,"",$2); print $2}')
BACKBONE=$( echo "$ROW" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
TBE=$(      echo "$ROW" | awk -F'|' '{gsub(/ /,"",$4); print $4}')
DEPTH=$(    echo "$ROW" | awk -F'|' '{gsub(/ /,"",$5); print $5}')
BASE_RST=$( echo "$ROW" | awk -F'|' '{gsub(/ /,"",$6); print $6}')
METHOD=$(   echo "$ROW" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
# optional 7th column; blank on every pre-existing row
: "${METHOD:=deyo_mlmp_adagate_continual}"

# -1 -2 ... -DEPTH
VO=$(seq -f '-%g' 1 "$DEPTH" | tr '\n' ' ')

PY=~/miniconda3/envs/MLMP/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

SAVE="save/ACDCDataset/backbone_ablation/${ARM}/"
LOG="save/_backbone_logs/${ARM}.log"
mkdir -p "$SAVE" save/_backbone_logs

echo "[$(date '+%m-%d %H:%M')] START ${ARM}: ${OVSS_TYPE} ${BACKBONE} tbe=${TBE} uaml=${DEPTH} base_rst=${BASE_RST} method=${METHOD} gpu=${GPU} rounds=${ROUNDS}"
CUDA_VISIBLE_DEVICES=$GPU $PY main_continual.py \
  --save_dir "$SAVE" --data_dir data/ACDC/ --prompt_dir prompts.yaml \
  --dataset ACDCDataset --workers 1 --init_resize 1120 560 \
  --patch_size 224 224 --patch_stride 112 --corruptions_list fog night rain snow \
  --phase1_rounds 0 --class_extensions --split val --subset_seed 0 \
  --corruption_severity 5 \
  --ovss_type "$OVSS_TYPE" --ovss_backbone "$BACKBONE" \
  --adapt --method "$METHOD" \
  --batch_size 1 --lr 5e-06 --steps 1 --continual_rounds "$ROUNDS" --seed 0 \
  --vision_outputs $VO \
  --deyo_margin_factor 0.5 --deyo_margin_e0_factor 0.4 --plpd_threshold 0.2 \
  --aug_type patch --patch_len 4 --reweight_ent 1 --reweight_plpd 1 \
  --top_block_exclude "$TBE" \
  --slope_window 10 --slope_deadzone 0.002 --lag_gain 1500.0 --base_rst "$BASE_RST" \
  --max_windows 2000 --h_drop_ratio 0.9 --maxlag_shallow 6 \
  --monitor_interval 50 --ln_ckpt_every 0 \
  --trend_stat mad --trend_thr 0.5 --trend_hist 50 \
  --lag_mode ecdf --lag_sat 1.5 --shallow_cap_mode growing_scaled \
  >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] DONE  ${ARM} (exit $?)"
tail -2 "$SAVE/results_all_rounds.txt" 2>/dev/null
