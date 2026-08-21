#!/bin/bash
# AdaGate sweep: make GDG-PA's two INERT absolute hyperparameters self-calibrating.
#
# Everything except the gate's trigger rule (A) and shallow-lag rule (B) is pinned
# to GDG-PA's best config (= save/*/hmgate2_prompt_S0_baseline), so any mIoU delta
# is attributable to A/B alone.
#
#   (A) trend_stat/trend_thr replace slope_deadzone=0.002, which measurement shows
#       degenerates to `slope > 0` (fires 46%/51%, vs P(slope>0)~50%).
#   (B) lag_mode replaces lag_gain=1500, which saturates the cap in 93%/99.7% of
#       active windows, making lag binary {0, maxlag_shallow}.
#
# Arms (ID: trend_stat trend_thr lag_mode lag_sat):
#   ctrl           abs   0.002  gain  -      <- must reproduce hmgate2 exactly
#   Amad0          mad   0.0    gain  -      A only, firing rate matched to ctrl
#   Amad05         mad   0.5    gain  -      A only, real SNR deadzone (fires ~30%)
#   Amad10         mad   1.0    gain  -      A only, strict deadzone (fires ~14%)
#   Atstat10       tstat 1.0    gain  -      A only, OLS significance test
#   Becdf          abs   0.002  ecdf  -      B only, trigger untouched
#   ABmad0_ecdf    mad   0.0    ecdf  -      A+B, firing matched
#   ABmad05_ecdf   mad   0.5    ecdf  -      A+B flagship (zero absolute constants)
#   ABmad10_ecdf   mad   1.0    ecdf  -      A+B strict
#   ABtstat10_ecdf tstat 1.0    ecdf  -      A+B, fully statistical
#   ABmad0_sat     mad   0.0    sat   1.5    A+B, saturating depth
#   ABmad05_sat    mad   0.5    sat   1.5    A+B, saturating depth + deadzone
#
# Usage (one dataset, space-sep arm IDs, CONCURRENT unless SEQ=1):
#   GPU=4 ROUNDS=150 bash bash/sweep_adagate.sh acdc "ABmad05_ecdf Becdf"
#   GPU=5 ROUNDS=3   SEQ=1 bash bash/sweep_adagate.sh acdc ctrl        # equivalence smoke
set -u
SEQ="${SEQ:-0}"
STAGGER="${STAGGER:-20}"
TAG="${TAG:-}"

DS="${1:?dataset key: acdc|v20|cityscapes}"
ARMS="${2:?space-sep arm IDs, e.g. \"ctrl ABmad05_ecdf\"}"
GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-150}"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENCV_NUM_THREADS=2

# Dataset configs are byte-identical to bash/sweep_prompt.sh so the GDG-PA
# reference run (hmgate2_prompt_S0_baseline) is directly comparable.
case "$DS" in
  acdc)
     DATASET=ACDCDataset;        DATA_DIR="data/ACDC/";        RESIZE="1120 560"
     CONDS="fog night rain snow";                SUBSET="--subset_size 50 --subset_seed 0";  SUBDIR="" ;;
  v20)
     DATASET=PascalVOC20Dataset; DATA_DIR="data/VOC/VOC2012/"; RESIZE="224 224"
     CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="v20_acdc_matched/" ;;
  cityscapes)
     DATASET=CityscapesDataset;  DATA_DIR="data/Cityscape/";   RESIZE="1120 560"
     CONDS="snow frost fog brightness contrast"; SUBSET="--subset_size 100 --subset_seed 0"; SUBDIR="" ;;
  *) echo "bad dataset key: $DS"; exit 1 ;;
esac

OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
# GDG-PA pinned config (bash/ACDC_10_round/deyo_mlmp_hmgate2_continual.sh)
SLOPE_WINDOW=10; SLOPE_DEADZONE=0.002; LAG_GAIN=1500; BASE_RST=0.01
H_DROP_RATIO=0.9; MAXLAG_SHALLOW=6; MONITOR_INTERVAL=50; TREND_HIST=50
mkdir -p save/_sweep_logs

# fields: trend_stat  trend_thr  lag_mode  lag_sat  maxlag_shallow
arm_cfg() {
  case "$1" in
    ctrl)           echo "abs   0.002 gain 1.5  6" ;;
    Amad0)          echo "mad   0.0   gain 1.5  6" ;;
    Amad05)         echo "mad   0.5   gain 1.5  6" ;;
    Amad10)         echo "mad   1.0   gain 1.5  6" ;;
    Atstat10)       echo "tstat 1.0   gain 1.5  6" ;;
    Becdf)          echo "abs   0.002 ecdf 1.5  6" ;;
    ABmad0_ecdf)    echo "mad   0.0   ecdf 1.5  6" ;;
    ABmad03_ecdf)   echo "mad   0.3   ecdf 1.5  6" ;;
    ABmad05_ecdf)   echo "mad   0.5   ecdf 1.5  6" ;;
    ABmad07_ecdf)   echo "mad   0.7   ecdf 1.5  6" ;;
    ABmad10_ecdf)   echo "mad   1.0   ecdf 1.5  6" ;;
    ABmad13_ecdf)   echo "mad   1.3   ecdf 1.5  6" ;;
    ABmad15_ecdf)   echo "mad   1.5   ecdf 1.5  6" ;;
    ABmad20_ecdf)   echo "mad   2.0   ecdf 1.5  6" ;;
    ABtstat10_ecdf) echo "tstat 1.0   ecdf 1.5  6" ;;
    ABmad0_sat)     echo "mad   0.0   sat  1.5  6" ;;
    ABmad05_sat)    echo "mad   0.5   sat  1.5  6" ;;
    # --- maxlag_shallow axis, pinned at the recommended mad 0.5 + ecdf ---
    # cap = min(maxlag_shallow, windows_since_min); ECDF then spreads lag over 1..cap,
    # so raising it deepens the reachable shallow anchor without changing the trigger.
    ABmad05_ecdf_ml10) echo "mad 0.5 ecdf 1.5 10" ;;
    ABmad05_ecdf_ml15) echo "mad 0.5 ecdf 1.5 15" ;;
    ABmad05_ecdf_ml20) echo "mad 0.5 ecdf 1.5 20" ;;
    # thr x maxlag interaction: thr 0.7 has the best mean on all three datasets but a
    # -0.75 VOC20 tail; maxlag 20 is the only arm non-degrading everywhere but has a
    # smaller mean. These cross them to see if the two gains compose.
    # deep end of the maxlag axis. windows_since_min is far larger than the old cap
    # (median 35/57/127, max 225/539 on VOC20/ACDC/Cityscapes), so 30..100 all bind:
    # 54%/17% of VOC20's shallow restores sit >30 / >100 windows from the grad-min.
    ABmad05_ecdf_ml30)  echo "mad 0.5 ecdf 1.5 30" ;;
    ABmad05_ecdf_ml50)  echo "mad 0.5 ecdf 1.5 50" ;;
    ABmad05_ecdf_ml75)  echo "mad 0.5 ecdf 1.5 75" ;;
    ABmad05_ecdf_ml100) echo "mad 0.5 ecdf 1.5 100" ;;
    ABmad07_ecdf_ml15) echo "mad 0.7 ecdf 1.5 15" ;;
    ABmad07_ecdf_ml20) echo "mad 0.7 ecdf 1.5 20" ;;
    ABmad03_ecdf_ml20) echo "mad 0.3 ecdf 1.5 20" ;;
    *) return 1 ;;
  esac
}

run_one() {
  local arm="$1"
  local cfg; cfg="$(arm_cfg "$arm")" || { echo "unknown arm: $arm"; return 1; }
  read -r TSTAT TTHR LMODE LSAT MAXLAG <<< "$cfg"
  local SAVE="save/${DATASET}/${SUBDIR}adagate_${arm}${TAG}/"
  local LOG="save/_sweep_logs/adagate_${DS}_${arm}${TAG}.log"
  echo "[$(date +%H:%M)] START $DS arm=$arm (stat=$TSTAT thr=$TTHR lag=$LMODE sat=$LSAT maxlag=$MAXLAG) gpu=$GPU rounds=$ROUNDS -> $SAVE"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n MLMP python main_continual.py \
     --adapt --method deyo_mlmp_adagate_continual \
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
     --dataset "$DATASET" --data_dir "$DATA_DIR" --init_resize $RESIZE \
     --patch_size 224 224 --patch_stride 112 --corruptions_list $CONDS $SUBSET \
     --workers 1 --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds "$ROUNDS" --seed 0 \
     --vision_outputs $OUT_VISION \
     --slope_window $SLOPE_WINDOW --slope_deadzone $SLOPE_DEADZONE --lag_gain $LAG_GAIN \
     --base_rst $BASE_RST --h_drop_ratio $H_DROP_RATIO --maxlag_shallow "$MAXLAG" \
     --monitor_interval $MONITOR_INTERVAL \
     --trend_stat "$TSTAT" --trend_thr "$TTHR" --trend_hist $TREND_HIST \
     --lag_mode "$LMODE" --lag_sat "$LSAT" \
     --save_dir "$SAVE" --class_extensions > "$LOG" 2>&1
  echo "[$(date +%H:%M)] DONE  $DS arm=$arm (exit $?)"
}

for arm in $ARMS; do
  if [ "$SEQ" = "1" ]; then run_one "$arm"; else run_one "$arm" & sleep "$STAGGER"; fi
done
wait
echo "[$(date +%H:%M)] $DS adagate sweep COMPLETE."
