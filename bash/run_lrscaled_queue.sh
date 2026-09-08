#!/bin/bash
# Batch-size-MATCHED ("LR-scaled") arm of the batch ablation.
#
# Why: at fixed LR 5e-6, batch=8 takes 8x fewer Adam steps than batch=1 over the same
# stream (measured: 159 vs 1218 gate windows), so ACDC b8 drifted only +0.08 mIoU over
# 150 rounds vs b1's +2.1 -- the b1-vs-b8 plot was measuring "not enough updates", not
# "does the method survive a large batch". Adam's per-step update is ~LR regardless of
# gradient magnitude, so total movement ~ n_steps * LR; matching b1 needs 8 * 5e-6 = 4e-5.
#
# TWO changes together, because either alone is misleading:
#   LR 5e-6 -> 4e-5      restores the AMOUNT of adaptation (8x fewer Adam steps at b8)
#   monitor_interval 50 -> 6   restores the gate's RESPONSE BANDWIDTH (152 -> ~1270
#                        windows, so windows_since_min is no longer hard-capped and the
#                        SHALLOW lag stops collapsing to ~2)
# Scaling only the LR would let the model drift while the gate still samples 8x too
# coarsely -- a collapse there would be misread as "the gate fails at large batch".
#
# 4 runs: {ACDC, Cityscapes} x {gradnorm_scaled, deyo_mlmp} at batch 8.
# deyo_mlmp (base_rst=0) is REQUIRED: without the no-gate arm at the same LR there is
# nothing to attribute stability to. It ignores monitor_interval functionally
# (_stochastic_restore is never called when rst=0, so no RNG is consumed either).
# no_adapt is unaffected (never updates) and mlmp_episodic keeps its own LR 1e-3 and
# per-sample reset -- both are reused as-is for the reference lines.
#
# THIS MACHINE IS SHARED (stanfc / phanfan / iceylemon / lightning all run here).
# ACDC/Cityscapes b8 peaks at ~79GB, so a job is only placed when a GPU genuinely has
# room; we never pre-empt anyone.
#
#   nohup bash bash/run_lrscaled_queue.sh > save/_batch_ablation_logs/lrscaled_queue.log 2>&1 &
set -u
POLL="${POLL:-300}"
SETTLE="${SETTLE:-120}"
BUFFER="${BUFFER:-8000}"
# GPU5 EXCLUDED 2026-08-26: lab restricted it for two days. Re-add it after that
# lifts. GPU1 is another user's; the free-VRAM check keeps us off it when busy.
GPUS="${GPUS:-0 1 2 3 4}"
MAXPER="${MAXPER:-1}"          # b8 needs ~79GB: at most one of ours per GPU
NEED="${NEED:-88000}"
LR="${LR:-0.00004}"
MONITOR="${MONITOR:-6}"      # ceil(50/8): gate clock in IMAGE units, matching b1
TAG="${TAG:-_matched}"
mkdir -p save/_batch_ablation_logs

# ORDER MATTERS: the queue places top-down. Both gated (ours) runs go first --
# they answer the primary question "does the method still work at batch 8 once the
# amount of adaptation and the gate's clock are matched to batch 1?". The two
# base_rst=0 no-gate controls run afterwards; they answer the follow-up "and is the
# gate the reason?", which is lower priority because the b1 panels already show that
# contrast starkly (ACDC 31.92 vs 6.04, Cityscapes 24.05 vs 2.79).
# NOTE: the two gated (gradnorm_scaled) arms are ALREADY RUNNING as of 2026-08-26
# (ACDC 103/150, Cityscapes 31/150) and are deliberately NOT listed here -- this
# driver has no notion of "already done", so listing them would launch a SECOND
# process writing the same save_dir and corrupt both runs. Only the two no-gate
# controls remain to be placed.
JOBS="
acdc:deyo_mlmp
cityscapes:deyo_mlmp
"

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
mine_on(){   # count only OUR batch_ablation jobs on this GPU
  local bus; bus=$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader -i "$1" | tr -d ' ')
  local n=0
  while IFS=, read -r pid b _; do
    pid="$(echo $pid|tr -d ' ')"; b="$(echo $b|tr -d ' ')"
    [ "$b" = "$bus" ] || continue
    ps -p "$pid" -o args= 2>/dev/null | grep -q "batch_ablation" && n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader)
  echo "$n"
}

pending="$(printf '%s' "$JOBS" | grep -v '^[[:space:]]*$')"
echo "[lq] $(date '+%m-%d %H:%M') start: $(printf '%s\n' "$pending" | grep -c .) jobs, LR=$LR monitor_interval=$MONITOR TAG=$TAG, need ${NEED}MiB each"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -A RES; for g in $GPUS; do RES[$g]=0; done
  progress=0; remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    ds="${job%%:*}"; mk="${job##*:}"
    placed=0
    for g in $GPUS; do
      free=$(free_mib "$g"); slots=$(mine_on "$g")
      avail=$(( free - ${RES[$g]} ))
      if [ "$slots" -lt "$MAXPER" ] && [ "$avail" -ge $(( NEED + BUFFER )) ]; then
        echo "[lq] $(date '+%H:%M') PLACE $ds/$mk -> GPU$g (free ${free})"
        LR_OVERRIDE="$LR" MONITOR_OVERRIDE="$MONITOR" TAG="$TAG" bash bash/batch_ablation.sh "$ds" "$mk" 8 "$g" 150 &
        RES[$g]=$(( ${RES[$g]} + NEED )); placed=1; progress=1; sleep "$SETTLE"; break
      fi
    done
    [ "$placed" = "0" ] && remaining+="$job"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  left=$(printf '%s\n' "$pending" | grep -c . || true)
  [ "$progress" = "0" ] && { echo "[lq] $(date '+%H:%M') machine busy; $left pending; sleep $POLL"; sleep "$POLL"; }
done
echo "[lq] all placed; waiting"; wait
echo "[lq] $(date '+%m-%d %H:%M') LR-SCALED QUEUE COMPLETE"
