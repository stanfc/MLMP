#!/bin/bash
# VRAM-aware queue for the batch-size ablation (see bash/batch_ablation.sh).
#
# Greedily places pending jobs onto GPUs with enough FREE VRAM (live query +
# local reservation accounting + BUFFER). This machine is SHARED -- GPU1 is
# another user's VLLM engine (~88GB) -- so we never assume a GPU is ours.
#
# NOTE: batch=64 is intentionally absent for acdc/cityscapes. At 1120x560 with
# patch 224 / stride 112 each image becomes 36 patches, so batch=64 = 2304
# patches; the bilinear upsample then builds a [2304,19,224,224] tensor =
# 2.19e9 elements, over INT_MAX -> hard RuntimeError, and it would need ~630GB
# anyway (batch=8 already measured at 79.2GB peak). Only v20 (1 patch/image)
# can do batch=64. See EXPERIMENT notes.
#
#   nohup bash bash/run_batch_ablation_queue.sh > save/_batch_ablation_logs/queue.log 2>&1 &
set -u
POLL="${POLL:-120}"
SETTLE="${SETTLE:-90}"
BUFFER="${BUFFER:-6000}"
GPUS="${GPUS:-0 2 3 4 5}"     # GPU1 excluded: another user's VLLM
MAXPER="${MAXPER:-3}"
mkdir -p save/_batch_ablation_logs

# "dataset:method:batch:rounds:mem_MiB"  -- biggest/longest first so they start soonest
JOBS="
acdc:gradnorm_scaled:8:150:88000
acdc:deyo_mlmp:8:150:88000
cityscapes:gradnorm_scaled:8:150:88000
cityscapes:deyo_mlmp:8:150:88000
acdc:gradnorm_scaled:1:150:16000
acdc:deyo_mlmp:1:150:16000
cityscapes:gradnorm_scaled:1:150:16000
cityscapes:deyo_mlmp:1:150:16000
v20:gradnorm_scaled:64:150:24000
v20:deyo_mlmp:64:150:24000
v20:gradnorm_scaled:8:150:10000
v20:deyo_mlmp:8:150:10000
v20:gradnorm_scaled:1:150:7000
v20:deyo_mlmp:1:150:7000
acdc:mlmp_episodic:8:1:88000
cityscapes:mlmp_episodic:8:1:88000
acdc:mlmp_episodic:1:1:16000
cityscapes:mlmp_episodic:1:1:16000
v20:mlmp_episodic:64:1:24000
v20:mlmp_episodic:8:1:10000
v20:mlmp_episodic:1:1:7000
acdc:no_adapt:1:3:14000
cityscapes:no_adapt:1:3:14000
v20:no_adapt:1:3:6000
"

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
mine_on(){
  local bus; bus=$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader -i "$1" | tr -d ' ')
  local n=0
  while IFS=, read -r pid b _; do
    pid="$(echo $pid|tr -d ' ')"; b="$(echo $b|tr -d ' ')"
    [ "$b" = "$bus" ] || continue
    ps -p "$pid" -o args= 2>/dev/null | grep -qE 'main_continual.py|main.py' && n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader)
  echo "$n"
}

pending="$(printf '%s' "$JOBS" | grep -v '^[[:space:]]*$')"
total=$(printf '%s\n' "$pending" | grep -c .)
echo "[bq] $(date '+%m-%d %H:%M') start: $total jobs on GPUs [$GPUS], max $MAXPER/GPU"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -A RES SLOTS FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); SLOTS[$g]=$(mine_on "$g"); done
  progress=0; remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    IFS=: read -r ds mk bs rounds need <<< "$job"
    placed=0
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "${SLOTS[$g]}" -lt "$MAXPER" ] && [ "$avail" -ge $(( need + BUFFER )) ]; then
        echo "[bq] $(date '+%H:%M') PLACE $ds/$mk/b$bs (need ${need}) -> GPU$g (free ${FREE[$g]}, res ${RES[$g]}, slots ${SLOTS[$g]})"
        bash bash/batch_ablation.sh "$ds" "$mk" "$bs" "$g" "$rounds" &
        RES[$g]=$(( ${RES[$g]} + need )); SLOTS[$g]=$(( ${SLOTS[$g]} + 1 ))
        placed=1; progress=1; sleep "$SETTLE"; break
      fi
    done
    [ "$placed" = "0" ] && remaining+="$job"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  left=$(printf '%s\n' "$pending" | grep -c . || true)
  [ "$progress" = "0" ] && { echo "[bq] $(date '+%H:%M') no room; $left pending; sleep $POLL"; sleep "$POLL"; }
done
echo "[bq] $(date '+%m-%d %H:%M') all placed; waiting"
wait
echo "[bq] $(date '+%m-%d %H:%M') QUEUE COMPLETE"
