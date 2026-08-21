#!/bin/bash
# GPU job QUEUE for the AdaGate Phase-S follow-up (20 jobs on GPUs 3/4/5).
#
# Two axes, both pinned to lag_mode=ecdf so only ONE thing varies per job:
#   (1) trend_thr grid  {0.3, 0.7, 1.3, 1.5} x {acdc, v20, cityscapes}
#       -> completes a 0.3/0.5/0.7/1.0/1.3/1.5 curve on all three datasets
#          (0.5 and 1.0 already exist everywhere; ACDC also has 1.5 and 2.0).
#          Goal: find a threshold that is good on ALL THREE, since mad 1.0 is
#          negative on Cityscapes and mad 0.5 is currently the only all-positive arm.
#   (2) maxlag_shallow {10, 15, 20} at mad 0.5 x all three datasets
#       -> cap = min(maxlag_shallow, windows_since_min); ECDF spreads lag over 1..cap,
#          so raising the cap deepens the reachable shallow anchor WITHOUT touching the
#          trigger. Goal: stop VOC20's tail decaying below GDG-PA (mad05 last 78.60 vs
#          GDG-PA 78.99, even though its mean is +0.69 higher).
#
# Each tick queries live free VRAM per GPU and greedily places as many pending jobs as
# fit (local reservation accounting + BUFFER), so it fills GPUs as room frees and will
# NOT OOM other users' jobs. This machine is shared -- a VLLM engine took 82GB of GPU 4
# mid-launch during the first wave and OOM-killed a run.
#
# Run in background:
#   nohup bash bash/run_adagate_queue.sh > save/_sweep_logs/adagate_queue_main.log 2>&1 &
set -u
ROUNDS="${ROUNDS:-150}"
POLL="${POLL:-90}"
SETTLE="${SETTLE:-120}"
BUFFER="${BUFFER:-4000}"
GPUS="${GPUS:-3 4 5}"
MAXPER="${MAXPER:-4}"          # cap concurrent jobs per GPU (compute contention, not VRAM)

mkdir -p save/_sweep_logs

# "dataset:arm:mem_MiB"   ACDC/Cityscapes 1120x560 ~13GB measured; VOC20 224 ~3GB.
JOBS=""
for a in ABmad03_ecdf ABmad07_ecdf ABmad13_ecdf; do
  JOBS+="acdc:${a}:15000"$'\n'
done
for a in ABmad03_ecdf ABmad07_ecdf ABmad13_ecdf ABmad15_ecdf; do
  JOBS+="v20:${a}:6000"$'\n'
  JOBS+="cityscapes:${a}:15000"$'\n'
done
for a in ABmad05_ecdf_ml10 ABmad05_ecdf_ml15 ABmad05_ecdf_ml20; do
  JOBS+="acdc:${a}:15000"$'\n'
  JOBS+="v20:${a}:6000"$'\n'
  JOBS+="cityscapes:${a}:15000"$'\n'
done

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
mine_on(){ # how many of MY adagate jobs currently hold memory on GPU $1
  local bus; bus=$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader -i "$1" | tr -d ' ')
  local n=0
  while read -r pid b _; do
    pid="${pid%,}"; b="${b%,}"
    [ "$b" = "$bus" ] || continue
    ps -p "$pid" -o args= 2>/dev/null | grep -q "deyo_mlmp_adagate_continual" && n=$((n+1))
  done < <(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader)
  echo "$n"
}

pending="$(printf '%s' "$JOBS" | grep -v '^[[:space:]]*$')"
total=$(printf '%s\n' "$pending" | grep -c .)
echo "[aq] $(date '+%m-%d %H:%M') start: $total jobs, GPUs: $GPUS, rounds=$ROUNDS, max $MAXPER/GPU"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -A RES SLOTS FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); SLOTS[$g]=$(mine_on "$g"); done
  progress=0; remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    ds="${job%%:*}"; rest="${job#*:}"; arm="${rest%%:*}"; need="${rest##*:}"
    placed=0
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "${SLOTS[$g]}" -lt "$MAXPER" ] && [ "$avail" -ge $(( need + BUFFER )) ]; then
        echo "[aq] $(date '+%H:%M') PLACE $ds $arm (need ${need}+${BUFFER}) -> GPU$g (free ${FREE[$g]}, reserved ${RES[$g]}, slots ${SLOTS[$g]}/$MAXPER)"
        GPU=$g SEQ=1 ROUNDS=$ROUNDS bash bash/sweep_adagate.sh "$ds" "$arm" \
          > "save/_sweep_logs/aq_${ds}_${arm}.log" 2>&1 &
        RES[$g]=$(( ${RES[$g]} + need )); SLOTS[$g]=$(( ${SLOTS[$g]} + 1 ))
        placed=1; progress=1
        sleep "$SETTLE"
        break
      fi
    done
    [ "$placed" = "0" ] && remaining+="$job"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  left=$(printf '%s\n' "$pending" | grep -c . || true)
  [ "$progress" = "0" ] && { echo "[aq] $(date '+%H:%M') no room; $left pending; sleep $POLL"; sleep "$POLL"; }
done
echo "[aq] $(date '+%m-%d %H:%M') all jobs placed; waiting for completion"
wait
echo "[aq] $(date '+%m-%d %H:%M') QUEUE COMPLETE"
