#!/bin/bash
# GPU job QUEUE for the GDG-PA PROMPT-SET sweep (9 sets x {ACDC, VOC20} = 18 jobs).
# Each tick queries live free VRAM per GPU and greedily places as many pending jobs
# as fit (with local reservation accounting + a BUFFER), so it fills GPUs as room
# frees and will NOT OOM-crash the user's existing composite/hmgate2 jobs.
#
# Each job -> bash/sweep_prompt.sh (SEQ=1, pinned GPU), keeping per-dataset config DRY.
#
# Run in background:
#   nohup bash bash/run_prompt_queue.sh > save/_sweep_logs/prompt_queue_main.log 2>&1 &
set -u
ROUNDS="${ROUNDS:-150}"
POLL="${POLL:-60}"
SETTLE="${SETTLE:-90}"
BUFFER="${BUFFER:-3000}"
GPUS="${GPUS:-0 1 2 3}"

mkdir -p save/_sweep_logs

# "dataset:promptID:mem_MiB"  (ACDC full-res ~13GB base; N=12 set costs more -> 20000
#  conservative. VOC20 224 ~6-8GB.)
PROMPTS="S0_baseline S1_minimal S2_photographic S3_street S4_weather S5_street_weather S6_rich S7_generic S8_artistic"
JOBS=""
for p in $PROMPTS; do
  JOBS+="acdc:${p}:20000"$'\n'
  JOBS+="v20:${p}:10000"$'\n'
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

pending="$(printf '%s' "$JOBS" | grep -v '^[[:space:]]*$')"
echo "[pq] $(date '+%m-%d %H:%M') start: $(printf '%s\n' "$pending" | grep -c .) jobs, GPUs: $GPUS, rounds=$ROUNDS"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -a RES FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); done
  progress=0
  remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    ds="${job%%:*}"; rest="${job#*:}"; pid="${rest%%:*}"; need="${rest##*:}"
    placed=0
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "$avail" -ge $(( need + BUFFER )) ]; then
        echo "[pq] $(date '+%H:%M') PLACE $ds prompt=$pid (need ${need}+${BUFFER}) -> GPU$g (free ${FREE[$g]}, reserved ${RES[$g]})"
        GPU=$g SEQ=1 ROUNDS=$ROUNDS bash bash/sweep_prompt.sh "$ds" "$pid" \
          > "save/_sweep_logs/pq_${ds}_${pid}.log" 2>&1 &
        RES[$g]=$(( ${RES[$g]} + need ))
        placed=1; progress=1
        break
      fi
    done
    [ "$placed" -eq 0 ] && remaining+="${job}"$'\n'
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  nleft=$(printf '%s\n' "$pending" | grep -c .)
  if [ "$progress" -eq 1 ]; then
    echo "[pq] $(date '+%H:%M') placed some; $nleft left; settling ${SETTLE}s"
    [ "$nleft" -gt 0 ] && sleep "$SETTLE"
  else
    echo "[pq] $(date '+%H:%M') nothing fit; $nleft left; waiting ${POLL}s"
    sleep "$POLL"
  fi
done

echo "[pq] $(date '+%H:%M') all jobs launched; waiting for training to finish..."
wait
echo "[pq] $(date '+%m-%d %H:%M') ALL PROMPT-SWEEP JOBS COMPLETE."
