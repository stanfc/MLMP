#!/bin/bash
# Mem-gated GPU queue for the divreg+composite sweep (lambda_div=0.3 fixed;
# conf_ceil bracketed below the diversity-lowered mean_conf peak per dataset).
# Same placement logic as run_divreg_queue.sh: fills GPUs as VRAM frees, gated by
# memory so it won't OOM the SHOT runs (or any other) already running.
#
# Run:  nohup bash bash/run_divreg_composite_queue.sh > save/_sweep_logs/queue_divregcomp.log 2>&1 &
set -u
ROUNDS="${ROUNDS:-150}"
POLL="${POLL:-60}"; SETTLE="${SETTLE:-90}"; BUFFER="${BUFFER:-3000}"
GPUS="${GPUS:-0 1 2 3}"
export LAMBDA_DIV="${LAMBDA_DIV:-0.3}"
mkdir -p save/_sweep_logs

# "dataset:conf_ceil:base_rst:grad_mult_max:mem_MiB"
JOBS="acdc:0.62:0.02:4:14000
acdc:0.66:0.02:4:14000
acdc:0.70:0.02:4:14000
v20:0.54:0.005:4:7000
v20:0.58:0.005:4:7000
cityscapes:0.48:0.005:4:14000
cityscapes:0.52:0.005:4:14000"

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

pending="$JOBS"
echo "[q-dc] $(date '+%m-%d %H:%M') start: $(printf '%s\n' "$pending"|grep -c .) jobs, lambda_div=$LAMBDA_DIV, GPUs: $GPUS"

while [ -n "$(printf '%s' "$pending" | grep -v '^[[:space:]]*$')" ]; do
  declare -a RES FREE
  for g in $GPUS; do RES[$g]=0; FREE[$g]=$(free_mib "$g"); done
  progress=0; remaining=""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    ds="${job%%:*}"; r1="${job#*:}"; cc="${r1%%:*}"; r2="${r1#*:}"; rst="${r2%%:*}"
    r3="${r2#*:}"; gm="${r3%%:*}"; need="${r3##*:}"
    # pick the GPU with the MOST available (free-reserved) that still fits -> balances load
    best=-1; bestavail=-1
    for g in $GPUS; do
      avail=$(( ${FREE[$g]} - ${RES[$g]} ))
      if [ "$avail" -ge $(( need + BUFFER )) ] && [ "$avail" -gt "$bestavail" ]; then
        bestavail=$avail; best=$g
      fi
    done
    if [ "$best" -ge 0 ]; then
      echo "[q-dc] $(date '+%H:%M') PLACE $ds cc=$cc rst=$rst gm=$gm -> GPU$best (avail ${bestavail})"
      GPU=$best SEQ=1 ROUNDS=$ROUNDS LAMBDA_DIV=$LAMBDA_DIV bash bash/sweep_divreg_composite.sh "$ds" "$cc:$rst:$gm" \
        > "save/_sweep_logs/qdc_${ds}_cc${cc}_rst${rst}.log" 2>&1 &
      RES[$best]=$(( ${RES[$best]} + need )); progress=1
    else
      remaining+="${job}"$'\n'
    fi
  done <<< "$pending"
  pending="$(printf '%s' "$remaining" | grep -v '^[[:space:]]*$')"
  nleft=$(printf '%s\n' "$pending" | grep -c .)
  if [ "$progress" -eq 1 ]; then echo "[q-dc] $(date '+%H:%M') placed some; $nleft left; settle ${SETTLE}s"; [ "$nleft" -gt 0 ] && sleep "$SETTLE"
  else echo "[q-dc] $(date '+%H:%M') nothing fit; $nleft left; wait ${POLL}s"; sleep "$POLL"; fi
done
echo "[q-dc] $(date '+%H:%M') all launched; waiting..."; wait
echo "[q-dc] $(date '+%m-%d %H:%M') ALL DIVREG+COMPOSITE JOBS COMPLETE."
