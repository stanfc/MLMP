#!/bin/bash
# Run a bash script and push a phone notification when it finishes.
# Usage: bash run_notify.sh bash/ACDC_10_round/deyo_mlmp_hmgate2_continual.sh
S="$1"
name=$(basename "$S")
if bash "$S"; then
  # try to grab the run's save dir + last mIoU line
  sd=$(grep -oE 'save/[^"]+' "$S" | head -1)
  last=$(tail -1 "$sd/results_all_rounds.txt" 2>/dev/null)
  bash "$(dirname "$0")/notify.sh" "✅ $name DONE  ${last:+| $last}" "MLMP ✅"
else
  bash "$(dirname "$0")/notify.sh" "❌ $name FAILED (exit $?)" "MLMP ❌"
fi
