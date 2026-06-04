#!/bin/bash
# Run all 3 sweep configurations assigned to GPU 0, sequentially.
# A1: H_HIGH=1.6, H_LOW=0.8, MAX_RST=0.05  (winner replica)
# A2: H_HIGH=1.6, H_LOW=0.8, MAX_RST=0.03
# A3: H_HIGH=1.6, H_LOW=0.8, MAX_RST=0.08
set -e
cd "$(dirname "$0")/../../.."
echo "[gpu0] starting at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/A1_gpu0_h1.6_0.8_max0.05.sh
echo "[gpu0] A1 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/A2_gpu0_h1.6_0.8_max0.03.sh
echo "[gpu0] A2 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/A3_gpu0_h1.6_0.8_max0.08.sh
echo "[gpu0] A3 done at $(date)"
echo "[gpu0] all 3 finished at $(date)"
