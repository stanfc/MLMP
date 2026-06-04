#!/bin/bash
# Run all 3 sweep configurations assigned to GPU 2, sequentially.
# C1: H_HIGH=1.7, H_LOW=0.9, MAX_RST=0.05
# C2: H_HIGH=1.5, H_LOW=0.7, MAX_RST=0.05
# C3: H_HIGH=1.8, H_LOW=0.6, MAX_RST=0.05
set -e
cd "$(dirname "$0")/../../.."
echo "[gpu2] starting at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/C1_gpu2_h1.7_0.9_max0.05.sh
echo "[gpu2] C1 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/C2_gpu2_h1.5_0.7_max0.05.sh
echo "[gpu2] C2 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/C3_gpu2_h1.8_0.6_max0.05.sh
echo "[gpu2] C3 done at $(date)"
echo "[gpu2] all 3 finished at $(date)"
