#!/bin/bash
# Run all 3 sweep configurations assigned to GPU 1, sequentially.
# B1: H_HIGH=1.6, H_LOW=0.8, MAX_RST=0.02
# B2: H_HIGH=1.6, H_LOW=0.8, MAX_RST=0.10
# B3: H_HIGH=1.6, H_LOW=0.8, MAX_RST=0.06
set -e
cd "$(dirname "$0")/../../.."
echo "[gpu1] starting at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/B1_gpu1_h1.6_0.8_max0.02.sh
echo "[gpu1] B1 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/B2_gpu1_h1.6_0.8_max0.1.sh
echo "[gpu1] B2 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/B3_gpu1_h1.6_0.8_max0.06.sh
echo "[gpu1] B3 done at $(date)"
echo "[gpu1] all 3 finished at $(date)"
