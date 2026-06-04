#!/bin/bash
# Run all 3 sweep configurations assigned to GPU 3, sequentially.
# D1: H_HIGH=1.6, H_LOW=1.0, MAX_RST=0.05
# D2: H_HIGH=1.6, H_LOW=0.5, MAX_RST=0.05
# D3: H_HIGH=1.5, H_LOW=0.9, MAX_RST=0.05
set -e
cd "$(dirname "$0")/../../.."
echo "[gpu3] starting at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/D1_gpu3_h1.6_1.0_max0.05.sh
echo "[gpu3] D1 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/D2_gpu3_h1.6_0.5_max0.05.sh
echo "[gpu3] D2 done at $(date)"
bash bash/ACDC_10_round/sweep_sigmoid/D3_gpu3_h1.5_0.9_max0.05.sh
echo "[gpu3] D3 done at $(date)"
echo "[gpu3] all 3 finished at $(date)"
