#!/bin/bash
# Launch all 4 GPU pipelines in parallel (each GPU runs 3 configs sequentially).
# Logs go to logs/sweep_sigmoid/gpu{0,1,2,3}.log; you can `tail -f` them to monitor.
cd "$(dirname "$0")/../../.."
mkdir -p logs/sweep_sigmoid
nohup bash bash/ACDC_10_round/sweep_sigmoid/run_gpu0.sh > logs/sweep_sigmoid/gpu0.log 2>&1 &
GPU0_PID=$!
nohup bash bash/ACDC_10_round/sweep_sigmoid/run_gpu1.sh > logs/sweep_sigmoid/gpu1.log 2>&1 &
GPU1_PID=$!
nohup bash bash/ACDC_10_round/sweep_sigmoid/run_gpu2.sh > logs/sweep_sigmoid/gpu2.log 2>&1 &
GPU2_PID=$!
nohup bash bash/ACDC_10_round/sweep_sigmoid/run_gpu3.sh > logs/sweep_sigmoid/gpu3.log 2>&1 &
GPU3_PID=$!

echo "Launched 4 GPU pipelines (PIDs: gpu0=$GPU0_PID gpu1=$GPU1_PID gpu2=$GPU2_PID gpu3=$GPU3_PID)"
echo "Logs:"
echo "  tail -f logs/sweep_sigmoid/gpu0.log"
echo "  tail -f logs/sweep_sigmoid/gpu1.log"
echo "  tail -f logs/sweep_sigmoid/gpu2.log"
echo "  tail -f logs/sweep_sigmoid/gpu3.log"
