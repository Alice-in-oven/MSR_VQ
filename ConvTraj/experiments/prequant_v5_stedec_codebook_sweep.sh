#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
SCRIPT="$ROOT/experiments/prequant_v5_stedec_codebook_worker.sh"
DATA_DIR="$ROOT/data"
SUMMARY_PATH="$DATA_DIR/prequant_v5_stedec_codebook_sweep_summary.json"

mkdir -p "$DATA_DIR"

echo "[SWEEP] launcher: $0"
echo "[SWEEP] worker: $SCRIPT"
echo "[SWEEP] summary: $SUMMARY_PATH"
echo "[SWEEP] method: prequant_v5_stedec"
echo "[SWEEP] grid: M in {4,8,16}, K in {16,64,256}"
echo "[SWEEP] epochs: 500"
echo "[SWEEP] eval epochs: 100,200,300,400,500"

nohup bash "$SCRIPT" cuda:0 4:16 8:16 16:16 > "$DATA_DIR/prequant_v5_stedec_worker_cuda0.log" 2>&1 &
PID0=$!
nohup bash "$SCRIPT" cuda:1 4:64 8:64 16:64 > "$DATA_DIR/prequant_v5_stedec_worker_cuda1.log" 2>&1 &
PID1=$!
nohup bash "$SCRIPT" cuda:2 4:256 8:256 16:256 > "$DATA_DIR/prequant_v5_stedec_worker_cuda2.log" 2>&1 &
PID2=$!

echo "[SWEEP] started cuda:0 worker pid=$PID0 log=$DATA_DIR/prequant_v5_stedec_worker_cuda0.log"
echo "[SWEEP] started cuda:1 worker pid=$PID1 log=$DATA_DIR/prequant_v5_stedec_worker_cuda1.log"
echo "[SWEEP] started cuda:2 worker pid=$PID2 log=$DATA_DIR/prequant_v5_stedec_worker_cuda2.log"
echo "[SWEEP] progress commands:"
echo "  tail -f $DATA_DIR/prequant_v5_stedec_worker_cuda0.log"
echo "  tail -f $DATA_DIR/prequant_v5_stedec_worker_cuda1.log"
echo "  tail -f $DATA_DIR/prequant_v5_stedec_worker_cuda2.log"
echo "  cat $SUMMARY_PATH"
