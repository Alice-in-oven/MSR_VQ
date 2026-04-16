#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj"
DATA_DIR="$ROOT/data"
SCRIPT="$ROOT/experiments/hybrid700_codebook_worker.sh"
SUMMARY_PATH="$DATA_DIR/hybrid700_codebook_sweep_summary.json"

mkdir -p "$DATA_DIR"

echo "[SWEEP] summary will be written to: $SUMMARY_PATH"
echo "[SWEEP] worker script: $SCRIPT"
echo "[SWEEP] recipe policy: m<8 -> improved_vq ; m>=8 -> baseline"
echo "[SWEEP] epoch_num: 700"
echo "[SWEEP] rerank_L: 100"

bash "$SCRIPT" cuda:1 4:16 8:16 16:16 > "$DATA_DIR/hybrid700_worker_cuda1.log" 2>&1 &
PID1=$!
bash "$SCRIPT" cuda:0 4:64 8:64 16:64 > "$DATA_DIR/hybrid700_worker_cuda0.log" 2>&1 &
PID0=$!
bash "$SCRIPT" cuda:2 4:256 8:256 16:256 > "$DATA_DIR/hybrid700_worker_cuda2.log" 2>&1 &
PID2=$!

echo "[SWEEP] started cuda:1 worker pid=$PID1 log=$DATA_DIR/hybrid700_worker_cuda1.log"
echo "[SWEEP] started cuda:0 worker pid=$PID0 log=$DATA_DIR/hybrid700_worker_cuda0.log"
echo "[SWEEP] started cuda:2 worker pid=$PID2 log=$DATA_DIR/hybrid700_worker_cuda2.log"
echo "[SWEEP] tail logs with:"
echo "  tail -f $DATA_DIR/hybrid700_worker_cuda1.log"
echo "  tail -f $DATA_DIR/hybrid700_worker_cuda0.log"
echo "  tail -f $DATA_DIR/hybrid700_worker_cuda2.log"
