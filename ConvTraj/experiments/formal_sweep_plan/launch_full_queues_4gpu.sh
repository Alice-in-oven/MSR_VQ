#!/usr/bin/env bash
set -euo pipefail
ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
PLAN_DIR="$ROOT/experiments/formal_sweep_plan"
RUNNER="$PLAN_DIR/run_train_from_config.py"
QUEUE_WORKER="$PLAN_DIR/run_queue_from_json.py"
QUEUE_DIR="$PLAN_DIR/queue_lists"
run_queue () {
  local session_name="$1"
  local visible_gpu="$2"
  local queue_json="$3"
  tmux new-session -d -s "$session_name" "bash -lc 'CUDA_VISIBLE_DEVICES=$visible_gpu python \"$QUEUE_WORKER\" --queue \"$queue_json\" --runner \"$RUNNER\"'"
}
run_queue formal_full_cuda0 0 "$QUEUE_DIR/cuda0_queue.json"
run_queue formal_full_cuda1 1 "$QUEUE_DIR/cuda1_queue.json"
run_queue formal_full_cuda2 2 "$QUEUE_DIR/cuda2_queue.json"
run_queue formal_full_cuda3 3 "$QUEUE_DIR/cuda3_queue.json"
