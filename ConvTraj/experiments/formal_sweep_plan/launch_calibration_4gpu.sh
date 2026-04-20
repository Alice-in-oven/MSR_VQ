#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
PLAN_DIR="$ROOT/experiments/formal_sweep_plan"
CFG_DIR="$PLAN_DIR/generated_configs"
RUNNER="$PLAN_DIR/run_train_from_config.py"

python "$PLAN_DIR/generate_formal_configs.py"

launch_job () {
  local session_name="$1"
  local gpu="$2"
  local config_path="$3"
  local log_path="$4"
  mkdir -p "$(dirname "$log_path")"
  tmux new-session -d -s "$session_name" "bash -lc 'cd /data3/menghaotian && CUDA_VISIBLE_DEVICES=$gpu python $RUNNER --config $config_path > $log_path 2>&1'"
  echo "[LAUNCHED] $session_name gpu=$gpu"
  echo "  config: $config_path"
  echo "  log:    $log_path"
}

launch_job \
  "formal_calib_porto_msr_cuda0" \
  "0" \
  "$CFG_DIR/quantized_main/porto/msr/porto_dtw_msr_pdtvq_m16_k64_e600_trueadc.json" \
  "$ROOT/exp/formal_sweep_runs/porto_dtw_msr_pdtvq_m16_k64_e600_trueadc/run.log"

launch_job \
  "formal_calib_geolife_msr_cuda1" \
  "1" \
  "$CFG_DIR/quantized_main/geolife/msr/geolife_dtw_msr_pdtvq_m16_k64_e500_trueadc.json" \
  "$ROOT/exp/formal_sweep_runs/geolife_dtw_msr_pdtvq_m16_k64_e500_trueadc/run.log"

launch_job \
  "formal_calib_geolife_neu_cuda2" \
  "2" \
  "$CFG_DIR/quantized_main/geolife/neutraj/geolife_dtw_neutraj_pdtvq_m16_k64_e500_trueadc.json" \
  "$ROOT/exp/formal_sweep_runs/geolife_dtw_neutraj_pdtvq_m16_k64_e500_trueadc/run.log"

launch_job \
  "formal_calib_geolife_sim_cuda3" \
  "3" \
  "$CFG_DIR/quantized_main/geolife/simformer/geolife_dtw_simformer_pdtvq_m16_k64_e500_trueadc.json" \
  "$ROOT/exp/formal_sweep_runs/geolife_dtw_simformer_pdtvq_m16_k64_e500_trueadc/run.log"
