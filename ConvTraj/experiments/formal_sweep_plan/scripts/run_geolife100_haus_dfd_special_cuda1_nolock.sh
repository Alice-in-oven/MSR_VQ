#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/formal_hotfix_runs"
DATA_ROOT="/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Geolife"
GPU="${GPU:-1}"

mkdir -p "$EXP_ROOT/script_logs"
MASTER_LOG="$EXP_ROOT/script_logs/run_geolife100_haus_dfd_special_cuda1_nolock_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[INFO] ROOT=$ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] GPU=$GPU"
echo "[INFO] MASTER_LOG=$MASTER_LOG"
echo "[INFO] lock=DISABLED"

run_job() {
  local dist="$1"
  local backbone="$2"
  local image_mode="$3"
  local run_name="$4"
  local batch_size="$5"
  local network_type
  local run_dir="$EXP_ROOT/$run_name"
  local metrics_path="$run_dir/metrics/${run_name}_e100.json"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"

  if [[ -s "$metrics_path" ]]; then
    echo "[SKIP] $run_name"
    echo "[SKIP] found existing metrics: $metrics_path"
    return 0
  fi

  mkdir -p "$log_dir"

  case "$backbone" in
    convtraj)
      network_type="TJCNN"
      ;;
    msr)
      network_type="TJCNN_MC_MSR"
      ;;
    *)
      echo "[ERROR] unknown backbone: $backbone" >&2
      exit 1
      ;;
  esac

  echo "[START] $run_name"
  echo "[LOG] $log_path"

  local cmd=(
    python train.py
    --target_size 128
    --batch_size "$batch_size"
    --sampling_num 1
    --learning_rate 0.001
    --epoch_num 100
    --dataset geolife
    --network_type "$network_type"
    --channel 8
    --loss_type triplet
    --cnn_feature_distance_type euclidean_sep
    --cnntotal_feature_distance_type euclidean
    --all_feature_distance_type euclidean
    --sampling_type distance_sampling1
    --root_write_path "$run_dir"
    --root_read_path "$DATA_ROOT"
    --artifact_prefix "$run_name"
    --grid_size 0.0
    --image_mode "$image_mode"
    --backbone_seq_max_length 200
    --simformer_num_layers 1
    --simformer_n_heads 16
    --simformer_dimfeedforward 256
    --simformer_pos_encoding fixed
    --neutraj_spatial_width 2
    --neutraj_incell True
    --disable_pdt_init_codebook
    --eval_embedding_type continuous
    --eval_search_mode decoded
    --freeze_backbone_epochs 0
    --pdt_loss_start_epoch 999999
    --pdt_loss_weight 0.0
    --loss_recipe baseline
    --consistency_weight 0.0
    --quantized_metric_weight 0.0
    --improved_qm_start_epoch 999999
    --improved_qm_warmup_epochs 0
    --improved_qm_max_weight 0.0
    --improved_pairwise_weight 0.0
    --improved_entropy_weight 0.0
    --improved_commit_weight 0.0
    --improved_uniform_weight 0.0
    --max_train_batches_per_epoch 0
    --triplet_pos_begin_pos 0
    --triplet_pos_end_pos 200
    --triplet_neg_begin_pos 0
    --triplet_neg_end_pos 200
    --train_ratio 1.0
    --mode train-directly
    --test_epoch 100
    --print_epoch 10
    --save_model True
    --save_model_epoch 100
    --eval_save_epochs 100
    --dist_type "$dist"
    --device "cuda:${GPU}"
    --train_flag "$run_name"
    --head_num 1
  )

  if [[ "$backbone" != "convtraj" ]]; then
    cmd+=(--embedding_backbone msr)
  fi

  (
    cd "$ROOT"
    stdbuf -oL -eL "${cmd[@]}"
  ) 2>&1 | tee "$log_path"

  echo "[DONE] $run_name"
}

run_job haus convtraj binary geolife_haus_convtraj_continuous_e100_cuda1 128
run_job haus msr shape5 geolife_haus_msr_shape5_continuous_e100_cuda1 96
run_job dfd convtraj binary geolife_dfd_convtraj_continuous_e100_cuda1 128
run_job dfd msr dfd7 geolife_dfd_msr_dfd7_continuous_e100_cuda1 96

echo "[ALL DONE] geolife haus/dfd special smoke finished"
