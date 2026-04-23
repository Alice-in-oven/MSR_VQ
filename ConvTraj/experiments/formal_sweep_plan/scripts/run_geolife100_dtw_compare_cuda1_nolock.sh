#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
DATA_ROOT="/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Geolife"
GPU="${GPU:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES

cd "$ROOT"

run_eval_existing() {
  local run_dir="$1"
  local run_name="$2"
  local network_type="$3"
  local image_mode="$4"
  local batch_size="$5"
  local metrics_path="$run_dir/metrics/${run_name}.json"
  local log_path="$run_dir/logs/eval_e100.log"

  mkdir -p "$run_dir/logs"
  if [[ -s "$metrics_path" ]]; then
    echo "[SKIP] $run_name already has $metrics_path"
    return 0
  fi

  echo "[START] eval $run_name @ e100"
  python train.py \
    --target_size 128 \
    --batch_size "$batch_size" \
    --sampling_num 1 \
    --learning_rate 0.001 \
    --epoch_num 100 \
    --dataset geolife \
    --network_type "$network_type" \
    --channel 8 \
    --loss_type triplet \
    --cnn_feature_distance_type euclidean_sep \
    --cnntotal_feature_distance_type euclidean \
    --all_feature_distance_type euclidean \
    --sampling_type distance_sampling1 \
    --root_write_path "$run_dir" \
    --root_read_path "$DATA_ROOT" \
    --artifact_prefix "$run_name" \
    --grid_size 0.0 \
    --image_mode "$image_mode" \
    --embedding_backbone msr \
    --backbone_seq_max_length 200 \
    --simformer_num_layers 1 \
    --simformer_n_heads 16 \
    --simformer_dimfeedforward 256 \
    --simformer_pos_encoding fixed \
    --neutraj_spatial_width 2 \
    --neutraj_incell True \
    --disable_pdt_init_codebook \
    --eval_embedding_type continuous \
    --eval_search_mode decoded \
    --freeze_backbone_epochs 0 \
    --pdt_loss_start_epoch 999999 \
    --pdt_loss_weight 0.0 \
    --loss_recipe baseline \
    --consistency_weight 0.0 \
    --quantized_metric_weight 0.0 \
    --improved_qm_start_epoch 999999 \
    --improved_qm_warmup_epochs 0 \
    --improved_qm_max_weight 0.0 \
    --improved_pairwise_weight 0.0 \
    --improved_entropy_weight 0.0 \
    --improved_commit_weight 0.0 \
    --improved_uniform_weight 0.0 \
    --max_train_batches_per_epoch 0 \
    --triplet_pos_begin_pos 0 \
    --triplet_pos_end_pos 200 \
    --triplet_neg_begin_pos 0 \
    --triplet_neg_end_pos 200 \
    --train_ratio 1.0 \
    --mode test \
    --test_epoch 100 \
    --print_epoch 10 \
    --save_model False \
    --eval_save_epochs 100 \
    --dist_type dtw \
    --device "cuda:${GPU}" \
    --train_flag "$run_name" \
    --head_num 1 \
    2>&1 | tee "$log_path"
}

run_train_dtw8() {
  local run_dir="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/formal_hotfix_runs/geolife_dtw_msr_dtw8_continuous_e100_cuda1"
  local run_name="geolife_dtw_msr_dtw8_continuous_e100_cuda1"
  local metrics_path="$run_dir/metrics/${run_name}_e100.json"
  local log_path="$run_dir/logs/run.log"

  mkdir -p "$run_dir/logs"
  if [[ -s "$metrics_path" ]]; then
    echo "[SKIP] $run_name already has $metrics_path"
    return 0
  fi

  echo "[START] train $run_name"
  python -u train.py \
    --target_size 128 \
    --batch_size 64 \
    --sampling_num 1 \
    --learning_rate 0.001 \
    --epoch_num 100 \
    --dataset geolife \
    --network_type TJCNN_MC_MSR \
    --channel 8 \
    --loss_type triplet \
    --cnn_feature_distance_type euclidean_sep \
    --cnntotal_feature_distance_type euclidean \
    --all_feature_distance_type euclidean \
    --sampling_type distance_sampling1 \
    --root_write_path "$run_dir" \
    --root_read_path "$DATA_ROOT" \
    --artifact_prefix "$run_name" \
    --grid_size 0.0 \
    --image_mode dtw8 \
    --embedding_backbone msr \
    --backbone_seq_max_length 200 \
    --simformer_num_layers 1 \
    --simformer_n_heads 16 \
    --simformer_dimfeedforward 256 \
    --simformer_pos_encoding fixed \
    --neutraj_spatial_width 2 \
    --neutraj_incell True \
    --disable_pdt_init_codebook \
    --eval_embedding_type continuous \
    --eval_search_mode decoded \
    --freeze_backbone_epochs 0 \
    --pdt_loss_start_epoch 999999 \
    --pdt_loss_weight 0.0 \
    --loss_recipe baseline \
    --consistency_weight 0.0 \
    --quantized_metric_weight 0.0 \
    --improved_qm_start_epoch 999999 \
    --improved_qm_warmup_epochs 0 \
    --improved_qm_max_weight 0.0 \
    --improved_pairwise_weight 0.0 \
    --improved_entropy_weight 0.0 \
    --improved_commit_weight 0.0 \
    --improved_uniform_weight 0.0 \
    --max_train_batches_per_epoch 0 \
    --triplet_pos_begin_pos 0 \
    --triplet_pos_end_pos 200 \
    --triplet_neg_begin_pos 0 \
    --triplet_neg_end_pos 200 \
    --train_ratio 1.0 \
    --mode train-directly \
    --test_epoch 100 \
    --print_epoch 10 \
    --save_model True \
    --save_model_epoch 100 \
    --eval_save_epochs 100 \
    --dist_type dtw \
    --device "cuda:${GPU}" \
    --train_flag "$run_name" \
    --head_num 1 \
    2>&1 | tee "$log_path"
}

echo "[INFO] ROOT=$ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] GPU=$GPU"
echo "[INFO] no lock enabled"

run_eval_existing \
  "/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/formal_hotfix_runs/geolife_dtw_convtraj_continuous_e500_cuda1" \
  "geolife_dtw_convtraj_continuous_e500_cuda1" \
  "TJCNN" \
  "binary" \
  "128"

run_eval_existing \
  "/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/formal_hotfix_runs/geolife_dtw_msr_continuous_e500_cuda1" \
  "geolife_dtw_msr_continuous_e500_cuda1" \
  "TJCNN_MC_MSR" \
  "motion6" \
  "256"

run_train_dtw8

echo "[ALL DONE] geolife dtw convtraj/msr/dtw8-msr @ e100"
