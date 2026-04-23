#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/formal_hotfix_runs"
GPU="${GPU:-3}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES

mkdir -p "$EXP_ROOT/script_logs"
MASTER_LOG="$EXP_ROOT/script_logs/run_geolife400_porto600_continuous_cuda3_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[INFO] ROOT=$ROOT"
echo "[INFO] GPU=$GPU"
echo "[INFO] MASTER_LOG=$MASTER_LOG"

run_backbone() {
  local dataset="$1"
  local backbone="$2"
  local epochs="$3"
  local eval_epoch="$4"
  local batch_size="$5"
  local root_read_path="$6"
  local train_size="$7"
  local query_size="$8"
  local base_size="$9"

  local run_name="${dataset}_dtw_${backbone}_continuous_e${epochs}_cuda${GPU}"
  local run_dir="$EXP_ROOT/$run_name"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"

  mkdir -p "$log_dir"

  echo "[START] $run_name"
  echo "[LOG] $log_path"

  local cmd=(
    python train.py
    --target_size 128
    --batch_size "$batch_size"
    --sampling_num 1
    --learning_rate 0.001
    --epoch_num "$epochs"
    --dataset "$dataset"
    --network_type TJCNN_MC_MSR
    --channel 8
    --loss_type triplet
    --cnn_feature_distance_type euclidean_sep
    --cnntotal_feature_distance_type euclidean
    --all_feature_distance_type euclidean
    --sampling_type distance_sampling1
    --root_write_path "$run_dir"
    --root_read_path "$root_read_path"
    --artifact_prefix "$run_name"
    --grid_size 0.0
    --image_mode motion6
    --embedding_backbone "$backbone"
    --backbone_seq_max_length 200
    --simformer_num_layers 1
    --simformer_n_heads 16
    --simformer_dimfeedforward 256
    --simformer_pos_encoding fixed
    --neutraj_spatial_width 2
    --neutraj_incell True
    --pdt_m 8
    --pdt_k 16
    --pdt_vq_type dpq
    --pdt_codebook_init uniform
    --qinco_h 256
    --qinco_L 1
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
    --test_epoch "$eval_epoch"
    --print_epoch 10
    --save_model True
    --save_model_epoch 100
    --eval_save_epochs "$eval_epoch"
    --dist_type dtw
    --device "cuda:${GPU}"
    --train_flag "$run_name"
    --head_num 1
    --train_set_size "$train_size"
    --query_set_size "$query_size"
    --base_set_size "$base_size"
  )

  if [[ "$backbone" == "neutraj" ]]; then
    echo "[INFO] $run_name uses NeuTraj default SAM-GRU cell"
  fi

  (
    cd "$ROOT"
    stdbuf -oL -eL "${cmd[@]}"
  ) 2>&1 | tee "$log_path"

  echo "[DONE] $run_name"
}

run_backbone geolife msr       400 400 96  "/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/data/0_geolife" 3000 1000 9386
run_backbone geolife simformer 400 400 96  "/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/data/0_geolife" 3000 1000 9386
run_backbone geolife neutraj   400 400 96  "/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/data/0_geolife" 3000 1000 9386

run_backbone porto   msr       600 600 256 "/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto" 3000 500 1597579
run_backbone porto   simformer 600 600 256 "/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto" 3000 500 1597579
run_backbone porto   neutraj   600 600 256 "/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto" 3000 500 1597579

echo "[ALL DONE] single-GPU serial queue finished"
