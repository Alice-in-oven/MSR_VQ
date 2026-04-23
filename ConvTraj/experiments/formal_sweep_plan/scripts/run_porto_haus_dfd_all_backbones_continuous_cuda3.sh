#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/formal_hotfix_runs"
DATA_ROOT="/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto"
GPU="${GPU:-3}"
CUDA3_LOCK_FILE="${CUDA3_LOCK_FILE:-$EXP_ROOT/script_locks/cuda3.lock}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES

mkdir -p "$EXP_ROOT/script_logs"
mkdir -p "$EXP_ROOT/script_locks"
MASTER_LOG="$EXP_ROOT/script_logs/run_porto_haus_dfd_all_backbones_continuous_cuda3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[INFO] ROOT=$ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] GPU=$GPU"
echo "[INFO] MASTER_LOG=$MASTER_LOG"
echo "[INFO] CUDA3_LOCK_FILE=$CUDA3_LOCK_FILE"

release_cuda3_lock() {
  if [[ -f "$CUDA3_LOCK_FILE" ]]; then
    local lock_pid
    lock_pid="$(cat "$CUDA3_LOCK_FILE" 2>/dev/null || true)"
    if [[ "$lock_pid" == "$$" ]]; then
      rm -f "$CUDA3_LOCK_FILE"
      echo "[LOCK] released $CUDA3_LOCK_FILE"
    fi
  fi
}

wait_and_acquire_cuda3_lock() {
  while true; do
    if [[ -f "$CUDA3_LOCK_FILE" ]]; then
      local lock_pid
      lock_pid="$(cat "$CUDA3_LOCK_FILE" 2>/dev/null || true)"
      if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "[LOCK] cuda:3 busy by pid=$lock_pid, waiting 60s on $CUDA3_LOCK_FILE"
        sleep 60
        continue
      fi
      echo "[LOCK] removing stale lock $CUDA3_LOCK_FILE"
      rm -f "$CUDA3_LOCK_FILE"
    fi

    if ( set -o noclobber; echo "$$" > "$CUDA3_LOCK_FILE" ) 2>/dev/null; then
      echo "[LOCK] acquired $CUDA3_LOCK_FILE with pid=$$"
      trap release_cuda3_lock EXIT INT TERM
      break
    fi

    echo "[LOCK] failed to acquire $CUDA3_LOCK_FILE, retry in 5s"
    sleep 5
  done
}

run_job() {
  local dist="$1"
  local backbone="$2"
  local epochs=600
  local eval_epoch=600
  local run_stem="porto_${dist}_${backbone}_continuous_e${epochs}"
  local run_name="porto_${dist}_${backbone}_continuous_e${epochs}_cuda${GPU}"
  local run_dir="$EXP_ROOT/$run_name"
  local summary_path="$run_dir/reports/${run_name}_summary.json"
  local metrics_path="$run_dir/metrics/${run_name}_e${eval_epoch}.json"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"
  local batch_size network_type image_mode
  local existing_summary=""
  local existing_metrics=""

  existing_summary=$(find "$EXP_ROOT" -maxdepth 3 -path "*/${run_stem}_cuda*/reports/*summary*.json" -type f -size +0c | head -n 1 || true)
  existing_metrics=$(find "$EXP_ROOT" -maxdepth 3 -path "*/${run_stem}_cuda*/metrics/${run_stem}_cuda*_e${eval_epoch}.json" -type f -size +0c | head -n 1 || true)
  if [[ -n "$existing_summary" || -n "$existing_metrics" ]]; then
    echo "[SKIP] $run_name"
    if [[ -n "$existing_summary" ]]; then
      echo "[SKIP] found existing summary: $existing_summary"
    fi
    if [[ -n "$existing_metrics" ]]; then
      echo "[SKIP] found existing metrics: $existing_metrics"
    fi
    return 0
  fi

  mkdir -p "$log_dir"

  case "$backbone" in
    convtraj)
      batch_size=128
      network_type="TJCNN"
      image_mode="binary"
      ;;
    msr)
      batch_size=128
      network_type="TJCNN_MC_MSR"
      image_mode="motion6"
      ;;
    simformer)
      batch_size=128
      network_type="TJCNN_MC_MSR"
      image_mode="motion6"
      ;;
    neutraj)
      batch_size=128
      network_type="TJCNN_MC_MSR"
      image_mode="motion6"
      ;;
    *)
      echo "[ERROR] unknown backbone: $backbone" >&2
      exit 1
      ;;
  esac

  echo "[START] $run_name"
  echo "[LOG] $log_path"
  echo "[SUMMARY] $summary_path"
  echo "[METRICS] $metrics_path"

  local cmd=(
    python train.py
    --target_size 128
    --batch_size "$batch_size"
    --sampling_num 1
    --learning_rate 0.001
    --epoch_num "$epochs"
    --dataset porto
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
    --test_epoch "$eval_epoch"
    --print_epoch 10
    --save_model True
    --save_model_epoch 100
    --eval_save_epochs "$eval_epoch"
    --dist_type "$dist"
    --device "cuda:${GPU}"
    --train_flag "$run_name"
    --head_num 1
    --train_set_size 3000
    --query_set_size 500
    --base_set_size 1597579
  )

  if [[ "$backbone" != "convtraj" ]]; then
    cmd+=(--embedding_backbone "$backbone")
  fi

  (
    cd "$ROOT"
    stdbuf -oL -eL "${cmd[@]}"
  ) 2>&1 | tee "$log_path"

  echo "[DONE] $run_name"
}

run_custom_porto_haus_multigrid3_msr() {
  local epochs=600
  local eval_epoch=600
  local run_stem="porto_haus_multigrid3_msr_continuous_e${epochs}"
  local run_name="${run_stem}_cuda${GPU}"
  local run_dir="$EXP_ROOT/$run_name"
  local summary_path="$run_dir/reports/${run_name}_summary.json"
  local metrics_path="$run_dir/metrics/${run_name}_e${eval_epoch}.json"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"
  local existing_summary=""
  local existing_metrics=""

  existing_summary=$(find "$EXP_ROOT" -maxdepth 3 -path "*/${run_stem}_cuda*/reports/*summary*.json" -type f -size +0c | head -n 1 || true)
  existing_metrics=$(find "$EXP_ROOT" -maxdepth 3 -path "*/${run_stem}_cuda*/metrics/${run_stem}_cuda*_e${eval_epoch}.json" -type f -size +0c | head -n 1 || true)
  if [[ -n "$existing_summary" || -n "$existing_metrics" ]]; then
    echo "[SKIP] $run_name"
    if [[ -n "$existing_summary" ]]; then
      echo "[SKIP] found existing summary: $existing_summary"
    fi
    if [[ -n "$existing_metrics" ]]; then
      echo "[SKIP] found existing metrics: $existing_metrics"
    fi
    return 0
  fi

  mkdir -p "$log_dir"

  echo "[START] $run_name"
  echo "[LOG] $log_path"
  echo "[SUMMARY] $summary_path"
  echo "[METRICS] $metrics_path"

  local cmd=(
    python train.py
    --target_size 128
    --batch_size 128
    --sampling_num 1
    --learning_rate 0.001
    --epoch_num "$epochs"
    --dataset porto
    --network_type TJCNN_MC_MSR
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
    --image_mode multigrid3
    --embedding_backbone msr
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
    --test_epoch "$eval_epoch"
    --print_epoch 10
    --save_model True
    --save_model_epoch 100
    --eval_save_epochs "$eval_epoch"
    --dist_type haus
    --device "cuda:${GPU}"
    --train_flag "$run_name"
    --head_num 1
    --train_set_size 3000
    --query_set_size 500
    --base_set_size 1597579
  )

  (
    cd "$ROOT"
    stdbuf -oL -eL "${cmd[@]}"
  ) 2>&1 | tee "$log_path"

  echo "[DONE] $run_name"
}

wait_and_acquire_cuda3_lock

for dist in dtw haus dfd; do
  for backbone in convtraj msr simformer neutraj; do
    run_job "$dist" "$backbone"
  done
done

run_custom_porto_haus_multigrid3_msr

echo "[ALL DONE] porto continuous grid finished"
