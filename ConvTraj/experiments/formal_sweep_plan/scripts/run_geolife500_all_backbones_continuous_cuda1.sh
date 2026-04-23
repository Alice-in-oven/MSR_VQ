#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/formal_hotfix_runs"
DATA_ROOT="/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Geolife"
GPU="${GPU:-1}"
CUDA1_LOCK_FILE="${CUDA1_LOCK_FILE:-$EXP_ROOT/script_locks/cuda1.lock}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES

mkdir -p "$EXP_ROOT/script_logs"
mkdir -p "$EXP_ROOT/script_locks"
MASTER_LOG="$EXP_ROOT/script_logs/run_geolife500_all_backbones_continuous_cuda1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[INFO] ROOT=$ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] GPU=$GPU"
echo "[INFO] MASTER_LOG=$MASTER_LOG"
echo "[INFO] CUDA1_LOCK_FILE=$CUDA1_LOCK_FILE"

release_cuda1_lock() {
  if [[ -f "$CUDA1_LOCK_FILE" ]]; then
    local lock_pid
    lock_pid="$(cat "$CUDA1_LOCK_FILE" 2>/dev/null || true)"
    if [[ "$lock_pid" == "$$" ]]; then
      rm -f "$CUDA1_LOCK_FILE"
      echo "[LOCK] released $CUDA1_LOCK_FILE"
    fi
  fi
}

wait_and_acquire_cuda1_lock() {
  while true; do
    if [[ -f "$CUDA1_LOCK_FILE" ]]; then
      local lock_pid
      lock_pid="$(cat "$CUDA1_LOCK_FILE" 2>/dev/null || true)"
      if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "[LOCK] cuda:1 busy by pid=$lock_pid, waiting 60s on $CUDA1_LOCK_FILE"
        sleep 60
        continue
      fi
      echo "[LOCK] removing stale lock $CUDA1_LOCK_FILE"
      rm -f "$CUDA1_LOCK_FILE"
    fi

    if ( set -o noclobber; echo "$$" > "$CUDA1_LOCK_FILE" ) 2>/dev/null; then
      echo "[LOCK] acquired $CUDA1_LOCK_FILE with pid=$$"
      trap release_cuda1_lock EXIT INT TERM
      break
    fi

    echo "[LOCK] failed to acquire $CUDA1_LOCK_FILE, retry in 5s"
    sleep 5
  done
}

run_job() {
  local dist="$1"
  local backbone="$2"
  local epochs=500
  local eval_epoch=500
  local run_stem="geolife_${dist}_${backbone}_continuous_e${epochs}"
  local run_name="geolife_${dist}_${backbone}_continuous_e${epochs}_cuda${GPU}"
  local run_dir="$EXP_ROOT/$run_name"
  local metrics_dir="$run_dir/metrics"
  local metrics_path="$metrics_dir/${run_name}_e${eval_epoch}.json"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"
  local batch_size network_type image_mode
  local existing_metrics=""

  existing_metrics=$(find "$EXP_ROOT" -maxdepth 3 -path "*/${run_stem}_cuda*/metrics/*.json" -type f -size +0c | head -n 1 || true)

  if [[ -n "$existing_metrics" ]]; then
    echo "[SKIP] $run_name"
    echo "[SKIP] found existing metrics: $existing_metrics"
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
      batch_size=96
      network_type="TJCNN_MC_MSR"
      image_mode="motion6"
      ;;
    simformer)
      batch_size=96
      network_type="TJCNN_MC_MSR"
      image_mode="motion6"
      ;;
    neutraj)
      batch_size=96
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
  echo "[METRICS] $metrics_path"

  local cmd=(
    python train.py
    --target_size 128
    --batch_size "$batch_size"
    --sampling_num 1
    --learning_rate 0.001
    --epoch_num "$epochs"
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
    --query_set_size 1000
    --base_set_size 9386
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

wait_and_acquire_cuda1_lock

for dist in dfd dtw haus; do
  for backbone in convtraj msr simformer neutraj; do
    run_job "$dist" "$backbone"
  done
done

echo "[ALL DONE] geolife continuous grid finished"
