#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/formal_hotfix_runs"
DATA_ROOT="/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto"
GPU="${GPU:-3}"
CUDA3_LOCK_FILE="${CUDA3_LOCK_FILE:-$EXP_ROOT/script_locks/cuda3.lock}"

unset CUDA_VISIBLE_DEVICES

mkdir -p "$EXP_ROOT/script_logs"
mkdir -p "$EXP_ROOT/script_locks"
MASTER_LOG="$EXP_ROOT/script_logs/run_porto_msr_opq_m16k64_per_metric_cuda3_$(date +%Y%m%d_%H%M%S).log"
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

ensure_continuous_run() {
  local dist="$1"
  local source_run_name="$2"
  local image_mode="$3"
  local batch_size="$4"
  local run_dir="$EXP_ROOT/$source_run_name"
  local metrics_path="$run_dir/metrics/${source_run_name}_e600.json"
  local summary_path="$run_dir/reports/${source_run_name}_summary.json"
  local train_path="$run_dir/embeddings/${source_run_name}_continuous_train_e600.pkl"
  local query_path="$run_dir/embeddings/${source_run_name}_continuous_query_e600.pkl"
  local base_path="$run_dir/embeddings/${source_run_name}_continuous_base_e600.pkl"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"

  if [[ -f "$train_path" && -f "$query_path" && -f "$base_path" ]]; then
    echo "[SKIP] continuous embeddings already exist for $source_run_name"
    return 0
  fi

  mkdir -p "$log_dir"

  echo "[START] bootstrap continuous run: $source_run_name"
  echo "[DIST] $dist | image_mode=$image_mode | batch_size=$batch_size"
  echo "[LOG] $log_path"
  echo "[SUMMARY] $summary_path"
  echo "[METRICS] $metrics_path"

  (
    cd "$ROOT"
    stdbuf -oL -eL python -u train.py \
      --target_size 128 \
      --batch_size "$batch_size" \
      --sampling_num 1 \
      --learning_rate 0.001 \
      --epoch_num 600 \
      --dataset porto \
      --network_type TJCNN_MC_MSR \
      --channel 8 \
      --loss_type triplet \
      --cnn_feature_distance_type euclidean_sep \
      --cnntotal_feature_distance_type euclidean \
      --all_feature_distance_type euclidean \
      --sampling_type distance_sampling1 \
      --root_write_path "$run_dir" \
      --root_read_path "$DATA_ROOT" \
      --artifact_prefix "$source_run_name" \
      --grid_size 0.0 \
      --image_mode "$image_mode" \
      --embedding_backbone msr \
      --pre_quant_residual_alpha_init 0.15 \
      --disable_pre_quant_learnable_alpha \
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
      --test_epoch 600 \
      --print_epoch 10 \
      --save_model True \
      --save_model_epoch 100 \
      --eval_save_epochs 600 \
      --dist_type "$dist" \
      --device "cuda:${GPU}" \
      --train_flag "$source_run_name" \
      --head_num 1 \
      --train_set_size 3000 \
      --query_set_size 500 \
      --base_set_size 1597579
  ) 2>&1 | tee "$log_path"

  for p in "$train_path" "$query_path" "$base_path"; do
    if [[ ! -f "$p" ]]; then
      echo "[ERROR] continuous bootstrap did not produce embedding file: $p" >&2
      return 1
    fi
  done

  echo "[DONE] bootstrap continuous run: $source_run_name"
}

run_opq_job() {
  local dist="$1"
  local source_run_name="$2"
  local image_mode="$3"
  local batch_size="$4"
  local artifact_prefix="$5"
  local gt_path="$6"
  local run_dir="$EXP_ROOT/$source_run_name"
  local emb_dir="$run_dir/embeddings"
  local output_dir="$run_dir/faiss_eval/e600/opq_m16_nbits6"
  local quant_stats_path="$output_dir/${artifact_prefix}_faiss_quant_stats.json"
  local train_path="$emb_dir/${source_run_name}_continuous_train_e600.pkl"
  local query_path="$emb_dir/${source_run_name}_continuous_query_e600.pkl"
  local base_path="$emb_dir/${source_run_name}_continuous_base_e600.pkl"
  local log_dir="$output_dir/logs"
  local log_path="$log_dir/run.log"

  if [[ -s "$quant_stats_path" ]]; then
    echo "[SKIP] ${artifact_prefix}"
    echo "[SKIP] found existing quant stats: $quant_stats_path"
    return 0
  fi

  ensure_continuous_run "$dist" "$source_run_name" "$image_mode" "$batch_size"

  for p in "$train_path" "$query_path" "$base_path"; do
    if [[ ! -f "$p" ]]; then
      echo "[ERROR] missing embedding file for ${artifact_prefix}: $p" >&2
      return 1
    fi
  done

  if [[ ! -f "$gt_path" ]]; then
    echo "[ERROR] missing gt_path for ${artifact_prefix}: $gt_path" >&2
    return 1
  fi

  mkdir -p "$log_dir"

  echo "[START] ${artifact_prefix}"
  echo "[SRC_RUN] $source_run_name"
  echo "[TRAIN] $train_path"
  echo "[QUERY] $query_path"
  echo "[BASE] $base_path"
  echo "[GT] $gt_path"
  echo "[OUT] $output_dir"
  echo "[LOG] $log_path"

  (
    cd "$ROOT"
    stdbuf -oL -eL python -u experiments/faiss_pq_opq_quantize.py \
      --input_mode split \
      --train_path "$train_path" \
      --query_path "$query_path" \
      --base_path "$base_path" \
      --quantizer opq \
      --m 16 \
      --nbits 6 \
      --opq_niter 50 \
      --gt_path "$gt_path" \
      --eval_on recon \
      --output_dir "$output_dir" \
      --artifact_prefix "$artifact_prefix" \
      --save_rotated
  ) 2>&1 | tee "$log_path"

  echo "[DONE] ${artifact_prefix}"
}

wait_and_acquire_cuda3_lock

run_opq_job \
  "dtw" \
  "porto_dtw_msr_dtw8_continuous_e600_cuda3" \
  "dtw8" \
  "256" \
  "porto_dtw_msr_dtw8_cont_e600_opq_m16_nbits6" \
  "$DATA_ROOT/DTW/dtw_test_distance_matrix_result"

run_opq_job \
  "dfd" \
  "porto_dfd_msr_dfd7_continuous_e600_cuda3" \
  "dfd7" \
  "128" \
  "porto_dfd_msr_dfd7_cont_e600_opq_m16_nbits6" \
  "$DATA_ROOT/DFD/dfd_test_distance_matrix_result"

run_opq_job \
  "haus" \
  "porto_haus_msr_haus6_continuous_e600_cuda3" \
  "haus6" \
  "128" \
  "porto_haus_msr_haus6_cont_e600_opq_m16_nbits6" \
  "$DATA_ROOT/Haus/haus_test_distance_matrix_result"

echo "[ALL DONE] porto msr per-metric opq finished"
