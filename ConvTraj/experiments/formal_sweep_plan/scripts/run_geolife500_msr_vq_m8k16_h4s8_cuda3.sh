#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/formal_hotfix_runs"
DATA_ROOT="$ROOT/data/0_geolife"
GPU="${GPU:-3}"
CUDA3_LOCK_FILE="${CUDA3_LOCK_FILE:-$EXP_ROOT/script_locks/cuda3.lock}"

EPOCHS=500
EVAL_EPOCH=500
SAVE_EPOCHS="300,400,500"
PDT_HEADS=4
PDT_STEPS=8

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES

mkdir -p "$EXP_ROOT/script_logs"
mkdir -p "$EXP_ROOT/script_locks"
MASTER_LOG="$EXP_ROOT/script_logs/run_geolife500_msr_vq_m8k16_h4s8_cuda3_$(date +%Y%m%d_%H%M%S).log"
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

image_mode_for_dist() {
  local dist="$1"
  case "$dist" in
    dtw) echo "dtw8" ;;
    dfd) echo "dfd7" ;;
    haus) echo "haus6" ;;
    *) echo "motion6" ;;
  esac
}

run_job() {
  local dist="$1"
  local image_mode
  image_mode="$(image_mode_for_dist "$dist")"
  local run_stem="geolife_${dist}_msr_${image_mode}_pdtvq_m8_k16_h${PDT_HEADS}_s${PDT_STEPS}_e${EPOCHS}"
  local run_name="${run_stem}_cuda${GPU}"
  local run_dir="$EXP_ROOT/$run_name"
  local metrics_path="$run_dir/metrics/${run_name}_e${EVAL_EPOCH}.json"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"
  local existing_metrics=""

  existing_metrics=$(find "$EXP_ROOT" -maxdepth 3 -path "*/${run_stem}_cuda*/metrics/${run_stem}_cuda*_e${EVAL_EPOCH}.json" -type f -size +0c | head -n 1 || true)
  if [[ -n "$existing_metrics" ]]; then
    echo "[SKIP] $run_name"
    echo "[SKIP] found existing metrics: $existing_metrics"
    return 0
  fi

  mkdir -p "$log_dir"

  echo "[START] $run_name"
  echo "[LOG] $log_path"
  echo "[METRICS] $metrics_path"

  local cmd=(
    python train.py
    --target_size 128
    --batch_size 256
    --sampling_num 1
    --learning_rate 0.001
    --epoch_num "$EPOCHS"
    --dataset geolife
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
    --image_mode "$image_mode"
    --embedding_backbone msr
    --backbone_seq_max_length 200
    --simformer_num_layers 1
    --simformer_n_heads 16
    --simformer_dimfeedforward 256
    --simformer_pos_encoding fixed
    --neutraj_spatial_width 2
    --neutraj_incell True
    --pdt_m 8
    --pdt_k 16
    --pdt_steps "$PDT_STEPS"
    --pdt_heads "$PDT_HEADS"
    --pdt_vq_type dpq
    --pdt_codebook_init faiss
    --eval_embedding_type both
    --eval_search_mode adc
    --freeze_backbone_epochs 0
    --pdt_loss_start_epoch 40
    --pdt_loss_weight 0.1
    --loss_recipe improved_vq
    --consistency_weight 0.1
    --quantized_metric_weight 0.3
    --improved_qm_start_epoch 80
    --improved_qm_warmup_epochs 80
    --improved_qm_max_weight 0.08
    --improved_pairwise_weight 0.05
    --improved_entropy_weight 0.02
    --improved_commit_weight 0.05
    --improved_uniform_weight 0.001
    --pre_quant_bottleneck_enabled
    --pre_quant_global_dim 48
    --pre_quant_local_dim 48
    --pre_quant_progress_dim 32
    --pre_quant_use_motion_stats
    --pre_quant_lambda_decor 0.01
    --pre_quant_lambda_stab 0.1
    --pre_quant_residual_alpha_init 0.15
    --disable_pre_quant_learnable_alpha
    --pre_quant_lr_multiplier 0.25
    --pre_quant_stab_late_epoch 100
    --pre_quant_stab_late_multiplier 4.0
    --pre_quant_refresh_start_epoch 100
    --pre_quant_refresh_period 50
    --pre_quant_refresh_end_epoch 400
    --pre_quant_raw_metric_weight 0.0
    --decoded_ste_metric_enabled
    --decoded_ste_metric_start_epoch 80
    --decoded_ste_metric_warmup_epochs 20
    --decoded_ste_metric_max_weight 0.03
    --late_finetune_start_epoch 400
    --late_finetune_main_lr_scale 0.1
    --late_finetune_pre_quant_lr_scale 0.1
    --max_train_batches_per_epoch 0
    --triplet_pos_begin_pos 0
    --triplet_pos_end_pos 200
    --triplet_neg_begin_pos 0
    --triplet_neg_end_pos 200
    --train_ratio 1.0
    --mode train-directly
    --test_epoch "$EVAL_EPOCH"
    --print_epoch 10
    --save_model True
    --save_model_epoch 100
    --eval_save_epochs "$SAVE_EPOCHS"
    --dist_type "$dist"
    --device "cuda:${GPU}"
    --train_flag "$run_name"
    --head_num 1
    --train_set_size 3000
    --query_set_size 1000
    --base_set_size 9386
  )

  (
    cd "$ROOT"
    stdbuf -oL -eL "${cmd[@]}"
  ) 2>&1 | tee "$log_path"

  echo "[DONE] $run_name"
}

wait_and_acquire_cuda3_lock

for dist in dtw haus dfd; do
  run_job "$dist"
done

echo "[ALL DONE] geolife msr_vq m8k16 h4 s8 grid finished"
