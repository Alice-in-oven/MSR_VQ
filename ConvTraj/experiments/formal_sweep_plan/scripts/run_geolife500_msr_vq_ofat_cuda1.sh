#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/formal_hotfix_runs"
DATA_ROOT="$ROOT/data/0_geolife"
GPU="${GPU:-1}"
CUDA1_LOCK_FILE="${CUDA1_LOCK_FILE:-$EXP_ROOT/script_locks/cuda1.lock}"

EPOCHS=500
EVAL_EPOCH=500
SAVE_EPOCHS="400,500"
PDT_HEADS=4
PDT_STEPS=8

# Baseline anchored on the Geolife DTW MSR_VQ main recipe, but with the requested
# VQ baseline switched to m=8,k=16 and fixed alpha (non-learnable).
BASE_G=48
BASE_L=48
BASE_P=32
BASE_STE=0.03
BASE_ALPHA=0.15
BASE_QM=0.08
BASE_M=8
BASE_K=16

DIM_CANDIDATES=(
  "32 32 64"
  "32 64 32"
  "48 48 32"
  "64 32 32"
  "64 48 16"
)
STE_CANDIDATES=("0.00" "0.01" "0.03" "0.05" "0.08")
QM_CANDIDATES=("0.00" "0.03" "0.05" "0.08" "0.12")
MK_CANDIDATES=(
  "8 16"
  "8 64"
  "16 16"
  "16 64"
)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES

mkdir -p "$EXP_ROOT/script_logs"
mkdir -p "$EXP_ROOT/script_locks"
MASTER_LOG="$EXP_ROOT/script_logs/run_geolife500_msr_vq_ofat_cuda1_$(date +%Y%m%d_%H%M%S).log"
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

image_mode_for_dist() {
  local dist="$1"
  case "$dist" in
    dtw) echo "dtw8" ;;
    dfd) echo "dfd7" ;;
    haus) echo "haus6" ;;
    *) echo "motion6" ;;
  esac
}

run_variant() {
  local dist="$1"
  local variant="$2"
  local g_dim="$3"
  local l_dim="$4"
  local p_dim="$5"
  local ste_max="$6"
  local alpha_init="$7"
  local qm_max="$8"
  local m_val="$9"
  local k_val="${10}"
  local image_mode
  image_mode="$(image_mode_for_dist "$dist")"

  local run_stem="geolife_${dist}_msr_${image_mode}_vq_${variant}_m${m_val}_k${k_val}_h${PDT_HEADS}_s${PDT_STEPS}_e${EPOCHS}"
  local run_name="${run_stem}_cuda${GPU}"
  local run_dir="$EXP_ROOT/$run_name"
  local log_dir="$run_dir/logs"
  local log_path="$log_dir/run.log"
  local existing_metrics=""

  existing_metrics=$(find "$EXP_ROOT" -maxdepth 3 -path "*/${run_stem}_cuda*/metrics/*.json" -type f -size +0c | head -n 1 || true)
  if [[ -n "$existing_metrics" ]]; then
    echo "[SKIP] $run_name"
    echo "[SKIP] found existing metrics: $existing_metrics"
    return 0
  fi

  mkdir -p "$log_dir"

  echo "[START] $run_name"
  echo "[LOG] $log_path"

  local cmd=(
    python train.py
    --target_size 128
    --batch_size 96
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
    --pdt_m "$m_val"
    --pdt_k "$k_val"
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
    --improved_qm_max_weight "$qm_max"
    --improved_pairwise_weight 0.05
    --improved_entropy_weight 0.02
    --improved_commit_weight 0.05
    --improved_uniform_weight 0.001
    --pre_quant_bottleneck_enabled
    --pre_quant_global_dim "$g_dim"
    --pre_quant_local_dim "$l_dim"
    --pre_quant_progress_dim "$p_dim"
    --pre_quant_use_motion_stats
    --pre_quant_lambda_decor 0.01
    --pre_quant_lambda_stab 0.1
    --pre_quant_residual_alpha_init "$alpha_init"
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
    --decoded_ste_metric_max_weight "$ste_max"
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

wait_and_acquire_cuda1_lock

for dist in dtw haus dfd; do
  # baseline
  run_variant "$dist" "base" "$BASE_G" "$BASE_L" "$BASE_P" "$BASE_STE" "$BASE_ALPHA" "$BASE_QM" "$BASE_M" "$BASE_K"

  # dims sweep
  for dims in "${DIM_CANDIDATES[@]}"; do
    read -r g l p <<< "$dims"
    if [[ "$g" == "$BASE_G" && "$l" == "$BASE_L" && "$p" == "$BASE_P" ]]; then
      continue
    fi
    run_variant "$dist" "dims_g${g}_l${l}_p${p}" "$g" "$l" "$p" "$BASE_STE" "$BASE_ALPHA" "$BASE_QM" "$BASE_M" "$BASE_K"
  done

  # decoded ste sweep
  for ste in "${STE_CANDIDATES[@]}"; do
    if [[ "$ste" == "$BASE_STE" ]]; then
      continue
    fi
    run_variant "$dist" "ste${ste}" "$BASE_G" "$BASE_L" "$BASE_P" "$ste" "$BASE_ALPHA" "$BASE_QM" "$BASE_M" "$BASE_K"
  done

  # improved qm sweep
  for qm in "${QM_CANDIDATES[@]}"; do
    if [[ "$qm" == "$BASE_QM" ]]; then
      continue
    fi
    run_variant "$dist" "qm${qm}" "$BASE_G" "$BASE_L" "$BASE_P" "$BASE_STE" "$BASE_ALPHA" "$qm" "$BASE_M" "$BASE_K"
  done

  # m/k sweep
  for mk in "${MK_CANDIDATES[@]}"; do
    read -r m k <<< "$mk"
    if [[ "$m" == "$BASE_M" && "$k" == "$BASE_K" ]]; then
      continue
    fi
    run_variant "$dist" "mk_m${m}_k${k}" "$BASE_G" "$BASE_L" "$BASE_P" "$BASE_STE" "$BASE_ALPHA" "$BASE_QM" "$m" "$k"
  done
done

echo "[ALL DONE] geolife msr_vq OFAT sweep finished"
