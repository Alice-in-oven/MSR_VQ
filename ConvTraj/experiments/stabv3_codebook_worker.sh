#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <cuda_device> <m:k> [<m:k> ...]" >&2
  exit 1
fi

ROOT="/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj"
DATA_DIR="$ROOT/data"
FEATURE_DIR="$DATA_DIR/feature_dir"

DEVICE="$1"
shift
SPECS=("$@")

EPOCHS="500"
RERANK_L="100"
BATCH_CANDIDATES=(256 192 160 128 96)

mkdir -p "$DATA_DIR" "$FEATURE_DIR"

run_one() {
  local m="$1"
  local k="$2"
  local train_flag="stabv3_impvq_m${m}_k${k}_e${EPOCHS}"
  local metrics_path="$FEATURE_DIR/metrics_${train_flag}.json"
  local log_path="$DATA_DIR/${train_flag}.log"

  if [[ -f "$metrics_path" ]]; then
    echo "[SKIP] existing metrics: $metrics_path"
    return 0
  fi

  local succeeded=0
  for batch_size in "${BATCH_CANDIDATES[@]}"; do
    {
      echo
      echo "[COMMAND] python $ROOT/train.py --train_flag $train_flag --pdt_m $m --pdt_k $k --network_type TJCNN_MC_MSR --image_mode motion6 --dataset geolife --epoch_num $EPOCHS --batch_size $batch_size --device $DEVICE --test_epoch $EPOCHS --print_epoch 25 --mode train-directly --pdt_vq_type dpq --pdt_codebook_init faiss --eval_embedding_type both --eval_search_mode decoded --enable_rerank --rerank_L $RERANK_L --rerank_source decoded --loss_recipe improved_vq --pdt_loss_weight 0.1 --pdt_loss_start_epoch 40 --improved_qm_start_epoch 80 --improved_qm_warmup_epochs 80 --improved_qm_max_weight 0.08 --improved_pairwise_weight 0.05 --improved_entropy_weight 0.02 --improved_commit_weight 0.05 --improved_uniform_weight 0.001"
    } >> "$log_path"

    set +e
    (
      cd "$ROOT"
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stdbuf -oL -eL python -u "$ROOT/train.py" \
        --train_flag "$train_flag" \
        --pdt_m "$m" \
        --pdt_k "$k" \
        --network_type TJCNN_MC_MSR \
        --image_mode motion6 \
        --dataset geolife \
        --epoch_num "$EPOCHS" \
        --batch_size "$batch_size" \
        --device "$DEVICE" \
        --test_epoch "$EPOCHS" \
        --print_epoch 25 \
        --mode train-directly \
        --pdt_vq_type dpq \
        --pdt_codebook_init faiss \
        --eval_embedding_type both \
        --eval_search_mode decoded \
        --enable_rerank \
        --rerank_L "$RERANK_L" \
        --rerank_source decoded \
        --loss_recipe improved_vq \
        --pdt_loss_weight 0.1 \
        --pdt_loss_start_epoch 40 \
        --improved_qm_start_epoch 80 \
        --improved_qm_warmup_epochs 80 \
        --improved_qm_max_weight 0.08 \
        --improved_pairwise_weight 0.05 \
        --improved_entropy_weight 0.02 \
        --improved_commit_weight 0.05 \
        --improved_uniform_weight 0.001
    ) 2>&1 | tee -a "$log_path"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ -f "$metrics_path" ]]; then
      succeeded=1
      break
    fi

    if grep -q "OutOfMemoryError" "$log_path"; then
      echo "[OOM] retrying $train_flag with smaller batch size after $batch_size" | tee -a "$log_path"
      rm -f "$ROOT/data/model/${train_flag}_epochs_${EPOCHS}" "$metrics_path"
      continue
    fi

    if [[ $exit_code -ne 0 ]]; then
      return "$exit_code"
    fi
  done

  if [[ $succeeded -ne 1 ]]; then
    echo "[FAIL] unable to finish $train_flag with batch sizes: ${BATCH_CANDIDATES[*]}" | tee -a "$log_path"
    return 1
  fi
}

for spec in "${SPECS[@]}"; do
  IFS=':' read -r m k <<< "$spec"
  run_one "$m" "$k"
done
