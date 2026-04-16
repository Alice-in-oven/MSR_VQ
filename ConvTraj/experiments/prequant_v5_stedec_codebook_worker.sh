#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <cuda_device> <m:k> [<m:k> ...]" >&2
  exit 1
fi

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
DATA_DIR="$ROOT/data"
FEATURE_DIR="$DATA_DIR/feature_dir"
MODEL_DIR="$DATA_DIR/model"
SUMMARY_PATH="$DATA_DIR/prequant_v5_stedec_codebook_sweep_summary.json"

DEVICE="$1"
shift
SPECS=("$@")

EPOCHS="500"
TEST_EPOCH="100"
MILESTONES=(100 200 300 400 500)
RERANK_L="100"
BATCH_CANDIDATES=(96 80 64 48 32)

mkdir -p "$DATA_DIR" "$FEATURE_DIR" "$MODEL_DIR"

update_summary() {
  local train_flag="$1"
  local m="$2"
  local k="$3"
  python - "$train_flag" "$m" "$k" "$SUMMARY_PATH" "$FEATURE_DIR" "$MODEL_DIR" <<'PY'
import json
import os
import sys

train_flag, m, k, summary_path, feature_dir, model_dir = sys.argv[1:]
m = int(m)
k = int(k)
milestones = [100, 200, 300, 400, 500]
metrics_path = os.path.join(feature_dir, f"metrics_{train_flag}_epochs_500.json")
ckpt_path = os.path.join(model_dir, f"{train_flag}_epochs_500")
if not os.path.exists(metrics_path):
    raise SystemExit(f"missing metrics: {metrics_path}")

with open(metrics_path, "r", encoding="utf-8") as handle:
    metrics = json.load(handle)

continuous = metrics.get("continuous", {})
continuous_raw = metrics.get("continuous_raw", {})
decoded = metrics.get("decoded", {})
rerank = metrics.get("rerank_decoded_L100", {})
code_usage = metrics.get("code_usage", {})

record = {
    "train_flag": train_flag,
    "m": m,
    "k": k,
    "metrics_path": metrics_path,
    "ckpt_path": ckpt_path,
    "milestones": [
        {
            "epoch": epoch,
            "metrics_path": os.path.join(feature_dir, f"metrics_{train_flag}_epochs_{epoch}.json"),
            "ckpt_path": os.path.join(model_dir, f"{train_flag}_epochs_{epoch}"),
        }
        for epoch in milestones
    ],
    "continuous_top10": float(continuous.get("top10_recall", -1.0)),
    "continuous_raw_top10": float(continuous_raw.get("top10_recall", -1.0)),
    "decoded_top10": float(decoded.get("top10_recall", -1.0)),
    "rerank_top10": float(rerank.get("top10_recall", -1.0)),
    "unique_tuple_ratio": float(code_usage.get("unique_tuple_ratio", -1.0)),
    "metrics": metrics,
}

if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
else:
    summary = {
        "method": "prequant_v5_stedec",
        "epoch_num": 500,
        "test_epoch": 100,
        "milestone_epochs": milestones,
        "rerank_L": 100,
        "runs": [],
    }

runs = [r for r in summary.get("runs", []) if r.get("train_flag") != train_flag]
runs.append(record)
runs.sort(key=lambda item: (item["m"], item["k"]))
summary["runs"] = runs
best_runs = sorted(runs, key=lambda item: item["rerank_top10"], reverse=True)
summary["best_rerank_top10"] = best_runs[0] if best_runs else None

with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
PY
}

run_one() {
  local m="$1"
  local k="$2"
  local train_flag="prequant_v5_stedec_sweep_m${m}_k${k}_e${EPOCHS}"
  local metrics_path="$FEATURE_DIR/metrics_${train_flag}_epochs_${EPOCHS}.json"
  local ckpt_path="$MODEL_DIR/${train_flag}_epochs_${EPOCHS}"
  local log_path="$DATA_DIR/${train_flag}.log"

  local all_milestones_ready=1
  for milestone in "${MILESTONES[@]}"; do
    if [[ ! -f "$FEATURE_DIR/metrics_${train_flag}_epochs_${milestone}.json" || ! -f "$MODEL_DIR/${train_flag}_epochs_${milestone}" ]]; then
      all_milestones_ready=0
      break
    fi
  done

  if [[ $all_milestones_ready -eq 1 ]]; then
    echo "[SKIP] existing run train_flag=$train_flag"
    update_summary "$train_flag" "$m" "$k"
    return 0
  fi

  local succeeded=0
  for batch_size in "${BATCH_CANDIDATES[@]}"; do
    {
      echo
      echo "[START] train_flag=$train_flag device=$DEVICE m=$m k=$k batch_size=$batch_size"
      echo "[COMMAND] python $ROOT/train.py --train_flag $train_flag --pdt_m $m --pdt_k $k --network_type TJCNN_MC_MSR --image_mode motion6 --dataset geolife --epoch_num $EPOCHS --batch_size $batch_size --device $DEVICE --test_epoch $TEST_EPOCH --print_epoch 10 --eval_save_epochs 100,200,300,400,500 --mode train-directly --eval_embedding_type both --eval_search_mode decoded --enable_rerank --rerank_L $RERANK_L --rerank_source decoded --pdt_vq_type dpq --pdt_codebook_init faiss --loss_recipe improved_vq --pdt_loss_weight 0.1 --pdt_loss_start_epoch 40 --improved_qm_start_epoch 80 --improved_qm_warmup_epochs 80 --improved_qm_max_weight 0.08 --improved_pairwise_weight 0.05 --improved_entropy_weight 0.02 --improved_commit_weight 0.05 --improved_uniform_weight 0.001 --pre_quant_bottleneck_enabled --pre_quant_use_motion_stats --pre_quant_lambda_decor 0.01 --pre_quant_lambda_stab 0.1 --pre_quant_stab_late_epoch 100 --pre_quant_stab_late_multiplier 4.0 --pre_quant_refresh_start_epoch 100 --pre_quant_refresh_period 50 --pre_quant_refresh_end_epoch 400 --pre_quant_lr_multiplier 0.25 --pre_quant_raw_metric_weight 0.0 --decoded_ste_metric_enabled --decoded_ste_metric_start_epoch 80 --decoded_ste_metric_warmup_epochs 20 --decoded_ste_metric_max_weight 0.03 --late_finetune_start_epoch 400 --late_finetune_main_lr_scale 0.1 --late_finetune_pre_quant_lr_scale 0.1"
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
        --test_epoch "$TEST_EPOCH" \
        --print_epoch 10 \
        --eval_save_epochs 100,200,300,400,500 \
        --mode train-directly \
        --eval_embedding_type both \
        --eval_search_mode decoded \
        --enable_rerank \
        --rerank_L "$RERANK_L" \
        --rerank_source decoded \
        --pdt_vq_type dpq \
        --pdt_codebook_init faiss \
        --loss_recipe improved_vq \
        --pdt_loss_weight 0.1 \
        --pdt_loss_start_epoch 40 \
        --improved_qm_start_epoch 80 \
        --improved_qm_warmup_epochs 80 \
        --improved_qm_max_weight 0.08 \
        --improved_pairwise_weight 0.05 \
        --improved_entropy_weight 0.02 \
        --improved_commit_weight 0.05 \
        --improved_uniform_weight 0.001 \
        --pre_quant_bottleneck_enabled \
        --pre_quant_use_motion_stats \
        --pre_quant_lambda_decor 0.01 \
        --pre_quant_lambda_stab 0.1 \
        --pre_quant_stab_late_epoch 100 \
        --pre_quant_stab_late_multiplier 4.0 \
        --pre_quant_refresh_start_epoch 100 \
        --pre_quant_refresh_period 50 \
        --pre_quant_refresh_end_epoch 400 \
        --pre_quant_lr_multiplier 0.25 \
        --pre_quant_raw_metric_weight 0.0 \
        --decoded_ste_metric_enabled \
        --decoded_ste_metric_start_epoch 80 \
        --decoded_ste_metric_warmup_epochs 20 \
        --decoded_ste_metric_max_weight 0.03 \
        --late_finetune_start_epoch 400 \
        --late_finetune_main_lr_scale 0.1 \
        --late_finetune_pre_quant_lr_scale 0.1
    ) >> "$log_path" 2>&1
    local exit_code=$?
    set -e

    if [[ -f "$metrics_path" ]]; then
      succeeded=1
      break
    fi

    if grep -q "OutOfMemoryError" "$log_path"; then
      echo "[OOM] retrying $train_flag with smaller batch size after $batch_size" | tee -a "$log_path"
      rm -f "$ckpt_path" "$metrics_path"
      continue
    fi

    if [[ $exit_code -ne 0 ]]; then
      echo "[FAIL] command exited with code $exit_code for $train_flag" | tee -a "$log_path"
      return "$exit_code"
    fi
  done

  if [[ $succeeded -ne 1 ]]; then
    echo "[FAIL] unable to finish $train_flag with batch sizes: ${BATCH_CANDIDATES[*]}" | tee -a "$log_path"
    return 1
  fi

  update_summary "$train_flag" "$m" "$k"

  python - "$metrics_path" "$train_flag" <<'PY'
import json
import sys

metrics_path, train_flag = sys.argv[1:]
with open(metrics_path, "r", encoding="utf-8") as handle:
    metrics = json.load(handle)
continuous = metrics.get("continuous", {})
decoded = metrics.get("decoded", {})
rerank = metrics.get("rerank_decoded_L100", {})
code_usage = metrics.get("code_usage", {})

print("[RESULT] train_flag={}".format(train_flag))
print("[RESULT] continuous_top10={:.4f}".format(float(continuous.get("top10_recall", -1.0))))
print("[RESULT] decoded_top10={:.4f}".format(float(decoded.get("top10_recall", -1.0))))
print("[RESULT] rerank_L100_top10={:.4f}".format(float(rerank.get("top10_recall", -1.0))))
print("[RESULT] unique_tuple_ratio={:.4f}".format(float(code_usage.get("unique_tuple_ratio", -1.0))))
PY
}

for spec in "${SPECS[@]}"; do
  IFS=':' read -r m k <<< "$spec"
  run_one "$m" "$k"
done

echo "[DONE] device=$DEVICE specs=${SPECS[*]} summary=$SUMMARY_PATH"
