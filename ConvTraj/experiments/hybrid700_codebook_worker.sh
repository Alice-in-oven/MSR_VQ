#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <cuda_device> <m:k> [<m:k> ...]" >&2
  exit 1
fi

ROOT="/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj"
DATA_DIR="$ROOT/data"
FEATURE_DIR="$DATA_DIR/feature_dir"
MODEL_DIR="$DATA_DIR/model"
SUMMARY_PATH="$DATA_DIR/hybrid700_codebook_sweep_summary.json"

DEVICE="$1"
shift
SPECS=("$@")

EPOCHS="700"
MILESTONES=(500 600 700)
RERANK_L="100"
BATCH_CANDIDATES=(256 192 160 128 96)

mkdir -p "$DATA_DIR" "$FEATURE_DIR" "$MODEL_DIR"

update_summary() {
  local train_flag="$1"
  local recipe="$2"
  local m="$3"
  local k="$4"
  python - "$train_flag" "$recipe" "$m" "$k" "$SUMMARY_PATH" "$FEATURE_DIR" "$MODEL_DIR" <<'PY'
import json
import os
import sys

train_flag, recipe, m, k, summary_path, feature_dir, model_dir = sys.argv[1:]
m = int(m)
k = int(k)
milestones = [500, 600, 700]
metrics_path = os.path.join(feature_dir, f"metrics_{train_flag}_epochs_700.json")
ckpt_path = os.path.join(model_dir, f"{train_flag}_epochs_700")
if not os.path.exists(metrics_path):
    raise SystemExit(f"missing metrics: {metrics_path}")

with open(metrics_path, "r", encoding="utf-8") as handle:
    metrics = json.load(handle)

rerank = metrics.get("rerank_decoded_L100", {})
decoded = metrics.get("decoded", {})
continuous = metrics.get("continuous", {})

record = {
    "train_flag": train_flag,
    "recipe": recipe,
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
    "decoded_top10": float(decoded.get("top10_recall", -1.0)),
    "rerank_top10": float(rerank.get("top10_recall", -1.0)),
    "metrics": metrics,
}

if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
else:
    summary = {
        "recipe_policy": {
            "m<8": "improved_vq",
            "m>=8": "baseline",
        },
        "epoch_num": 700,
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
  local recipe=""
  local train_flag=""
  local metrics_path=""
  local ckpt_path=""
  local log_path=""

  local -a recipe_args
  if (( m < 8 )); then
    recipe="improved_vq"
    train_flag="hybrid700_impvq_m${m}_k${k}_e${EPOCHS}"
    recipe_args=(
      --pdt_vq_type dpq
      --pdt_codebook_init faiss
      --loss_recipe improved_vq
      --pdt_loss_weight 0.1
      --pdt_loss_start_epoch 40
      --improved_qm_start_epoch 80
      --improved_qm_warmup_epochs 80
      --improved_qm_max_weight 0.08
      --improved_pairwise_weight 0.05
      --improved_entropy_weight 0.02
      --improved_commit_weight 0.05
      --improved_uniform_weight 0.001
    )
  else
    recipe="baseline"
    train_flag="hybrid700_base_m${m}_k${k}_e${EPOCHS}"
    recipe_args=(
      --pdt_vq_type dpq
      --pdt_codebook_init uniform
      --loss_recipe baseline
      --pdt_loss_weight 0.3
      --pdt_loss_start_epoch 20
    )
  fi

  metrics_path="$FEATURE_DIR/metrics_${train_flag}_epochs_${EPOCHS}.json"
  ckpt_path="$MODEL_DIR/${train_flag}_epochs_${EPOCHS}"
  log_path="$DATA_DIR/${train_flag}.log"

  local all_milestones_ready=1
  for milestone in "${MILESTONES[@]}"; do
    if [[ ! -f "$FEATURE_DIR/metrics_${train_flag}_epochs_${milestone}.json" || ! -f "$MODEL_DIR/${train_flag}_epochs_${milestone}" ]]; then
      all_milestones_ready=0
      break
    fi
  done

  if [[ $all_milestones_ready -eq 1 ]]; then
    echo "[SKIP] existing metrics: $metrics_path"
    echo "[SKIP] existing ckpt: $ckpt_path"
    update_summary "$train_flag" "$recipe" "$m" "$k"
    return 0
  fi

  local succeeded=0
  for batch_size in "${BATCH_CANDIDATES[@]}"; do
    {
      echo
      echo "[START] train_flag=$train_flag recipe=$recipe device=$DEVICE batch_size=$batch_size"
      echo "[COMMAND] python $ROOT/train.py --train_flag $train_flag --pdt_m $m --pdt_k $k --network_type TJCNN_MC_MSR --image_mode motion6 --dataset geolife --epoch_num $EPOCHS --batch_size $batch_size --device $DEVICE --test_epoch $EPOCHS --print_epoch 25 --eval_save_epochs 500,600,700 --mode train-directly --eval_embedding_type both --eval_search_mode decoded --enable_rerank --rerank_L $RERANK_L --rerank_source decoded ${recipe_args[*]}"
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
        --eval_save_epochs 500,600,700 \
        --mode train-directly \
        --eval_embedding_type both \
        --eval_search_mode decoded \
        --enable_rerank \
        --rerank_L "$RERANK_L" \
        --rerank_source decoded \
        "${recipe_args[@]}"
    ) 2>&1 | tee -a "$log_path"
    local exit_code=${PIPESTATUS[0]}
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

  update_summary "$train_flag" "$recipe" "$m" "$k"

  python - "$metrics_path" "$ckpt_path" "$train_flag" "$recipe" "$FEATURE_DIR" "$MODEL_DIR" <<'PY'
import json
import os
import sys

metrics_path, ckpt_path, train_flag, recipe, feature_dir, model_dir = sys.argv[1:]
with open(metrics_path, "r", encoding="utf-8") as handle:
    metrics = json.load(handle)

continuous = metrics.get("continuous", {})
decoded = metrics.get("decoded", {})
rerank = metrics.get("rerank_decoded_L100", {})

print("[RESULT] train_flag={} recipe={}".format(train_flag, recipe))
print("[RESULT] continuous_top10={:.4f}".format(float(continuous.get("top10_recall", -1.0))))
print("[RESULT] decoded_top10={:.4f}".format(float(decoded.get("top10_recall", -1.0))))
print("[RESULT] rerank_L100_top10={:.4f}".format(float(rerank.get("top10_recall", -1.0))))
for epoch in (500, 600, 700):
    print("[RESULT] epoch={} metrics_path={}".format(epoch, os.path.join(feature_dir, f"metrics_{train_flag}_epochs_{epoch}.json")))
    print("[RESULT] epoch={} ckpt_path={}".format(epoch, os.path.join(model_dir, f"{train_flag}_epochs_{epoch}")))
PY
}

for spec in "${SPECS[@]}"; do
  IFS=':' read -r m k <<< "$spec"
  run_one "$m" "$k"
done

echo "[DONE] device=$DEVICE specs=${SPECS[*]} summary=$SUMMARY_PATH"
