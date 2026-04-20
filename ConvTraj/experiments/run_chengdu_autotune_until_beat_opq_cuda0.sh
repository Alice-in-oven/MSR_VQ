#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
DATA_ROOT="$ROOT/data/0_chengdu"
CUDA_INDEX="${CUDA_INDEX:-0}"
DEVICE="${DEVICE:-cuda:${CUDA_INDEX}}"
TARGET_TOP10="0.2899"
MEMORY_FILE="$ROOT/CHENGDU_POLICY_MEMORY.md"

TEACHER_CKPT="$ROOT/exp/chengdu_cont_opq/checkpoints/chengdu_cont_msr_e300_e300.pt"
SOURCE_CACHE_DIR="$ROOT/exp/chengdu_cont_opq/eval_preproc_cache"
OPQ_DIR="$ROOT/exp/chengdu_cont_opq/faiss_opq_m8_nbits4"
OPQ_TRAIN_RECON="$OPQ_DIR/chengdu_cont_opq_m8_nbits4_train_faiss_recon.pkl"
OPQ_TRAIN_ROTATED="$OPQ_DIR/chengdu_cont_opq_m8_nbits4_train_faiss_rotated.pkl"
OPQ_CODEBOOK="$OPQ_DIR/chengdu_cont_opq_m8_nbits4_faiss_pq_codebook.npy"

AUTOTUNE_ROOT="$ROOT/exp/chengdu_autotune"
mkdir -p "$AUTOTUNE_ROOT/logs"

echo "[ChengduAutoTune] policy memory: $MEMORY_FILE"
sed -n '1,80p' "$MEMORY_FILE"

wait_for_target_gpu() {
  while true; do
    local gpu_uuid
    gpu_uuid="$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | sed -n "$((CUDA_INDEX + 1))p" | tr -d ' ')"
    local active_pids
    active_pids="$(
      nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
        | awk -F', ' -v target_uuid="$gpu_uuid" '$1 == target_uuid {print $2}'
    )"
    local gpu_mem
    gpu_mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((CUDA_INDEX + 1))p" | tr -d ' ')"
    if [ -z "${active_pids}" ] && [ "${gpu_mem:-0}" -lt 200 ]; then
      echo "[ChengduAutoTune] ${DEVICE} is free, starting next stage."
      break
    fi
    echo "[ChengduAutoTune] waiting for ${DEVICE} to become free... used=${gpu_mem:-unknown}MiB"
    sleep 120
  done
}

run_experiment() {
  local exp_name="$1"
  shift

  local exp_root="$ROOT/exp/$exp_name"
  local train_flag="${exp_name}_cuda0"
  local artifact_prefix="$exp_name"
  local log_path="$exp_root/logs/${train_flag}.log"
  local summary_path="$exp_root/reports/${artifact_prefix}_summary.json"

  mkdir -p "$exp_root/logs"
  if [ -e "$SOURCE_CACHE_DIR" ] && [ ! -e "$exp_root/eval_preproc_cache" ]; then
    ln -s "$SOURCE_CACHE_DIR" "$exp_root/eval_preproc_cache"
  fi

  wait_for_target_gpu
  echo "[ChengduAutoTune] start experiment: $exp_name"
  (
    cd "$ROOT"
    python -u train.py \
      --train_flag "$train_flag" \
      --network_type TJCNN_MC_MSR \
      --image_mode motion6 \
      --dataset chengdu \
      --root_read_path "$DATA_ROOT" \
      --root_write_path "$exp_root" \
      --artifact_prefix "$artifact_prefix" \
      --grid_size 0.0005 \
      --epoch_num 400 \
      --batch_size 96 \
      --device "$DEVICE" \
      --test_epoch 100 \
      --eval_save_epochs 100,200,300,400 \
      --print_epoch 10 \
      --mode train-directly \
      --eval_embedding_type both \
      --eval_search_mode decoded \
      --enable_rerank \
      --rerank_L 100 \
      --rerank_source decoded \
      --pdt_m 8 \
      --pdt_k 16 \
      --pdt_vq_type dpq \
      --pdt_codebook_init faiss \
      --loss_recipe improved_vq \
      --backbone_checkpoint "$TEACHER_CKPT" \
      --pre_quant_bottleneck_enabled \
      --pre_quant_use_motion_stats \
      --pre_quant_lambda_decor 0.01 \
      --pre_quant_lambda_stab 0.08 \
      --pre_quant_stab_late_epoch 180 \
      --pre_quant_stab_late_multiplier 2.0 \
      --pre_quant_lr_multiplier 0.25 \
      --pre_quant_refresh_period 0 \
      --disable_improved_vq_adaptive_low_codebook \
      "$@" 2>&1 | tee "$log_path"
  )

  if [ ! -f "$summary_path" ]; then
    echo "[ChengduAutoTune] missing summary: $summary_path" >&2
    return 2
  fi

  python - "$summary_path" "$TARGET_TOP10" <<'PY'
import json, sys
summary_path = sys.argv[1]
target = float(sys.argv[2])
with open(summary_path) as f:
    data = json.load(f)
decoded = float((data.get("best_records", {}).get("decoded") or {}).get("top10_recall", -1.0))
rerank_rec = (data.get("best_records", {}).get("strict_rerank") or {})
rerank = float(rerank_rec.get("top10_recall", -1.0)) if rerank_rec else -1.0
continuous = float((data.get("best_records", {}).get("continuous") or {}).get("top10_recall", -1.0))
print("[ChengduAutoTune] summary:", json.dumps({
    "summary_path": summary_path,
    "continuous_top10": continuous,
    "decoded_top10": decoded,
    "strict_rerank_top10": rerank,
    "target_top10": target,
    "beat_target": decoded > target,
}, ensure_ascii=False))
sys.exit(0 if decoded > target else 1)
PY
}

cd "$ROOT"

if run_experiment \
  "chengdu_vq_teacher_stage1" \
  --freeze_backbone_epochs 80 \
  --pdt_loss_start_epoch 120 \
  --pdt_loss_weight 0.05 \
  --quantized_metric_weight 0.30 \
  --improved_qm_start_epoch 80 \
  --improved_qm_warmup_epochs 80 \
  --improved_qm_max_weight 0.12 \
  --improved_pairwise_weight 0.05 \
  --improved_entropy_weight 0.01 \
  --improved_commit_weight 0.02 \
  --improved_uniform_weight 0.001 \
  --pre_quant_raw_metric_weight 0.10 \
  --decoded_ste_metric_enabled \
  --decoded_ste_metric_start_epoch 120 \
  --decoded_ste_metric_warmup_epochs 40 \
  --decoded_ste_metric_max_weight 0.05 \
  --late_finetune_start_epoch 320 \
  --late_finetune_main_lr_scale 0.2 \
  --late_finetune_pre_quant_lr_scale 0.2; then
  echo "[ChengduAutoTune] success at stage1"
  exit 0
fi

if run_experiment \
  "chengdu_vq_teacher_stage2_neighbor_landmark" \
  --freeze_backbone_epochs 80 \
  --pdt_loss_start_epoch 120 \
  --pdt_loss_weight 0.05 \
  --quantized_metric_weight 0.30 \
  --improved_qm_start_epoch 80 \
  --improved_qm_warmup_epochs 80 \
  --improved_qm_max_weight 0.12 \
  --improved_pairwise_weight 0.05 \
  --improved_entropy_weight 0.01 \
  --improved_commit_weight 0.02 \
  --improved_uniform_weight 0.001 \
  --pre_quant_raw_metric_weight 0.12 \
  --decoded_ste_metric_enabled \
  --decoded_ste_metric_start_epoch 120 \
  --decoded_ste_metric_warmup_epochs 40 \
  --decoded_ste_metric_max_weight 0.05 \
  --pre_quant_neighbor_enabled \
  --pre_quant_neighbor_use_btn \
  --pre_quant_neighbor_use_dec \
  --pre_quant_neighbor_topk 20 \
  --pre_quant_neighbor_lambda_btn 0.03 \
  --pre_quant_neighbor_lambda_dec 0.12 \
  --pre_quant_neighbor_start_epoch 80 \
  --pre_quant_neighbor_warmup_epochs 40 \
  --pre_quant_neighbor_teacher_mode offline_gt \
  --pre_quant_neighbor_dec_stop_backbone \
  --pre_quant_landmark_enabled \
  --pre_quant_landmark_use_dec \
  --pre_quant_landmark_dec_ste_to_btn \
  --pre_quant_landmark_num 128 \
  --pre_quant_landmark_lambda_dec 0.10 \
  --pre_quant_landmark_start_epoch 120 \
  --pre_quant_landmark_warmup_epochs 40 \
  --pre_quant_landmark_teacher_checkpoint "$TEACHER_CKPT" \
  --late_finetune_start_epoch 320 \
  --late_finetune_main_lr_scale 0.2 \
  --late_finetune_pre_quant_lr_scale 0.2; then
  echo "[ChengduAutoTune] success at stage2"
  exit 0
fi

if run_experiment \
  "chengdu_vq_teacher_stage3_opq_teacher" \
  --freeze_backbone_epochs 60 \
  --pdt_loss_start_epoch 100 \
  --pdt_loss_weight 0.05 \
  --quantized_metric_weight 0.30 \
  --improved_qm_start_epoch 60 \
  --improved_qm_warmup_epochs 80 \
  --improved_qm_max_weight 0.12 \
  --improved_pairwise_weight 0.05 \
  --improved_entropy_weight 0.01 \
  --improved_commit_weight 0.02 \
  --improved_uniform_weight 0.001 \
  --pre_quant_raw_metric_weight 0.12 \
  --decoded_ste_metric_enabled \
  --decoded_ste_metric_start_epoch 100 \
  --decoded_ste_metric_warmup_epochs 40 \
  --decoded_ste_metric_max_weight 0.06 \
  --pre_quant_neighbor_enabled \
  --pre_quant_neighbor_use_dec \
  --pre_quant_neighbor_topk 20 \
  --pre_quant_neighbor_lambda_dec 0.12 \
  --pre_quant_neighbor_start_epoch 80 \
  --pre_quant_neighbor_warmup_epochs 40 \
  --pre_quant_neighbor_teacher_mode offline_gt \
  --pre_quant_neighbor_dec_stop_backbone \
  --pre_quant_landmark_enabled \
  --pre_quant_landmark_use_dec \
  --pre_quant_landmark_dec_ste_to_btn \
  --pre_quant_landmark_num 128 \
  --pre_quant_landmark_lambda_dec 0.10 \
  --pre_quant_landmark_start_epoch 100 \
  --pre_quant_landmark_warmup_epochs 40 \
  --pre_quant_landmark_teacher_checkpoint "$TEACHER_CKPT" \
  --porto_opq_warmup_train_recon_path "$OPQ_TRAIN_RECON" \
  --porto_opq_warmup_start_epoch 0 \
  --porto_opq_warmup_end_epoch 60 \
  --porto_opq_warmup_max_weight 0.08 \
  --porto_opq_teacher_rotated_train_path "$OPQ_TRAIN_ROTATED" \
  --porto_opq_teacher_codebook_path "$OPQ_CODEBOOK" \
  --porto_opq_teacher_start_epoch 40 \
  --porto_opq_teacher_end_epoch 180 \
  --porto_opq_teacher_z_weight 0.03 \
  --porto_opq_teacher_partition_weight 0.02 \
  --late_finetune_start_epoch 320 \
  --late_finetune_main_lr_scale 0.2 \
  --late_finetune_pre_quant_lr_scale 0.2; then
  echo "[ChengduAutoTune] success at stage3"
  exit 0
fi

echo "[ChengduAutoTune] all configured stages finished but none beat the OPQ target."
exit 1
