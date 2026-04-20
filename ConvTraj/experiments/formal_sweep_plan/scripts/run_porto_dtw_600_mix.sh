#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
RUN_ROOT="${REPO_DIR}/exp/formal_hotfix_runs"
SCRIPT_LOG_DIR="${RUN_ROOT}/script_logs"
mkdir -p "${SCRIPT_LOG_DIR}"

CUDA_VQ=2
CUDA_CONT=3

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${SCRIPT_LOG_DIR}/run_porto_dtw_600_mix_${TS}.log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[INFO] repo=${REPO_DIR}"
echo "[INFO] master_log=${MASTER_LOG}"
echo "[INFO] cuda_vq=${CUDA_VQ} cuda_cont=${CUDA_CONT}"

run_vq_from_config() {
  local gpu="$1"
  local config_path="$2"
  local label="$3"
  echo "[START][VQ][gpu=${gpu}] ${label}"
  (
    cd "${REPO_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python experiments/formal_sweep_plan/run_train_from_config.py \
      --config "${config_path}"
  )
  echo "[DONE][VQ][gpu=${gpu}] ${label}"
}

run_continuous_backbone() {
  local gpu="$1"
  local backbone="$2"
  local train_flag="$3"
  local artifact_prefix="$4"
  local run_dir="$5"

  mkdir -p "${run_dir}/logs"
  echo "[START][CONT][gpu=${gpu}] ${train_flag}"
  (
    cd "${REPO_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python train.py \
      --target_size 128 \
      --batch_size 256 \
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
      --root_write_path "${run_dir}" \
      --root_read_path /data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto \
      --artifact_prefix "${artifact_prefix}" \
      --grid_size 0.0 \
      --image_mode motion6 \
      --embedding_backbone "${backbone}" \
      --backbone_seq_max_length 200 \
      --simformer_num_layers 1 \
      --simformer_n_heads 16 \
      --simformer_dimfeedforward 256 \
      --simformer_pos_encoding fixed \
      --neutraj_spatial_width 2 \
      --neutraj_incell True \
      --pdt_m 8 \
      --pdt_k 16 \
      --pdt_vq_type dpq \
      --pdt_codebook_init uniform \
      --qinco_h 256 \
      --qinco_L 1 \
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
      --dist_type dtw \
      --device cuda:0 \
      --train_flag "${train_flag}" \
      --head_num 1 \
      --train_set_size 3000 \
      --query_set_size 500 \
      --base_set_size 1597579
  )
  echo "[DONE][CONT][gpu=${gpu}] ${train_flag}"
}

VQ_MSR_CFG="${REPO_DIR}/experiments/formal_sweep_plan/generated_configs/quantized_main/porto/msr/porto_dtw_msr_pdtvq_m16_k64_e600_trueadc.json"
VQ_SIM_CFG="${REPO_DIR}/experiments/formal_sweep_plan/generated_configs/quantized_main/porto/simformer/porto_dtw_simformer_pdtvq_m16_k64_e600_trueadc.json"

CONT_MSR_DIR="${RUN_ROOT}/porto_dtw_msr_continuous_e600_clean_cuda3"
CONT_SIM_DIR="${RUN_ROOT}/porto_dtw_simformer_continuous_e600_clean_cuda3"

(
  run_vq_from_config "${CUDA_VQ}" "${VQ_MSR_CFG}" "porto_dtw_msr_pdtvq_m16_k64_e600_trueadc"
  run_vq_from_config "${CUDA_VQ}" "${VQ_SIM_CFG}" "porto_dtw_simformer_pdtvq_m16_k64_e600_trueadc"
) &
PID_VQ=$!

(
  run_continuous_backbone "${CUDA_CONT}" "msr" "porto_dtw_msr_continuous_e600_clean_cuda3" "porto_dtw_msr_continuous_e600_clean_cuda3" "${CONT_MSR_DIR}"
  run_continuous_backbone "${CUDA_CONT}" "simformer" "porto_dtw_simformer_continuous_e600_clean_cuda3" "porto_dtw_simformer_continuous_e600_clean_cuda3" "${CONT_SIM_DIR}"
) &
PID_CONT=$!

echo "[INFO] started background queues: vq_pid=${PID_VQ} cont_pid=${PID_CONT}"

wait "${PID_VQ}"
wait "${PID_CONT}"

echo "[DONE] all porto dtw 600-epoch jobs finished"
