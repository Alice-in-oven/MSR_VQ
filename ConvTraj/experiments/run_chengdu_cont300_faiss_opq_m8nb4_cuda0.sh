#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/chengdu_cont_opq"
DATA_ROOT="$ROOT/data/0_chengdu"
TRAIN_FLAG="chengdu_msr_cont_only_e300_cuda0"
ARTIFACT_PREFIX="chengdu_cont_msr_e300"
GT_PATH="$DATA_ROOT/DTW/dtw_test_distance_matrix_result"
LOG_PATH="$EXP_ROOT/logs/${TRAIN_FLAG}.log"

TRAIN_EMB="$EXP_ROOT/embeddings/${ARTIFACT_PREFIX}_continuous_train_e300.pkl"
QUERY_EMB="$EXP_ROOT/embeddings/${ARTIFACT_PREFIX}_continuous_query_e300.pkl"
BASE_EMB="$EXP_ROOT/embeddings/${ARTIFACT_PREFIX}_continuous_base_e300.pkl"

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/train_config"

cd "$ROOT"

python -u train.py \
  --train_flag "$TRAIN_FLAG" \
  --artifact_prefix "$ARTIFACT_PREFIX" \
  --network_type TJCNN_MC_MSR \
  --embedding_backbone msr \
  --image_mode motion6 \
  --dataset chengdu \
  --root_read_path "$DATA_ROOT" \
  --root_write_path "$EXP_ROOT" \
  --dist_type dtw \
  --grid_size 0.0005 \
  --epoch_num 300 \
  --batch_size 96 \
  --device cuda:0 \
  --test_epoch 300 \
  --eval_save_epochs 300 \
  --print_epoch 10 \
  --mode train-directly \
  --eval_embedding_type continuous \
  --triplet_pos_begin_pos 1 \
  --triplet_pos_end_pos 60 \
  --triplet_neg_begin_pos 180 \
  --triplet_neg_end_pos 900 \
  --pdt_m 8 \
  --pdt_k 16 \
  --loss_recipe baseline \
  --pdt_loss_weight 0.0 \
  --pdt_loss_start_epoch 999999 \
  --disable_pdt_init_codebook \
  2>&1 | tee "$LOG_PATH"

python -u experiments/faiss_pq_opq_quantize.py \
  --input_mode split \
  --train_path "$TRAIN_EMB" \
  --query_path "$QUERY_EMB" \
  --base_path "$BASE_EMB" \
  --quantizer opq \
  --m 8 \
  --nbits 4 \
  --opq_niter 50 \
  --gt_path "$GT_PATH" \
  --output_dir "$EXP_ROOT/faiss_opq_m8_nbits4" \
  --artifact_prefix chengdu_cont_opq_m8_nbits4 \
  --save_rotated \
  2>&1 | tee -a "$LOG_PATH"
