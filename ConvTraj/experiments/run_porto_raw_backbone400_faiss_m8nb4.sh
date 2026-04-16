#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/porto_raw_backbone400_faiss_m8nb4"
TRAIN_FLAG="porto_raw_backbone_only_e400_cuda1"
GT_PATH="/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto/DTW/dtw_test_distance_matrix_result"

TRAIN_EMB="$EXP_ROOT/feature_dir/train_dtw_feature_continuous_${TRAIN_FLAG}_400"
QUERY_EMB="$EXP_ROOT/feature_dir/query_dtw_feature_continuous_${TRAIN_FLAG}_400"
BASE_EMB="$EXP_ROOT/feature_dir/base_dtw_feature_continuous_${TRAIN_FLAG}_400"

mkdir -p "$EXP_ROOT/logs"

cd "$ROOT"

python -u train.py \
  --train_flag "$TRAIN_FLAG" \
  --network_type TJCNN_MC_MSR \
  --image_mode motion6 \
  --dataset porto \
  --root_read_path /data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto \
  --root_write_path "$EXP_ROOT" \
  --dist_type dtw \
  --epoch_num 400 \
  --batch_size 256 \
  --device cuda:1 \
  --test_epoch 400 \
  --eval_save_epochs 400 \
  --print_epoch 10 \
  --mode train-directly \
  --eval_embedding_type continuous \
  --pdt_m 8 \
  --pdt_k 16 \
  --loss_recipe baseline \
  --pdt_loss_weight 0.0 \
  --pdt_loss_start_epoch 999999 \
  --disable_pdt_init_codebook

python -u experiments/faiss_pq_opq_quantize.py \
  --input_mode split \
  --train_path "$TRAIN_EMB" \
  --query_path "$QUERY_EMB" \
  --base_path "$BASE_EMB" \
  --quantizer pq \
  --m 8 \
  --nbits 4 \
  --gt_path "$GT_PATH" \
  --output_dir "$EXP_ROOT/faiss_pq_m8_nbits4" \
  --artifact_prefix porto_raw_backbone_cont_pq_m8_nbits4

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
  --artifact_prefix porto_raw_backbone_cont_opq_m8_nbits4 \
  --save_rotated
