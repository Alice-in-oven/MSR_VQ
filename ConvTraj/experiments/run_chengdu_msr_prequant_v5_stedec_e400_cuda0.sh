#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/chengdu_msr_vq"
DATA_ROOT="$ROOT/data/0_chengdu"
TRAIN_FLAG="chengdu_msr_prequant_v5_stedec_m8_k16_e400_cuda0"
ARTIFACT_PREFIX="chengdu_msr_vq_m8_k16"
LOG_PATH="$EXP_ROOT/logs/${TRAIN_FLAG}.log"

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/train_config"

cd "$ROOT"
python train.py \
  --train_flag "$TRAIN_FLAG" \
  --artifact_prefix "$ARTIFACT_PREFIX" \
  --network_type TJCNN_MC_MSR \
  --embedding_backbone msr \
  --image_mode motion6 \
  --dataset chengdu \
  --root_read_path "$DATA_ROOT" \
  --root_write_path "$EXP_ROOT" \
  --dist_type dtw \
  --epoch_num 400 \
  --batch_size 256 \
  --device cuda:0 \
  --test_epoch 100 \
  --print_epoch 10 \
  --eval_save_epochs 100,200,300,400 \
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
  --late_finetune_start_epoch 320 \
  --late_finetune_main_lr_scale 0.1 \
  --late_finetune_pre_quant_lr_scale 0.1 \
  2>&1 | tee "$LOG_PATH"
