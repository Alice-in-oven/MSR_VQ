#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
EXP_ROOT="$ROOT/exp/porto_raw_m16k64_opqpart_nofreeze_e400_cuda0"
TRAIN_FLAG="porto_raw_m16k64_opqpart_nofreeze_e400_cuda0"
DEVICE="cuda:0"

BACKBONE_CKPT="$ROOT/exp/porto_raw_backbone400_faiss_m8nb4/model/porto_raw_backbone_only_e400_cuda1_epochs_400"
SOURCE_CACHE_DIR="$ROOT/exp/porto_raw_backbone400_faiss_m8nb4/eval_preproc_cache"
PORTO_OPQ_ROT="$ROOT/exp/porto_raw_backbone400_faiss_m16_nbits6_opq/porto_raw_backbone_cont_opq_m16_nbits6_train_faiss_rotated.pkl"
PORTO_OPQ_CODEBOOK="$ROOT/exp/porto_raw_backbone400_faiss_m16_nbits6_opq/porto_raw_backbone_cont_opq_m16_nbits6_faiss_pq_codebook.npy"

mkdir -p "$EXP_ROOT/logs"

if [ ! -e "$EXP_ROOT/eval_preproc_cache" ]; then
  ln -s "$SOURCE_CACHE_DIR" "$EXP_ROOT/eval_preproc_cache"
fi

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
  --device "$DEVICE" \
  --test_epoch 400 \
  --eval_save_epochs 400 \
  --print_epoch 10 \
  --mode train-directly \
  --eval_embedding_type quantized \
  --eval_search_mode decoded \
  --pdt_m 16 \
  --pdt_k 64 \
  --pdt_vq_type dpq \
  --pdt_codebook_init faiss \
  --backbone_checkpoint "$BACKBONE_CKPT" \
  --freeze_backbone_epochs 0 \
  --loss_recipe improved_vq \
  --pdt_loss_weight 0.05 \
  --pdt_loss_start_epoch 0 \
  --disable_improved_vq_adaptive_low_codebook \
  --improved_qm_start_epoch 0 \
  --improved_qm_warmup_epochs 1 \
  --improved_qm_max_weight 0.0 \
  --improved_pairwise_weight 0.0 \
  --decoded_ste_metric_enabled \
  --decoded_ste_metric_start_epoch 10 \
  --decoded_ste_metric_warmup_epochs 30 \
  --decoded_ste_metric_max_weight 0.10 \
  --porto_opq_teacher_rotated_train_path "$PORTO_OPQ_ROT" \
  --porto_opq_teacher_codebook_path "$PORTO_OPQ_CODEBOOK" \
  --porto_opq_teacher_start_epoch 0 \
  --porto_opq_teacher_end_epoch 80 \
  --porto_opq_teacher_z_weight 0.20 \
  --porto_opq_teacher_partition_weight 0.20 \
  --late_finetune_start_epoch 360 \
  --late_finetune_main_lr_scale 0.2 \
  --late_finetune_pre_quant_lr_scale 0.2
