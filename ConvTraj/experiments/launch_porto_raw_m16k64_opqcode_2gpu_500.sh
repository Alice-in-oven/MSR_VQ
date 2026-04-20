#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
BACKBONE_CKPT="$ROOT/exp/porto_raw_backbone400_faiss_m8nb4/model/porto_raw_backbone_only_e400_cuda1_epochs_400"
SOURCE_CACHE_DIR="$ROOT/exp/porto_raw_backbone400_faiss_m8nb4/eval_preproc_cache"
PORTO_OPQ_ROT="$ROOT/exp/porto_raw_backbone400_faiss_m16_nbits6_opq/porto_raw_backbone_cont_opq_m16_nbits6_train_faiss_rotated.pkl"
PORTO_OPQ_CODEBOOK="$ROOT/exp/porto_raw_backbone400_faiss_m16_nbits6_opq/porto_raw_backbone_cont_opq_m16_nbits6_faiss_pq_codebook.npy"

launch_variant() {
  local train_flag="$1"
  local device="$2"
  local teacher_z="$3"
  local teacher_part="$4"
  local teacher_end="$5"
  local freeze_end="$6"
  local pdt_start="$7"
  local pdt_weight="$8"
  local ste_start="$9"
  local ste_warmup="${10}"
  local ste_max="${11}"
  local entropy_w="${12}"
  local uniform_w="${13}"
  local late_start="${14}"
  local realign_flag="${15}"

  local exp_root="$ROOT/exp/$train_flag"
  mkdir -p "$exp_root/logs"
  if [ ! -e "$exp_root/eval_preproc_cache" ]; then
    ln -s "$SOURCE_CACHE_DIR" "$exp_root/eval_preproc_cache"
  fi

  local log_path="$exp_root/logs/run.log"
  echo "[LAUNCH] $train_flag on $device -> $log_path"

  local maybe_realign=""
  if [ "$realign_flag" = "1" ]; then
    maybe_realign="--porto_opq_teacher_realign_codebook_on_unfreeze"
  fi

  nohup bash -lc "
    cd '$ROOT'
    python -u train.py \
      --train_flag '$train_flag' \
      --network_type TJCNN_MC_MSR \
      --image_mode motion6 \
      --dataset porto \
      --root_read_path /data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto \
      --root_write_path '$exp_root' \
      --dist_type dtw \
      --epoch_num 500 \
      --batch_size 256 \
      --device '$device' \
      --test_epoch 500 \
      --eval_save_epochs 400,500 \
      --print_epoch 5 \
      --mode train-directly \
      --eval_embedding_type quantized \
      --eval_search_mode decoded \
      --pdt_m 16 \
      --pdt_k 64 \
      --pdt_vq_type dpq \
      --pdt_codebook_init faiss \
      --backbone_checkpoint '$BACKBONE_CKPT' \
      --freeze_backbone_epochs 0 \
      --loss_recipe improved_vq \
      --pdt_loss_start_epoch '$pdt_start' \
      --pdt_loss_weight '$pdt_weight' \
      --disable_improved_vq_adaptive_low_codebook \
      --improved_qm_start_epoch 0 \
      --improved_qm_warmup_epochs 1 \
      --improved_qm_max_weight 0.0 \
      --improved_pairwise_weight 0.0 \
      --improved_entropy_weight '$entropy_w' \
      --improved_commit_weight 0.02 \
      --improved_uniform_weight '$uniform_w' \
      --decoded_ste_metric_enabled \
      --decoded_ste_metric_start_epoch '$ste_start' \
      --decoded_ste_metric_warmup_epochs '$ste_warmup' \
      --decoded_ste_metric_max_weight '$ste_max' \
      --porto_opq_teacher_rotated_train_path '$PORTO_OPQ_ROT' \
      --porto_opq_teacher_codebook_path '$PORTO_OPQ_CODEBOOK' \
      --porto_opq_teacher_start_epoch 0 \
      --porto_opq_teacher_end_epoch '$teacher_end' \
      --porto_opq_teacher_z_weight '$teacher_z' \
      --porto_opq_teacher_partition_weight '$teacher_part' \
      --porto_opq_teacher_codebook_freeze_end_epoch '$freeze_end' \
      $maybe_realign \
      --late_finetune_start_epoch '$late_start' \
      --late_finetune_main_lr_scale 0.2 \
      --late_finetune_pre_quant_lr_scale 0.2
  " > "$log_path" 2>&1 &

  local pid=$!
  echo "$pid" > "$exp_root/logs/pid.txt"
  echo "[PID] $train_flag -> $pid"
}

echo "[PLAN] Two Porto-only OPQ-code-supervision 500-epoch runs"
echo "[PLAN] cuda:0 code-imprint (long frozen codebook)"
echo "[PLAN] cuda:1 code-imprint -> GT handoff (earlier unfreeze + realign)"

launch_variant "porto_raw_m16k64_opqcode_frozen500_cuda0" "cuda:0" "0.00" "1.00" "280" "280" "240" "0.020" "120" "40" "0.12" "0.002" "0.00005" "430" "0"
launch_variant "porto_raw_m16k64_opqcode_handoff500_cuda1" "cuda:1" "0.02" "0.90" "240" "180" "160" "0.030" "80" "30" "0.16" "0.003" "0.00010" "420" "1"

echo "[DONE] Both runs launched."
