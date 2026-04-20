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
  local pdt_start="$6"
  local pdt_weight="$7"
  local ste_start="$8"
  local ste_warmup="$9"
  local ste_max="${10}"
  local entropy_w="${11}"
  local uniform_w="${12}"
  local late_start="${13}"
  local qm_start="${14}"
  local qm_warmup="${15}"
  local qm_max="${16}"
  local pairwise_w="${17}"

  local exp_root="$ROOT/exp/$train_flag"
  mkdir -p "$exp_root/logs"
  if [ ! -e "$exp_root/eval_preproc_cache" ]; then
    ln -s "$SOURCE_CACHE_DIR" "$exp_root/eval_preproc_cache"
  fi

  local log_path="$exp_root/logs/run.log"
  echo "[LAUNCH] $train_flag on $device -> $log_path"

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
      --epoch_num 400 \
      --batch_size 256 \
      --device '$device' \
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
      --backbone_checkpoint '$BACKBONE_CKPT' \
      --freeze_backbone_epochs 0 \
      --loss_recipe improved_vq \
      --pdt_loss_start_epoch '$pdt_start' \
      --pdt_loss_weight '$pdt_weight' \
      --disable_improved_vq_adaptive_low_codebook \
      --improved_qm_start_epoch '$qm_start' \
      --improved_qm_warmup_epochs '$qm_warmup' \
      --improved_qm_max_weight '$qm_max' \
      --improved_pairwise_weight '$pairwise_w' \
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
      --late_finetune_start_epoch '$late_start' \
      --late_finetune_main_lr_scale 0.2 \
      --late_finetune_pre_quant_lr_scale 0.2
  " > "$log_path" 2>&1 &

  local pid=$!
  echo "$pid" > "$exp_root/logs/pid.txt"
  echo "[PID] $train_flag -> $pid"
}

echo "[PLAN] Four Porto-only direct-decoded ablations"
echo "[PLAN] cuda:0 balanced tuned"
echo "[PLAN] cuda:1 stronger/longer OPQ teacher"
echo "[PLAN] cuda:2 delay PDT to reduce early interference"
echo "[PLAN] cuda:3 tiny late qmetric/pairwise ablation"

launch_variant "porto_raw_m16k64_direct_balanced_e400_cuda0" "cuda:0" "0.12" "0.45" "200" "20" "0.03" "0" "20" "0.15" "0.005" "0.0002" "380" "0" "1" "0.0" "0.0"
launch_variant "porto_raw_m16k64_direct_longteacher_e400_cuda1" "cuda:1" "0.08" "0.55" "260" "20" "0.03" "0" "20" "0.15" "0.005" "0.0002" "380" "0" "1" "0.0" "0.0"
launch_variant "porto_raw_m16k64_direct_delaypdt_e400_cuda2" "cuda:2" "0.12" "0.45" "200" "40" "0.02" "0" "20" "0.15" "0.005" "0.0002" "-1" "0" "1" "0.0" "0.0"
launch_variant "porto_raw_m16k64_direct_tinyqm_e400_cuda3" "cuda:3" "0.12" "0.45" "200" "20" "0.03" "0" "20" "0.15" "0.005" "0.0002" "380" "220" "60" "0.01" "0.005"

echo "[DONE] All four runs launched."
