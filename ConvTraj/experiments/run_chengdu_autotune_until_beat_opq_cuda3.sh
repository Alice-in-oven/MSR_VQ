#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj"
export CUDA_INDEX=3
export DEVICE="cuda:3"

exec "$ROOT/experiments/run_chengdu_autotune_until_beat_opq_cuda0.sh"
