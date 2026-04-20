#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import subprocess
from pathlib import Path

queue_path = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/experiments/formal_sweep_plan/queue_lists/cuda3_faiss_queue.json")
faiss_script = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/experiments/faiss_pq_opq_quantize.py")

with open(queue_path) as f:
    rows = json.load(f)

for idx, row in enumerate(rows, 1):
    missing = [key for key in ["train_path", "query_path", "base_path", "gt_path"] if not Path(row[key]).exists()]
    if missing:
        print(f"[FAISS-SKIP] missing={missing} train_flag={row['train_flag']}", flush=True)
        continue
    cmd = [
        "python", str(faiss_script),
        "--input_mode", "split",
        "--train_path", row["train_path"],
        "--query_path", row["query_path"],
        "--base_path", row["base_path"],
        "--quantizer", row["quantizer"],
        "--m", str(row["m"]),
        "--nbits", str(row["nbits"]),
        "--gt_path", row["gt_path"],
        "--output_dir", row["output_dir"],
        "--artifact_prefix", row["artifact_prefix"],
    ]
    if row["quantizer"] == "opq":
        cmd.extend(["--opq_niter", "50", "--save_rotated"])
    print(f"[FAISS] {idx}/{len(rows)} :: {row['train_flag']}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[FAISS-STOP] rc={rc} train_flag={row['train_flag']}", flush=True)
        raise SystemExit(rc)
print("[FAISS-DONE]", queue_path, flush=True)
PY
