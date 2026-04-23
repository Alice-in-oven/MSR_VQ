#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto/_specific_image_cache/arrays/dtw8}"
LOG_GLOB_DEFAULT="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/exp/formal_hotfix_runs/porto_*dtw8*/logs/run.log"
LOG_GLOB="${LOG_GLOB:-$LOG_GLOB_DEFAULT}"
MODE="dry-run"

usage() {
  cat <<'EOF'
Usage:
  cleanup_porto_dtw8_cache.sh [--apply]

Behavior:
  - Default is dry-run.
  - Deletes only:
    1. files matching *.tmp-*.npy
    2. dtw8 cache files not referenced by Porto dtw8 run logs

Environment overrides:
  CACHE_DIR=/path/to/dtw8/cache
  LOG_GLOB='/path/pattern/porto_*dtw8*/logs/run.log'
EOF
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --apply)
      MODE="apply"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
fi

python - "$CACHE_DIR" "$LOG_GLOB" "$MODE" <<'PY'
import glob
import pathlib
import re
import sys

cache_dir = pathlib.Path(sys.argv[1])
log_glob = sys.argv[2]
mode = sys.argv[3]

if not cache_dir.is_dir():
    raise SystemExit(f"[ERROR] cache dir not found: {cache_dir}")

logs = sorted(glob.glob(log_glob))
keep = set()
pat = re.compile(r"/Porto/_specific_image_cache/arrays/dtw8/([0-9a-f]{40}\.npy)")

for log_path in logs:
    try:
        text = pathlib.Path(log_path).read_text(errors="ignore")
    except Exception:
        continue
    keep.update(pat.findall(text))

files = sorted(p for p in cache_dir.iterdir() if p.is_file())
remove = []
for path in files:
    name = path.name
    if ".tmp-" in name:
        remove.append(path)
    elif name not in keep:
        remove.append(path)

remove_bytes = sum(p.stat().st_size for p in remove if p.exists())
keep_files = [p for p in files if p not in remove]
keep_bytes = sum(p.stat().st_size for p in keep_files if p.exists())

print(f"[INFO] logs_scanned={len(logs)}")
for log_path in logs:
    print(f"[INFO] log={log_path}")
print(f"[INFO] cache_dir={cache_dir}")
print(f"[INFO] total_files={len(files)}")
print(f"[INFO] keep_refs_from_logs={len(keep)}")
print(f"[INFO] delete_candidates={len(remove)}")
print(f"[INFO] keep_bytes={keep_bytes}")
print(f"[INFO] delete_bytes={remove_bytes}")

for path in remove:
    size = path.stat().st_size if path.exists() else 0
    reason = "tmp" if ".tmp-" in path.name else "not-in-keep-list"
    print(f"[DELETE:{reason}] {size}\t{path}")

if mode != "apply":
    print("[INFO] dry-run only; pass --apply to delete.")
    raise SystemExit(0)

for path in remove:
    if path.exists():
        path.unlink()
        print(f"[REMOVED] {path}")

print("[INFO] apply done.")
PY
