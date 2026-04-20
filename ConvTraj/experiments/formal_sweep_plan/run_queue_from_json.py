import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=str)
    parser.add_argument("--runner", required=True, type=str)
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    runner = Path(args.runner).resolve()

    with open(queue_path, "r") as f:
        rows = json.load(f)

    for idx, row in enumerate(rows, 1):
        cmd = ["python", str(runner), "--config", row["config_path"]]
        print(f"[QUEUE] {idx}/{len(rows)} :: {row['train_flag']}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[QUEUE-STOP] rc={rc} train_flag={row['train_flag']}", flush=True)
            raise SystemExit(rc)

    print(f"[QUEUE-DONE] {queue_path}", flush=True)


if __name__ == "__main__":
    main()
