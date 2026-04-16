import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
CONVTRAJ_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = CONVTRAJ_ROOT.parent
DATA_ROOT = CONVTRAJ_ROOT / "data"
FEATURE_DIR = DATA_ROOT / "feature_dir"
TRAIN_SCRIPT = CONVTRAJ_ROOT / "train.py"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def summary_path(train_flag):
    return FEATURE_DIR / "summary_{}.json".format(train_flag)


def smoke_summary_path(train_flag):
    return FEATURE_DIR / "smoke_summary_{}.json".format(train_flag)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_and_stream(cmd, log_path):
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("[RUN]", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(CONVTRAJ_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("Command failed with exit code {}: {}".format(return_code, " ".join(cmd)))


def common_args(device):
    return [
        "--network_type", "TJCNN_MC_MSR",
        "--image_mode", "motion6",
        "--dataset", "geolife",
        "--device", device,
        "--root_write_path", str(DATA_ROOT),
        "--root_read_path", str(DATA_ROOT),
        "--pdt_m", "8",
        "--pdt_k", "16",
        "--pdt_vq_type", "dpq",
        "--pdt_codebook_init", "faiss",
        "--loss_recipe", "improved_vq",
        "--pdt_loss_weight", "0.1",
        "--pdt_loss_start_epoch", "40",
        "--improved_qm_start_epoch", "80",
        "--improved_qm_warmup_epochs", "80",
        "--improved_qm_max_weight", "0.08",
        "--improved_pairwise_weight", "0.05",
        "--improved_entropy_weight", "0.02",
        "--improved_commit_weight", "0.05",
        "--improved_uniform_weight", "0.001",
        "--eval_embedding_type", "both",
        "--eval_search_mode", "decoded",
        "--enable_rerank",
        "--rerank_L", "100",
        "--rerank_source", "decoded",
        "--learning_rate", "0.001",
        "--random_seed", "666",
    ]


def build_command(device, train_flag, mode, epochs, batch_size, bottleneck_enabled=False, extra_args=None):
    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    cmd.extend(common_args(device))
    cmd.extend([
        "--train_flag", train_flag,
        "--mode", mode,
        "--epoch_num", str(epochs),
        "--batch_size", str(batch_size),
    ])
    if bottleneck_enabled:
        cmd.extend([
            "--pre_quant_bottleneck_enabled",
            "--pre_quant_global_dim", "48",
            "--pre_quant_local_dim", "48",
            "--pre_quant_progress_dim", "32",
            "--pre_quant_lambda_decor", "0.01",
            "--pre_quant_lambda_stab", "0.1",
        ])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_smoke(device):
    configs = [
        ("prequant_smoke_baseline", False),
        ("prequant_smoke_bottleneck", True),
    ]
    results = {}
    for train_flag, enabled in configs:
        cmd = build_command(
            device=device,
            train_flag=train_flag,
            mode="smoke-test",
            epochs=1,
            batch_size=8,
            bottleneck_enabled=enabled,
            extra_args=[
                "--train_set_size", "64",
                "--query_set_size", "16",
                "--base_set_size", "64",
                "--test_epoch", "1",
                "--print_epoch", "1",
                "--rerank_L", "20",
                "--max_train_batches_per_epoch", "1",
            ],
        )
        log_path = DATA_ROOT / "{}.log".format(train_flag)
        run_and_stream(cmd, log_path)
        results[train_flag] = load_json(smoke_summary_path(train_flag))
    return results


def run_train_pair(stage_name,
                   device,
                   epochs,
                   batch_size,
                   train_set_size=None,
                   query_set_size=None,
                   base_set_size=None,
                   max_train_batches_per_epoch=0,
                   test_epoch=1,
                   print_epoch=1,
                   baseline_test_epoch=None,
                   bottleneck_test_epoch=None):
    configs = [
        ("{}_baseline".format(stage_name), False),
        ("{}_bottleneck".format(stage_name), True),
    ]
    results = {}
    for train_flag, enabled in configs:
        mode_test_epoch = test_epoch
        if enabled and bottleneck_test_epoch is not None:
            mode_test_epoch = int(bottleneck_test_epoch)
        if (not enabled) and baseline_test_epoch is not None:
            mode_test_epoch = int(baseline_test_epoch)
        extra_args = [
            "--test_epoch", str(mode_test_epoch),
            "--print_epoch", str(print_epoch),
        ]
        if train_set_size is not None:
            extra_args.extend(["--train_set_size", str(train_set_size)])
        if query_set_size is not None:
            extra_args.extend(["--query_set_size", str(query_set_size)])
        if base_set_size is not None:
            extra_args.extend(["--base_set_size", str(base_set_size)])
        if max_train_batches_per_epoch > 0:
            extra_args.extend(["--max_train_batches_per_epoch", str(max_train_batches_per_epoch)])
        cmd = build_command(
            device=device,
            train_flag=train_flag,
            mode="train-directly",
            epochs=epochs,
            batch_size=batch_size,
            bottleneck_enabled=enabled,
            extra_args=extra_args,
        )
        log_path = DATA_ROOT / "{}.log".format(train_flag)
        run_and_stream(cmd, log_path)
        results[train_flag] = load_json(summary_path(train_flag))
    return results


def collect_comparison(stage_name, summaries):
    comparison = {
        "stage_name": stage_name,
        "rows": [],
    }
    for train_flag, summary in summaries.items():
        best_records = summary.get("best_records", {})
        comparison["rows"].append({
            "train_flag": train_flag,
            "best_continuous_top10": (best_records.get("continuous") or {}).get("top10_recall"),
            "best_decoded_top10": (best_records.get("decoded") or {}).get("top10_recall"),
            "best_strict_rerank_top10": (best_records.get("strict_rerank") or {}).get("top10_recall"),
            "best_continuous_ckpt": (best_records.get("continuous") or {}).get("checkpoint_path"),
            "best_decoded_ckpt": (best_records.get("decoded") or {}).get("checkpoint_path"),
            "best_strict_rerank_ckpt": (best_records.get("strict_rerank") or {}).get("checkpoint_path"),
            "summary_path": str(summary_path(train_flag)),
        })
    out_path = FEATURE_DIR / "comparison_{}.json".format(stage_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print("[COMPARISON] saved to {}".format(out_path))
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all", choices=["smoke", "sanity", "train150", "all"])
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--baseline_test_epoch", type=int, default=None)
    parser.add_argument("--bottleneck_test_epoch", type=int, default=None)
    parser.add_argument("--train150_stage_name", type=str, default="prequant_train150")
    args = parser.parse_args()

    ensure_dir(FEATURE_DIR)
    results = {}

    if args.stage in ["smoke", "all"]:
        print("[STAGE] smoke")
        results["smoke"] = run_smoke(args.device)

    if args.stage in ["sanity", "all"]:
        print("[STAGE] sanity")
        sanity_results = run_train_pair(
            stage_name="prequant_sanity5",
            device=args.device,
            epochs=5,
            batch_size=16,
            train_set_size=256,
            query_set_size=64,
            base_set_size=256,
            max_train_batches_per_epoch=4,
            test_epoch=1,
            print_epoch=1,
        )
        results["sanity"] = collect_comparison("prequant_sanity5", sanity_results)

    if args.stage in ["train150", "all"]:
        print("[STAGE] train150")
        train150_results = run_train_pair(
            stage_name=args.train150_stage_name,
            device=args.device,
            epochs=150,
            batch_size=96,
            train_set_size=None,
            query_set_size=None,
            base_set_size=None,
            max_train_batches_per_epoch=0,
            test_epoch=10,
            print_epoch=10,
            baseline_test_epoch=args.baseline_test_epoch,
            bottleneck_test_epoch=args.bottleneck_test_epoch,
        )
        results["train150"] = collect_comparison(args.train150_stage_name, train150_results)

    overall_path = FEATURE_DIR / "suite_results.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[SUITE] results saved to {}".format(overall_path))


if __name__ == "__main__":
    main()
