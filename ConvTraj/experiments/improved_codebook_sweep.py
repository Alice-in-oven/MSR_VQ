import json
import os
import subprocess
import sys


ROOT = "/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj"
TRAIN_PY = os.path.join(ROOT, "train.py")
DATA_DIR = os.path.join(ROOT, "data")
FEATURE_DIR = os.path.join(DATA_DIR, "feature_dir")
SUMMARY_PATH = os.path.join(DATA_DIR, "improved_codebook_sweep_summary.json")

DEVICE = "cuda:1"
EPOCHS = 500
RERANK_L = 100

M_VALUES = [4, 8, 16]
K_VALUES = [16, 64, 256]


def metrics_path(train_flag):
    return os.path.join(FEATURE_DIR, f"metrics_{train_flag}.json")


def log_path(train_flag):
    return os.path.join(DATA_DIR, f"{train_flag}.log")


def run_command(cmd, output_log):
    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    with open(output_log, "a", encoding="utf-8") as log_file:
        log_file.write("\n[COMMAND] {}\n".format(" ".join(cmd)))
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("Command failed with exit code {}: {}".format(return_code, " ".join(cmd)))


def load_metrics(train_flag):
    with open(metrics_path(train_flag), "r", encoding="utf-8") as handle:
        return json.load(handle)


def rerank_top10(metrics):
    return float(metrics["rerank_decoded_L100"]["top10_recall"])


def save_summary(summary):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main():
    summary = {
        "recipe": "improved_vq",
        "device": DEVICE,
        "epoch_num": EPOCHS,
        "rerank_L": RERANK_L,
        "runs": [],
    }

    shared_args = [
        "--network_type", "TJCNN_MC_MSR",
        "--image_mode", "motion6",
        "--dataset", "geolife",
        "--epoch_num", str(EPOCHS),
        "--batch_size", "256",
        "--device", DEVICE,
        "--test_epoch", str(EPOCHS),
        "--print_epoch", "25",
        "--mode", "train-directly",
        "--pdt_vq_type", "dpq",
        "--pdt_codebook_init", "faiss",
        "--eval_embedding_type", "both",
        "--eval_search_mode", "decoded",
        "--enable_rerank",
        "--rerank_L", str(RERANK_L),
        "--rerank_source", "decoded",
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
    ]

    for m in M_VALUES:
        for k in K_VALUES:
            train_flag = f"impv1_faiss_warm_m{m}_k{k}_e{EPOCHS}"
            metric_file = metrics_path(train_flag)
            if not os.path.exists(metric_file):
                cmd = [
                    "python", TRAIN_PY,
                    "--train_flag", train_flag,
                    "--pdt_m", str(m),
                    "--pdt_k", str(k),
                ] + shared_args
                run_command(cmd, log_path(train_flag))

            metrics = load_metrics(train_flag)
            summary["runs"].append({
                "train_flag": train_flag,
                "m": m,
                "k": k,
                "metrics_path": metric_file,
                "rerank_top10": rerank_top10(metrics),
                "metrics": metrics,
            })
            save_summary(summary)

    summary["runs"].sort(key=lambda item: item["rerank_top10"], reverse=True)
    summary["best"] = summary["runs"][0] if summary["runs"] else None
    save_summary(summary)


if __name__ == "__main__":
    main()
