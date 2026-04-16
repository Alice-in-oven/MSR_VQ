import json
import os
import subprocess
import sys
from copy import deepcopy


ROOT = "/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj"
TRAIN_PY = os.path.join(ROOT, "train.py")
DATA_DIR = os.path.join(ROOT, "data")
FEATURE_DIR = os.path.join(DATA_DIR, "feature_dir")
SUMMARY_PATH = os.path.join(DATA_DIR, "incremental_pipeline_m8_k16_summary.json")


def metrics_path(train_flag):
    return os.path.join(FEATURE_DIR, f"metrics_{train_flag}.json")


def train_log_path(train_flag):
    return os.path.join(DATA_DIR, f"{train_flag}.log")


def run_command(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n[COMMAND] {}\n".format(" ".join(cmd)))
        log_file.flush()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("Command failed with exit code {}: {}".format(return_code, " ".join(cmd)))


def load_metrics(train_flag):
    with open(metrics_path(train_flag), "r", encoding="utf-8") as f:
        return json.load(f)


def section_top10(metrics, section):
    if section is None:
        return float(metrics["top10_recall"])
    payload = metrics.get(section)
    if payload is None:
        return float("-inf")
    if isinstance(payload, dict) and payload.get("available", True) is False:
        return float("-inf")
    return float(payload["top10_recall"])


def choose_best_search_section(metrics):
    candidates = []
    if "decoded" in metrics:
        candidates.append(("decoded", section_top10(metrics, "decoded")))
    if "adc" in metrics and isinstance(metrics["adc"], dict) and metrics["adc"].get("available", True) is not False:
        candidates.append(("adc", section_top10(metrics, "adc")))
    if not candidates and "top10_recall" in metrics:
        candidates.append((None, float(metrics["top10_recall"])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def train_or_eval(train_flag, extra_args, device, mode="train-directly", load_model_train_flag=None, force=False):
    metric_file = metrics_path(train_flag)
    if os.path.exists(metric_file) and not force:
        return load_metrics(train_flag)

    checkpoint_flag = load_model_train_flag or train_flag
    checkpoint_path = os.path.join(DATA_DIR, "model", f"{checkpoint_flag}_epochs_200")
    effective_mode = mode
    if mode == "train-directly" and os.path.exists(checkpoint_path):
        effective_mode = "test"

    cmd = [
        "python", TRAIN_PY,
        "--network_type", "TJCNN_MC_MSR",
        "--image_mode", "motion6",
        "--dataset", "geolife",
        "--epoch_num", "200",
        "--batch_size", "256",
        "--device", device,
        "--train_flag", train_flag,
        "--test_epoch", "200",
        "--print_epoch", "10",
        "--pdt_m", "8",
        "--pdt_k", "16",
        "--mode", effective_mode,
    ] + extra_args
    if load_model_train_flag is not None:
        cmd.extend(["--load_model_train_flag", load_model_train_flag])
    run_command(cmd, train_log_path(train_flag))
    return load_metrics(train_flag)


def make_record(step_name, train_flag, metrics, primary_section, pdt_vq_type):
    return {
        "step_name": step_name,
        "train_flag": train_flag,
        "pdt_vq_type": pdt_vq_type,
        "metrics_path": metrics_path(train_flag),
        "primary_section": primary_section,
        "primary_top10_recall": section_top10(metrics, primary_section),
        "metrics": metrics,
    }


def save_summary(summary):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    summary = {
        "primary_metric": "top10_recall",
        "base_config": {
            "epoch_num": 200,
            "pdt_m": 8,
            "pdt_k": 16,
        },
        "steps": [],
        "discarded_steps": [],
    }

    current_best = None

    baseline_flag = "inc0_baseline_dpq_m8k16_e200"
    baseline_metrics = train_or_eval(
        baseline_flag,
        [
            "--pdt_vq_type", "dpq",
            "--pdt_codebook_init", "uniform",
            "--eval_embedding_type", "both",
            "--eval_search_mode", "both",
        ],
        device="cuda:1",
    )
    baseline_section, _ = choose_best_search_section(baseline_metrics)
    current_best = make_record("baseline", baseline_flag, baseline_metrics, baseline_section, "dpq")
    summary["steps"].append(current_best)
    save_summary(summary)

    qinco_flag = "inc2_qinco_m8k16_e200"
    qinco_metrics = train_or_eval(
        qinco_flag,
        [
            "--pdt_vq_type", "qinco",
            "--pdt_codebook_init", "uniform",
            "--eval_embedding_type", "both",
            "--eval_search_mode", "decoded",
        ],
        device="cuda:1",
    )
    qinco_section, _ = choose_best_search_section(qinco_metrics)
    qinco_record = make_record("qinco", qinco_flag, qinco_metrics, qinco_section, "qinco")
    if qinco_record["primary_top10_recall"] > current_best["primary_top10_recall"]:
        current_best = qinco_record
        summary["steps"].append(qinco_record)
    else:
        summary["discarded_steps"].append(qinco_record)
    save_summary(summary)

    recipe_candidates = [
        ("consistency", "inc3_consistency", "cuda:1", []),
        # Restrict all follow-up runs to cuda:1 and keep a slightly smaller batch size
        # for the quantized_metric recipe to reduce peak memory pressure.
        ("quantized_metric", "inc3_quantized_metric", "cuda:1", ["--batch_size", "192"]),
    ]
    base_recipe = current_best
    for recipe_name, flag_prefix, recipe_device, recipe_extra_args in recipe_candidates:
        candidate_flag = "{}_{}".format(flag_prefix, base_recipe["train_flag"])
        candidate_metrics = train_or_eval(
            candidate_flag,
            [
                "--pdt_vq_type", base_recipe["pdt_vq_type"],
                "--pdt_codebook_init", "uniform",
                "--eval_embedding_type", "both",
                "--eval_search_mode", "both" if base_recipe["pdt_vq_type"] == "dpq" else "decoded",
                "--loss_recipe", recipe_name,
            ] + recipe_extra_args,
            device=recipe_device,
        )
        candidate_section, _ = choose_best_search_section(candidate_metrics)
        candidate_record = make_record(recipe_name, candidate_flag, candidate_metrics, candidate_section, base_recipe["pdt_vq_type"])
        if candidate_record["primary_top10_recall"] > current_best["primary_top10_recall"]:
            current_best = candidate_record
            summary["steps"].append(candidate_record)
            base_recipe = candidate_record
        else:
            summary["discarded_steps"].append(candidate_record)
        save_summary(summary)

    if base_recipe["step_name"] == "consistency":
        combo_flag = "inc3_consistency_quantized_metric_{}".format(base_recipe["train_flag"])
        combo_metrics = train_or_eval(
            combo_flag,
            [
                "--pdt_vq_type", base_recipe["pdt_vq_type"],
                "--pdt_codebook_init", "uniform",
                "--eval_embedding_type", "both",
                "--eval_search_mode", "both" if base_recipe["pdt_vq_type"] == "dpq" else "decoded",
                "--loss_recipe", "consistency_quantized_metric",
            ],
            device="cuda:1",
        )
        combo_section, _ = choose_best_search_section(combo_metrics)
        combo_record = make_record("consistency_quantized_metric", combo_flag, combo_metrics, combo_section, base_recipe["pdt_vq_type"])
        if combo_record["primary_top10_recall"] > current_best["primary_top10_recall"]:
            current_best = combo_record
            summary["steps"].append(combo_record)
        else:
            summary["discarded_steps"].append(combo_record)
        save_summary(summary)

    rerank_flag = "inc4_rerank_{}_{}".format(current_best["primary_section"] or "decoded", current_best["train_flag"])
    rerank_metrics = train_or_eval(
        rerank_flag,
        [
            "--pdt_vq_type", current_best["pdt_vq_type"],
            "--pdt_codebook_init", "uniform",
            "--eval_embedding_type", "both",
            "--eval_search_mode", "both" if current_best["primary_section"] == "adc" else "decoded",
            "--enable_rerank",
            "--rerank_L", "100",
            "--rerank_source", current_best["primary_section"] or "decoded",
        ],
        device="cuda:1",
        mode="test",
        load_model_train_flag=current_best["train_flag"],
    )
    rerank_key = "rerank_{}_L100".format(current_best["primary_section"] or "decoded")
    rerank_record = make_record("rerank", rerank_flag, rerank_metrics, rerank_key, current_best["pdt_vq_type"])
    if rerank_record["primary_top10_recall"] > current_best["primary_top10_recall"]:
        summary["steps"].append(rerank_record)
        current_best = rerank_record
    else:
        summary["discarded_steps"].append(rerank_record)
    summary["final_best"] = current_best
    save_summary(summary)


if __name__ == "__main__":
    main()
