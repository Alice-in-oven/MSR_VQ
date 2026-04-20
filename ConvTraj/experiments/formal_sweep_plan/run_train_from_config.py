import argparse
import json
import shlex
import subprocess
from pathlib import Path


STORE_TRUE_KEYS = {
    "save_feature_distance",
    "report_only_eval",
    "neutraj_use_standard_gru",
    "pre_quant_bottleneck_enabled",
    "pre_quant_use_motion_stats",
    "pre_quant_neighbor_enabled",
    "pre_quant_neighbor_use_btn",
    "pre_quant_neighbor_use_dec",
    "pre_quant_neighbor_dec_stop_backbone",
    "pre_quant_landmark_enabled",
    "pre_quant_landmark_use_btn",
    "pre_quant_landmark_use_dec",
    "pre_quant_landmark_dec_ste_to_btn",
    "decoded_ste_metric_enabled",
    "enable_rerank",
    "qinco_identity_init",
    "porto_opq_teacher_realign_codebook_on_unfreeze",
}

INVERTED_BOOL_KEYS = {
    "pdt_init_codebook": "disable_pdt_init_codebook",
    "print_code_usage": "disable_code_usage_stats",
    "improved_vq_adaptive_low_codebook": "disable_improved_vq_adaptive_low_codebook",
}

KEY_ALIASES = {
    "train_set": "train_set_size",
    "query_set": "query_set_size",
    "base_set": "base_set_size",
}

IGNORE_KEYS = {
    "lon_input_size",
    "lat_input_size",
    "save_model_path",
    "LDS",
    "FDS",
}


def build_train_command(config_path: Path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    train_py = Path(__file__).resolve().parents[2] / "train.py"
    cmd = ["python", str(train_py)]
    cmd.extend(["--config_path_placeholder_not_used", ""])  # placeholder removed below
    cmd = ["python", str(train_py)]

    for key, value in cfg.items():
        if key in IGNORE_KEYS:
            continue
        key = KEY_ALIASES.get(key, key)
        if key in STORE_TRUE_KEYS:
            if bool(value):
                cmd.append(f"--{key}")
            continue
        if key in INVERTED_BOOL_KEYS:
            if value is False:
                cmd.append(f"--{INVERTED_BOOL_KEYS[key]}")
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            cmd.extend([f"--{key}", "True" if value else "False"])
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cmd = build_train_command(config_path)
    print("[CONFIG]", config_path)
    print("[COMMAND]")
    print(" ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
