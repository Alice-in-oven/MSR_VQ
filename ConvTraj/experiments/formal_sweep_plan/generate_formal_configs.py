import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
OUT_ROOT = ROOT / "experiments" / "formal_sweep_plan" / "generated_configs"
RUN_ROOT = ROOT / "exp" / "formal_sweep_runs"
FAISS_EVAL_EPOCHS = [400, 500]

GEOLIFE_SWEEP_CFG_DIR = ROOT / "data" / "train_config"
CHENGDU_MAIN_CFG = ROOT / "exp" / "chengdu_vq_teacher_stage1" / "train_config" / "chengdu_vq_teacher_stage1_cuda0.json"
PORTO_MAIN_CFG = ROOT / "exp" / "porto_raw_m16k64_opqcode_frozen700_slowhandoff_cuda1" / "train_config" / "porto_raw_m16k64_opqcode_frozen700_slowhandoff_cuda1.json"
NEU_CFG = ROOT / "exp" / "neu_lightprequant_vq" / "train_config" / "neu_lightprequant_vq_m8_k16_e400.json"
SIM_CFG = ROOT / "exp" / "sim_lightprequant_vq" / "train_config" / "sim_lightprequant_vq_m8_k16_e400.json"

CODEBOOKS = [(16, 64)]
DATASETS = ["geolife", "chengdu", "porto"]
DISTS = ["haus", "dtw", "dfd"]
MODELS = ["msr", "neutraj", "simformer"]

DATASET_ROOTS = {
    "geolife": str(ROOT / "data" / "0_geolife"),
    "chengdu": str(ROOT / "data" / "0_chengdu"),
    "porto": "/data3/menghaotian/Traj_sim/SequenceDistanceDataset/Trajectory/Porto",
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=True)


def assign_training_queue(train_flag: str):
    if train_flag.startswith("porto_"):
        return "cuda0"
    if train_flag.startswith("chengdu_"):
        return "cuda1"
    if train_flag.startswith("geolife_") and "_continuous_" not in train_flag:
        if "_simformer_" in train_flag:
            return "cuda3"
        return "cuda2"
    if train_flag.startswith("geolife_") and "_continuous_" in train_flag:
        return "cuda3"
    raise ValueError(f"Unrecognized train_flag for queue assignment: {train_flag}")


def geolife_msr_seed(m: int, k: int):
    path = GEOLIFE_SWEEP_CFG_DIR / f"prequant_v5_stedec_sweep_m{m}_k{k}_e500.json"
    return load_json(path)


def dataset_main_recipe(dataset: str, m: int, k: int):
    if dataset == "geolife":
        return geolife_msr_seed(m, k)
    if dataset == "chengdu":
        return load_json(CHENGDU_MAIN_CFG)
    if dataset == "porto":
        return load_json(PORTO_MAIN_CFG)
    raise ValueError(dataset)


def model_seed(model: str, dataset: str, m: int, k: int):
    if model == "msr":
        return dataset_main_recipe(dataset, m, k)
    if model == "neutraj":
        return load_json(NEU_CFG)
    if model == "simformer":
        return load_json(SIM_CFG)
    raise ValueError(model)


def schedule_for_dataset(dataset: str):
    if dataset in ["geolife", "chengdu"]:
        return {
            "epoch_num": 500,
            "test_epoch": 500,
            "eval_save_epochs": "400,500",
        }
    if dataset == "porto":
        return {
            "epoch_num": 600,
            "test_epoch": 600,
            "eval_save_epochs": "600",
        }
    raise ValueError(dataset)


def apply_common_schedule(cfg, train_flag, dataset, dist, m, k):
    schedule = schedule_for_dataset(dataset)
    cfg["train_flag"] = train_flag
    cfg["dataset"] = dataset
    cfg["dist_type"] = dist
    cfg["root_read_path"] = DATASET_ROOTS[dataset]
    cfg["pdt_m"] = m
    cfg["pdt_k"] = k
    cfg["epoch_num"] = schedule["epoch_num"]
    cfg["test_epoch"] = schedule["test_epoch"]
    cfg["eval_save_epochs"] = schedule["eval_save_epochs"]
    cfg["save_model_epoch"] = 100
    cfg["save_model"] = True
    cfg["eval_embedding_type"] = "both"
    cfg["eval_search_mode"] = "adc"
    cfg["enable_rerank"] = False
    cfg["rerank_source"] = "adc"
    cfg["rerank_L"] = 100
    cfg["report_only_eval"] = False
    cfg["save_feature_distance"] = False
    cfg["mode"] = "train-directly"
    cfg["device"] = "cuda:0"
    cfg["artifact_prefix"] = train_flag
    cfg["root_write_path"] = str(RUN_ROOT / train_flag)
    cfg["save_model_path"] = str((RUN_ROOT / train_flag) / "checkpoints")
    return cfg


def overlay_recipe(seed_cfg, recipe_cfg, dataset, model):
    merged = deepcopy(seed_cfg)
    overlay_keys = [
        "loss_recipe",
        "pdt_loss_start_epoch",
        "pdt_loss_weight",
        "freeze_backbone_epochs",
        "decoded_ste_metric_enabled",
        "decoded_ste_metric_start_epoch",
        "decoded_ste_metric_warmup_epochs",
        "decoded_ste_metric_max_weight",
        "pre_quant_bottleneck_enabled",
        "pre_quant_use_motion_stats",
        "pre_quant_global_dim",
        "pre_quant_local_dim",
        "pre_quant_progress_dim",
        "pre_quant_lambda_decor",
        "pre_quant_lambda_stab",
        "pre_quant_residual_alpha_init",
        "pre_quant_lr_multiplier",
        "pre_quant_stab_late_epoch",
        "pre_quant_stab_late_multiplier",
        "pre_quant_refresh_start_epoch",
        "pre_quant_refresh_period",
        "pre_quant_refresh_end_epoch",
        "pre_quant_raw_metric_weight",
        "quantized_metric_weight",
        "improved_qm_start_epoch",
        "improved_qm_warmup_epochs",
        "improved_qm_max_weight",
        "improved_pairwise_weight",
        "improved_entropy_weight",
        "improved_commit_weight",
        "improved_uniform_weight",
        "improved_vq_adaptive_low_codebook",
        "late_finetune_start_epoch",
        "late_finetune_main_lr_scale",
        "late_finetune_pre_quant_lr_scale",
        "pdt_codebook_init",
        "pdt_vq_type",
        "qinco_h",
        "qinco_L",
        "qinco_identity_init",
        "pdt_init_codebook",
    ]
    for key in overlay_keys:
        if key in recipe_cfg:
            merged[key] = recipe_cfg[key]

    if model == "msr":
        for key in [
            "backbone_checkpoint",
            "porto_opq_teacher_rotated_train_path",
            "porto_opq_teacher_codebook_path",
            "porto_opq_teacher_start_epoch",
            "porto_opq_teacher_end_epoch",
            "porto_opq_teacher_z_weight",
            "porto_opq_teacher_partition_weight",
            "porto_opq_teacher_codebook_freeze_end_epoch",
            "porto_opq_teacher_realign_codebook_on_unfreeze",
            "porto_opq_warmup_train_recon_path",
            "porto_opq_warmup_start_epoch",
            "porto_opq_warmup_end_epoch",
            "porto_opq_warmup_max_weight",
        ]:
            if key in recipe_cfg:
                merged[key] = recipe_cfg[key]
    else:
        merged["backbone_checkpoint"] = None
        merged["porto_opq_teacher_rotated_train_path"] = ""
        merged["porto_opq_teacher_codebook_path"] = ""
        merged["porto_opq_teacher_start_epoch"] = 0
        merged["porto_opq_teacher_end_epoch"] = 0
        merged["porto_opq_teacher_z_weight"] = 0.0
        merged["porto_opq_teacher_partition_weight"] = 0.0
        merged["porto_opq_teacher_codebook_freeze_end_epoch"] = 0
        merged["porto_opq_teacher_realign_codebook_on_unfreeze"] = False
        merged["porto_opq_warmup_train_recon_path"] = ""
        merged["porto_opq_warmup_start_epoch"] = 0
        merged["porto_opq_warmup_end_epoch"] = 0
        merged["porto_opq_warmup_max_weight"] = 0.0

    if dataset == "porto" and model != "msr":
        merged["pre_quant_bottleneck_enabled"] = False

    return merged


def build_quantized_config(dataset, dist, model, m, k):
    end_epoch = schedule_for_dataset(dataset)["epoch_num"]
    train_flag = f"{dataset}_{dist}_{model}_pdtvq_m{m}_k{k}_e{end_epoch}_trueadc"
    seed_cfg = model_seed(model, dataset, m, k)
    recipe_cfg = dataset_main_recipe(dataset, m, k)
    cfg = overlay_recipe(seed_cfg, recipe_cfg, dataset, model)
    cfg = apply_common_schedule(cfg, train_flag, dataset, dist, m, k)
    return cfg


def build_continuous_config(dist):
    base = geolife_msr_seed(16, 64)
    train_flag = f"geolife_{dist}_msr_continuous_e500"
    cfg = deepcopy(base)
    cfg["embedding_backbone"] = "msr"
    cfg["train_flag"] = train_flag
    cfg["dataset"] = "geolife"
    cfg["dist_type"] = dist
    cfg["root_read_path"] = DATASET_ROOTS["geolife"]
    cfg["epoch_num"] = 500
    cfg["test_epoch"] = 500
    cfg["eval_save_epochs"] = "400,500"
    cfg["save_model_epoch"] = 100
    cfg["save_model"] = True
    cfg["mode"] = "train-directly"
    cfg["device"] = "cuda:0"
    cfg["artifact_prefix"] = train_flag
    cfg["root_write_path"] = str(RUN_ROOT / train_flag)
    cfg["save_model_path"] = str((RUN_ROOT / train_flag) / "checkpoints")
    cfg["eval_embedding_type"] = "continuous"
    cfg["eval_search_mode"] = "decoded"
    cfg["enable_rerank"] = False
    cfg["loss_recipe"] = "baseline"
    cfg["pdt_loss_weight"] = 0.0
    cfg["pdt_loss_start_epoch"] = 999999
    cfg["pdt_init_codebook"] = False
    cfg["pre_quant_bottleneck_enabled"] = False
    cfg["decoded_ste_metric_enabled"] = False
    cfg["backbone_checkpoint"] = None
    return cfg


def generate_all():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    quant_root = OUT_ROOT / "quantized_main"
    cont_root = OUT_ROOT / "continuous"
    queue_dir = ROOT / "experiments" / "formal_sweep_plan" / "queue_lists"
    manifest = []
    faiss_manifest = []

    for dataset in DATASETS:
        for dist in DISTS:
            for model in MODELS:
                for m, k in CODEBOOKS:
                    cfg = build_quantized_config(dataset, dist, model, m, k)
                    path = quant_root / dataset / model / f"{cfg['train_flag']}.json"
                    save_json(path, cfg)
                    manifest.append({"group": "quantized_main", "config_path": str(path), "train_flag": cfg["train_flag"]})

    for dist in DISTS:
        cfg = build_continuous_config(dist)
        path = cont_root / f"{cfg['train_flag']}.json"
        save_json(path, cfg)
        manifest.append({"group": "continuous_train", "config_path": str(path), "train_flag": cfg["train_flag"]})
        artifact_prefix = cfg["artifact_prefix"]
        embeddings_dir = Path(cfg["root_write_path"]) / "embeddings"
        gt_path = Path(cfg["root_read_path"]) / f"{dist}_test_distance_matrix_result"
        for epoch in FAISS_EVAL_EPOCHS:
            for quantizer in ["pq", "opq"]:
                faiss_train_flag = f"geolife_{dist}_{quantizer}_m16_k64_from_cont_e{epoch}"
                faiss_manifest.append({
                    "group": "faiss_offline",
                    "dataset": "geolife",
                    "dist": dist,
                    "quantizer": quantizer,
                    "m": 16,
                    "k": 64,
                    "nbits": 6,
                    "epoch": epoch,
                    "train_flag": faiss_train_flag,
                    "continuous_train_flag": cfg["train_flag"],
                    "train_path": str(embeddings_dir / f"{artifact_prefix}_continuous_train_e{epoch}.pkl"),
                    "query_path": str(embeddings_dir / f"{artifact_prefix}_continuous_query_e{epoch}.pkl"),
                    "base_path": str(embeddings_dir / f"{artifact_prefix}_continuous_base_e{epoch}.pkl"),
                    "gt_path": str(gt_path),
                    "output_dir": str(Path(cfg["root_write_path"]) / "faiss_eval" / f"e{epoch}" / quantizer),
                    "artifact_prefix": faiss_train_flag,
                })

    manifest_path = OUT_ROOT / "generated_config_manifest.json"
    save_json(manifest_path, manifest)
    flat_plan_dir = ROOT / "experiments" / "formal_sweep_plan"
    train_queues = {"cuda0": [], "cuda1": [], "cuda2": [], "cuda3": []}
    for row in manifest:
        queue_name = assign_training_queue(row["train_flag"])
        train_queues[queue_name].append(row)
    flat_rows = []
    for row in manifest:
        cfg = load_json(Path(row["config_path"]))
        flat_rows.append({
            "group": row["group"],
            "dataset": cfg["dataset"],
            "dist": cfg["dist_type"],
            "model": cfg.get("embedding_backbone", "msr") if row["group"] == "quantized_main" else "msr_continuous",
            "m": cfg.get("pdt_m", "-"),
            "k": cfg.get("pdt_k", "-"),
            "train_flag": cfg["train_flag"],
            "epoch_num": cfg["epoch_num"],
            "eval_epochs": cfg["eval_save_epochs"],
            "ckpt_every": cfg["save_model_epoch"],
            "eval_embedding_type": cfg["eval_embedding_type"],
            "eval_search_mode": cfg["eval_search_mode"],
            "config_path": row["config_path"],
        })
    save_json(flat_plan_dir / "formal_experiment_manifest.json", flat_rows)
    save_json(flat_plan_dir / "faiss_manifest.json", faiss_manifest)
    for queue_name, rows in train_queues.items():
        save_json(queue_dir / f"{queue_name}_queue.json", rows)
    save_json(queue_dir / "cuda3_faiss_queue.json", faiss_manifest)
    try:
        import csv
        with open(flat_plan_dir / "formal_experiment_manifest.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
        if faiss_manifest:
            with open(flat_plan_dir / "faiss_manifest.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(faiss_manifest[0].keys()))
                writer.writeheader()
                writer.writerows(faiss_manifest)
    except Exception:
        pass
    print(manifest_path)
    print("generated", len(manifest), "configs")
    print(flat_plan_dir / "faiss_manifest.json")
    print("generated", len(faiss_manifest), "faiss tasks")


if __name__ == "__main__":
    generate_all()
