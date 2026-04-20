import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
BASE_CFG = (
    ROOT
    / "exp"
    / "porto_raw_backbone400_faiss_m8nb4"
    / "train_config"
    / "porto_raw_backbone_only_e400_cuda1.json"
)


def main():
    cfg = json.loads(BASE_CFG.read_text())
    cfg = deepcopy(cfg)

    train_flag = "porto_dtw_msr_continuous_e600_cuda3"
    root_write = ROOT / "exp" / "formal_hotfix_runs" / train_flag
    out_cfg = (
        ROOT
        / "experiments"
        / "formal_sweep_plan"
        / "generated_configs"
        / "hotfix"
        / f"{train_flag}.json"
    )

    cfg["train_flag"] = train_flag
    cfg["artifact_prefix"] = train_flag
    cfg["root_write_path"] = str(root_write)
    cfg["save_model_path"] = str(root_write / "checkpoints")
    cfg["device"] = "cuda:0"

    cfg["epoch_num"] = 600
    cfg["test_epoch"] = 600
    cfg["eval_save_epochs"] = "600"
    cfg["save_model_epoch"] = 100
    cfg["eval_embedding_type"] = "continuous"
    cfg["eval_search_mode"] = "decoded"
    cfg["enable_rerank"] = False
    cfg["rerank_source"] = "decoded"

    cfg["pdt_m"] = 8
    cfg["pdt_k"] = 16
    cfg["pdt_codebook_init"] = "uniform"
    cfg["pdt_init_codebook"] = False
    cfg["pdt_loss_start_epoch"] = 999999
    cfg["pdt_loss_weight"] = 0.0
    cfg["loss_recipe"] = "baseline"
    cfg["decoded_ste_metric_enabled"] = False

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=True))
    print(out_cfg)


if __name__ == "__main__":
    main()
