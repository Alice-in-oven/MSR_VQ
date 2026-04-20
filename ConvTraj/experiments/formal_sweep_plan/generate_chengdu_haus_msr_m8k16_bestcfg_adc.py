import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
BASE_CFG = ROOT / "exp" / "chengdu_vq_teacher_stage1" / "train_config" / "chengdu_vq_teacher_stage1_cuda0.json"
OUT_CFG = ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "hotfix" / "chengdu_haus_msr_pdtvq_m8_k16_bestcfg_e400_trueadc.json"


def main():
    cfg = json.loads(BASE_CFG.read_text())
    cfg = deepcopy(cfg)

    train_flag = "chengdu_haus_msr_pdtvq_m8_k16_bestcfg_e400_trueadc"
    root_write = ROOT / "exp" / "formal_hotfix_runs" / train_flag

    cfg["train_flag"] = train_flag
    cfg["artifact_prefix"] = train_flag
    cfg["root_write_path"] = str(root_write)
    cfg["save_model_path"] = str(root_write / "checkpoints")

    cfg["epoch_num"] = 400
    cfg["test_epoch"] = 400
    cfg["eval_save_epochs"] = "300,400"
    cfg["save_model_epoch"] = 100

    cfg["dataset"] = "chengdu"
    cfg["dist_type"] = "haus"
    cfg["root_read_path"] = str(ROOT / "data" / "0_chengdu")
    cfg["device"] = "cuda:0"
    cfg["train_set_size"] = 1000
    cfg["query_set_size"] = 1000
    cfg["base_set_size"] = 3000

    cfg["eval_embedding_type"] = "both"
    cfg["eval_search_mode"] = "adc"
    cfg["enable_rerank"] = False
    cfg["rerank_source"] = "adc"

    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CFG.write_text(json.dumps(cfg, ensure_ascii=True))
    print(OUT_CFG)


if __name__ == "__main__":
    main()
