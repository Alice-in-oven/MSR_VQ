import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
BASE_CFG = (
    ROOT
    / "exp"
    / "chengdu_vq_teacher_stage1"
    / "train_config"
    / "chengdu_vq_teacher_stage1_cuda0.json"
)


def main():
    cfg = json.loads(BASE_CFG.read_text())
    cfg = deepcopy(cfg)

    train_flag = "chengdu_dtw_msr_pdtvq_m8_k16_bestcfg_e400_cuda2"
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

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=True))
    print(out_cfg)


if __name__ == "__main__":
    main()
