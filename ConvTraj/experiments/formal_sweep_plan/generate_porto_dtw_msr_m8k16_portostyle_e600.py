import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
BASE_CFG = (
    ROOT
    / "exp"
    / "porto_raw_m16k64_opqcode_frozen700_slowhandoff_cuda1"
    / "train_config"
    / "porto_raw_m16k64_opqcode_frozen700_slowhandoff_cuda1.json"
)
BACKBONE_CKPT = (
    ROOT
    / "exp"
    / "porto_raw_backbone400_faiss_m8nb4"
    / "model"
    / "porto_raw_backbone_only_e400_cuda1_epochs_400"
)
OPQ_DIR = ROOT / "exp" / "porto_raw_backbone400_faiss_m8nb4" / "faiss_opq_m8_nbits4"
OPQ_ROTATED_TRAIN = OPQ_DIR / "porto_raw_backbone_cont_opq_m8_nbits4_train_faiss_rotated.pkl"
OPQ_CODEBOOK = OPQ_DIR / "porto_raw_backbone_cont_opq_m8_nbits4_faiss_pq_codebook.npy"


def main():
    cfg = json.loads(BASE_CFG.read_text())
    cfg = deepcopy(cfg)

    train_flag = "porto_dtw_msr_pdtvq_m8_k16_portostyle_e600_trueadc"
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

    cfg["pdt_m"] = 8
    cfg["pdt_k"] = 16
    cfg["backbone_checkpoint"] = str(BACKBONE_CKPT)

    cfg["porto_opq_teacher_rotated_train_path"] = str(OPQ_ROTATED_TRAIN)
    cfg["porto_opq_teacher_codebook_path"] = str(OPQ_CODEBOOK)
    cfg["porto_opq_teacher_start_epoch"] = 0
    cfg["porto_opq_teacher_end_epoch"] = 360
    cfg["porto_opq_teacher_z_weight"] = 0.0
    cfg["porto_opq_teacher_partition_weight"] = 1.0
    cfg["porto_opq_teacher_codebook_freeze_end_epoch"] = 360
    cfg["porto_opq_teacher_realign_codebook_on_unfreeze"] = False

    cfg["epoch_num"] = 600
    cfg["test_epoch"] = 600
    cfg["eval_save_epochs"] = "600"
    cfg["save_model_epoch"] = 100
    cfg["dist_type"] = "dtw"
    cfg["device"] = "cuda:0"
    cfg["eval_embedding_type"] = "both"
    cfg["eval_search_mode"] = "adc"
    cfg["enable_rerank"] = False
    cfg["rerank_source"] = "adc"

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=True))
    print(out_cfg)


if __name__ == "__main__":
    main()
