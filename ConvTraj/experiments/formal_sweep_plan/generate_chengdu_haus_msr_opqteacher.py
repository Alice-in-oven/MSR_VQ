import argparse
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
BASE_CFG = ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "hotfix" / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc.json"
HAUS_HOTFIX_CKPT = ROOT / "exp" / "formal_hotfix_runs" / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc" / "checkpoints" / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc_e400.pt"
OPQ_DIR = ROOT / "exp" / "formal_hotfix_runs" / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc" / "faiss_eval" / "e400" / "opq_m16_nbits6"
OPQ_ROTATED_TRAIN = OPQ_DIR / "chengdu_haus_msr_hotfix_cont_e400_opq_m16_nbits6_train_faiss_rotated.pkl"
OPQ_CODEBOOK = OPQ_DIR / "chengdu_haus_msr_hotfix_cont_e400_opq_m16_nbits6_faiss_pq_codebook.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["longfreeze", "shorthandoff"], required=True)
    args = parser.parse_args()

    cfg = json.loads(BASE_CFG.read_text())
    cfg = deepcopy(cfg)

    if args.variant == "longfreeze":
        train_flag = "chengdu_haus_msr_pdtvq_m16_k64_opqteacher_longfreeze_e400_trueadc"
        teacher_end = 280
        freeze_end = 180
        realign = False
        ste_max = 0.08
    else:
        train_flag = "chengdu_haus_msr_pdtvq_m16_k64_opqteacher_shorthandoff_e400_trueadc"
        teacher_end = 220
        freeze_end = 120
        realign = True
        ste_max = 0.10

    root_write = ROOT / "exp" / "formal_hotfix_runs" / train_flag
    out_cfg = ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "hotfix" / f"{train_flag}.json"

    cfg["train_flag"] = train_flag
    cfg["artifact_prefix"] = train_flag
    cfg["root_write_path"] = str(root_write)
    cfg["save_model_path"] = str(root_write / "checkpoints")
    cfg["backbone_checkpoint"] = str(HAUS_HOTFIX_CKPT)

    cfg["porto_opq_teacher_rotated_train_path"] = str(OPQ_ROTATED_TRAIN)
    cfg["porto_opq_teacher_codebook_path"] = str(OPQ_CODEBOOK)
    cfg["porto_opq_teacher_start_epoch"] = 0
    cfg["porto_opq_teacher_end_epoch"] = teacher_end
    cfg["porto_opq_teacher_z_weight"] = 0.0
    cfg["porto_opq_teacher_partition_weight"] = 1.0
    cfg["porto_opq_teacher_codebook_freeze_end_epoch"] = freeze_end
    cfg["porto_opq_teacher_realign_codebook_on_unfreeze"] = realign

    cfg["freeze_backbone_epochs"] = 0
    cfg["pdt_loss_start_epoch"] = 20
    cfg["pdt_loss_weight"] = 0.03
    cfg["decoded_ste_metric_enabled"] = True
    cfg["decoded_ste_metric_start_epoch"] = 20
    cfg["decoded_ste_metric_warmup_epochs"] = 40
    cfg["decoded_ste_metric_max_weight"] = ste_max

    cfg["pre_quant_bottleneck_enabled"] = False
    cfg["quantized_metric_weight"] = 0.0
    cfg["improved_qm_start_epoch"] = 999999
    cfg["improved_qm_warmup_epochs"] = 0
    cfg["improved_qm_max_weight"] = 0.0
    cfg["improved_pairwise_weight"] = 0.0

    cfg["epoch_num"] = 400
    cfg["test_epoch"] = 400
    cfg["eval_save_epochs"] = "300,400"
    cfg["save_model_epoch"] = 100
    cfg["train_set_size"] = 1000
    cfg["query_set_size"] = 1000
    cfg["base_set_size"] = 3000

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=True))
    print(out_cfg)


if __name__ == "__main__":
    main()
