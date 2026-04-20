import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
BASE_CFG = ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "hotfix" / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc.json"
HAUS_BACKBONE_CKPT = (
    ROOT
    / "exp"
    / "formal_hotfix_runs"
    / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc"
    / "checkpoints"
    / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc_e400.pt"
)
OPQ_DIR = (
    ROOT
    / "exp"
    / "formal_hotfix_runs"
    / "chengdu_haus_msr_pdtvq_m16_k64_hotfix_e400_trueadc"
    / "faiss_eval"
    / "e400"
    / "opq_m8_nbits4"
)
OPQ_ROTATED_TRAIN = OPQ_DIR / "chengdu_haus_msr_hotfix_cont_e400_opq_m8_nbits4_train_faiss_rotated.pkl"
OPQ_CODEBOOK = OPQ_DIR / "chengdu_haus_msr_hotfix_cont_e400_opq_m8_nbits4_faiss_pq_codebook.npy"


def main():
    cfg = json.loads(BASE_CFG.read_text())
    cfg = deepcopy(cfg)

    train_flag = "chengdu_haus_msr_pdtvq_m8_k16_portostyle_e400_trueadc"
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
    cfg["backbone_checkpoint"] = str(HAUS_BACKBONE_CKPT)

    # Porto slow-handoff strategy, scaled from 700 epochs to 400 epochs.
    cfg["freeze_backbone_epochs"] = 0
    cfg["pdt_loss_start_epoch"] = 183
    cfg["pdt_loss_weight"] = 0.02
    cfg["decoded_ste_metric_enabled"] = True
    cfg["decoded_ste_metric_start_epoch"] = 80
    cfg["decoded_ste_metric_warmup_epochs"] = 30
    cfg["decoded_ste_metric_max_weight"] = 0.12
    cfg["late_finetune_start_epoch"] = 354
    cfg["late_finetune_main_lr_scale"] = 0.2
    cfg["late_finetune_pre_quant_lr_scale"] = 0.2

    cfg["pre_quant_bottleneck_enabled"] = False
    cfg["quantized_metric_weight"] = 0.3
    cfg["improved_qm_start_epoch"] = 0
    cfg["improved_qm_warmup_epochs"] = 1
    cfg["improved_qm_max_weight"] = 0.0
    cfg["improved_pairwise_weight"] = 0.0
    cfg["improved_entropy_weight"] = 0.002
    cfg["improved_commit_weight"] = 0.02
    cfg["improved_uniform_weight"] = 5e-05
    cfg["improved_vq_adaptive_low_codebook"] = False

    cfg["porto_opq_teacher_rotated_train_path"] = str(OPQ_ROTATED_TRAIN)
    cfg["porto_opq_teacher_codebook_path"] = str(OPQ_CODEBOOK)
    cfg["porto_opq_teacher_start_epoch"] = 0
    cfg["porto_opq_teacher_end_epoch"] = 206
    cfg["porto_opq_teacher_z_weight"] = 0.0
    cfg["porto_opq_teacher_partition_weight"] = 1.0
    cfg["porto_opq_teacher_codebook_freeze_end_epoch"] = 206
    cfg["porto_opq_teacher_realign_codebook_on_unfreeze"] = False

    cfg["epoch_num"] = 400
    cfg["test_epoch"] = 400
    cfg["eval_save_epochs"] = "300,400"
    cfg["save_model_epoch"] = 100
    cfg["dist_type"] = "haus"
    cfg["eval_embedding_type"] = "both"
    cfg["eval_search_mode"] = "adc"
    cfg["enable_rerank"] = False
    cfg["rerank_source"] = "adc"
    cfg["train_set_size"] = 1000
    cfg["query_set_size"] = 1000
    cfg["base_set_size"] = 3000

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=True))
    print(out_cfg)


if __name__ == "__main__":
    main()
