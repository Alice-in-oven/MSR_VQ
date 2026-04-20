import argparse
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj")
MODEL_TO_BASE = {
    "msr": ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "quantized_main" / "chengdu" / "msr" / "chengdu_haus_msr_pdtvq_m16_k64_e500_trueadc.json",
    "neutraj": ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "quantized_main" / "chengdu" / "neutraj" / "chengdu_haus_neutraj_pdtvq_m16_k64_e500_trueadc.json",
    "simformer": ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "quantized_main" / "chengdu" / "simformer" / "chengdu_haus_simformer_pdtvq_m16_k64_e500_trueadc.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["msr", "neutraj", "simformer"], default="msr")
    parser.add_argument("--prequant", choices=["off", "on"], default="off")
    args = parser.parse_args()

    base_cfg = MODEL_TO_BASE[args.model]
    cfg = json.loads(base_cfg.read_text())
    cfg = deepcopy(cfg)

    suffix = "prequant_on" if args.prequant == "on" else "hotfix"
    train_flag = f"chengdu_haus_{args.model}_pdtvq_m16_k64_{suffix}_e400_trueadc"
    root_write = ROOT / "exp" / "formal_hotfix_runs" / train_flag
    out_cfg = ROOT / "experiments" / "formal_sweep_plan" / "generated_configs" / "hotfix" / f"{train_flag}.json"

    cfg["train_flag"] = train_flag
    cfg["artifact_prefix"] = train_flag
    cfg["root_write_path"] = str(root_write)
    cfg["save_model_path"] = str(root_write / "checkpoints")

    cfg["epoch_num"] = 400
    cfg["test_epoch"] = 400
    cfg["eval_save_epochs"] = "300,400"
    cfg["save_model_epoch"] = 100
    cfg["train_set_size"] = 1000
    cfg["query_set_size"] = 1000
    cfg["base_set_size"] = 3000

    # Chengdu Haus m16k64 collapsed under the aggressive formal recipe.
    # Use a conservative schedule: no backbone freeze, early GT supervision,
    # and no qmetric / pairwise branches. Pre-quant is ablated via flag.
    cfg["freeze_backbone_epochs"] = 0
    cfg["pdt_loss_start_epoch"] = 20
    cfg["pdt_loss_weight"] = 0.03

    cfg["decoded_ste_metric_enabled"] = True
    cfg["decoded_ste_metric_start_epoch"] = 20
    cfg["decoded_ste_metric_warmup_epochs"] = 40
    cfg["decoded_ste_metric_max_weight"] = 0.08

    cfg["quantized_metric_weight"] = 0.0
    cfg["improved_qm_start_epoch"] = 999999
    cfg["improved_qm_warmup_epochs"] = 0
    cfg["improved_qm_max_weight"] = 0.0
    cfg["improved_pairwise_weight"] = 0.0

    if args.prequant == "on":
        cfg["pre_quant_bottleneck_enabled"] = True
    else:
        cfg["pre_quant_bottleneck_enabled"] = False

    cfg["late_finetune_start_epoch"] = 260
    cfg["late_finetune_main_lr_scale"] = 0.2
    cfg["late_finetune_pre_quant_lr_scale"] = 0.2

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=True))
    print(out_cfg)


if __name__ == "__main__":
    main()
