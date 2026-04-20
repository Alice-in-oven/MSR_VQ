#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config.new_config import MyEncoder
from experiments.eval_true_adc import (
    resolve_dataset,
    build_trainer,
    compute_true_adc_pred_knn,
)
from tools import function, test_methods


def collect_eval_arrays(trainer, model):
    total_continuous = []
    total_quantized = []
    total_transformed = []
    total_codes = []
    total_time = 0.0

    eval_chunk_size = 10240
    eval_batch_size = 1024
    begin_pos, end_pos = 0, min(eval_chunk_size, len(trainer.traj_list))

    while True:
        print(begin_pos, end_pos)
        (
            trainer.pad_total_lon_onehot,
            trainer.pad_total_lat_onehot,
            trainer.pad_total_lon_lat_image,
            eval_seq_lengths,
            trainer.pad_total_lon_grid,
            trainer.pad_total_lat_grid,
        ) = trainer._build_eval_tensors(begin_pos, end_pos)

        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start is not None:
            start.record()

        outputs = trainer._collect_embedding_outputs(
            model,
            trainer.pad_total_lon_onehot,
            trainer.pad_total_lat_onehot,
            trainer.pad_total_lon_lat_image,
            eval_seq_lengths,
            lon_grid_tensor=trainer.pad_total_lon_grid,
            lat_grid_tensor=trainer.pad_total_lat_grid,
            test_batch=eval_batch_size,
            embedding_type="both",
            collect_code_usage=True,
            collect_transformed=True,
            collect_reconstructed=False,
            collect_raw_continuous=False,
        )

        total_continuous.append(outputs["continuous"].numpy())
        total_quantized.append(outputs["quantized"].numpy())
        total_transformed.append(outputs["transformed"].numpy())
        total_codes.append(outputs["codes"].numpy())

        if end is not None:
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end) / 1000.0

        if end_pos == len(trainer.traj_list):
            break
        begin_pos = end_pos
        end_pos = min(end_pos + eval_chunk_size, len(trainer.traj_list))

    return (
        np.concatenate(total_continuous, axis=0),
        np.concatenate(total_quantized, axis=0),
        np.concatenate(total_transformed, axis=0),
        np.concatenate(total_codes, axis=0),
        total_time,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate true ADC shortlist + decoded rerank on a saved checkpoint.")
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_epoch", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rerank_L", type=int, default=100)
    parser.add_argument("--output_path", default="")
    args = parser.parse_args()

    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.device:
        cfg["device"] = str(args.device)
    cfg["mode"] = "test"
    cfg["report_only_eval"] = True

    trainer = build_trainer(cfg)
    model = function.initialize_model(trainer.my_config.my_dict, trainer.max_traj_length).to(trainer.my_config.my_dict["device"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=trainer.my_config.my_dict["device"]))
    model.eval()

    continuous, quantized, transformed, codes, inference_time = collect_eval_arrays(trainer, model)
    codebook = model.PDT_model.vq.quantizer.codebook.detach().cpu().numpy()

    coarse_k = max(100, int(args.rerank_L))
    adc_pred_knn = compute_true_adc_pred_knn(trainer, codebook, transformed, codes, topk=coarse_k)
    adc_metrics = test_methods.metrics_from_pred_knn(trainer.test_knn, adc_pred_knn)

    query_continuous, _ = trainer._split_eval_array(continuous)
    _, base_quantized = trainer._split_eval_array(quantized)
    rerank_pred_knn = trainer._compute_rerank_pred_knn(
        query_continuous,
        base_quantized,
        adc_pred_knn,
        rerank_L=int(args.rerank_L),
    )
    rerank_metrics = test_methods.metrics_from_pred_knn(trainer.test_knn, rerank_pred_knn)
    code_usage = trainer._summarize_code_usage(torch.from_numpy(codes))

    payload = {
        "protocol": "true_adc_coarse_decoded_rerank",
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "eval_epoch": int(args.eval_epoch),
        "device": str(trainer.my_config.my_dict["device"]),
        "rerank_L": int(args.rerank_L),
        "inference_time_seconds": float(inference_time),
        "continuous_shape": list(continuous.shape),
        "quantized_shape": list(quantized.shape),
        "transformed_shape": list(transformed.shape),
        "codes_shape": list(codes.shape),
        "codebook_shape": list(codebook.shape),
        "adc": adc_metrics,
        f"rerank_decoded_L{int(args.rerank_L)}": rerank_metrics,
        "code_usage": code_usage,
    }

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        feature_dir = Path(cfg["root_write_path"]) / "feature_dir"
        feature_dir.mkdir(parents=True, exist_ok=True)
        output_path = feature_dir / f"metrics_true_adc_rerankL{int(args.rerank_L)}_{cfg['train_flag']}_epochs_{int(args.eval_epoch)}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=MyEncoder)

    print(
        "True ADC Top-5 {:.4f} | Top-10 {:.4f} | Top-50 {:.4f} | Top-100 {:.4f} | Top-10@50 {:.4f}".format(
            adc_metrics["top5_recall"],
            adc_metrics["top10_recall"],
            adc_metrics["top50_recall"],
            adc_metrics["top100_recall"],
            adc_metrics["top10_at_50_recall"],
        )
    )
    print(
        "ADC->Decoded Rerank L{} Top-5 {:.4f} | Top-10 {:.4f} | Top-50 {:.4f} | Top-100 {:.4f} | Top-10@50 {:.4f}".format(
            int(args.rerank_L),
            rerank_metrics["top5_recall"],
            rerank_metrics["top10_recall"],
            rerank_metrics["top50_recall"],
            rerank_metrics["top100_recall"],
            rerank_metrics["top10_at_50_recall"],
        )
    )
    print("Metrics saved to:", output_path)


if __name__ == "__main__":
    main()
