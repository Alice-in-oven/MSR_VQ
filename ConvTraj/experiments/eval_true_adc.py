#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from config.new_config import MyEncoder
from mynn.Traj_KNN import NeuTrajTrainer
from tools import function, grid, test_methods


def resolve_default_dataset_root(dataset_name, requested_root_path):
    requested = str(requested_root_path or "").strip()
    requested_path = Path(requested).expanduser()
    dataset_key = str(dataset_name or "").strip().lower()
    default_data_root = (Path(__file__).resolve().parents[1] / "data").resolve()
    try:
        requested_resolved = requested_path.resolve()
    except FileNotFoundError:
        requested_resolved = requested_path
    if requested_resolved != default_data_root:
        return str(requested_path)
    candidate_root = default_data_root / f"0_{dataset_key}"
    if candidate_root.exists():
        return str(candidate_root)
    return str(requested_path)


def resolve_dataset(cfg):
    root_read_path = resolve_default_dataset_root(cfg.get("dataset", ""), cfg["root_read_path"])
    dataset = str(cfg.get("dataset", ""))
    dist_type = str(cfg.get("dist_type", "dtw"))

    if function.has_split_dataset(root_read_path):
        traj_list, train_set, query_set, base_set, train_dist_matrix, test_dist_matrix = function.load_split_dataset(
            root_read_path,
            dist_type,
        )
    elif dataset == "porto" and "0_porto_all" in root_read_path:
        train_set, query_set, base_set = function.set_dataset(root_read_path, dist_type)
        traj_list = pickle.load(open(Path(root_read_path) / "traj_list", "rb"))
        train_dist_matrix = pickle.load(open(Path(root_read_path) / f"{dist_type}_train_distance_matrix_result", "rb"))
        test_dist_matrix = pickle.load(open(Path(root_read_path) / f"{dist_type}_test_distance_matrix_result", "rb"))
    else:
        train_set, query_set, base_set = function.set_dataset(root_read_path, dist_type)
        traj_list = pickle.load(open(Path(root_read_path) / "traj_list", "rb"))
        train_dist_matrix = pickle.load(open(Path(root_read_path) / "dtw_train_distance_matrix_result", "rb"))
        test_dist_matrix = pickle.load(open(Path(root_read_path) / "dtw_test_distance_matrix_result", "rb"))

    total_required = int(train_set) + int(query_set) + int(base_set)
    traj_list = traj_list[:total_required]
    train_dist_matrix = np.asarray(train_dist_matrix[:train_set, :train_set])
    test_dist_matrix = np.asarray(test_dist_matrix[:query_set, :base_set])
    return traj_list, int(train_set), int(query_set), int(base_set), train_dist_matrix, test_dist_matrix


def build_trainer(config_dict):
    traj_list, train_set, query_set, base_set, train_dist_matrix, test_dist_matrix = resolve_dataset(config_dict)
    effective_grid_size = float(config_dict.get("grid_size", 0.0))
    if effective_grid_size <= 0.0:
        effective_grid_size = None
    lon_grid_id_list, lat_grid_id_list, lon_input_size, lat_input_size, lon_list, lat_list = grid.split_traj_into_equal_grid(
        traj_list,
        grid_size=effective_grid_size,
    )

    config_dict = dict(config_dict)
    config_dict["lon_input_size"] = int(lon_input_size)
    config_dict["lat_input_size"] = int(lat_input_size)
    config_dict["train_set"] = int(train_set)
    config_dict["query_set"] = int(query_set)
    config_dict["base_set"] = int(base_set)
    config_dict["device"] = torch.device(config_dict["device"])

    if config_dict["network_type"] in ["TJCNN", "TJCNN_MC_MSR"]:
        lon_onehot = lon_list
        lat_onehot = lat_list
    else:
        lon_onehot = [[[value] for value in traj] for traj in lon_grid_id_list]
        lat_onehot = [[[value] for value in traj] for traj in lat_grid_id_list]

    trainer = NeuTrajTrainer(SimpleNamespace(my_dict=config_dict))
    trainer.data_prepare(
        traj_list,
        train_dist_matrix,
        test_dist_matrix,
        lon_onehot=lon_onehot,
        lat_onehot=lat_onehot,
        lon_grid_id_list=lon_grid_id_list,
        lat_grid_id_list=lat_grid_id_list,
    )
    return trainer


def collect_transformed_and_codes(trainer, model):
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
            embedding_type="continuous",
            collect_code_usage=True,
            collect_transformed=True,
            collect_reconstructed=False,
            collect_raw_continuous=False,
        )
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

    transformed = np.concatenate(total_transformed, axis=0)
    codes = np.concatenate(total_codes, axis=0)
    return transformed, codes, total_time


def compute_true_adc_pred_knn(trainer, codebook, transformed_embeddings, codes, topk=100):
    query_trans, _ = trainer._split_eval_array(transformed_embeddings)
    _, base_codes = trainer._split_eval_array(codes)

    query_trans = np.asarray(query_trans, dtype=np.float32)
    base_codes = np.asarray(base_codes, dtype=np.int32)
    codebook = np.asarray(codebook, dtype=np.float32)

    nq = query_trans.shape[0]
    nb = base_codes.shape[0]
    M, K, dsub = codebook.shape
    query_groups = query_trans.reshape(nq, M, dsub)
    adc_distance = np.zeros((nq, nb), dtype=np.float32)

    for group_idx in range(M):
        query_group = query_groups[:, group_idx, :]
        centroids = codebook[group_idx]
        query_norm = np.sum(query_group ** 2, axis=1, keepdims=True)
        centroid_norm = np.sum(centroids ** 2, axis=1, keepdims=True).T
        lookup = query_norm - 2.0 * np.matmul(query_group, centroids.T) + centroid_norm
        adc_distance += lookup[:, base_codes[:, group_idx]]

    topk = min(int(topk), nb)
    partition = np.argpartition(adc_distance, kth=topk - 1, axis=1)[:, :topk]
    topk_dist = np.take_along_axis(adc_distance, partition, axis=1)
    topk_order = np.argsort(topk_dist, axis=1)
    pred_knn = np.take_along_axis(partition, topk_order, axis=1)
    return pred_knn


def main():
    parser = argparse.ArgumentParser(description="Evaluate true ADC / asymmetric scoring on a saved Porto checkpoint.")
    parser.add_argument("--train_config", required=True, help="Path to saved train_config json.")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint.")
    parser.add_argument("--eval_epoch", type=int, required=True, help="Epoch label used in the output file.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0")
    parser.add_argument("--output_path", default="", help="Optional explicit output json path.")
    args = parser.parse_args()

    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.device:
        cfg["device"] = str(args.device)
    cfg["mode"] = "test"
    cfg["eval_embedding_type"] = "quantized"
    cfg["eval_search_mode"] = "decoded"
    cfg["enable_rerank"] = False
    cfg["report_only_eval"] = True

    trainer = build_trainer(cfg)
    model = function.initialize_model(trainer.my_config.my_dict, trainer.max_traj_length).to(trainer.my_config.my_dict["device"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=trainer.my_config.my_dict["device"]))
    model.eval()

    transformed, codes, inference_time = collect_transformed_and_codes(trainer, model)
    codebook = model.PDT_model.vq.quantizer.codebook.detach().cpu().numpy()
    pred_knn = compute_true_adc_pred_knn(trainer, codebook, transformed, codes, topk=100)
    metrics = test_methods.metrics_from_pred_knn(trainer.test_knn, pred_knn)
    code_usage = trainer._summarize_code_usage(torch.from_numpy(codes))

    payload = {
        "protocol": "true_adc_asymmetric",
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "eval_epoch": int(args.eval_epoch),
        "device": str(trainer.my_config.my_dict["device"]),
        "inference_time_seconds": float(inference_time),
        "transformed_shape": list(transformed.shape),
        "codes_shape": list(codes.shape),
        "codebook_shape": list(codebook.shape),
        "adc": metrics,
        "code_usage": code_usage,
    }

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        feature_dir = Path(cfg["root_write_path"]) / "feature_dir"
        feature_dir.mkdir(parents=True, exist_ok=True)
        output_path = feature_dir / f"metrics_true_adc_{cfg['train_flag']}_epochs_{int(args.eval_epoch)}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=MyEncoder)

    adc = metrics
    print(
        "True ADC Top-5 {:.4f} | Top-10 {:.4f} | Top-50 {:.4f} | Top-100 {:.4f} | Top-10@50 {:.4f}".format(
            adc["top5_recall"],
            adc["top10_recall"],
            adc["top50_recall"],
            adc["top100_recall"],
            adc["top10_at_50_recall"],
        )
    )
    print("Metrics saved to:", output_path)


if __name__ == "__main__":
    main()
