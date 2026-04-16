import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from tools import function
from tools import grid
from tools import pre_rep
from PDT_VQ.utils.engine import inference_vetor
from PDT_VQ.utils.engine import batch_forward
from PDT_VQ.utils.engine import re_ranking
from PDT_VQ.utils.get_nn import get_nearestneighbors


REPO_ROOT = Path("/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj")
DATA_ROOT = REPO_ROOT / "data"
GEOLIFE_ROOT = DATA_ROOT / "0_geolife"
OUTPUT_ROOT = DATA_ROOT / "inference_mode_eval"


def parse_args():
    parser = argparse.ArgumentParser(description = "Evaluate MSR+PDT_VQ with different inference modes on geolife.")
    parser.add_argument("--checkpoint_path", type = str, required = True)
    parser.add_argument("--config_path", type = str, default = None)
    parser.add_argument("--root_read_path", type = str, default = str(DATA_ROOT))
    parser.add_argument("--root_write_path", type = str, default = str(OUTPUT_ROOT))
    parser.add_argument("--device", type = str, default = "cuda:0")
    parser.add_argument("--batch_size", type = int, default = 1024)
    parser.add_argument("--inference_mode", type = str, required = True, choices = ["exact", "rerank", "symmetric"])
    parser.add_argument("--rerank_L_values", type = str, default = "10,20,50,100,200,500,1000")
    parser.add_argument("--topk_eval", type = int, default = 100)
    parser.add_argument("--cache_dir", type = str, default = None)
    parser.add_argument("--force_reencode", action = "store_true")
    parser.add_argument("--random_seed", type = int, default = 666)
    return parser.parse_args()


def resolve_config_path(checkpoint_path, config_path):
    if config_path is not None:
        return str(config_path)
    checkpoint = Path(checkpoint_path)
    train_flag = checkpoint.name.replace("_epochs_500", "")
    candidate = checkpoint.parent.parent / "train_config" / "{}.json".format(train_flag)
    if not candidate.exists():
        raise FileNotFoundError("Config file not found for checkpoint: {}".format(checkpoint_path))
    return str(candidate)


def clone_feature_list(feature_list):
    return [[point[:] for point in traj] for traj in feature_list]


def load_model_and_data(args):
    config_path = resolve_config_path(args.checkpoint_path, args.config_path)
    with open(config_path, "r", encoding = "utf-8") as f:
        config_dict = json.load(f)
    config_dict["device"] = torch.device(args.device)
    config_dict["root_read_path"] = args.root_read_path
    config_dict["root_write_path"] = args.root_read_path

    with open(GEOLIFE_ROOT / "traj_list", "rb") as f:
        traj_list = pickle.load(f)
    max_traj_length = max(len(traj) for traj in traj_list)
    my_net = function.initialize_model(config_dict, max_traj_length).to(config_dict["device"])
    my_net.load_state_dict(torch.load(args.checkpoint_path, map_location = config_dict["device"]))
    my_net.eval()

    lon_grid_id_list, lat_grid_id_list, lon_input_size, lat_input_size, lon_list, lat_list = grid.split_traj_into_equal_grid(traj_list)

    with open(GEOLIFE_ROOT / "dtw_train_distance_matrix_result", "rb") as f:
        train_distance_matrix = pickle.load(f)
    with open(GEOLIFE_ROOT / "dtw_test_distance_matrix_result", "rb") as f:
        test_distance_matrix = pickle.load(f)
    train_max = np.max(train_distance_matrix)
    test_distance_matrix = test_distance_matrix / train_max
    true_knn = np.argsort(test_distance_matrix, axis = 1)

    return {
        "model": my_net,
        "config_dict": config_dict,
        "config_path": config_path,
        "traj_list": traj_list,
        "max_traj_length": max_traj_length,
        "lon_grid_id_list": lon_grid_id_list,
        "lat_grid_id_list": lat_grid_id_list,
        "lon_input_size": lon_input_size,
        "lat_input_size": lat_input_size,
        "lon_list": lon_list,
        "lat_list": lat_list,
        "true_knn": true_knn,
        "train_set": 3000,
        "query_set": 1000,
        "base_set": 9386,
    }


def build_batch_tensors(batch_indices, state):
    lon_batch = clone_feature_list([state["lon_list"][index] for index in batch_indices])
    lat_batch = clone_feature_list([state["lat_list"][index] for index in batch_indices])
    image_batch = pre_rep.build_traj_image([state["lon_grid_id_list"][index] for index in batch_indices],
                                           [state["lat_grid_id_list"][index] for index in batch_indices],
                                           state["lon_input_size"],
                                           state["lat_input_size"],
                                           image_mode = state["config_dict"].get("image_mode", "binary"),
                                           traj_list = [state["traj_list"][index] for index in batch_indices])
    lon_tensor = torch.tensor(function.pad_traj_list(lon_batch, state["max_traj_length"], pad_value = 0.0),
                              dtype = torch.float32,
                              device = state["config_dict"]["device"])
    lat_tensor = torch.tensor(function.pad_traj_list(lat_batch, state["max_traj_length"], pad_value = 0.0),
                              dtype = torch.float32,
                              device = state["config_dict"]["device"])
    image_tensor = torch.tensor(image_batch, dtype = torch.float32, device = state["config_dict"]["device"])
    return lon_tensor, lat_tensor, image_tensor


def encode_continuous_split(split_indices, state, batch_size):
    outputs = []
    with torch.no_grad():
        for begin in range(0, len(split_indices), batch_size):
            end = min(begin + batch_size, len(split_indices))
            batch_indices = split_indices[begin:end]
            lon_tensor, lat_tensor, image_tensor = build_batch_tensors(batch_indices, state)
            embedding = state["model"].inference_continuous(lon_tensor, lat_tensor, image_tensor)
            outputs.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis = 0)


def l2_distance_matrix(query_embeddings, base_embeddings):
    query_embeddings = np.asarray(query_embeddings, dtype = np.float32)
    base_embeddings = np.asarray(base_embeddings, dtype = np.float32)
    q2 = np.sum(query_embeddings ** 2, axis = 1, keepdims = True)
    b2 = np.sum(base_embeddings ** 2, axis = 1)[None, :]
    dist = q2 + b2 - 2.0 * (query_embeddings @ base_embeddings.T)
    dist[dist < 0.0] = 0.0
    return dist.astype(np.float32)


def metrics_from_pred_knn(true_knn, pred_knn):
    def intersect_sizes(true_ids, pred_ids):
        return np.array([len(np.intersect1d(true_value, pred_value)) for true_value, pred_value in zip(true_ids, pred_ids)])

    metrics = {
        "top5_recall": float(np.mean(intersect_sizes(true_knn[:, :5], pred_knn[:, :5]) / 5.0)),
        "top10_recall": float(np.mean(intersect_sizes(true_knn[:, :10], pred_knn[:, :10]) / 10.0)),
        "top50_recall": float(np.mean(intersect_sizes(true_knn[:, :50], pred_knn[:, :50]) / 50.0)),
        "top100_recall": float(np.mean(intersect_sizes(true_knn[:, :100], pred_knn[:, :100]) / 100.0)),
        "top10_at_50_recall": float(np.mean(intersect_sizes(true_knn[:, :10], pred_knn[:, :50]) / 10.0)),
    }
    metrics["metrics_list"] = [
        metrics["top5_recall"],
        metrics["top10_recall"],
        metrics["top50_recall"],
        metrics["top100_recall"],
        metrics["top10_at_50_recall"],
    ]
    return metrics


def build_pred_knn_from_distance_matrix(distance_matrix, topk_eval):
    topk_eval = min(topk_eval, distance_matrix.shape[1])
    pred_knn = np.empty((distance_matrix.shape[0], topk_eval), dtype = np.int32)
    for i in range(distance_matrix.shape[0]):
        row = distance_matrix[i]
        idx = np.argpartition(row, topk_eval - 1)[:topk_eval]
        idx = idx[np.argsort(row[idx])]
        pred_knn[i] = idx
    return pred_knn


def load_or_build_continuous_cache(state, args):
    cache_root = Path(args.cache_dir) if args.cache_dir is not None else Path(args.root_write_path)
    cache_root.mkdir(parents = True, exist_ok = True)
    ckpt_stem = Path(args.checkpoint_path).name
    cache_npz = cache_root / "{}_continuous_cache.npz".format(ckpt_stem)
    cache_meta = cache_root / "{}_continuous_cache_meta.json".format(ckpt_stem)

    if (not args.force_reencode) and cache_npz.exists() and cache_meta.exists():
        meta = json.load(open(cache_meta, "r", encoding = "utf-8"))
        if meta.get("checkpoint_path") == args.checkpoint_path:
            cache = np.load(cache_npz)
            print("Continuous embedding cache: HIT ->", cache_npz)
            return {
                "query_continuous": np.asarray(cache["query_continuous"], dtype = np.float32),
                "base_continuous": np.asarray(cache["base_continuous"], dtype = np.float32),
                "cache_npz": str(cache_npz),
                "cache_meta": str(cache_meta),
                "cache_used": True,
            }

    print("Continuous embedding cache: MISS, encoding query/base continuous embeddings...")
    query_indices = np.arange(state["train_set"], state["train_set"] + state["query_set"], dtype = np.int32)
    base_indices = np.arange(state["train_set"] + state["query_set"], state["train_set"] + state["query_set"] + state["base_set"], dtype = np.int32)
    query_continuous = encode_continuous_split(query_indices, state, args.batch_size)
    base_continuous = encode_continuous_split(base_indices, state, args.batch_size)
    np.savez(cache_npz,
             query_continuous = query_continuous,
             base_continuous = base_continuous)
    with open(cache_meta, "w", encoding = "utf-8") as f:
        json.dump({
            "checkpoint_path": args.checkpoint_path,
            "config_path": state["config_path"],
            "query_set": int(state["query_set"]),
            "base_set": int(state["base_set"]),
        }, f, indent = 2, ensure_ascii = False)
    print("Continuous embedding cache saved to:", cache_npz)
    return {
        "query_continuous": query_continuous,
        "base_continuous": base_continuous,
        "cache_npz": str(cache_npz),
        "cache_meta": str(cache_meta),
        "cache_used": False,
    }


def evaluate_symmetric(cache_payload, state, args):
    start = time.perf_counter()
    q = torch.from_numpy(cache_payload["query_continuous"]).to(state["config_dict"]["device"])
    b = torch.from_numpy(cache_payload["base_continuous"]).to(state["config_dict"]["device"])
    query_quantized = inference_vetor(state["model"].PDT_model, q, state["model"].pdt_args).detach().cpu().numpy().astype(np.float32)
    base_quantized = inference_vetor(state["model"].PDT_model, b, state["model"].pdt_args).detach().cpu().numpy().astype(np.float32)
    distance_matrix = l2_distance_matrix(query_quantized, base_quantized)
    pred_knn = build_pred_knn_from_distance_matrix(distance_matrix, args.topk_eval)
    metrics = metrics_from_pred_knn(state["true_knn"], pred_knn)
    metrics["avg_query_time_ms"] = 1000.0 * (time.perf_counter() - start) / max(1, len(query_quantized))
    metrics["retrieval_type"] = "symmetric_quantized"
    return metrics


def evaluate_exact(cache_payload, state, args):
    start = time.perf_counter()
    q = cache_payload["query_continuous"]
    b = torch.from_numpy(cache_payload["base_continuous"]).to(state["config_dict"]["device"])
    base_quantized = inference_vetor(state["model"].PDT_model, b, state["model"].pdt_args).detach().cpu().numpy().astype(np.float32)
    distance_matrix = l2_distance_matrix(q, base_quantized)
    pred_knn = build_pred_knn_from_distance_matrix(distance_matrix, args.topk_eval)
    metrics = metrics_from_pred_knn(state["true_knn"], pred_knn)
    metrics["avg_query_time_ms"] = 1000.0 * (time.perf_counter() - start) / max(1, len(q))
    metrics["retrieval_type"] = "exact_asymmetric"
    return metrics


def evaluate_rerank(cache_payload, state, args):
    device = state["config_dict"]["device"]
    q = torch.from_numpy(cache_payload["query_continuous"]).to(device)
    b = torch.from_numpy(cache_payload["base_continuous"]).to(device)
    start = time.perf_counter()
    xb_trans = batch_forward(state["model"].PDT_model.encode, b, args.batch_size, state["model"].pdt_args.steps)[-1]
    codes = batch_forward(state["model"].PDT_model.get_codes, xb_trans, args.batch_size)
    xb_recon = batch_forward(state["model"].PDT_model.reconstruction, codes, args.batch_size)
    xb_recon_model = xb_recon
    xq_trans = batch_forward(state["model"].PDT_model.encode, q, args.batch_size, state["model"].pdt_args.steps)[-1]
    max_L = max(args.rerank_L_values)
    coarse_pred_nn = get_nearestneighbors(xb_recon.detach().cpu().numpy(),
                                          xq_trans.detach().cpu().numpy(),
                                          max_L,
                                          args.device)
    coarse_time = time.perf_counter() - start

    rerank_metrics = {}
    for L in args.rerank_L_values:
        rerank_start = time.perf_counter()
        pred_nn = torch.from_numpy(coarse_pred_nn[:, :L]).to(device)
        one_step_size = max(1, int(args.batch_size / max(1, L)))
        reranked = re_ranking(xb_recon_model,
                              q,
                              pred_nn,
                              state["model"].PDT_model.decode,
                              L,
                              state["model"].pdt_args.steps,
                              batch_size = args.batch_size,
                              one_step_size = one_step_size)
        reranked_np = reranked.detach().cpu().numpy()[:, :args.topk_eval]
        metrics = metrics_from_pred_knn(state["true_knn"], reranked_np)
        metrics["avg_query_time_ms"] = 1000.0 * (coarse_time + (time.perf_counter() - rerank_start)) / max(1, q.shape[0])
        metrics["rerank_L"] = int(L)
        rerank_metrics["L{}".format(L)] = metrics
    return rerank_metrics


def main():
    args = parse_args()
    function.setup_seed(args.random_seed)
    args.rerank_L_values = [int(x) for x in args.rerank_L_values.split(",") if x]

    state = load_model_and_data(args)
    cache_payload = load_or_build_continuous_cache(state, args)

    output_root = Path(args.root_write_path)
    output_root.mkdir(parents = True, exist_ok = True)
    ckpt_stem = Path(args.checkpoint_path).name.replace(".pth", "").replace(".pt", "")
    metrics_path = output_root / "metrics_{}_{}.json".format(ckpt_stem, args.inference_mode)

    if args.inference_mode == "symmetric":
        metrics = evaluate_symmetric(cache_payload, state, args)
    elif args.inference_mode == "exact":
        metrics = evaluate_exact(cache_payload, state, args)
    elif args.inference_mode == "rerank":
        metrics = evaluate_rerank(cache_payload, state, args)
    else:
        raise ValueError("Unsupported inference mode: {}".format(args.inference_mode))

    payload = {
        "checkpoint_path": args.checkpoint_path,
        "config_path": state["config_path"],
        "device": args.device,
        "inference_mode": args.inference_mode,
        "cache_npz": cache_payload["cache_npz"],
        "cache_meta": cache_payload["cache_meta"],
        "cache_used": cache_payload["cache_used"],
        "metrics": metrics,
    }
    with open(metrics_path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, ensure_ascii = False, indent = 2)

    print("Metrics saved to:", metrics_path)
    if args.inference_mode != "rerank":
        print(json.dumps(metrics, ensure_ascii = False, indent = 2))
    else:
        for key, value in metrics.items():
            print(key, json.dumps(value, ensure_ascii = False))


if __name__ == "__main__":
    main()
