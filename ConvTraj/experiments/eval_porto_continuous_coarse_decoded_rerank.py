import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from tools import test_methods


def load_pickle_array(path: Path, dtype=np.float32):
    with open(path, "rb") as f:
        value = pickle.load(f)
    return np.asarray(value, dtype=dtype)


def chunked_topk_l2(query_embeddings: np.ndarray, base_embeddings: np.ndarray, k: int, chunk_size: int):
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    base_embeddings = np.asarray(base_embeddings, dtype=np.float32)
    query_norm = np.sum(query_embeddings * query_embeddings, axis=1, keepdims=True)
    base_norm = np.sum(base_embeddings * base_embeddings, axis=1)

    best_dist = np.full((query_embeddings.shape[0], k), np.inf, dtype=np.float32)
    best_idx = np.full((query_embeddings.shape[0], k), -1, dtype=np.int32)

    for begin in range(0, base_embeddings.shape[0], chunk_size):
        end = min(begin + chunk_size, base_embeddings.shape[0])
        base_chunk = base_embeddings[begin:end]
        dist = query_norm + base_norm[begin:end][None, :] - 2.0 * (query_embeddings @ base_chunk.T)
        idx = np.arange(begin, end, dtype=np.int32)[None, :].repeat(query_embeddings.shape[0], axis=0)

        candidate_dist = np.concatenate([best_dist, dist], axis=1)
        candidate_idx = np.concatenate([best_idx, idx], axis=1)
        select = np.argpartition(candidate_dist, kth=k - 1, axis=1)[:, :k]
        best_dist = np.take_along_axis(candidate_dist, select, axis=1)
        best_idx = np.take_along_axis(candidate_idx, select, axis=1)
        order = np.argsort(best_dist, axis=1)
        best_dist = np.take_along_axis(best_dist, order, axis=1)
        best_idx = np.take_along_axis(best_idx, order, axis=1)

    return best_idx, best_dist


def rerank_with_decoded(query_decoded: np.ndarray, base_decoded: np.ndarray, coarse_pred_knn: np.ndarray, rerank_L: int):
    coarse_pred_knn = np.asarray(coarse_pred_knn, dtype=np.int32)
    effective_L = min(int(rerank_L), coarse_pred_knn.shape[1])
    shortlist = coarse_pred_knn[:, :effective_L]
    rerank_return_k = min(100, effective_L)
    reranked = np.empty((shortlist.shape[0], rerank_return_k), dtype=np.int32)

    for i in range(shortlist.shape[0]):
        candidates = base_decoded[shortlist[i]]
        diff = candidates - query_decoded[i : i + 1]
        dist = np.sum(diff * diff, axis=1)
        order = np.argsort(dist)[:rerank_return_k]
        reranked[i] = shortlist[i][order]
    return reranked


def main():
    parser = argparse.ArgumentParser(description="Porto-only hybrid evaluation: continuous coarse shortlist + decoded rerank")
    parser.add_argument("--root_write_path", required=True, type=str)
    parser.add_argument("--train_flag", required=True, type=str)
    parser.add_argument("--epoch", default=400, type=int)
    parser.add_argument("--dist_type", default="dtw", type=str)
    parser.add_argument("--gt_path", required=True, type=str)
    parser.add_argument("--rerank_L", default=1000, type=int)
    parser.add_argument("--chunk_size", default=100000, type=int)
    parser.add_argument("--output_json", default="", type=str)
    args = parser.parse_args()

    feature_dir = Path(args.root_write_path) / "feature_dir"
    query_cont_path = feature_dir / f"query_{args.dist_type}_feature_continuous_{args.train_flag}_{args.epoch}"
    base_cont_path = feature_dir / f"base_{args.dist_type}_feature_continuous_{args.train_flag}_{args.epoch}"
    query_dec_path = feature_dir / f"query_{args.dist_type}_feature_decoded_{args.train_flag}_{args.epoch}"
    base_dec_path = feature_dir / f"base_{args.dist_type}_feature_decoded_{args.train_flag}_{args.epoch}"

    if not query_cont_path.exists():
        raise FileNotFoundError(f"Missing continuous query embeddings: {query_cont_path}")
    if not base_cont_path.exists():
        raise FileNotFoundError(f"Missing continuous base embeddings: {base_cont_path}")
    if not query_dec_path.exists():
        raise FileNotFoundError(f"Missing decoded query embeddings: {query_dec_path}")
    if not base_dec_path.exists():
        raise FileNotFoundError(f"Missing decoded base embeddings: {base_dec_path}")

    query_cont = load_pickle_array(query_cont_path)
    base_cont = load_pickle_array(base_cont_path)
    query_dec = load_pickle_array(query_dec_path)
    base_dec = load_pickle_array(base_dec_path)
    with open(args.gt_path, "rb") as f:
        true_knn = pickle.load(f)

    topk = max(100, int(args.rerank_L))
    coarse_pred_knn, _ = chunked_topk_l2(query_cont, base_cont, k=topk, chunk_size=int(args.chunk_size))
    coarse_metrics = test_methods.metrics_from_pred_knn(true_knn, coarse_pred_knn[:, :100])

    rerank_pred_knn = rerank_with_decoded(query_dec, base_dec, coarse_pred_knn, rerank_L=int(args.rerank_L))
    rerank_metrics = test_methods.metrics_from_pred_knn(true_knn, rerank_pred_knn)

    payload = {
        "protocol": "porto_continuous_coarse_decoded_rerank",
        "train_flag": args.train_flag,
        "epoch": int(args.epoch),
        "rerank_L": int(args.rerank_L),
        "continuous_coarse_top100": coarse_metrics,
        "continuous_coarse_decoded_rerank": rerank_metrics,
        "embedding_paths": {
            "query_continuous": str(query_cont_path),
            "base_continuous": str(base_cont_path),
            "query_decoded": str(query_dec_path),
            "base_decoded": str(base_dec_path),
        },
    }

    output_json = Path(args.output_json) if args.output_json else feature_dir / f"metrics_hybrid_continuous_coarse_decoded_L{int(args.rerank_L)}_{args.train_flag}_e{int(args.epoch)}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Hybrid metrics saved to:", output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
