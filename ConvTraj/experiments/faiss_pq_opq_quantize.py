#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize continuous trajectory embeddings with Faiss PQ/OPQ."
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="split",
        choices=["split", "all"],
        help="Use separate train/query/base files or a single all-embedding file plus split sizes.",
    )
    parser.add_argument("--train_path", type=str, default="", help="Pickle path for train embeddings.")
    parser.add_argument("--query_path", type=str, default="", help="Pickle path for query embeddings.")
    parser.add_argument("--base_path", type=str, default="", help="Pickle path for base embeddings.")
    parser.add_argument("--all_path", type=str, default="", help="Pickle path for all embeddings.")
    parser.add_argument("--train_size", type=int, default=0, help="Train split size when using --input_mode all.")
    parser.add_argument("--query_size", type=int, default=0, help="Query split size when using --input_mode all.")
    parser.add_argument("--base_size", type=int, default=0, help="Base split size when using --input_mode all.")
    parser.add_argument(
        "--quantizer",
        type=str,
        default="pq",
        choices=["pq", "opq"],
        help="Faiss quantizer backend.",
    )
    parser.add_argument("--m", type=int, required=True, help="Number of sub-quantizers.")
    parser.add_argument("--nbits", type=int, default=8, help="Bits per sub-quantizer.")
    parser.add_argument(
        "--opq_out_dim",
        type=int,
        default=0,
        help="Optional OPQ output dimension. 0 means keep the original dimension.",
    )
    parser.add_argument("--opq_niter", type=int, default=50, help="OPQ training iterations.")
    parser.add_argument("--pq_train_limit", type=int, default=0, help="Max train vectors used to fit PQ/OPQ. 0 uses all.")
    parser.add_argument("--seed", type=int, default=123456, help="Random seed for train subsampling.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save quantized artifacts.")
    parser.add_argument("--artifact_prefix", type=str, default="", help="Prefix for saved artifact filenames.")
    parser.add_argument(
        "--save_rotated",
        action="store_true",
        help="For OPQ, also save rotated vectors and reconstructed rotated vectors.",
    )
    parser.add_argument(
        "--skip_all_dump",
        action="store_true",
        help="Do not save concatenated all-split reconstructed vectors and codes.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="",
        help="Optional pickle path for the query-base ground-truth distance matrix.",
    )
    parser.add_argument(
        "--eval_on",
        type=str,
        default="recon",
        choices=["recon", "rotated"],
        help="Embedding space used for retrieval evaluation when --gt_path is provided.",
    )
    parser.add_argument(
        "--save_distance_matrix",
        action="store_true",
        help="Also save the predicted query-base distance matrix used for evaluation.",
    )
    return parser.parse_args()


def _load_pickle_array(path):
    with open(path, "rb") as f:
        array = pickle.load(f)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape} from {path}")
    return np.ascontiguousarray(array)


def load_embeddings(args):
    if args.input_mode == "split":
        if not args.train_path or not args.query_path or not args.base_path:
            raise ValueError("--input_mode split requires --train_path --query_path --base_path")
        train = _load_pickle_array(args.train_path)
        query = _load_pickle_array(args.query_path)
        base = _load_pickle_array(args.base_path)
        all_embeddings = np.concatenate([train, query, base], axis=0)
        return train, query, base, all_embeddings

    if not args.all_path:
        raise ValueError("--input_mode all requires --all_path")
    if args.train_size <= 0 or args.query_size <= 0 or args.base_size <= 0:
        raise ValueError("--input_mode all requires positive --train_size --query_size --base_size")

    all_embeddings = _load_pickle_array(args.all_path)
    expected = args.train_size + args.query_size + args.base_size
    if all_embeddings.shape[0] != expected:
        raise ValueError(
            f"Split sizes do not match all embeddings: {all_embeddings.shape[0]} != {expected}"
        )
    train_end = args.train_size
    query_end = train_end + args.query_size
    train = all_embeddings[:train_end]
    query = all_embeddings[train_end:query_end]
    base = all_embeddings[query_end:]
    return train, query, base, all_embeddings


def maybe_subsample_train(train, limit, seed):
    if limit <= 0 or train.shape[0] <= limit:
        return train
    rng = np.random.default_rng(seed)
    indices = rng.choice(train.shape[0], size=limit, replace=False)
    indices.sort()
    return np.ascontiguousarray(train[indices])


def build_quantizer(train_embeddings, quantizer_type, m, nbits, opq_out_dim, opq_niter):
    dim = int(train_embeddings.shape[1])
    if dim % m != 0:
        raise ValueError(f"Embedding dim {dim} must be divisible by m={m}")

    if quantizer_type == "pq":
        pq = faiss.ProductQuantizer(dim, m, nbits)
        pq.train(train_embeddings)
        return {
            "quantizer_type": "pq",
            "dim_in": dim,
            "dim_pq": dim,
            "pq": pq,
            "opq": None,
        }

    dim_out = int(opq_out_dim) if int(opq_out_dim) > 0 else dim
    if dim_out % m != 0:
        raise ValueError(f"OPQ output dim {dim_out} must be divisible by m={m}")
    if train_embeddings.shape[0] < 256:
        raise ValueError(
            "Faiss OPQ needs at least 256 training vectors in this environment; "
            f"got {train_embeddings.shape[0]}. Increase --pq_train_limit or use a larger train embedding set."
        )

    opq = faiss.OPQMatrix(dim, m, dim_out)
    opq.niter = int(opq_niter)
    opq.train(train_embeddings)
    train_rot = np.ascontiguousarray(opq.apply_py(train_embeddings).astype(np.float32, copy=False))
    pq = faiss.ProductQuantizer(dim_out, m, nbits)
    pq.train(train_rot)
    return {
        "quantizer_type": "opq",
        "dim_in": dim,
        "dim_pq": dim_out,
        "pq": pq,
        "opq": opq,
    }


def quantize_split(x, bundle):
    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    opq = bundle["opq"]
    pq = bundle["pq"]

    if opq is not None:
        rotated = np.ascontiguousarray(opq.apply_py(x).astype(np.float32, copy=False))
        codes = np.ascontiguousarray(pq.compute_codes(rotated))
        recon_rot = np.ascontiguousarray(pq.decode(codes).astype(np.float32, copy=False))
        recon = np.ascontiguousarray(opq.reverse_transform(recon_rot).astype(np.float32, copy=False))
        return {
            "codes": codes,
            "recon": recon,
            "rotated": rotated,
            "recon_rotated": recon_rot,
        }

    codes = np.ascontiguousarray(pq.compute_codes(x))
    recon = np.ascontiguousarray(pq.decode(codes).astype(np.float32, copy=False))
    return {
        "codes": codes,
        "recon": recon,
        "rotated": None,
        "recon_rotated": None,
    }


def mse(a, b):
    diff = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.mean(diff * diff))


def prefix_name(prefix, stem):
    prefix = str(prefix or "").strip()
    return f"{prefix}_{stem}" if prefix else stem


def save_pickle(path, value):
    with open(path, "wb") as f:
        pickle.dump(value, f, protocol=4)


def l2_dist(query_embeddings, base_embeddings):
    query_sq = np.sum(query_embeddings * query_embeddings, axis=1, keepdims=True)
    base_sq = np.sum(base_embeddings * base_embeddings, axis=1, keepdims=True).T
    dist = query_sq + base_sq - 2.0 * np.matmul(query_embeddings, base_embeddings.T)
    np.maximum(dist, 0.0, out=dist)
    return dist


def get_knn_from_dist(dist):
    return np.argsort(dist, axis=1)


def intersect_sizes(true_list, test_list):
    return np.array([len(np.intersect1d(true_value, list(test_value))) for true_value, test_value in zip(true_list, test_list)])


def metrics_from_pred_knn(true_knn, pred_knn):
    top_test_dict = {5: [5], 10: [10], 50: [50, 10], 100: [100]}
    results = {}
    top10_at_50 = None
    for tem_test_num, top_true_list in top_test_dict.items():
        test_top_id = pred_knn[:, :tem_test_num]
        intersect_list = [
            intersect_sizes(true_knn[:, :tem_true_num], test_top_id) / float(tem_true_num)
            for tem_true_num in top_true_list
        ]
        recall_list = [float(np.mean(tem_list)) for tem_list in intersect_list]
        for pos, tem_true_num in enumerate(top_true_list):
            if tem_test_num == 5 and tem_true_num == 5:
                results["top5_recall"] = recall_list[pos]
            if tem_test_num == 10 and tem_true_num == 10:
                results["top10_recall"] = recall_list[pos]
            if tem_test_num == 50 and tem_true_num == 50:
                results["top50_recall"] = recall_list[pos]
            if tem_test_num == 100 and tem_true_num == 100:
                results["top100_recall"] = recall_list[pos]
            if tem_test_num == 50 and tem_true_num == 10:
                top10_at_50 = recall_list[pos]
    results["top10_at_50_recall"] = top10_at_50
    results["metrics_list"] = [
        results["top5_recall"],
        results["top10_recall"],
        results["top50_recall"],
        results["top100_recall"],
        results["top10_at_50_recall"],
    ]
    return results


def evaluate_retrieval(query_embeddings, base_embeddings, gt_dist):
    if query_embeddings.shape[0] != gt_dist.shape[0]:
        raise ValueError(
            f"Query count mismatch for evaluation: embeddings={query_embeddings.shape[0]} gt={gt_dist.shape[0]}"
        )
    if base_embeddings.shape[0] != gt_dist.shape[1]:
        raise ValueError(
            f"Base count mismatch for evaluation: embeddings={base_embeddings.shape[0]} gt={gt_dist.shape[1]}"
        )
    pred_dist = l2_dist(query_embeddings, base_embeddings)
    pred_knn = get_knn_from_dist(pred_dist)
    true_knn = get_knn_from_dist(gt_dist)
    return pred_dist, metrics_from_pred_knn(true_knn, pred_knn)


def save_artifacts(output_dir, prefix, split_name, result, save_rotated):
    recon_path = output_dir / prefix_name(prefix, f"{split_name}_faiss_recon.pkl")
    codes_path = output_dir / prefix_name(prefix, f"{split_name}_faiss_codes.npy")
    save_pickle(recon_path, result["recon"])
    np.save(codes_path, result["codes"])

    extra = {
        "recon_path": str(recon_path),
        "codes_path": str(codes_path),
    }
    if save_rotated and result["rotated"] is not None:
        rotated_path = output_dir / prefix_name(prefix, f"{split_name}_faiss_rotated.pkl")
        recon_rotated_path = output_dir / prefix_name(prefix, f"{split_name}_faiss_recon_rotated.pkl")
        save_pickle(rotated_path, result["rotated"])
        save_pickle(recon_rotated_path, result["recon_rotated"])
        extra["rotated_path"] = str(rotated_path)
        extra["recon_rotated_path"] = str(recon_rotated_path)
    return extra


def export_model_artifacts(output_dir, prefix, bundle):
    pq = bundle["pq"]
    opq = bundle["opq"]

    pq_path = output_dir / prefix_name(prefix, "faiss_pq.index")
    faiss.write_ProductQuantizer(pq, str(pq_path))

    artifacts = {"pq_path": str(pq_path)}
    if opq is not None:
        opq_path = output_dir / prefix_name(prefix, "faiss_opq_transform.faiss")
        faiss.write_VectorTransform(opq, str(opq_path))
        artifacts["opq_path"] = str(opq_path)

    centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
    codebook_path = output_dir / prefix_name(prefix, "faiss_pq_codebook.npy")
    np.save(codebook_path, centroids.astype(np.float32, copy=False))
    artifacts["pq_codebook_path"] = str(codebook_path)
    return artifacts


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, query, base, all_embeddings = load_embeddings(args)
    dim = int(train.shape[1])
    if query.shape[1] != dim or base.shape[1] != dim:
        raise ValueError("train/query/base embedding dimensions must match")

    train_fit = maybe_subsample_train(train, args.pq_train_limit, args.seed)
    min_required = 1 << int(args.nbits)
    if train_fit.shape[0] < min_required:
        raise ValueError(
            f"Not enough training vectors for nbits={args.nbits}: {train_fit.shape[0]} < {min_required}"
        )

    print(
        f"[FaissQuant] mode={args.quantizer} dim={dim} m={args.m} nbits={args.nbits} "
        f"train={train.shape[0]} query={query.shape[0]} base={base.shape[0]} fit={train_fit.shape[0]}"
    )
    bundle = build_quantizer(
        train_embeddings=train_fit,
        quantizer_type=args.quantizer,
        m=args.m,
        nbits=args.nbits,
        opq_out_dim=args.opq_out_dim,
        opq_niter=args.opq_niter,
    )

    split_inputs = {
        "train": train,
        "query": query,
        "base": base,
    }
    split_results = {}
    for split_name, split_array in split_inputs.items():
        print(f"[FaissQuant] quantizing {split_name} shape={split_array.shape}")
        split_results[split_name] = quantize_split(split_array, bundle)

    if not args.skip_all_dump:
        all_recon = np.concatenate([split_results["train"]["recon"], split_results["query"]["recon"], split_results["base"]["recon"]], axis=0)
        all_codes = np.concatenate([split_results["train"]["codes"], split_results["query"]["codes"], split_results["base"]["codes"]], axis=0)
        split_results["all"] = {
            "recon": all_recon,
            "codes": all_codes,
            "rotated": None,
            "recon_rotated": None,
        }
        if args.save_rotated and bundle["opq"] is not None:
            split_results["all"]["rotated"] = np.concatenate(
                [split_results["train"]["rotated"], split_results["query"]["rotated"], split_results["base"]["rotated"]],
                axis=0,
            )
            split_results["all"]["recon_rotated"] = np.concatenate(
                [split_results["train"]["recon_rotated"], split_results["query"]["recon_rotated"], split_results["base"]["recon_rotated"]],
                axis=0,
            )

    saved_paths = {}
    for split_name, result in split_results.items():
        saved_paths[split_name] = save_artifacts(output_dir, args.artifact_prefix, split_name, result, args.save_rotated)

    model_artifacts = export_model_artifacts(output_dir, args.artifact_prefix, bundle)

    stats = {
        "args": vars(args),
        "quantizer": args.quantizer,
        "input_dim": dim,
        "pq_dim": bundle["dim_pq"],
        "m": args.m,
        "nbits": args.nbits,
        "code_size_bytes": int(bundle["pq"].code_size),
        "train_size": int(train.shape[0]),
        "query_size": int(query.shape[0]),
        "base_size": int(base.shape[0]),
        "fit_train_size": int(train_fit.shape[0]),
        "mse": {
            "train": mse(train, split_results["train"]["recon"]),
            "query": mse(query, split_results["query"]["recon"]),
            "base": mse(base, split_results["base"]["recon"]),
        },
        "saved_paths": saved_paths,
        "model_artifacts": model_artifacts,
    }
    if not args.skip_all_dump:
        stats["mse"]["all"] = mse(all_embeddings, split_results["all"]["recon"])

    if args.gt_path:
        gt_dist = np.asarray(pickle.load(open(args.gt_path, "rb")), dtype=np.float32)
        if args.eval_on == "rotated":
            if bundle["opq"] is None:
                raise ValueError("--eval_on rotated requires --quantizer opq")
            query_eval = split_results["query"]["rotated"]
            base_eval = split_results["base"]["rotated"]
        else:
            query_eval = split_results["query"]["recon"]
            base_eval = split_results["base"]["recon"]
        pred_dist, retrieval_metrics = evaluate_retrieval(query_eval, base_eval, gt_dist)
        stats["retrieval_eval"] = {
            "gt_path": args.gt_path,
            "eval_on": args.eval_on,
            **retrieval_metrics,
        }
        print(
            "[FaissQuant] retrieval Top-5 {:.4f} | Top-10 {:.4f} | Top-50 {:.4f} | Top-100 {:.4f} | Top-10@50 {:.4f}".format(
                retrieval_metrics["top5_recall"],
                retrieval_metrics["top10_recall"],
                retrieval_metrics["top50_recall"],
                retrieval_metrics["top100_recall"],
                retrieval_metrics["top10_at_50_recall"],
            )
        )
        if args.save_distance_matrix:
            dist_path = output_dir / prefix_name(args.artifact_prefix, f"query_base_{args.eval_on}_l2_dist.pkl")
            save_pickle(dist_path, pred_dist.astype(np.float32, copy=False))
            stats["retrieval_eval"]["pred_distance_path"] = str(dist_path)

    stats_path = output_dir / prefix_name(args.artifact_prefix, "faiss_quant_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[FaissQuant] done")
    print(json.dumps({"stats_path": str(stats_path), "model_artifacts": model_artifacts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
