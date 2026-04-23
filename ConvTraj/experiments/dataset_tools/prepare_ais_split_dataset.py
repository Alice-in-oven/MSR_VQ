#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


DIST_SOURCE_MAP = {
    "dtw": "dis_matrix_dtw_10k.pkl",
    "haus": "dis_matrix_haus_10k.pkl",
    # Keep the main-chain naming convention even though the raw file is named dfrec.
    "dfd": "dis_matrix_dfrec_10k.pkl",
}


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dump_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the raw AIS dataset into ConvTraj split-dataset format."
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="/data3/menghaotian/Landmarks-TrajSim/datasets/AIS",
        help="Raw AIS dataset directory containing traj_list.pkl and full distance matrices.",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="/data3/menghaotian/Traj_sim/trajectory_specific_VQ/ConvTraj/data/0_ais",
        help="Output directory in ConvTraj split-dataset format.",
    )
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--query_size", type=int, default=1000)
    parser.add_argument("--base_size", type=int, default=6000)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Target dtype for saved distance matrices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()

    total_required = int(args.train_size) + int(args.query_size) + int(args.base_size)
    target_dtype = np.float32 if args.dtype == "float32" else np.float64

    traj_list_path = src_root / "traj_list.pkl"
    if not traj_list_path.exists():
        raise FileNotFoundError(f"Missing trajectory list: {traj_list_path}")

    raw_trajs = _load_pickle(traj_list_path)
    if isinstance(raw_trajs, tuple):
        raw_trajs = list(raw_trajs)
    else:
        raw_trajs = list(raw_trajs)

    if len(raw_trajs) < total_required:
        raise ValueError(
            f"AIS traj_list is too short: len={len(raw_trajs)} < required={total_required}"
        )

    train_end = int(args.train_size)
    query_end = train_end + int(args.query_size)
    base_end = query_end + int(args.base_size)

    train_list = raw_trajs[:train_end]
    query_list = raw_trajs[train_end:query_end]
    base_list = raw_trajs[query_end:base_end]
    combined_list = train_list + query_list + base_list

    _dump_pickle(dst_root / "train_list", train_list)
    _dump_pickle(dst_root / "query_list", query_list)
    _dump_pickle(dst_root / "base_list", base_list)
    _dump_pickle(dst_root / "traj_list", combined_list)

    for dist_type, src_name in DIST_SOURCE_MAP.items():
        src_path = src_root / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"Missing distance matrix for {dist_type}: {src_path}")

        full_matrix = np.asarray(_load_pickle(src_path), dtype=target_dtype)
        if full_matrix.shape[0] < total_required or full_matrix.shape[1] < total_required:
            raise ValueError(
                f"{dist_type} full matrix shape {full_matrix.shape} cannot cover required={total_required}"
            )

        train_matrix = np.asarray(full_matrix[:train_end, :train_end], dtype=target_dtype)
        test_matrix = np.asarray(full_matrix[train_end:query_end, query_end:base_end], dtype=target_dtype)

        dist_dir_name = {
            "dtw": "DTW",
            "haus": "Haus",
            "dfd": "DFD",
        }[dist_type]
        dist_dir = dst_root / dist_dir_name
        _dump_pickle(dist_dir / f"{dist_type}_train_distance_matrix_result", train_matrix)
        _dump_pickle(dist_dir / f"{dist_type}_test_distance_matrix_result", test_matrix)

    meta = {
        "source_root": str(src_root),
        "target_root": str(dst_root),
        "train_size": int(args.train_size),
        "query_size": int(args.query_size),
        "base_size": int(args.base_size),
        "total_required": total_required,
        "saved_dtype": args.dtype,
        "distance_sources": DIST_SOURCE_MAP,
    }
    (dst_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
