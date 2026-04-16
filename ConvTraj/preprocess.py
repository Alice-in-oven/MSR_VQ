# import os
# import json
# import random
# import pickle
# import multiprocessing as mp
# import time

# import numpy as np
# from dtaidistance import dtw_ndim
# from frechetdist import frdist
# from scipy.spatial.distance import directed_hausdorff

# from NeuTraj_preprocess import trajectory_feature_generation

# LatLon_range = {
#     "Porto": [[41.10, 41.24], [-8.73, -8.50]]
# }

# paths = {
#     "Porto": "./Porto/porto_trajs.pkl"
# }

# min_length = {
#     "Porto": 10
# }


# DATASET = "Porto"
# BASE_SIZE = 1_000_000          # 例如 100万
# BASE_INNER_SAMPLE = 100        # base 内每条随机采样 100 条
# QUERY_SIZE = 100               # 从剩余轨迹里采 100 条作为 query
# SEED = 42
# METRIC = "dtw"                 # "dtw" | "frechet" | "hausdorff"

# # NUM_PROCESSES = max(1, (os.cpu_count() or 8) - 1)
# NUM_PROCESSES = 16
# CHUNK_SIZE = 32

# OUT_DIR = "./features_sampling"
# os.makedirs(OUT_DIR, exist_ok=True)

# def now():
#     return time.perf_counter()

# def fmt(sec):
#     if sec < 60:
#         return f"{sec:.1f}s"
#     elif sec < 3600:
#         return f"{sec/60:.1f}min"
#     else:
#         return f"{sec/3600:.2f}h"


# # =========================
# # 读取/保存工具
# # =========================

# def load_traj_coord(traj_coord_path):
#     # traj_coord: (trajs, [], max_len)
#     with open(traj_coord_path, "rb") as f:
#         trajs, _, max_len = pickle.load(f)
#     return trajs, max_len

# def load_traj_grid(traj_grid_path):
#     # traj_grid: (all_trajs_grids_xy, [], max_len)
#     with open(traj_grid_path, "rb") as f:
#         grids, _, max_len = pickle.load(f)
#     return grids, max_len

# def save_subset_pickle(out_path, data, max_len):
#     with open(out_path, "wb") as f:
#         pickle.dump((data, [], max_len), f, protocol=pickle.HIGHEST_PROTOCOL)

# def infer_traj_grid_path_from_coord(traj_coord_path, data_name):
#     # 你的 preprocessing 输出固定是 ./features/{fname}_traj_coord / _traj_grid
#     # trajectory_feature_generation 返回的是 traj_coord_path 和 fname(data_name)
#     feat_dir = "./features"
#     return os.path.join(feat_dir, f"{data_name}_traj_grid")

# # =========================
# # 距离函数（按 METRIC 切换）
# # =========================

# def dist_dtw(traj_a, traj_b):
#     a = np.asarray(traj_a, dtype=np.float32)
#     b = np.asarray(traj_b, dtype=np.float32)
#     return float(dtw_ndim.distance(a, b))

# def dist_frechet(traj_a, traj_b):
#     # frechetdist 接受 list[tuple/list]
#     return float(frdist(traj_a, traj_b))

# def dist_hausdorff(traj_a, traj_b):
#     a = np.asarray(traj_a, dtype=np.float32)
#     b = np.asarray(traj_b, dtype=np.float32)
#     d1 = directed_hausdorff(a, b)[0]
#     d2 = directed_hausdorff(b, a)[0]
#     return float(max(d1, d2))

# def trajectory_distance(traj_a, traj_b):
#     if METRIC == "dtw":
#         return dist_dtw(traj_a, traj_b)
#     elif METRIC == "frechet":
#         return dist_frechet(traj_a, traj_b)
#     elif METRIC == "hausdorff":
#         return dist_hausdorff(traj_a, traj_b)
#     else:
#         raise ValueError(f"Unknown METRIC={METRIC}")

# # =========================
# # 多进程：Base 内采样相似度
# # =========================

# # worker 全局
# G_BASE_TRAJS = None
# G_BASE_N = 0
# G_SEED = 0
# G_SAMPLE = 100

# def _init_base_inner_worker(base_trajs, seed, sample_k):
#     global G_BASE_TRAJS, G_BASE_N, G_SEED, G_SAMPLE
#     G_BASE_TRAJS = base_trajs          # 只保存 base 子集
#     G_BASE_N = len(base_trajs)
#     G_SEED = seed
#     G_SAMPLE = sample_k

# def _sample_base_local(rng, n, k, exclude):
#     k = min(k, n - 1)
#     chosen = set()
#     while len(chosen) < k:
#         j = rng.randrange(n)
#         if j != exclude:
#             chosen.add(j)
#     return list(chosen)

# def _base_inner_task(base_lid):
#     rng = random.Random(G_SEED + base_lid * 1000003)

#     traj_i = G_BASE_TRAJS[base_lid]
#     js = _sample_base_local(rng, G_BASE_N, G_SAMPLE, exclude=base_lid)

#     out = []
#     for j in js:
#         d = trajectory_distance(traj_i, G_BASE_TRAJS[j])
#         out.append((j, d))   # j 是 base-local id
#     return base_lid, out

# # =========================
# # 多进程：Query vs Base 全量相似度（memmap）
# # =========================

# G_Q_TRAJS = None
# G_BASE_TRAJS_QB = None

# def _init_query_base_worker(q_trajs, base_trajs):
#     global G_Q_TRAJS, G_BASE_TRAJS_QB
#     G_Q_TRAJS = q_trajs
#     G_BASE_TRAJS_QB = base_trajs

# def _query_one_to_base_task(q_idx):
#     q = G_Q_TRAJS[q_idx]
#     dists = np.empty((len(G_BASE_TRAJS_QB),), dtype=np.float32)
#     for k, b in enumerate(G_BASE_TRAJS_QB):
#         dists[k] = trajectory_distance(q, b)
#     return q_idx, dists

# if __name__ == "__main__":

#     print(f"[INFO] dataset={DATASET} metric={METRIC} base_size={BASE_SIZE} query_size={QUERY_SIZE}")
#     print(f"[INFO] processes={NUM_PROCESSES}")

#     coor_path, data_name = trajectory_feature_generation(path=paths[DATASET], 
#                                                          lat_range=LatLon_range[DATASET][0], 
#                                                          lon_range=LatLon_range[DATASET][1],
#                                                          min_length=min_length[DATASET])
#     grid_path = infer_traj_grid_path_from_coord(coor_path, data_name)

#     trajs_coord, max_len_coord = load_traj_coord(coor_path)
#     trajs_grid, max_len_grid = load_traj_grid(grid_path)
#     assert len(trajs_coord) == len(trajs_grid), "traj_coord 和 traj_grid 数量不一致"

#     n_all = len(trajs_coord)
#     print(f"[INFO] total trajs loaded: {n_all}")

#     if BASE_SIZE + QUERY_SIZE > n_all:
#         raise ValueError(f"BASE_SIZE+QUERY_SIZE={BASE_SIZE+QUERY_SIZE} > total={n_all}")

#     # 3) 采样 base / query 索引（全局索引）
#     base_coord_path = os.path.join(OUT_DIR, f"{DATASET}_base_traj_coord.pkl")
#     base_grid_path  = os.path.join(OUT_DIR, f"{DATASET}_base_traj_grid.pkl")
#     query_coord_path = os.path.join(OUT_DIR, f"{DATASET}_query_traj_coord.pkl")
#     query_grid_path  = os.path.join(OUT_DIR, f"{DATASET}_query_traj_grid.pkl")

#     subset_exists = (
#         os.path.exists(base_coord_path)
#         and os.path.exists(base_grid_path)
#         and os.path.exists(query_coord_path)
#         and os.path.exists(query_grid_path)
#     )
#     if subset_exists:
#         print("[INFO] base/query subset already exists, loading from disk")

#         base_coord, _ = load_traj_coord(base_coord_path)
#         base_grid, _  = load_traj_grid(base_grid_path)
#         query_coord, _ = load_traj_coord(query_coord_path)
#         query_grid, _  = load_traj_grid(query_grid_path)

#         BASE_SIZE = len(base_coord)
#         QUERY_SIZE = len(query_coord)

#         print(f"[INFO] loaded base={BASE_SIZE}, query={QUERY_SIZE}")

#     else:
#         print("[INFO] base/query subset not found, sampling new subset")
#         rng = random.Random(SEED)
#         all_indices = list(range(n_all))
#         rng.shuffle(all_indices)

#         base_indices = all_indices[:BASE_SIZE]
#         remaining = all_indices[BASE_SIZE:]
#         query_indices = remaining[:QUERY_SIZE]

#         base_indices_np = np.asarray(base_indices, dtype=np.int64)
#         query_indices_np = np.asarray(query_indices, dtype=np.int64)

#         np.save(os.path.join(OUT_DIR, f"{DATASET}_base_indices.npy"), base_indices_np)
#         np.save(os.path.join(OUT_DIR, f"{DATASET}_query_indices.npy"), query_indices_np)

#         print(f"[INFO] sampled base={len(base_indices)} query={len(query_indices)}")

#         # 4) 保存本次实验用到的 base/query 子集轨迹与网格（便于复现 & 后续重复跑）
#         base_coord = [trajs_coord[i] for i in base_indices]
#         base_grid  = [trajs_grid[i] for i in base_indices]
#         query_coord = [trajs_coord[i] for i in query_indices]
#         query_grid  = [trajs_grid[i] for i in query_indices]

#         save_subset_pickle(base_coord_path, base_coord, max_len_coord)
#         save_subset_pickle(base_grid_path, base_grid, max_len_grid)
#         save_subset_pickle(query_coord_path, query_coord, max_len_coord)
#         save_subset_pickle(query_grid_path, query_grid, max_len_grid)

#         print(f"[INFO] saved subset pickles into {OUT_DIR}")

#     # 5) Base 内：每条随机采 100 条计算相似度（多进程 + jsonl 流式写）
#     base_inner_out = os.path.join(OUT_DIR, f"{DATASET}_base_inner_{METRIC}.jsonl")
#     ctx = mp.get_context("fork")

#     print("[INFO] computing base-inner sampled similarities...")
#     t0 = now()

#     processed = 0
#     total = len(base_indices)
#     with ctx.Pool(
#         processes=NUM_PROCESSES,
#         initializer=_init_base_inner_worker,
#         initargs=(base_coord, SEED, BASE_INNER_SAMPLE),
#     ) as pool, open(base_inner_out, "w", encoding="utf-8") as fout:

#         # 注意：这里遍历的是 base_indices（全局索引）
#         for base_lid, pairs in pool.imap_unordered(_base_inner_task, range(BASE_SIZE), chunksize=CHUNK_SIZE):
#             # 输出全局索引，避免后续混淆
#             rec = {"base_lid": int(base_lid), "metric": METRIC, "pairs": [[int(j), float(d)] for j, d in pairs]}
#             fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
#             processed += 1
#             if processed % 1000 == 0:
#                 elapsed = now() - t0
#                 speed = processed / elapsed
#                 eta = (total - processed) / speed if speed > 0 else 0
#                 print(
#                     f"[BASE-INNER] {processed}/{total} "
#                     f"elapsed={fmt(elapsed)} "
#                     f"speed={speed:.2f} traj/s "
#                     f"ETA={fmt(eta)}"
#                 )

#     t1 = now()
#     print(f"[DONE] base-inner finished in {fmt(t1 - t0)}")
#     print(f"[DONE] base-inner output: {base_inner_out}")

#     # 6) Query vs Base：计算 100 条 query 与 100万 base 的全量距离矩阵（写 memmap，不占内存）
#     #    输出是一个 (QUERY_SIZE, BASE_SIZE) 的 float32 距离矩阵
#     query_base_memmap_path = os.path.join(OUT_DIR, f"{DATASET}_query_base_{METRIC}.f32mmap")
#     print("[INFO] computing query-to-base full similarities (memmap on disk)...")
#     print(f"[INFO] memmap shape = ({QUERY_SIZE}, {BASE_SIZE}) float32")

#     t0 = now()
#     processed = 0

#     # 建立 memmap 文件
#     dist_mm = np.memmap(
#         query_base_memmap_path,
#         dtype=np.float32,
#         mode="w+",
#         shape=(QUERY_SIZE, BASE_SIZE),
#     )

#     # 为了让 query->base 更省内存、逻辑更清晰：worker 用子集 base_coord / query_coord
#     # 并行计算每个 query 对所有 base 的距离向量
#     with ctx.Pool(
#         processes=NUM_PROCESSES,
#         initializer=_init_query_base_worker,
#         initargs=(query_coord, base_coord),
#     ) as pool:
#         for q_idx, dists in pool.imap_unordered(_query_one_to_base_task, range(QUERY_SIZE), chunksize=1):
#             dist_mm[q_idx, :] = dists  # 写入磁盘映射
#             if q_idx % 10 == 0:
#                 print(f"[INFO] query processed {q_idx}/{QUERY_SIZE}")

#             elapsed = now() - t0
#             speed = processed / elapsed
#             eta = (QUERY_SIZE - processed) / speed if speed > 0 else 0

#             print(
#                 f"[QUERY-BASE] {processed}/{QUERY_SIZE} "
#                 f"elapsed={fmt(elapsed)} "
#                 f"speed={speed:.2f} query/s "
#                 f"ETA={fmt(eta)}"
#             )

#     dist_mm.flush()
#     del dist_mm

#     t1 = now()
#     print(f"[DONE] query-base finished in {fmt(t1 - t0)}")
#     print(f"[DONE] query-base memmap: {query_base_memmap_path}")

#     # 同时写一个元信息文件（方便你之后读取）
#     meta = {
#         "dataset": DATASET,
#         "metric": METRIC,
#         "base_size": BASE_SIZE,
#         "query_size": QUERY_SIZE,
#         "dtype": "float32",
#         "memmap_path": query_base_memmap_path,
#         "base_indices_path": os.path.join(OUT_DIR, f"{DATASET}_base_indices.npy"),
#         "query_indices_path": os.path.join(OUT_DIR, f"{DATASET}_query_indices.npy"),
#         "base_coord_path": base_coord_path,
#         "query_coord_path": query_coord_path,
#         "base_grid_path": base_grid_path,
#         "query_grid_path": query_grid_path,
#         "seed": SEED,
#     }
#     meta_path = os.path.join(OUT_DIR, f"{DATASET}_query_base_{METRIC}.meta.json")
#     with open(meta_path, "w", encoding="utf-8") as f:
#         json.dump(meta, f, ensure_ascii=False, indent=2)

#     print(f"[DONE] query-base memmap: {query_base_memmap_path}")
#     print(f"[DONE] meta saved: {meta_path}")

import os
import json
import random
import pickle
import argparse
import multiprocessing as mp
import time

import numpy as np
from dtaidistance import dtw_ndim
from frechetdist import frdist
from curvesimilarities import dfd, dtw
from scipy.spatial.distance import directed_hausdorff

from NeuTraj_preprocess import trajectory_feature_generation


LatLon_range = {
    "Porto": [[41.10, 41.24], [-8.73, -8.50]],
    "Chengdu": [[104.04215, 104.12958], [30.65294, 30.72775]]
}

paths = {
    "Porto": "./Porto/porto_trajs.pkl",
    "Chengdu": "./Chengdu/chengdu_trajs.pkl"
}

min_length = {
    "Porto": 10,
    "Chengdu": 10
}


# -------------------------
# time helpers
# -------------------------
def now():
    return time.perf_counter()

def fmt(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}min"
    return f"{sec/3600:.2f}h"


# -------------------------
# IO helpers
# -------------------------
def load_traj_coord(traj_coord_path):
    with open(traj_coord_path, "rb") as f:
        trajs, _, max_len = pickle.load(f)
    return trajs, max_len

def load_traj_grid(traj_grid_path):
    with open(traj_grid_path, "rb") as f:
        grids, _, max_len = pickle.load(f)
    return grids, max_len

def save_subset_pickle(out_path, data, max_len):
    with open(out_path, "wb") as f:
        pickle.dump((data, [], max_len), f, protocol=pickle.HIGHEST_PROTOCOL)

def infer_traj_grid_path_from_coord(data_name):
    feat_dir = "./features"
    return os.path.join(feat_dir, f"{data_name}_traj_grid")


# -------------------------
# distance functions
# -------------------------
def trajs_to_np(trajs, dtype=np.float32):
    out = []
    for traj in trajs:
        if not traj or len(traj) < 2:
            out.append(None)   # 或者 np.empty((0,2))
            continue
        arr = np.asarray(traj, dtype=dtype)
        # 防御：确保是 (T,2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            out.append(None)
        else:
            out.append(arr)
    return out

def dist_dtw(traj_a, traj_b):
    # return float(dtw_ndim.distance(traj_a, traj_b))
    return float(dtw(traj_a, traj_b))

def dist_frechet(traj_a, traj_b):
    return float(dfd(traj_a, traj_b))

def dist_hausdorff(traj_a, traj_b):
    d1 = directed_hausdorff(traj_a, traj_b)[0]
    d2 = directed_hausdorff(traj_b, traj_a)[0]
    return float(max(d1, d2))

def make_distance_fn(metric: str):
    metric = metric.lower()
    if metric == "dtw":
        return dist_dtw
    if metric == "frechet":
        return dist_frechet
    if metric == "hausdorff":
        return dist_hausdorff
    raise ValueError(f"Unknown metric: {metric}")

# =========================================================
# Base-inner multiprocessing (base-local ids)
# =========================================================
G_BASE_TRAJS = None
G_BASE_N = 0
G_SEED = 0
G_SAMPLE = 100
G_DIST_FN = None

def _init_base_inner_worker(base_trajs, seed, sample_k, metric):
    global G_BASE_TRAJS, G_BASE_N, G_SEED, G_SAMPLE, G_DIST_FN
    G_BASE_TRAJS = base_trajs
    G_BASE_N = len(base_trajs)
    G_SEED = seed
    G_SAMPLE = sample_k
    G_DIST_FN = make_distance_fn(metric)

def _sample_base_local(rng, n, k, exclude):
    k = min(k, n - 1)
    chosen = set()
    while len(chosen) < k:
        j = rng.randrange(n)
        if j != exclude:
            chosen.add(j)
    return list(chosen)

def _base_inner_task(base_lid):
    rng = random.Random(G_SEED + base_lid * 1000003)
    traj_i = G_BASE_TRAJS[base_lid]
    js = _sample_base_local(rng, G_BASE_N, G_SAMPLE, exclude=base_lid)

    out = []
    for j in js:
        d = G_DIST_FN(traj_i, G_BASE_TRAJS[j])
        out.append((j, d))  # j is base-local id
    return base_lid, out


# =========================================================
# Query-base multiprocessing (query-local vs base-local)
# =========================================================
G_Q_TRAJS = None
G_BASE_TRAJS_QB = None
G_DIST_FN_QB = None

def _init_query_base_worker(q_trajs, base_trajs, metric):
    global G_Q_TRAJS, G_BASE_TRAJS_QB, G_DIST_FN_QB
    G_Q_TRAJS = q_trajs
    G_BASE_TRAJS_QB = base_trajs
    G_DIST_FN_QB = make_distance_fn(metric)

def _query_one_to_base_task(q_idx):
    q = G_Q_TRAJS[q_idx]
    dists = np.empty((len(G_BASE_TRAJS_QB),), dtype=np.float32)
    for k, b in enumerate(G_BASE_TRAJS_QB):
        dists[k] = G_DIST_FN_QB(q, b)
    return q_idx, dists

# =========================================================
# Core runner for one metric
# =========================================================
def run_one_metric(
    dataset: str,
    metric: str,
    out_dir: str,
    base_size: int,
    base_inner_sample: int,
    query_size: int,
    seed: int,
    processes: int,
    chunk_size: int,
):
    metric = metric.lower()
    print(f"\n===== RUN metric={metric} dataset={dataset} =====")

    # Ensure preprocess outputs exist
    coor_path, data_name = trajectory_feature_generation(
        path=paths[dataset],
        lat_range=LatLon_range[dataset][0],
        lon_range=LatLon_range[dataset][1],
        min_length=min_length[dataset],
    )
    # Paths for subset pickles (reused across metrics)
    base_coord_path = os.path.join(out_dir, f"{dataset}_base_traj_coord.pkl")
    base_grid_path  = os.path.join(out_dir, f"{dataset}_base_traj_grid.pkl")
    query_coord_path = os.path.join(out_dir, f"{dataset}_query_traj_coord.pkl")
    query_grid_path  = os.path.join(out_dir, f"{dataset}_query_traj_grid.pkl")

    # If subset exists, load it (do not resample)
    subset_exists = (
        os.path.exists(base_coord_path)
        and os.path.exists(base_grid_path)
        and os.path.exists(query_coord_path)
        and os.path.exists(query_grid_path)
    )

    if subset_exists:
        print("[INFO] base/query subset exists; loading subset pickles (no resampling)")
        base_coord, _ = load_traj_coord(base_coord_path)
        base_grid, _ = load_traj_grid(base_grid_path)
        query_coord, _ = load_traj_coord(query_coord_path)
        query_grid, _ = load_traj_grid(query_grid_path)
        base_size = len(base_coord)
        query_size = len(query_coord)
        print(f"[INFO] loaded base={base_size} query={query_size}")
    else:
        grid_path = infer_traj_grid_path_from_coord(data_name)
        # Load full preprocessed datasets
        trajs_coord_all, max_len_coord = load_traj_coord(coor_path)
        trajs_grid_all, max_len_grid = load_traj_grid(grid_path)
        n_all = len(trajs_coord_all)
        assert n_all == len(trajs_grid_all), "traj_coord and traj_grid counts differ"
        if base_size + query_size > n_all:
            raise ValueError(f"BASE_SIZE+QUERY_SIZE={base_size+query_size} > total={n_all}")

        print("[INFO] subset not found; sampling and saving subset pickles")
        rng = random.Random(seed)
        all_indices = list(range(n_all))
        rng.shuffle(all_indices)
        base_indices = all_indices[:base_size]
        query_indices = all_indices[base_size:base_size + query_size]

        np.save(os.path.join(out_dir, f"{dataset}_base_indices.npy"), np.asarray(base_indices, dtype=np.int64))
        np.save(os.path.join(out_dir, f"{dataset}_query_indices.npy"), np.asarray(query_indices, dtype=np.int64))

        base_coord = [trajs_coord_all[i] for i in base_indices]
        base_grid  = [trajs_grid_all[i] for i in base_indices]
        query_coord = [trajs_coord_all[i] for i in query_indices]
        query_grid  = [trajs_grid_all[i] for i in query_indices]

        save_subset_pickle(base_coord_path, base_coord, max_len_coord)
        save_subset_pickle(base_grid_path, base_grid, max_len_grid)
        save_subset_pickle(query_coord_path, query_coord, max_len_coord)
        save_subset_pickle(query_grid_path, query_grid, max_len_grid)

        print(f"[INFO] subset saved to {out_dir}")

    base_coord = trajs_to_np(base_coord)
    query_coord = trajs_to_np(query_coord)

    # Outputs per metric (won't overwrite across metrics)
    base_inner_out = os.path.join(out_dir, f"{dataset}_base_inner_{metric}.jsonl")
    query_base_memmap_path = os.path.join(out_dir, f"{dataset}_query_base_{metric}.f32mmap")
    meta_path = os.path.join(out_dir, f"{dataset}_query_base_{metric}.meta.json")

    ctx = mp.get_context("fork")

    # ---- Step 5: base-inner sampled similarities ----
    print("[INFO] Step5: computing base-inner sampled similarities...")
    t0 = now()
    processed = 0
    total = base_size

    with ctx.Pool(
        processes=processes,
        initializer=_init_base_inner_worker,
        initargs=(base_coord, seed, base_inner_sample, metric),
    ) as pool, open(base_inner_out, "w", encoding="utf-8") as fout:

        for base_lid, pairs in pool.imap_unordered(_base_inner_task, range(base_size), chunksize=chunk_size):
            rec = {"base_lid": int(base_lid), "metric": metric,
                   "pairs": [[int(j), float(d)] for j, d in pairs]}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            processed += 1
            if processed % 1000 == 0 or processed == total:
                elapsed = now() - t0
                speed = processed / elapsed if elapsed > 0 else 0.0
                eta = (total - processed) / speed if speed > 0 else 0.0
                print(f"[BASE-INNER] {processed}/{total} elapsed={fmt(elapsed)} speed={speed:.2f} traj/s ETA={fmt(eta)}")

    t1 = now()
    print(f"[DONE] Step5 finished in {fmt(t1 - t0)} -> {base_inner_out}")

    # ---- Step 6: query-base full similarities (memmap) ----
    print("[INFO] Step6: computing query-to-base full similarities (memmap)...")
    print(f"[INFO] memmap shape=({query_size}, {base_size}) float32")
    t0 = now()

    dist_mm = np.memmap(query_base_memmap_path, dtype=np.float32, mode="w+", shape=(query_size, base_size))

    q_done = 0
    with ctx.Pool(
        processes=processes,
        initializer=_init_query_base_worker,
        initargs=(query_coord, base_coord, metric),
    ) as pool:
        for q_idx, dists in pool.imap_unordered(_query_one_to_base_task, range(query_size), chunksize=1):
            dist_mm[q_idx, :] = dists
            q_done += 1

            elapsed = now() - t0
            speed = q_done / elapsed if elapsed > 0 else 0.0
            eta = (query_size - q_done) / speed if speed > 0 else 0.0
            print(f"[QUERY-BASE] {q_done}/{query_size} elapsed={fmt(elapsed)} speed={speed:.2f} query/s ETA={fmt(eta)}")

    dist_mm.flush()
    del dist_mm

    t1 = now()
    print(f"[DONE] Step6 finished in {fmt(t1 - t0)} -> {query_base_memmap_path}")

    meta = {
        "dataset": dataset,
        "metric": metric,
        "base_size": base_size,
        "query_size": query_size,
        "dtype": "float32",
        "memmap_path": query_base_memmap_path,
        "base_subset_coord_path": base_coord_path,
        "query_subset_coord_path": query_coord_path,
        "base_subset_grid_path": base_grid_path,
        "query_subset_grid_path": query_grid_path,
        "seed": seed,
        "id_convention": {
            "base": "base_lid in [0, base_size)",
            "query": "query_lid in [0, query_size)",
            "matrix": "dist[q_lid, b_lid]"
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] meta saved: {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Porto", choices=list(paths.keys()))
    ap.add_argument("--metric", default="dtw", choices=["dtw", "frechet", "hausdorff"])
    ap.add_argument("--run_all_metrics", action="store_true", help="Run dtw, frechet, hausdorff sequentially")
    ap.add_argument("--out_dir", default="./features_sampling")

    ap.add_argument("--base_size", type=int, default=1_000_000)
    ap.add_argument("--base_inner_sample", type=int, default=100)
    ap.add_argument("--query_size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=32)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = ["dtw", "hausdorff", "frechet"] if args.run_all_metrics else [args.metric]

    for m in metrics:
        run_one_metric(
            dataset=args.dataset,
            metric=m,
            out_dir=args.out_dir,
            base_size=args.base_size,
            base_inner_sample=args.base_inner_sample,
            query_size=args.query_size,
            seed=args.seed,
            processes=args.processes,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()