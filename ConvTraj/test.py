import torch
import argparse
import pickle

from tools import function
from tools import grid
from tools import pre_rep
from tools import test_methods

from config.new_config import ConfigClass
from mynn.Traj_KNN import NeuTrajTrainer

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for Traj Embedding")

    parser.add_argument("--fmt",                            type=str,   default="Our", help="dataset format")
    parser.add_argument("--dataset",                        type=str,   default="Porto", help="dataset name")
    parser.add_argument("--data_dir",                       type=str,   default="../TrajPQ/data/features_v2", help="dataset name")

    parser.add_argument("--network_type",                   type=str,   default="TJCNN", help="network type")
    parser.add_argument("--loss_type",                      type=str,   default="triplet", help="loss type")
    parser.add_argument("--cnn_feature_distance_type",      type=str,   default="euclidean_sep", help="cnn feature distance type")
    parser.add_argument("--cnntotal_feature_distance_type", type=str,   default="euclidean", help="@Deprecated")
    parser.add_argument("--all_feature_distance_type",      type=str,   default="euclidean", help="all feature distance type")
    parser.add_argument("--sampling_type",                  type=str,   default="distance_sampling1", help="sampling type")
    parser.add_argument("--train_flag",                     type=str,   default="test", help="train flag")
    parser.add_argument("--head_num",                       type=str,   default=1, help="mlp head num")
    parser.add_argument("--area_path",                      type=str,   default="", help="selection area path")
    
    
    parser.add_argument("--target_size",                    type=int,   default=128, help="mlp target size")
    parser.add_argument("--channel",                        type=int,   default=8, help="channel num")
    parser.add_argument("--sampling_num",                   type=int,   default=1, help="sampling num for each sampling type")
    parser.add_argument("--epoch_num",                      type=int,   default=200, help="epoch num")
    parser.add_argument("--device",                         type=str,   default="cuda:0", help="device")
    
    parser.add_argument("--learning_rate",                  type=float, default=0.001, help="learning rate")
    parser.add_argument("--train_ratio",                    type=float, default=1, help="train ratio")
    parser.add_argument("--batch_size",                     type=int,   default=128, help="batch size")
    parser.add_argument("--random_seed",                    type=int,   default=666, help="random seed")
    parser.add_argument("--mode",                           type=str,   default="train-directly", help="mode")
    parser.add_argument("--test_epoch",                     type=int,   default=5, help="test epoch")
    parser.add_argument("--print_epoch",                    type=int,   default=1, help="print epoch")
    parser.add_argument("--save_model",                     type=bool,  default=False, help="save model")
    parser.add_argument("--save_model_epoch",               type=int,   default=5, help="save model epoch")

    parser.add_argument("--root_write_path",                type=str,   default="./data/0_geolife", help="root write path")
    parser.add_argument("--root_read_path",                 type=str,   default="./data/0_geolife", help="root read path")
    parser.add_argument("--dist_type",                      type=str,   default="dtw", help="distance type")
    # parser.add_argument("--dist_type",                      type=str,   default="frechet", help="distance type")
    # parser.add_argument("--dist_type",                      type=str,   default="haus", help="distance type")

    args = parser.parse_args()

    return args

def load_custom(traj_data_path: str, train_dist_path: str, test_top100_path: str):
    """
    Custom v2 format:
      - traj_data_{DATASET}.pkl: dict with 'trajs' (list[np.ndarray(T,2)]) and 'splits'
      - train_dist_{DATASET}_{METRIC}.f32mmap: float32 memmap (TRAIN_N, TRAIN_N)
      - test_top100_{DATASET}_{METRIC}.jsonl: per test_gid top100 base_gid+dist (GT)
    Returns:
      trajs, (train_s,train_e), (test_s,test_e), (base_s,base_e), train_mm, gt_top100_dict
    """
    with open(traj_data_path, "rb") as f:
        data = pickle.load(f)
    trajs = data["trajs"]
    splits = data["splits"]
    train_s, train_e = splits["train"]
    test_s, test_e = splits["test"]
    base_s, base_e = splits["base"]
    # train_n = train_e - train_s

    # train_mm = np.memmap(train_dist_path, dtype=np.float32, mode="r", shape=(train_n, train_n))
    # gt_top100 = {}
    # with open(test_top100_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         rec = json.loads(line)
    #         tg = int(rec["test_gid"])
    #         top = [(int(b), float(d)) for b, d in rec["top100"]]
    #         top.sort(key=lambda x: x[1])
    #         gt_top100[tg] = top
    return trajs, (train_s,train_e), (test_s,test_e), (base_s,base_e)

def _intersect_sizes(true_ids: np.ndarray, pred_ids: np.ndarray) -> np.ndarray:
    """
    true_ids: (Q, M)
    pred_ids: (Q, K)
    return: (Q,) intersection sizes for each query
    """
    # membership via broadcasting
    return (true_ids[:, :, None] == pred_ids[:, None, :]).any(axis=2).sum(axis=1)

def recall_K_M(true_knn: np.ndarray, pred_knn: np.ndarray, K: int, M: int) -> float:
    """
    Mean recall: |GT@M ∩ Pred@K| / M, averaged over queries.
    """
    gt = true_knn[:, :M]
    pr = pred_knn[:, :K]
    inter = _intersect_sizes(gt, pr)
    return float(np.mean(inter / float(M)))

def convtraj_metrics(true_knn: np.ndarray, pred_knn: np.ndarray):
    """
    Matches the spirit of ConvTraj's test_all_print_new:
      HR@1   = recall(1,1)
      HR@10  = recall(10,10)
      HR@50  = recall(50,50)
      HR10@50= recall(50,10)
    Notes:
      - true_knn and pred_knn should be full rankings (Q,B) or at least cover top50/100.
    """
    return {
        "HR@1": recall_K_M(true_knn, pred_knn, 1, 1),
        "HR@10": recall_K_M(true_knn, pred_knn, 10, 10),
        "HR@50": recall_K_M(true_knn, pred_knn, 50, 50),
        "HR10@50": recall_K_M(true_knn, pred_knn, 50, 10),
    }

def evaluate_custom(test_emb, base_emb, test_range, base_range, gt_top100, device, pred_topk=100):
    test_s, test_e = test_range
    base_s, base_e = base_range
    test_ids = list(range(test_s, test_e))
    base_ids = list(range(base_s, base_e))
    test_n = len(test_ids)
    base_n = len(base_ids)

    base_norm = (base_emb * base_emb).sum(axis=1)
    true_knn = np.empty((test_n, 100), dtype=np.int64)
    pred_knn = np.empty((test_n, pred_topk), dtype=np.int64)
    for qi, gid in enumerate(test_ids):
        top = gt_top100.get(gid)
        if top is None or len(top) < 100:
            arr = np.full((100,), -1, dtype=np.int64)
        else:
            arr = np.asarray([b for b,_ in top[:100]], dtype=np.int64) - base_s
        true_knn[qi] = arr

        q = test_emb[qi]
        qn = float((q * q).sum())
        d2 = base_norm + qn - 2.0 * (base_emb @ q)
        k = min(pred_topk, base_n)
        idx = np.argpartition(d2, k-1)[:k]
        idx = idx[np.argsort(d2[idx])]
        pred_knn[qi,:k] = idx
        if k < pred_topk:
            pred_knn[qi,k:] = -1

    return convtraj_metrics(true_knn, pred_knn)

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("GPU is Not Available.")
        exit()

    args   = get_args()
    device = torch.device(args.device)
    print("Device is:", device)

    function.setup_seed(args.random_seed)

    if args.fmt == "Our":
        import os
        traj_data_path = '/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj/data/features_v2/traj_data_Porto.pkl'
        train_dist_matrix_path = '/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj/data/features_v2/train_dist_Porto_dtw.f32mmap'
        test_dist_matrix_path = '/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj/data/features_v2/test_top100_Porto_dtw.jsonl'
        traj_list, train_range, query_range, base_range = load_custom(
            traj_data_path, train_dist_matrix_path, test_dist_matrix_path
        )
        train_set = train_range[1] - train_range[0]
        query_set = query_range[1] - query_range[0]
        base_set = base_range[1] - base_range[0]


        import json
        gt_top100 = {}
        with open(test_dist_matrix_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tg = int(rec["test_gid"])
                top = [(int(b), float(d)) for b, d in rec["top100"]]
                top.sort(key=lambda x: x[1])
                gt_top100[tg] = top
        
        test_flag = 'LSTMfeature'

        query_embeddings = pickle.load(open('/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj/data/feature_dir/p_query_dtw_feature_test_200', "rb"))
        base_embeddings = pickle.load(open('/data3/menghaotian/Traj_sim/ConvTraj_DeepVQ/ConvTraj/data/feature_dir/p_base_dtw_feature_test_200', "rb"))
        # test_distance_matrix = test_methods.get_feature_distance(query_embeddings, base_embeddings, 'cnnfeature', my_config)
        # results = test_methods.topk_recall(list(range(len(query_embeddings))), list(range(len(base_embeddings))), distance_matrix, test_distance_matrix)
        # results = test_methods.test_all_print_new(None, None, test_knn, test_distance_matrix)
        print(query_embeddings.shape)
        results = evaluate_custom(query_embeddings, base_embeddings, query_range, base_range, gt_top100, None, pred_topk=100)
        print(results)

