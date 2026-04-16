import os
import pickle
import numpy as np
import tqdm
from tools import feature_distance
import json


def _artifact_prefix(my_config):
    return str(my_config.my_dict.get("artifact_prefix", "") or "").strip()


def _use_standard_artifacts(my_config):
    return bool(_artifact_prefix(my_config))


def _report_only_eval(my_config):
    return bool(my_config.my_dict.get("report_only_eval", False))


def _ensure_subdir(root_write_path, dirname):
    path = os.path.join(root_write_path, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_embedding_flag(test_flag):
    return str(test_flag).replace("feature_", "", 1)


def _embedding_path(my_config, test_flag, split_name, epoch):
    root_write_path = my_config.my_dict["root_write_path"]
    if not _use_standard_artifacts(my_config):
        train_flag = my_config.my_dict["train_flag"]
        dist_type = my_config.my_dict["dist_type"]
        if epoch == -1:
            filename = "{}_{}_{}_{}".format(split_name, dist_type, test_flag, train_flag)
        else:
            filename = "{}_{}_{}_{}_{}".format(split_name, dist_type, test_flag, train_flag, epoch)
        return os.path.join(_ensure_subdir(root_write_path, "feature_dir"), filename)

    artifact_prefix = _artifact_prefix(my_config)
    embedding_dir = _ensure_subdir(root_write_path, "embeddings")
    suffix = "" if epoch == -1 else "_e{}".format(int(epoch))
    filename = "{}_{}_{}{}.pkl".format(artifact_prefix, _normalize_embedding_flag(test_flag), split_name, suffix)
    return os.path.join(embedding_dir, filename)


def _feature_distance_path(my_config, split_name, test_flag, epoch):
    root_write_path = my_config.my_dict["root_write_path"]
    if not _use_standard_artifacts(my_config):
        train_flag = my_config.my_dict["train_flag"]
        dist_type = my_config.my_dict["dist_type"]
        if epoch == -1:
            filename = "{}_{}_{}_{}".format(split_name, dist_type, test_flag, train_flag)
        else:
            filename = "{}_{}_{}_{}_{}".format(split_name, dist_type, test_flag, train_flag, epoch)
        return os.path.join(_ensure_subdir(root_write_path, "feature_distance_dir"), filename)

    artifact_prefix = _artifact_prefix(my_config)
    feature_distance_dir = _ensure_subdir(root_write_path, "feature_distance")
    suffix = "" if epoch == -1 else "_e{}".format(int(epoch))
    filename = "{}_{}_{}{}.pkl".format(artifact_prefix, _normalize_embedding_flag(test_flag), split_name, suffix)
    return os.path.join(feature_distance_dir, filename)


def load_dtw_matrix_auto():
        """
        自动读取 meta.json 并加载 memmap 矩阵
        """
        # 1. 构建文件路径
        memmap_path = '/data3/menghaotian/Traj_sim/datasets/features_sampling/Porto_query_base_dtw.f32mmap'
        meta_path = '/data3/menghaotian/Traj_sim/datasets/features_sampling/Porto_query_base_dtw.meta.json'
        
        # 2. 读取元数据以获取 shape
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        query_size = meta["query_size"]
        base_size = meta["base_size"]
        dtype_str = meta.get("dtype", "float32") # 默认 float32
        
        print(f"[Info] Loading Matrix: Shape=({query_size}, {base_size}), Dtype={dtype_str}")
        
        # 3. 使用 np.memmap 读取
        # mode='r' 表示只读，不会把整个文件加载到内存，适合大文件
        matrix = np.memmap(memmap_path, dtype=dtype_str, mode='r', shape=(query_size, base_size))
        
        return matrix
def test_all_log(total_embeddings, my_config, test_flag, train_distance_matrix, query_dist, epoch = -1):
    
    root_write_path  = my_config.my_dict["root_write_path"]
    train_flag       = my_config.my_dict["train_flag"]
    save_feature_distance = bool(my_config.my_dict.get("save_feature_distance", False))
    save_embeddings = not _report_only_eval(my_config)
    train_embeddings = total_embeddings[:my_config.my_dict["train_set"]]
    query_embeddings = total_embeddings[my_config.my_dict["train_set"]:my_config.my_dict["train_set"] + my_config.my_dict["query_set"]]
    print("query_embeddings shape:", query_embeddings.shape)
    base_embeddings  = total_embeddings[my_config.my_dict["train_set"] + my_config.my_dict["query_set"]:]
    print('base_size:',base_embeddings.shape)
    test_distance_matrix = get_feature_distance(query_embeddings, base_embeddings)
    print("test_distance_matrix shape:", test_distance_matrix.shape)

    if save_embeddings and not _use_standard_artifacts(my_config):
        _ensure_subdir(root_write_path, "feature_dir")

    if save_embeddings and save_feature_distance:
        pickle.dump(np.array(train_distance_matrix), open(_feature_distance_path(my_config, "train_dist", test_flag, epoch), "wb"))
        pickle.dump(np.array(test_distance_matrix), open(_feature_distance_path(my_config, "query_base_dist", test_flag, epoch), "wb"))
    else:
        print("[EvalArtifacts] skip feature_distance_dir dump for {}".format(test_flag))

    if save_embeddings:
        pickle.dump(np.array(train_embeddings), open(_embedding_path(my_config, test_flag, "train", epoch), "wb"))
        pickle.dump(np.array(query_embeddings), open(_embedding_path(my_config, test_flag, "query", epoch), "wb"))
        pickle.dump(np.array(base_embeddings), open(_embedding_path(my_config, test_flag, "base", epoch), "wb"), protocol=4)
        pickle.dump(np.array(total_embeddings), open(_embedding_path(my_config, test_flag, "all", epoch), "wb"), protocol=4)
    else:
        print("[EvalArtifacts] skip embedding dump for {}".format(test_flag))
    #truth = pickle.load(open('/data3/menghaotian/Traj_sim/ConvTraj/data/0_geolife/dtw_test_distance_matrix_result', "rb"))
    truth = query_dist
    query_knn = get_knn(query_dist)
    print("query_knn.shape:", query_knn.shape)
    idx = np.arange(len(total_embeddings))
    query_ids = idx[my_config.my_dict["train_set"]:my_config.my_dict["train_set"] + my_config.my_dict["query_set"]]
    print("query_ids.shape:", query_ids.shape)
    base_ids  = idx[my_config.my_dict["train_set"] + my_config.my_dict["query_set"]:]
    print("base_ids.shape:", base_ids.shape)
    return test_all_print_new(query_ids, base_ids, query_knn, test_distance_matrix)
def topk_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    result = []
    s_test_distance_dict = {}
    s_true_distance_dict = {}
    for query_id in query_id_list:
        test_distance = []
        true_distance = []
        for base_id in base_id_list:
            test_distance.append((base_id, test_distance_matrix[query_id][base_id]))
            true_distance.append((base_id, distance_matrix[query_id][base_id]))
        s_test_distance    = sorted(test_distance, key = lambda a: a[1], reverse = False)
        s_true_distance    = sorted(true_distance, key = lambda a: a[1], reverse = False)
        s_test_distance_dict[query_id] = s_test_distance
        s_true_distance_dict[query_id] = s_true_distance

    top_test_num = [1]
    top_true_num = [1]
    
    for tem_test_num in top_test_num:
        top_count_list = [0 for i in range(len(top_true_num))]
        for query_id in query_id_list:
            s_test_distance = s_test_distance_dict[query_id]
            s_true_distance = s_true_distance_dict[query_id]
            for pos in range(len(top_true_num)):
                tem_true_num = top_true_num[pos]
                tem_top_list = [l[0] for l in s_test_distance[:tem_test_num] if l[0] in [j[0] for j in s_true_distance[:tem_true_num]]]
                if tem_test_num == 1 and tem_true_num == 1 and s_test_distance[0][0] != s_true_distance[0][0]:
                    top11_true_list = [s_true_distance[0][0]]
                    for id, value in s_true_distance:
                        if value == s_true_distance[0][1]:
                            top11_true_list.append(id)
                    if s_test_distance[0][0] in top11_true_list:
                        tem_top_list.append(s_test_distance[0][0])
                top_count_list[pos] += len(tem_top_list)
        for pos in range(len(top_true_num)):
            tem_recall = top_count_list[pos] / (len(query_id_list) * top_true_num[pos])
            result.append(tem_recall)
    return result

def test_all_print(query_id_list, base_id_list, distance_matrix, test_distance_matrix):
    recall_list  = topk_recall(query_id_list, base_id_list, distance_matrix, test_distance_matrix)


    return recall_list

def intersect_sizes(true_list, test_list):
    return np.array([len(np.intersect1d(true_value, list(test_value))) for true_value, test_value in zip(true_list, test_list)])

def get_knn(dist):
    knn = np.empty(dtype=np.int32, shape=(len(dist), len(dist[0])))
    for i in tqdm.tqdm(range(len(dist)), desc="# sorting for KNN indices"):
        knn[i, :] = np.argsort(dist[i, :])
    return knn
def test_all_print_new(query_id_list, base_id_list, true_knn, test_distance_matrix):
    
    test_knn = np.argsort(test_distance_matrix)
    #print("test_knn.shape:", test_knn.shape)

    top_test_dict = {5:[5], 10:[10], 50:[50, 10], 100:[100]}

    for tem_test_num in top_test_dict.keys():
        top_true_list  = top_test_dict[tem_test_num]
        test_top_id    = test_knn[:, :tem_test_num]
        intersect_list = [intersect_sizes(true_knn[:, :tem_true_num], test_top_id) / float(tem_true_num) for tem_true_num in top_true_list]
        recall_list    = [np.mean(tem_list) for tem_list in intersect_list]
        for pos in range(len(top_true_list)):
            if tem_test_num == 5 and top_true_list[pos] == 5:
                top5_recall = recall_list[pos]
            if tem_test_num == 10 and top_true_list[pos] == 10:
                top10_recall = recall_list[pos]
            if tem_test_num == 50 and top_true_list[pos] == 50:
                top50_recall = recall_list[pos]
            if tem_test_num == 100 and top_true_list[pos] == 100:
                top100_recall = recall_list[pos]
            if tem_test_num == 50 and top_true_list[pos] == 10:
                top10_50_recall = recall_list[pos]
            
    total_list = []
    recall_list = [top5_recall, top10_recall, top50_recall, top100_recall, top10_50_recall]
    print("Top-5 Recall: {:.4f}, Top-10 Recall: {:.4f}, Top-50 Recall: {:.4f}, Top-100 Recall: {:.4f}, Top-10@50 Recall: {:.4f}".format(top5_recall, top10_recall, top50_recall, top100_recall, top10_50_recall))
    total_list.extend(recall_list)
    return {
        "top5_recall": top5_recall,
        "top10_recall": top10_recall,
        "top50_recall": top50_recall,
        "top100_recall": top100_recall,
        "top10_at_50_recall": top10_50_recall,
        "metrics_list": total_list,
    }


def _ensure_true_knn(true_knn):
    true_knn = np.asarray(true_knn)
    if np.issubdtype(true_knn.dtype, np.integer):
        return true_knn
    return np.argsort(true_knn, axis=1)


def metrics_from_pred_knn(true_knn, pred_knn):
    true_knn = _ensure_true_knn(true_knn)
    pred_knn = np.asarray(pred_knn, dtype=np.int32)
    top_test_dict = {5:[5], 10:[10], 50:[50, 10], 100:[100]}

    for tem_test_num in top_test_dict.keys():
        top_true_list = top_test_dict[tem_test_num]
        test_top_id = pred_knn[:, :tem_test_num]
        intersect_list = [intersect_sizes(true_knn[:, :tem_true_num], test_top_id) / float(tem_true_num) for tem_true_num in top_true_list]
        recall_list = [np.mean(tem_list) for tem_list in intersect_list]
        for pos in range(len(top_true_list)):
            if tem_test_num == 5 and top_true_list[pos] == 5:
                top5_recall = recall_list[pos]
            if tem_test_num == 10 and top_true_list[pos] == 10:
                top10_recall = recall_list[pos]
            if tem_test_num == 50 and top_true_list[pos] == 50:
                top50_recall = recall_list[pos]
            if tem_test_num == 100 and top_true_list[pos] == 100:
                top100_recall = recall_list[pos]
            if tem_test_num == 50 and top_true_list[pos] == 10:
                top10_50_recall = recall_list[pos]

    total_list = [top5_recall, top10_recall, top50_recall, top100_recall, top10_50_recall]
    print("Top-5 Recall: {:.4f}, Top-10 Recall: {:.4f}, Top-50 Recall: {:.4f}, Top-100 Recall: {:.4f}, Top-10@50 Recall: {:.4f}".format(top5_recall, top10_recall, top50_recall, top100_recall, top10_50_recall))
    return {
        "top5_recall": top5_recall,
        "top10_recall": top10_recall,
        "top50_recall": top50_recall,
        "top100_recall": top100_recall,
        "top10_at_50_recall": top10_50_recall,
        "metrics_list": total_list,
    }


def get_feature_distance(query_embeddings, base_embeddings):
    test_distance_matrix = feature_distance.l2_dist(query_embeddings, base_embeddings)
    '''
    cnn_feature_distance_type      = my_config.my_dict["cnn_feature_distance_type"]
    cnntotal_feature_distance_type = my_config.my_dict["cnntotal_feature_distance_type"]
    all_feature_distance_type      = my_config.my_dict["all_feature_distance_type"]
    if test_flag == "cnnfeature" and "euclidean" in cnn_feature_distance_type:
        test_distance_matrix = feature_distance.l2_dist_separate(query_embeddings, base_embeddings)
    elif test_flag == "cnnfeature" and "manhattan" in cnn_feature_distance_type:
        test_distance_matrix = feature_distance.manhattan_dist_separate(query_embeddings, base_embeddings)
    elif test_flag == "cnnfeature" and "hyperbolic" in cnn_feature_distance_type:
        test_distance_matrix = feature_distance.hyperbolic_dist_separate(query_embeddings, base_embeddings)
    elif test_flag == "cnntotalfeature" and cnntotal_feature_distance_type == "euclidean":
        test_distance_matrix = feature_distance.l2_dist(query_embeddings, base_embeddings)
    elif test_flag == "cnntotalfeature" and cnntotal_feature_distance_type == "hyperbolic":
        test_distance_matrix = feature_distance.hyperbolic_dist(query_embeddings, base_embeddings)
    elif test_flag == "allfeature" and all_feature_distance_type == "euclidean":
        test_distance_matrix = feature_distance.l2_dist(query_embeddings, base_embeddings)
    elif test_flag == "allfeature" and all_feature_distance_type == "manhattan":
        test_distance_matrix = feature_distance.manhattan_dist(query_embeddings, base_embeddings)
    elif test_flag == "allfeature" and all_feature_distance_type == "hyperbolic":
        test_distance_matrix = feature_distance.hyperbolic_dist(query_embeddings, base_embeddings)
    else:
        raise ValueError('Unsupported Test Flag: {}'.format(test_flag))
    '''
    return test_distance_matrix
    
