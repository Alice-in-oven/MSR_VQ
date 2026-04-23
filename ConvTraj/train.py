import torch
import argparse
import pickle
from pathlib import Path

from tools import function
from tools import grid
from tools import pre_rep

from config.new_config import ConfigClass
from mynn.Traj_KNN import NeuTrajTrainer

import pickle
import numpy as np
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVTRAJ_ROOT = REPO_ROOT / "ConvTraj"
DATA_ROOT = CONVTRAJ_ROOT / "data"


def resolve_default_dataset_root(dataset_name: str, requested_root_path: str) -> str:
    requested = str(requested_root_path or "").strip()
    if not requested:
        requested = str(DATA_ROOT)
    requested_path = Path(requested).expanduser()
    dataset_key = str(dataset_name or "").strip().lower()
    if not dataset_key:
        return str(requested_path)
    default_data_root = DATA_ROOT.resolve()
    try:
        requested_resolved = requested_path.resolve()
    except FileNotFoundError:
        requested_resolved = requested_path
    if requested_resolved != default_data_root:
        return str(requested_path)
    candidate_root = DATA_ROOT / "0_{}".format(dataset_key)
    if candidate_root.exists():
        return str(candidate_root)
    return str(requested_path)


def load_custom(traj_data_path: str, train_dist_path: str, test_top100_path: str):
    """
    Custom v2 format:
      - traj_data_{DATASET}.pkl: dict with 'trajs' (list[np.ndarray(T,2)]) and 'splits'
      - train_dist_{DATASET}_{METRIC}.f32mmap: float32 memmap (TRAIN_N, TRAIN_N)
      - test_top100_{DATASET}_{METRIC}.jsonl: per test_gid top100 base_gid+dist (GT)
    Returns:
      trajs, (train_s,train_e), (test_s,test_e), (base_s,base_e), train_mm, gt_top100_dict
    """
   
    data = np.load(traj_data_path, allow_pickle=True)
    trajs = data["trajs"]
    splits = data["splits"]
    train_s, train_e = splits["train"]
    test_s, test_e = splits["test"]
    base_s, base_e = splits["base"]
    train_n = train_e - train_s

    train_mm = np.memmap(train_dist_path, dtype=np.float32, mode="r", shape=(train_n, train_n))
    gt_top100 = np.load(DATA_ROOT / 'features_v2' / 'knn_indices.npy')
    return trajs, (train_s,train_e), (test_s,test_e), (base_s,base_e), train_mm, gt_top100
def pad_traj_list(dist_type, seq_list, max_length, pad_value=0.0):
    value = [1.0 * pad_value for i in range(len(seq_list[0][0]))]
    final_pad_seq_list = []
    for seq in seq_list:
        assert len(seq) <= max_length, "Sequence length {} is larger than max_length {}".format(len(seq), max_length)

        if dist_type == "dtw":
            value = [1.0 * val for val in seq[len(seq) - 1]]
        for j in range(max_length - len(seq)):
            seq.append(value)
        final_pad_seq_list.append(seq)
    return final_pad_seq_list
def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for Traj Embedding")

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
    parser.add_argument("--device",                         type=str,   default="cuda:3", help="device")
    
    parser.add_argument("--learning_rate",                  type=float, default=0.001, help="learning rate")
    parser.add_argument("--train_ratio",                    type=float, default=1, help="train ratio")
    parser.add_argument("--batch_size",                     type=int,   default=256, help="batch size")
    parser.add_argument("--random_seed",                    type=int,   default=666, help="random seed")
    parser.add_argument("--mode",                           type=str,   default="train-directly", help="mode")
    parser.add_argument("--test_epoch",                     type=int,   default=1, help="test epoch")
    parser.add_argument("--print_epoch",                    type=int,   default=1, help="print epoch")
    parser.add_argument("--save_model",                     type=bool,  default=True, help="save model")
    parser.add_argument("--save_model_epoch",               type=int,   default=5, help="save model epoch")
    parser.add_argument("--eval_save_epochs",               type=str,   default="", help="comma-separated epochs to save and evaluate, e.g. 500,600,700")
    parser.add_argument("--save_feature_distance",          action="store_true", help="persist large train/test feature-distance matrices during evaluation")
    parser.add_argument("--report_only_eval",               action="store_true", help="run evaluation and print metrics without saving checkpoints, metric json, or feature embeddings")
    parser.add_argument("--dataset",                        type=str,   default="geolife", help="dataset name")
    parser.add_argument("--root_write_path",                type=str,   default=str(DATA_ROOT), help="root write path")
    parser.add_argument("--root_read_path",                 type=str,   default=str(DATA_ROOT), help="root read path")
    parser.add_argument("--artifact_prefix",                type=str,   default="", help="standard artifact prefix for checkpoints, metrics, embeddings, and reports")
    parser.add_argument("--dist_type",                      type=str,   default="dtw", help="distance type")
    parser.add_argument("--grid_size",                      type=float, default=0.0, help="optional override for equal-grid meshing size; <=0 keeps dataset default")
    parser.add_argument("--image_mode",                     type=str,   default="binary", choices=["binary", "motion6", "dtw8", "motion6_pyr2", "multigrid3", "shape5", "shape5_pyr2", "haus6", "dfd7"], help="2D trajectory image mode")
    parser.add_argument("--disable_specific_image_disk_cache", action="store_true", help="disable on-disk caching for specific image modes such as dtw8/dfd7/haus6")
    parser.add_argument("--embedding_backbone",             type=str,   default="msr", choices=["msr", "neutraj", "simformer"], help="continuous embedding backbone before PDT-VQ")
    parser.add_argument("--backbone_seq_max_length",        type=int,   default=200, help="maximum per-trajectory sequence length fed into non-MSR backbones via uniform downsampling")
    parser.add_argument("--simformer_num_layers",           type=int,   default=1, help="number of SIMformer encoder layers")
    parser.add_argument("--simformer_n_heads",              type=int,   default=16, help="number of SIMformer attention heads")
    parser.add_argument("--simformer_dimfeedforward",       type=int,   default=256, help="SIMformer feedforward width")
    parser.add_argument("--simformer_pos_encoding",         type=str,   default="fixed", choices=["fixed", "learnable"], help="SIMformer positional encoding type")
    parser.add_argument("--neutraj_spatial_width",          type=int,   default=2, help="spatial neighborhood width for the NeuTraj SAM-GRU memory")
    parser.add_argument("--neutraj_incell",                 type=lambda s: str(s).lower() in ["1", "true", "yes", "y"], default=True, help="enable in-cell memory updates for the NeuTraj backbone")
    parser.add_argument("--neutraj_use_standard_gru",       action="store_true", help="replace the NeuTraj SAM-GRU cell with a plain GRU cell")
    parser.add_argument("--pdt_m",                          type=int,   default=16, help="PDT-VQ codebook count M")
    parser.add_argument("--pdt_k",                          type=int,   default=2**8, help="PDT-VQ codebook size K")
    parser.add_argument("--pdt_steps",                      type=int,   default=1, help="PDT transform step count")
    parser.add_argument("--pdt_heads",                      type=int,   default=1, help="PDT transform head count")
    parser.add_argument("--pdt_vq_type",                   type=str,   default="dpq", choices=["dpq", "drq", "qinco"], help="PDT-VQ quantizer type")
    parser.add_argument("--pdt_codebook_init",             type=str,   default="uniform", choices=["uniform", "faiss"], help="PDT-VQ codebook initialization strategy")
    parser.add_argument("--qinco_h",                       type=int,   default=256, help="hidden width for QINCo residual blocks")
    parser.add_argument("--qinco_L",                       type=int,   default=1, help="residual block depth for each QINCo step")
    parser.add_argument("--qinco_identity_init",           action="store_true", help="initialize QINCo refinement blocks close to identity so codebook init is preserved")
    parser.add_argument("--disable_pdt_init_codebook",      action="store_true", help="disable PDT-VQ codebook initialization from trajectory embeddings")
    parser.add_argument("--eval_embedding_type",            type=str,   default="quantized", choices=["quantized", "continuous", "both"], help="which embedding type to evaluate/export")
    parser.add_argument("--eval_search_mode",              type=str,   default="decoded", choices=["decoded", "adc", "both"], help="search mode for quantized retrieval evaluation; adc means true asymmetric distance computation with query transformed embeddings vs base codes+codebook")
    parser.add_argument("--enable_rerank",                 action="store_true", help="enable reranking on top of coarse retrieval")
    parser.add_argument("--rerank_L",                      type=int,   default=100, help="shortlist size for reranking")
    parser.add_argument("--rerank_source",                 type=str,   default="decoded", choices=["decoded", "adc"], help="coarse retrieval source for reranking; adc uses the true ADC shortlist when eval_search_mode includes adc")
    parser.add_argument("--disable_code_usage_stats",       action="store_true", help="disable quantized code usage statistics")
    parser.add_argument("--load_model_train_flag",         type=str,   default=None, help="load checkpoint from another train_flag when running evaluation-only")
    parser.add_argument("--backbone_checkpoint",            type=str,   default=None, help="optional pretrained MSR backbone checkpoint")
    parser.add_argument("--freeze_backbone_epochs",         type=int,   default=0, help="freeze backbone for the first N epochs and train PDT only")
    parser.add_argument("--pdt_loss_start_epoch",           type=int,   default=20, help="epoch to start PDT loss when not using a frozen warmup")
    parser.add_argument("--pdt_loss_weight",                type=float, default=0.3, help="weight for the PDT loss term")
    parser.add_argument("--loss_recipe",                   type=str,   default="baseline", choices=["baseline", "consistency", "quantized_metric", "consistency_quantized_metric", "improved_vq"], help="incremental training objective recipe")
    parser.add_argument("--consistency_weight",            type=float, default=0.1, help="weight for continuous-quantized consistency loss")
    parser.add_argument("--quantized_metric_weight",       type=float, default=0.3, help="weight for quantized metric loss")
    parser.add_argument("--improved_qm_start_epoch",       type=int,   default=60, help="epoch to start the improved quantized metric warmup")
    parser.add_argument("--improved_qm_warmup_epochs",     type=int,   default=80, help="warmup length for the improved quantized metric weight")
    parser.add_argument("--improved_qm_max_weight",        type=float, default=0.12, help="maximum weight for the improved quantized metric term")
    parser.add_argument("--improved_pairwise_weight",      type=float, default=0.05, help="weight for continuous-quantized pairwise distance distillation")
    parser.add_argument("--improved_entropy_weight",       type=float, default=0.01, help="weight for code usage entropy regularization")
    parser.add_argument("--improved_commit_weight",        type=float, default=0.02, help="weight for soft quantization commitment/codebook alignment")
    parser.add_argument("--improved_uniform_weight",       type=float, default=0.001, help="weight for codebook spread regularization")
    parser.add_argument("--disable_improved_vq_adaptive_low_codebook", action="store_true", help="disable the automatic low-codebook improved_vq schedule so explicit training settings are respected")
    parser.add_argument("--pre_quant_bottleneck_enabled",  action="store_true", help="enable the pre-quant trajectory bottleneck before PDT-VQ")
    parser.add_argument("--pre_quant_global_dim",          type=int,   default=48, help="global subspace dimension")
    parser.add_argument("--pre_quant_local_dim",           type=int,   default=48, help="local subspace dimension")
    parser.add_argument("--pre_quant_progress_dim",        type=int,   default=32, help="progress subspace dimension")
    parser.add_argument("--pre_quant_use_motion_stats",    action="store_true", help="use lightweight motion6 summaries for the progress subspace")
    parser.add_argument("--pre_quant_lambda_decor",        type=float, default=0.01, help="decorrelation regularizer weight for the bottleneck")
    parser.add_argument("--pre_quant_lambda_stab",         type=float, default=0.1, help="stabilization regularizer weight for the bottleneck")
    parser.add_argument("--pre_quant_residual_alpha_init", type=float, default=0.15, help="initial residual mixing strength for the bottleneck")
    parser.add_argument("--disable_pre_quant_learnable_alpha", action="store_true", help="disable learnable pre-quant residual alpha and fall back to direct residual addition")
    parser.add_argument("--pre_quant_lr_multiplier",       type=float, default=0.25, help="learning-rate multiplier for bottleneck parameters")
    parser.add_argument("--pre_quant_stab_late_epoch",     type=int,   default=100, help="epoch to strengthen bottleneck stabilization")
    parser.add_argument("--pre_quant_stab_late_multiplier",type=float, default=4.0, help="multiplier for bottleneck stabilization after the late epoch")
    parser.add_argument("--pre_quant_refresh_start_epoch", type=int,   default=100, help="start epoch for extra PDT codebook refresh in bottleneck mode")
    parser.add_argument("--pre_quant_refresh_period",      type=int,   default=50, help="period for extra PDT codebook refresh in bottleneck mode")
    parser.add_argument("--pre_quant_refresh_end_epoch",   type=int,   default=0, help="exclusive upper bound for extra PDT refresh in bottleneck mode; <=0 keeps the old schedule")
    parser.add_argument("--pre_quant_raw_metric_weight",   type=float, default=0.2, help="auxiliary metric-loss weight on raw e_cont when bottleneck mode is enabled")
    parser.add_argument("--pre_quant_neighbor_enabled",    action="store_true", help="enable retrieval-neighborhood consistency losses in bottleneck mode")
    parser.add_argument("--pre_quant_neighbor_use_btn",    action="store_true", help="apply neighborhood consistency on e_bottleneck")
    parser.add_argument("--pre_quant_neighbor_use_dec",    action="store_true", help="apply neighborhood consistency on decoded e_hat")
    parser.add_argument("--pre_quant_neighbor_topk",       type=int,   default=10, help="teacher neighborhood size for retrieval consistency")
    parser.add_argument("--pre_quant_neighbor_tau_btn",    type=float, default=0.07, help="temperature for bottleneck neighborhood logits")
    parser.add_argument("--pre_quant_neighbor_tau_dec",    type=float, default=0.07, help="temperature for decoded neighborhood logits")
    parser.add_argument("--pre_quant_neighbor_lambda_btn", type=float, default=0.05, help="weight for bottleneck neighborhood consistency")
    parser.add_argument("--pre_quant_neighbor_lambda_dec", type=float, default=0.10, help="weight for decoded neighborhood consistency")
    parser.add_argument("--pre_quant_neighbor_start_epoch", type=int, default=0, help="epoch to start neighborhood consistency losses")
    parser.add_argument("--pre_quant_neighbor_warmup_epochs", type=int, default=0, help="warmup epochs for neighborhood-consistency weights")
    parser.add_argument("--pre_quant_neighbor_dec_stop_backbone", action="store_true", help="apply decoded-neighborhood loss through PDT only by detaching the bottleneck input")
    parser.add_argument("--pre_quant_neighbor_teacher_mode", type=str, default="batch_cont", choices=["batch_cont", "offline_gt"], help="teacher source for neighborhood consistency")
    parser.add_argument("--pre_quant_neighbor_offline_path", type=str, default="", help="path to a saved offline neighborhood artifact")
    parser.add_argument("--pre_quant_landmark_enabled", action="store_true", help="enable GT landmark distance-profile supervision in bottleneck mode")
    parser.add_argument("--pre_quant_landmark_num", type=int, default=64, help="number of GT landmark trajectories")
    parser.add_argument("--pre_quant_landmark_select", type=str, default="fps", choices=["fps"], help="landmark selection strategy")
    parser.add_argument("--pre_quant_landmark_profile_transform", type=str, default="log1p_zscore", choices=["log1p_zscore", "zscore", "none"], help="row-wise transform for GT/student landmark profiles")
    parser.add_argument("--pre_quant_landmark_use_btn", action="store_true", help="apply GT landmark profile loss on e_bottleneck")
    parser.add_argument("--pre_quant_landmark_use_dec", action="store_true", help="apply GT landmark profile loss on decoded e_hat")
    parser.add_argument("--pre_quant_landmark_dec_ste_to_btn", action="store_true", help="for decoded landmark loss, keep decoded forward values but route gradients to the pre-quant embedding through a straight-through estimator")
    parser.add_argument("--pre_quant_landmark_dec_bank_source", type=str, default="decoded", choices=["decoded", "bottleneck"], help="landmark anchor space used by the decoded landmark loss")
    parser.add_argument("--pre_quant_landmark_rank_weight", type=float, default=0.0, help="extra listwise ranking weight inside the landmark profile loss")
    parser.add_argument("--pre_quant_landmark_rank_tau", type=float, default=0.5, help="temperature for landmark ranking distributions")
    parser.add_argument("--pre_quant_landmark_lambda_btn", type=float, default=0.03, help="weight for bottleneck landmark profile loss")
    parser.add_argument("--pre_quant_landmark_lambda_dec", type=float, default=0.10, help="weight for decoded landmark profile loss")
    parser.add_argument("--pre_quant_landmark_start_epoch", type=int, default=100, help="epoch to start GT landmark profile supervision")
    parser.add_argument("--pre_quant_landmark_warmup_epochs", type=int, default=20, help="warmup epochs for GT landmark profile supervision")
    parser.add_argument("--pre_quant_landmark_bank_refresh_epochs", type=int, default=10, help="refresh interval for landmark embedding banks")
    parser.add_argument("--pre_quant_landmark_artifact_path", type=str, default="", help="path to a saved GT landmark profile artifact")
    parser.add_argument("--pre_quant_landmark_fixed_bank_path", type=str, default="", help="path to a saved fixed landmark embedding bank artifact")
    parser.add_argument("--pre_quant_landmark_teacher_checkpoint", type=str, default="", help="checkpoint used to build a fixed landmark embedding bank")
    parser.add_argument("--decoded_ste_metric_enabled", action="store_true", help="enable straight-through decoded metric alignment on the hard decoded retrieval space")
    parser.add_argument("--decoded_ste_metric_start_epoch", type=int, default=80, help="epoch to start the decoded STE metric loss")
    parser.add_argument("--decoded_ste_metric_warmup_epochs", type=int, default=20, help="warmup epochs for the decoded STE metric loss weight")
    parser.add_argument("--decoded_ste_metric_max_weight", type=float, default=0.03, help="maximum weight for the decoded STE metric loss")
    parser.add_argument("--porto_opq_warmup_train_recon_path", type=str, default="", help="Porto-only path to offline OPQ reconstructed train embeddings used as an early decoded warm-start target")
    parser.add_argument("--porto_opq_warmup_start_epoch", type=int, default=0, help="Porto-only epoch to start the OPQ decoded warm-start")
    parser.add_argument("--porto_opq_warmup_end_epoch", type=int, default=0, help="Porto-only exclusive end epoch for the OPQ decoded warm-start; <= start disables it")
    parser.add_argument("--porto_opq_warmup_max_weight", type=float, default=0.0, help="Porto-only maximum weight for the OPQ decoded warm-start loss")
    parser.add_argument("--porto_opq_teacher_rotated_train_path", type=str, default="", help="Porto-only path to offline OPQ rotated train embeddings used for z-space teacher alignment")
    parser.add_argument("--porto_opq_teacher_codebook_path", type=str, default="", help="Porto-only path to offline OPQ PQ codebook used for teacher partition distillation / optional codebook init")
    parser.add_argument("--porto_opq_teacher_start_epoch", type=int, default=0, help="Porto-only epoch to start OPQ z/code partition teacher distillation")
    parser.add_argument("--porto_opq_teacher_end_epoch", type=int, default=0, help="Porto-only exclusive end epoch for OPQ z/code partition teacher distillation; <= start disables it")
    parser.add_argument("--porto_opq_teacher_z_weight", type=float, default=0.0, help="Porto-only max weight for z-space alignment to offline OPQ rotated embeddings")
    parser.add_argument("--porto_opq_teacher_partition_weight", type=float, default=0.0, help="Porto-only max weight for offline OPQ partition/code distillation")
    parser.add_argument("--porto_opq_teacher_codebook_freeze_end_epoch", type=int, default=0, help="Porto-only exclusive end epoch to keep the student PDT codebook frozen to the OPQ teacher")
    parser.add_argument("--porto_opq_teacher_realign_codebook_on_unfreeze", action="store_true", help="Porto-only: copy the OPQ teacher codebook back into the student once when the frozen codebook stage ends")
    parser.add_argument("--late_finetune_start_epoch",     type=int,   default=-1, help="epoch to rebuild the optimizer with smaller late-stage learning rates")
    parser.add_argument("--late_finetune_main_lr_scale",   type=float, default=1.0, help="late-stage multiplier for the main optimizer group learning rate")
    parser.add_argument("--late_finetune_pre_quant_lr_scale", type=float, default=1.0, help="late-stage multiplier for the bottleneck optimizer group learning rate")
    parser.add_argument("--train_set_size",                type=int,   default=None, help="optional override for the training split size")
    parser.add_argument("--query_set_size",                type=int,   default=None, help="optional override for the query split size")
    parser.add_argument("--base_set_size",                 type=int,   default=None, help="optional override for the base split size")
    parser.add_argument("--max_train_batches_per_epoch",   type=int,   default=0, help="limit the number of batches per epoch for smoke/sanity runs")
    parser.add_argument("--triplet_pos_begin_pos",         type=int,   default=0, help="inclusive lower bound of the positive neighbor sampling window")
    parser.add_argument("--triplet_pos_end_pos",           type=int,   default=200, help="inclusive upper bound of the positive neighbor sampling window")
    parser.add_argument("--triplet_neg_begin_pos",         type=int,   default=0, help="inclusive lower bound of the negative neighbor sampling window")
    parser.add_argument("--triplet_neg_end_pos",           type=int,   default=200, help="inclusive upper bound of the negative neighbor sampling window")

    args = parser.parse_args()
    if args.network_type == "TJCNN" and args.image_mode != "binary":
        raise ValueError("TJCNN must use --image_mode binary to preserve the original baseline.")
    if args.network_type == "TJCNN_MC_MSR" and args.image_mode not in ["motion6", "dtw8", "motion6_pyr2", "multigrid3", "shape5", "shape5_pyr2", "haus6", "dfd7"]:
        raise ValueError("TJCNN_MC_MSR must use --image_mode motion6, dtw8, motion6_pyr2, multigrid3, shape5, shape5_pyr2, haus6, or dfd7.")
    if args.embedding_backbone != "msr" and args.network_type != "TJCNN_MC_MSR":
        raise ValueError("Non-MSR embedding_backbone currently only supports network_type='TJCNN_MC_MSR'.")

    return args
    

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("GPU is Not Available.")
        exit()

    args   = get_args()
    device = torch.device(args.device)
    print("Device is:", device)

    function.setup_seed(args.random_seed)

    root_read_path = resolve_default_dataset_root(args.dataset, args.root_read_path)
    if root_read_path != str(args.root_read_path):
        print("Resolved dataset root_read_path to {}".format(root_read_path))
    if function.has_split_dataset(root_read_path):
        print("Dataset: {} (split dataset)".format(args.dataset))
        traj_list, train_set, query_set, base_set, train_dist_matrix, test_dist_matrix = function.load_split_dataset(
            root_read_path,
            args.dist_type,
        )
    elif args.dataset == "porto" and "0_porto_all" in root_read_path:
        print("Dataset: Porto (legacy 0_porto_all)")
        train_set, query_set, base_set = function.set_dataset(root_read_path, args.dist_type)
        traj_list_path = Path(root_read_path) / 'traj_list'
        train_dist_matrix_path = Path(root_read_path) / '{}_train_distance_matrix_result'.format(args.dist_type)
        test_dist_matrix_path = Path(root_read_path) / '{}_test_distance_matrix_result'.format(args.dist_type)
        traj_list = pickle.load(open(traj_list_path, "rb"))
        train_dist_matrix = pickle.load(open(train_dist_matrix_path, 'rb'))
        test_dist_matrix = pickle.load(open(test_dist_matrix_path, 'rb'))
    else:
        train_set, query_set, base_set = function.set_dataset(root_read_path, args.dist_type)
        train_dist_matrix_path = Path(root_read_path) / 'dtw_train_distance_matrix_result'
        test_dist_matrix_path  = Path(root_read_path) / 'dtw_test_distance_matrix_result'
        traj_list_path         = Path(root_read_path) / 'traj_list'
        traj_list = pickle.load(open(traj_list_path, "rb"))
        train_dist_matrix = pickle.load(open(train_dist_matrix_path,'rb'))
        test_dist_matrix = pickle.load(open(test_dist_matrix_path,'rb'))

    if args.train_set_size is not None:
        train_set = int(args.train_set_size)
    if args.query_set_size is not None:
        query_set = int(args.query_set_size)
    if args.base_set_size is not None:
        base_set = int(args.base_set_size)

    total_required = train_set + query_set + base_set
    if len(traj_list) < total_required:
        raise ValueError(
            "traj_list is too short for configured split: len(traj_list)={} < train+query+base={}".format(
                len(traj_list), total_required
            )
        )
    train_dist_shape = np.asarray(train_dist_matrix).shape
    test_dist_shape = np.asarray(test_dist_matrix).shape
    if train_dist_shape[0] < train_set or train_dist_shape[1] < train_set:
        raise ValueError(
            "train_dist_matrix shape {} is incompatible with train_set={}".format(
                train_dist_shape, train_set
            )
        )
    if test_dist_shape[0] < query_set or test_dist_shape[1] < base_set:
        raise ValueError(
            "test_dist_matrix shape {} is incompatible with query/base=({}, {})".format(
                test_dist_shape, query_set, base_set
            )
        )
    traj_list = traj_list[:total_required]
    train_dist_matrix = train_dist_matrix[:train_set, :train_set]
    test_dist_matrix = np.asarray(test_dist_matrix[:query_set, :base_set])
    artifact_prefix = str(args.artifact_prefix or "").strip()
    if artifact_prefix:
        save_model_path = str(Path(args.root_write_path) / "checkpoints")
    else:
        save_model_path = args.root_write_path + "/model"
    #query_traj = pickle.load(open('/data3/menghaotian/Traj_sim/datasets/features_sampling/Porto_query_traj_coord.pkl', "rb"))
    #query_traj = query_traj[0]
    print("traj num:", len(traj_list))
    #traj_list = pad_traj_list('dtw', traj_list, max_len, pad_value=1.0)
    #traj_list = np.array(traj_list)

    print("Time to meshing...")
    effective_grid_size = float(args.grid_size) if float(args.grid_size) > 0.0 else None
    if effective_grid_size is not None:
        print("Using custom grid_size:", effective_grid_size)
    lon_grid_id_list, lat_grid_id_list, lon_input_size, lat_input_size, lon_list, lat_list = grid.split_traj_into_equal_grid(
        traj_list,
        grid_size=effective_grid_size,
    )
    print("lon_input_size:", lon_input_size, " lat_input_size:", lat_input_size)
    #lon_grid_id_list, lat_grid_id_list, lon_input_size, lat_input_size, lon_list, lat_list =pickle.load(open('/data3/menghaotian/Traj_sim/datasets/features_sampling/grid.pkl', 'rb'))
    #l = [lon_grid_id_list, lat_grid_id_list, lon_input_size, lat_input_size, lon_list, lat_list]
    #with open('/data3/menghaotian/Traj_sim/datasets/features_sampling/grid.pkl', 'wb') as file:
    #    pickle.dump(l, file)


    my_config = ConfigClass(lon_input_size                 = lon_input_size,
                            lat_input_size                 = lat_input_size,
                            target_size                    = args.target_size,
                            batch_size                     = args.batch_size,
                            sampling_num                   = args.sampling_num,
                            learning_rate                  = args.learning_rate,
                            epoch_num                      = args.epoch_num,
                            dataset                        = args.dataset,
                            network_type                   = args.network_type,
                            channel                        = args.channel,
                            loss_type                      = args.loss_type,
                            cnn_feature_distance_type      = args.cnn_feature_distance_type,
                            cnntotal_feature_distance_type = args.cnntotal_feature_distance_type,
                            all_feature_distance_type      = args.all_feature_distance_type,
                            sampling_type                  = args.sampling_type,
                            root_write_path                = args.root_write_path,
                            root_read_path                 = args.root_read_path,
                            artifact_prefix                = artifact_prefix,
                            grid_size                      = args.grid_size,
                            image_mode                     = args.image_mode,
                            specific_image_disk_cache_enabled = not args.disable_specific_image_disk_cache,
                            embedding_backbone             = args.embedding_backbone,
                            backbone_seq_max_length        = args.backbone_seq_max_length,
                            simformer_num_layers           = args.simformer_num_layers,
                            simformer_n_heads              = args.simformer_n_heads,
                            simformer_dimfeedforward       = args.simformer_dimfeedforward,
                            simformer_pos_encoding         = args.simformer_pos_encoding,
                            neutraj_spatial_width          = args.neutraj_spatial_width,
                            neutraj_incell                 = args.neutraj_incell,
                            neutraj_use_standard_gru       = args.neutraj_use_standard_gru,
                            pdt_m                          = args.pdt_m,
                            pdt_k                          = args.pdt_k,
                            pdt_steps                      = args.pdt_steps,
                            pdt_heads                      = args.pdt_heads,
                            pdt_vq_type                    = args.pdt_vq_type,
                            pdt_codebook_init              = args.pdt_codebook_init,
                            qinco_h                        = args.qinco_h,
                            qinco_L                        = args.qinco_L,
                            qinco_identity_init            = args.qinco_identity_init,
                            pdt_init_codebook              = not args.disable_pdt_init_codebook,
                            eval_embedding_type            = args.eval_embedding_type,
                            eval_search_mode               = args.eval_search_mode,
                            enable_rerank                  = args.enable_rerank,
                            rerank_L                       = args.rerank_L,
                            rerank_source                  = args.rerank_source,
                            print_code_usage               = not args.disable_code_usage_stats,
                            load_model_train_flag          = args.load_model_train_flag,
                            backbone_checkpoint            = args.backbone_checkpoint,
                            freeze_backbone_epochs         = args.freeze_backbone_epochs,
                            pdt_loss_start_epoch           = args.pdt_loss_start_epoch,
                            pdt_loss_weight                = args.pdt_loss_weight,
                            loss_recipe                    = args.loss_recipe,
                            consistency_weight             = args.consistency_weight,
                            quantized_metric_weight        = args.quantized_metric_weight,
                            improved_qm_start_epoch        = args.improved_qm_start_epoch,
                            improved_qm_warmup_epochs      = args.improved_qm_warmup_epochs,
                            improved_qm_max_weight         = args.improved_qm_max_weight,
                            improved_pairwise_weight       = args.improved_pairwise_weight,
                            improved_entropy_weight        = args.improved_entropy_weight,
                            improved_commit_weight         = args.improved_commit_weight,
                            improved_uniform_weight        = args.improved_uniform_weight,
                            improved_vq_adaptive_low_codebook = not args.disable_improved_vq_adaptive_low_codebook,
                            pre_quant_bottleneck_enabled   = args.pre_quant_bottleneck_enabled,
                            pre_quant_global_dim           = args.pre_quant_global_dim,
                            pre_quant_local_dim            = args.pre_quant_local_dim,
                            pre_quant_progress_dim         = args.pre_quant_progress_dim,
                            pre_quant_use_motion_stats     = args.pre_quant_use_motion_stats,
                            pre_quant_lambda_decor         = args.pre_quant_lambda_decor,
                            pre_quant_lambda_stab          = args.pre_quant_lambda_stab,
                            pre_quant_residual_alpha_init  = args.pre_quant_residual_alpha_init,
                            pre_quant_learnable_alpha      = not args.disable_pre_quant_learnable_alpha,
                            pre_quant_lr_multiplier        = args.pre_quant_lr_multiplier,
                            pre_quant_stab_late_epoch      = args.pre_quant_stab_late_epoch,
                            pre_quant_stab_late_multiplier = args.pre_quant_stab_late_multiplier,
                            pre_quant_refresh_start_epoch  = args.pre_quant_refresh_start_epoch,
                            pre_quant_refresh_period       = args.pre_quant_refresh_period,
                            pre_quant_refresh_end_epoch    = args.pre_quant_refresh_end_epoch,
                            pre_quant_raw_metric_weight    = args.pre_quant_raw_metric_weight,
                            pre_quant_neighbor_enabled     = args.pre_quant_neighbor_enabled,
                            pre_quant_neighbor_use_btn     = args.pre_quant_neighbor_use_btn,
                            pre_quant_neighbor_use_dec     = args.pre_quant_neighbor_use_dec,
                            pre_quant_neighbor_topk        = args.pre_quant_neighbor_topk,
                            pre_quant_neighbor_tau_btn     = args.pre_quant_neighbor_tau_btn,
                            pre_quant_neighbor_tau_dec     = args.pre_quant_neighbor_tau_dec,
                            pre_quant_neighbor_lambda_btn  = args.pre_quant_neighbor_lambda_btn,
                            pre_quant_neighbor_lambda_dec  = args.pre_quant_neighbor_lambda_dec,
                            pre_quant_neighbor_start_epoch = args.pre_quant_neighbor_start_epoch,
                            pre_quant_neighbor_warmup_epochs = args.pre_quant_neighbor_warmup_epochs,
                            pre_quant_neighbor_dec_stop_backbone = args.pre_quant_neighbor_dec_stop_backbone,
                            pre_quant_neighbor_teacher_mode = args.pre_quant_neighbor_teacher_mode,
                            pre_quant_neighbor_offline_path = args.pre_quant_neighbor_offline_path,
                            pre_quant_landmark_enabled     = args.pre_quant_landmark_enabled,
                            pre_quant_landmark_num         = args.pre_quant_landmark_num,
                            pre_quant_landmark_select      = args.pre_quant_landmark_select,
                            pre_quant_landmark_profile_transform = args.pre_quant_landmark_profile_transform,
                            pre_quant_landmark_use_btn     = args.pre_quant_landmark_use_btn,
                            pre_quant_landmark_use_dec     = args.pre_quant_landmark_use_dec,
                            pre_quant_landmark_dec_ste_to_btn = args.pre_quant_landmark_dec_ste_to_btn,
                            pre_quant_landmark_dec_bank_source = args.pre_quant_landmark_dec_bank_source,
                            pre_quant_landmark_rank_weight = args.pre_quant_landmark_rank_weight,
                            pre_quant_landmark_rank_tau    = args.pre_quant_landmark_rank_tau,
                            pre_quant_landmark_lambda_btn  = args.pre_quant_landmark_lambda_btn,
                            pre_quant_landmark_lambda_dec  = args.pre_quant_landmark_lambda_dec,
                            pre_quant_landmark_start_epoch = args.pre_quant_landmark_start_epoch,
                            pre_quant_landmark_warmup_epochs = args.pre_quant_landmark_warmup_epochs,
                            pre_quant_landmark_bank_refresh_epochs = args.pre_quant_landmark_bank_refresh_epochs,
                            pre_quant_landmark_artifact_path = args.pre_quant_landmark_artifact_path,
                            pre_quant_landmark_fixed_bank_path = args.pre_quant_landmark_fixed_bank_path,
                            pre_quant_landmark_teacher_checkpoint = args.pre_quant_landmark_teacher_checkpoint,
                            decoded_ste_metric_enabled      = args.decoded_ste_metric_enabled,
                            decoded_ste_metric_start_epoch  = args.decoded_ste_metric_start_epoch,
                            decoded_ste_metric_warmup_epochs = args.decoded_ste_metric_warmup_epochs,
                            decoded_ste_metric_max_weight   = args.decoded_ste_metric_max_weight,
                            porto_opq_warmup_train_recon_path = args.porto_opq_warmup_train_recon_path,
                            porto_opq_warmup_start_epoch   = args.porto_opq_warmup_start_epoch,
                            porto_opq_warmup_end_epoch     = args.porto_opq_warmup_end_epoch,
                            porto_opq_warmup_max_weight    = args.porto_opq_warmup_max_weight,
                            porto_opq_teacher_rotated_train_path = args.porto_opq_teacher_rotated_train_path,
                            porto_opq_teacher_codebook_path = args.porto_opq_teacher_codebook_path,
                            porto_opq_teacher_start_epoch  = args.porto_opq_teacher_start_epoch,
                            porto_opq_teacher_end_epoch    = args.porto_opq_teacher_end_epoch,
                            porto_opq_teacher_z_weight     = args.porto_opq_teacher_z_weight,
                            porto_opq_teacher_partition_weight = args.porto_opq_teacher_partition_weight,
                            porto_opq_teacher_codebook_freeze_end_epoch = args.porto_opq_teacher_codebook_freeze_end_epoch,
                            porto_opq_teacher_realign_codebook_on_unfreeze = args.porto_opq_teacher_realign_codebook_on_unfreeze,
                            late_finetune_start_epoch      = args.late_finetune_start_epoch,
                            late_finetune_main_lr_scale    = args.late_finetune_main_lr_scale,
                            late_finetune_pre_quant_lr_scale = args.late_finetune_pre_quant_lr_scale,
                            max_train_batches_per_epoch    = args.max_train_batches_per_epoch,
                            triplet_pos_begin_pos          = args.triplet_pos_begin_pos,
                            triplet_pos_end_pos            = args.triplet_pos_end_pos,
                            triplet_neg_begin_pos          = args.triplet_neg_begin_pos,
                            triplet_neg_end_pos            = args.triplet_neg_end_pos,
                            train_ratio                    = args.train_ratio,
                            mode                           = args.mode,
                            test_epoch                     = args.test_epoch,
                            print_epoch                    = args.print_epoch,
                            save_model_epoch               = args.save_model_epoch,
                            save_model                     = args.save_model,
                            eval_save_epochs               = args.eval_save_epochs,
                            save_feature_distance          = args.save_feature_distance,
                            report_only_eval               = args.report_only_eval,
                            save_model_path                = save_model_path,
                            dist_type                      = args.dist_type,
                            device                         = device,
                            LDS                            = False,
                            FDS                            = False,
                            train_flag                     = args.train_flag,
                            head_num                       = int(args.head_num),
                            area_path                      = args.area_path,
                            train_set                      = train_set,
                            query_set                      = query_set,
                            base_set                       = base_set)
    
    if my_config.my_dict["network_type"] in ["TJCNN", "TJCNN_MC_MSR"]:
        lon_onehot = lon_list
        lat_onehot = lat_list
    else:
        new_lon_grid_id_list = []
        for traj in lon_grid_id_list:
            tem_lon_list = []
            for value in traj:
                tem_lon_list.append([value])
            new_lon_grid_id_list.append(tem_lon_list)
        new_lat_grid_id_list = []
        for traj in lat_grid_id_list:
            tem_lat_list = []
            for value in traj:
                tem_lat_list.append([value])
            new_lat_grid_id_list.append(tem_lat_list)
        lon_onehot = new_lon_grid_id_list
        lat_onehot = new_lat_grid_id_list



    traj_network = NeuTrajTrainer(my_config)

    traj_network.data_prepare(traj_list,
                              train_dist_matrix,
                              test_dist_matrix,
                              lon_onehot = lon_onehot,
                              lat_onehot = lat_onehot,
                              lon_grid_id_list = lon_grid_id_list,
                              lat_grid_id_list = lat_grid_id_list)
    
    mode = my_config.my_dict["mode"]
    if mode == "test":
        traj_network.extract_feature()
    elif mode == "build-neighbor-gt":
        traj_network.build_gt_neighbor_teacher()
    elif mode == "build-landmark-profile":
        traj_network.build_gt_landmark_profile()
    elif mode == "build-landmark-bank":
        traj_network.build_pre_quant_landmark_bank()
    elif mode == "smoke-test":
        traj_network.run_smoke_test()
    elif mode == "train-directly":
        traj_network.train()
    else:
        raise ValueError("Train Mode Value Error!")
