import torch
import random
import numpy as np
import sys
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PDT_ROOT = REPO_ROOT / "PDT_VQ"
for candidate in (REPO_ROOT, PDT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)
from mynn.transformer import Transformer
from PDT_VQ.model import PDT
from PDT_VQ.train import get_args

from tools import pre_rep
from mynn.shortcut import ShortCutCNN as ConvTraj
from mynn.shortcut import ResidualMultiScaleMotionCanvasCNN
from mynn.simplest_cnn import SimpleCNN
from mynn.external_backbones import ExternalSequenceBackbonePDTModel




from loss.triplet import TripletLoss


def has_split_dataset(root_read_path):
    root = Path(root_read_path)
    return all((root / name).exists() for name in ["train_list", "query_list", "base_list"])


def _split_dist_dir(root_read_path, dist_type):
    root = Path(root_read_path)
    dist_dir_map = {
        "dtw": "DTW",
        "dfd": "DFD",
        "edr": "EDR",
        "haus": "Haus",
    }
    dist_key = str(dist_type).lower()
    if dist_key not in dist_dir_map:
        raise ValueError("Unsupported dist_type for split dataset: {}".format(dist_type))
    return root / dist_dir_map[dist_key]


def load_split_dataset(root_read_path, dist_type):
    root = Path(root_read_path)
    if not has_split_dataset(root):
        raise ValueError("split dataset files not found under {}".format(root))

    train_list = pickle.load(open(root / "train_list", "rb"))
    query_list = pickle.load(open(root / "query_list", "rb"))
    base_list = pickle.load(open(root / "base_list", "rb"))

    dist_root = _split_dist_dir(root, dist_type)
    train_dist_matrix = pickle.load(open(dist_root / "{}_train_distance_matrix_result".format(dist_type), "rb"))
    test_dist_matrix = pickle.load(open(dist_root / "{}_test_distance_matrix_result".format(dist_type), "rb"))

    traj_list = list(train_list) + list(query_list) + list(base_list)
    return traj_list, len(train_list), len(query_list), len(base_list), train_dist_matrix, test_dist_matrix


def set_dataset(root_read_path, dist_type="dtw"):
    if has_split_dataset(root_read_path):
        _, train_set, query_set, base_set, _, _ = load_split_dataset(root_read_path, dist_type)
    elif "0_geolife" in root_read_path:
        train_set, query_set, base_set = 3000, 1000, 13386 - 4000
    elif "0_porto_all" in root_read_path:
        train_set, query_set, base_set = 3000, 1000, 1601579 - 4000
    else:
        raise Exception("root_read_path error")
    return train_set, query_set, base_set


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pad_traj_list(seq_list, max_length, pad_value = 0.0):
    value = [1.0 * pad_value for i in range(len(seq_list[0][0]))]
    final_pad_seq_list = []
    for seq in seq_list:
        assert len(seq) <= max_length, "Sequence length {} is larger than max_length {}".format(len(seq), max_length)

        for j in range(max_length - len(seq)):
            seq.append(value)
        final_pad_seq_list.append(seq)
    return final_pad_seq_list



def initialize_loss(my_dict):
    batch_size   = my_dict["batch_size"]
    sampling_num = my_dict["sampling_num"]
    epoch_num    = my_dict["epoch_num"]
    if my_dict["loss_type"] == "triplet":
        my_loss = TripletLoss(epoch_num)
    else:
        raise ValueError("Loss Type Error")
    print("Init {} Loss Done !!!".format(my_dict["loss_type"]))
    return my_loss

def initialize_model(my_dict, max_traj_length):

    lon_input_size = my_dict["lon_input_size"]
    lat_input_size = my_dict["lat_input_size"]
    target_size    = my_dict["target_size"]
    batch_size     = my_dict["batch_size"]
    sampling_num   = my_dict["sampling_num"]
    device         = my_dict["device"]
    channel        = my_dict["channel"]
    head_num       = my_dict["head_num"]
    image_mode     = my_dict.get("image_mode", "binary")
    pdt_m          = my_dict.get("pdt_m", 16)
    pdt_k          = my_dict.get("pdt_k", 256)
    pdt_vq_type    = my_dict.get("pdt_vq_type", "dpq")
    pdt_codebook_init = my_dict.get("pdt_codebook_init", "uniform")
    qinco_h        = my_dict.get("qinco_h", 256)
    qinco_L        = my_dict.get("qinco_L", 1)
    qinco_identity_init = my_dict.get("qinco_identity_init", False)
    pre_quant_bottleneck_enabled = my_dict.get("pre_quant_bottleneck_enabled", False)
    pre_quant_global_dim = my_dict.get("pre_quant_global_dim", 48)
    pre_quant_local_dim = my_dict.get("pre_quant_local_dim", 48)
    pre_quant_progress_dim = my_dict.get("pre_quant_progress_dim", 32)
    pre_quant_use_motion_stats = my_dict.get("pre_quant_use_motion_stats", False)
    pre_quant_lambda_decor = my_dict.get("pre_quant_lambda_decor", 0.01)
    pre_quant_lambda_stab = my_dict.get("pre_quant_lambda_stab", 0.1)
    pre_quant_residual_alpha_init = my_dict.get("pre_quant_residual_alpha_init", 0.15)
    embedding_backbone = my_dict.get("embedding_backbone", "msr")
    backbone_seq_max_length = my_dict.get("backbone_seq_max_length", 200)
    simformer_num_layers = my_dict.get("simformer_num_layers", 1)
    simformer_n_heads = my_dict.get("simformer_n_heads", 16)
    simformer_dimfeedforward = my_dict.get("simformer_dimfeedforward", 256)
    simformer_pos_encoding = my_dict.get("simformer_pos_encoding", "fixed")
    neutraj_spatial_width = my_dict.get("neutraj_spatial_width", 2)
    neutraj_incell = my_dict.get("neutraj_incell", True)
    neutraj_use_standard_gru = my_dict.get("neutraj_use_standard_gru", False)

    if embedding_backbone not in ["msr", "neutraj", "simformer"]:
        raise ValueError("Unsupported embedding_backbone: {}".format(embedding_backbone))

    if my_dict["network_type"] == "TJCNN":
        my_net = ConvTraj(lon_input_size, lat_input_size, target_size, batch_size, sampling_num, max_traj_length, channel, device, head_num, pdt_m=pdt_m, pdt_k=pdt_k, pdt_vq_type=pdt_vq_type, pdt_codebook_init=pdt_codebook_init, qinco_h=qinco_h, qinco_L=qinco_L, qinco_identity_init=qinco_identity_init, pre_quant_bottleneck_enabled=pre_quant_bottleneck_enabled, pre_quant_global_dim=pre_quant_global_dim, pre_quant_local_dim=pre_quant_local_dim, pre_quant_progress_dim=pre_quant_progress_dim, pre_quant_use_motion_stats=pre_quant_use_motion_stats, pre_quant_lambda_decor=pre_quant_lambda_decor, pre_quant_lambda_stab=pre_quant_lambda_stab, pre_quant_residual_alpha_init=pre_quant_residual_alpha_init)
    elif my_dict["network_type"] == "TJCNN_MC_MSR":
        if image_mode not in ["motion6", "motion6_pyr2", "multigrid3"]:
            raise ValueError("TJCNN_MC_MSR must use image_mode='motion6', 'motion6_pyr2', or 'multigrid3'.")
        if embedding_backbone == "msr":
            my_net = ResidualMultiScaleMotionCanvasCNN(lon_input_size,
                                                       lat_input_size,
                                                       target_size,
                                                       batch_size,
                                                       sampling_num,
                                                       max_traj_length,
                                                       channel,
                                                       device,
                                                       head_num,
                                                       image_channels=pre_rep.get_image_mode_channels(image_mode),
                                                       pdt_m=pdt_m,
                                                       pdt_k=pdt_k,
                                                       pdt_vq_type=pdt_vq_type,
                                                       pdt_codebook_init=pdt_codebook_init,
                                                       qinco_h=qinco_h,
                                                       qinco_L=qinco_L,
                                                       qinco_identity_init=qinco_identity_init,
                                                       pre_quant_bottleneck_enabled=pre_quant_bottleneck_enabled,
                                                       pre_quant_global_dim=pre_quant_global_dim,
                                                       pre_quant_local_dim=pre_quant_local_dim,
                                                       pre_quant_progress_dim=pre_quant_progress_dim,
                                                       pre_quant_use_motion_stats=pre_quant_use_motion_stats,
                                                       pre_quant_lambda_decor=pre_quant_lambda_decor,
                                                       pre_quant_lambda_stab=pre_quant_lambda_stab,
                                                       pre_quant_residual_alpha_init=pre_quant_residual_alpha_init)
        else:
            my_net = ExternalSequenceBackbonePDTModel(
                backbone_kind=embedding_backbone,
                lon_input_size=lon_input_size,
                lat_input_size=lat_input_size,
                target_size=target_size,
                max_seq_length=max_traj_length,
                device=device,
                pdt_m=pdt_m,
                pdt_k=pdt_k,
                pdt_vq_type=pdt_vq_type,
                pdt_codebook_init=pdt_codebook_init,
                qinco_h=qinco_h,
                qinco_L=qinco_L,
                qinco_identity_init=qinco_identity_init,
                backbone_seq_max_length=backbone_seq_max_length,
                simformer_num_layers=simformer_num_layers,
                simformer_n_heads=simformer_n_heads,
                simformer_dimfeedforward=simformer_dimfeedforward,
                simformer_pos_encoding=simformer_pos_encoding,
                neutraj_spatial_width=neutraj_spatial_width,
                neutraj_incell=neutraj_incell,
                neutraj_use_standard_gru=neutraj_use_standard_gru,
                pre_quant_bottleneck_enabled=pre_quant_bottleneck_enabled,
                pre_quant_global_dim=pre_quant_global_dim,
                pre_quant_local_dim=pre_quant_local_dim,
                pre_quant_progress_dim=pre_quant_progress_dim,
                pre_quant_use_motion_stats=pre_quant_use_motion_stats,
                pre_quant_lambda_decor=pre_quant_lambda_decor,
                pre_quant_lambda_stab=pre_quant_lambda_stab,
                pre_quant_residual_alpha_init=pre_quant_residual_alpha_init,
            )
    elif my_dict["network_type"] == "SimpleCNN" and ("Test_porto" in my_dict["root_read_path"]):
        my_net = SimpleCNN(lon_input_size, lat_input_size, target_size, batch_size, sampling_num, max_traj_length, channel, device, head_num)
    elif my_dict["network_type"] == "Global_T":
        my_net = Transformer(lon_input_size, lat_input_size, target_size, max_traj_length, device)
    elif "Local_T" in my_dict["network_type"]:
        windows_size = int(my_dict["network_type"][7:])
        my_net = Transformer(lon_input_size, lat_input_size, target_size, max_traj_length, device, mask = "local", k = windows_size)
    else:
        raise ValueError("Network Type Error")
    
    print("Init {} Model Done !!!".format(my_dict["network_type"]))
    return my_net

if __name__ == '__main__':
    pass
