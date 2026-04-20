import os
import json
import numpy as np

import torch

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, type(bytes)):
            return str(obj, encoding='utf-8')
        elif type(obj) == int:
            return str(obj)
        elif type(obj) == float:
            return str(obj)
        elif type(obj) == bool:
            return str(obj)
        elif type(obj) == torch.device:
            return str(obj)
        else:
            print(type(obj))
            return super(MyEncoder, self).default(obj)

class ConfigClass():
    def __init__(self,
                 lon_input_size                 = None,
                 lat_input_size                 = None,
                 target_size                    = None,
                 batch_size                     = None,
                 sampling_num                   = None,
                 learning_rate                  = None,
                 epoch_num                      = None,
                 dataset                        = None,
                 network_type                   = None,
                 channel                        = None,
                 loss_type                      = None,
                 cnn_feature_distance_type      = None,
                 cnntotal_feature_distance_type = None,
                 all_feature_distance_type      = None,
                 sampling_type                  = None,
                 root_write_path                = None,
                 root_read_path                = None,
                 artifact_prefix                = "",
                 grid_size                      = 0.0,
                 image_mode                     = "binary",
                 embedding_backbone             = "msr",
                 backbone_seq_max_length        = 200,
                 simformer_num_layers           = 1,
                 simformer_n_heads              = 16,
                 simformer_dimfeedforward       = 256,
                 simformer_pos_encoding         = "fixed",
                 neutraj_spatial_width          = 2,
                 neutraj_incell                 = True,
                 neutraj_use_standard_gru       = False,
                 pdt_m                          = 16,
                 pdt_k                          = 256,
                 pdt_vq_type                    = "dpq",
                 pdt_codebook_init              = "uniform",
                 qinco_h                        = 256,
                 qinco_L                        = 1,
                 qinco_identity_init            = False,
                 pdt_init_codebook              = True,
                 eval_embedding_type            = "quantized",
                 eval_search_mode               = "decoded",
                 enable_rerank                  = False,
                 rerank_L                       = 100,
                 rerank_source                  = "decoded",
                 print_code_usage               = True,
                 load_model_train_flag          = None,
                 backbone_checkpoint            = None,
                 freeze_backbone_epochs         = 0,
                 pdt_loss_start_epoch           = 20,
                 pdt_loss_weight                = 0.3,
                 loss_recipe                    = "baseline",
                 consistency_weight             = 0.1,
                 quantized_metric_weight        = 0.3,
                 improved_qm_start_epoch        = 60,
                 improved_qm_warmup_epochs      = 80,
                 improved_qm_max_weight         = 0.12,
                 improved_pairwise_weight       = 0.05,
                 improved_entropy_weight        = 0.01,
                 improved_commit_weight         = 0.02,
                 improved_uniform_weight        = 0.001,
                 improved_vq_adaptive_low_codebook = True,
                 pre_quant_bottleneck_enabled   = False,
                 pre_quant_global_dim           = 48,
                 pre_quant_local_dim            = 48,
                 pre_quant_progress_dim         = 32,
                 pre_quant_use_motion_stats     = False,
                 pre_quant_lambda_decor         = 0.01,
                 pre_quant_lambda_stab          = 0.1,
                 pre_quant_residual_alpha_init  = 0.15,
                 pre_quant_lr_multiplier        = 0.25,
                 pre_quant_stab_late_epoch      = 100,
                 pre_quant_stab_late_multiplier = 4.0,
                 pre_quant_refresh_start_epoch  = 100,
                 pre_quant_refresh_period       = 50,
                 pre_quant_refresh_end_epoch    = 0,
                 pre_quant_raw_metric_weight    = 0.2,
                 pre_quant_neighbor_enabled     = False,
                 pre_quant_neighbor_use_btn     = False,
                 pre_quant_neighbor_use_dec     = False,
                 pre_quant_neighbor_topk        = 10,
                 pre_quant_neighbor_tau_btn     = 0.07,
                 pre_quant_neighbor_tau_dec     = 0.07,
                 pre_quant_neighbor_lambda_btn  = 0.05,
                 pre_quant_neighbor_lambda_dec  = 0.10,
                 pre_quant_neighbor_start_epoch = 0,
                 pre_quant_neighbor_warmup_epochs = 0,
                 pre_quant_neighbor_dec_stop_backbone = False,
                 pre_quant_neighbor_teacher_mode = "batch_cont",
                 pre_quant_neighbor_offline_path = "",
                 pre_quant_landmark_enabled     = False,
                 pre_quant_landmark_num         = 64,
                 pre_quant_landmark_select      = "fps",
                 pre_quant_landmark_profile_transform = "log1p_zscore",
                 pre_quant_landmark_use_btn     = False,
                 pre_quant_landmark_use_dec     = True,
                 pre_quant_landmark_dec_ste_to_btn = False,
                 pre_quant_landmark_dec_bank_source = "decoded",
                 pre_quant_landmark_rank_weight = 0.0,
                 pre_quant_landmark_rank_tau = 0.5,
                 pre_quant_landmark_lambda_btn  = 0.03,
                 pre_quant_landmark_lambda_dec  = 0.10,
                 pre_quant_landmark_start_epoch = 100,
                 pre_quant_landmark_warmup_epochs = 20,
                 pre_quant_landmark_bank_refresh_epochs = 10,
                 pre_quant_landmark_artifact_path = "",
                 pre_quant_landmark_fixed_bank_path = "",
                 pre_quant_landmark_teacher_checkpoint = "",
                 decoded_ste_metric_enabled      = False,
                 decoded_ste_metric_start_epoch  = 80,
                 decoded_ste_metric_warmup_epochs = 20,
                 decoded_ste_metric_max_weight   = 0.03,
                 porto_opq_warmup_train_recon_path = "",
                 porto_opq_warmup_start_epoch   = 0,
                 porto_opq_warmup_end_epoch     = 0,
                 porto_opq_warmup_max_weight    = 0.0,
                 porto_opq_teacher_rotated_train_path = "",
                 porto_opq_teacher_codebook_path = "",
                 porto_opq_teacher_start_epoch  = 0,
                 porto_opq_teacher_end_epoch    = 0,
                 porto_opq_teacher_z_weight     = 0.0,
                 porto_opq_teacher_partition_weight = 0.0,
                 porto_opq_teacher_codebook_freeze_end_epoch = 0,
                 porto_opq_teacher_realign_codebook_on_unfreeze = False,
                 late_finetune_start_epoch      = -1,
                 late_finetune_main_lr_scale    = 1.0,
                 late_finetune_pre_quant_lr_scale = 1.0,
                 max_train_batches_per_epoch    = 0,
                 triplet_pos_begin_pos          = 0,
                 triplet_pos_end_pos            = 200,
                 triplet_neg_begin_pos          = 0,
                 triplet_neg_end_pos            = 200,
                 train_ratio                    = None,
                 mode                           = None,
                 test_epoch                     = None,
                 print_epoch                    = None,
                 save_model_epoch               = None,
                 save_model                     = None,
                 eval_save_epochs               = "",
                 save_feature_distance          = False,
                 report_only_eval               = False,
                 save_model_path                = None,
                 dist_type                      = None,
                 device                         = "cpu",
                 LDS                            = False,
                 FDS                            = True,
                 train_flag                     = "train.log",
                 head_num                       = None,
                 area_path                      = None,
                 train_set                      = None,
                 query_set                      = None,
                 base_set                       = None):
        self.my_dict = {}
        self.my_dict["lon_input_size"]                 = lon_input_size
        self.my_dict["lat_input_size"]                 = lat_input_size
        self.my_dict["target_size"]                    = target_size
        self.my_dict["batch_size"]                     = batch_size
        self.my_dict["sampling_num"]                   = sampling_num
        self.my_dict["learning_rate"]                  = learning_rate
        self.my_dict["epoch_num"]                      = epoch_num
        self.my_dict["dataset"]                        = dataset
        self.my_dict["network_type"]                   = network_type
        self.my_dict["channel"]                        = channel
        self.my_dict["loss_type"]                      = loss_type
        self.my_dict["cnn_feature_distance_type"]      = cnn_feature_distance_type
        self.my_dict["cnntotal_feature_distance_type"] = cnntotal_feature_distance_type
        self.my_dict["all_feature_distance_type"]      = all_feature_distance_type
        self.my_dict["sampling_type"]                  = sampling_type
        self.my_dict["root_write_path"]                = root_write_path
        self.my_dict["root_read_path"]                = root_read_path
        self.my_dict["artifact_prefix"]                = artifact_prefix
        self.my_dict["grid_size"]                      = grid_size
        self.my_dict["image_mode"]                     = image_mode
        self.my_dict["embedding_backbone"]             = embedding_backbone
        self.my_dict["backbone_seq_max_length"]        = backbone_seq_max_length
        self.my_dict["simformer_num_layers"]           = simformer_num_layers
        self.my_dict["simformer_n_heads"]              = simformer_n_heads
        self.my_dict["simformer_dimfeedforward"]       = simformer_dimfeedforward
        self.my_dict["simformer_pos_encoding"]         = simformer_pos_encoding
        self.my_dict["neutraj_spatial_width"]          = neutraj_spatial_width
        self.my_dict["neutraj_incell"]                 = neutraj_incell
        self.my_dict["neutraj_use_standard_gru"]       = neutraj_use_standard_gru
        self.my_dict["pdt_m"]                          = pdt_m
        self.my_dict["pdt_k"]                          = pdt_k
        self.my_dict["pdt_vq_type"]                    = pdt_vq_type
        self.my_dict["pdt_codebook_init"]              = pdt_codebook_init
        self.my_dict["qinco_h"]                        = qinco_h
        self.my_dict["qinco_L"]                        = qinco_L
        self.my_dict["qinco_identity_init"]            = qinco_identity_init
        self.my_dict["pdt_init_codebook"]              = pdt_init_codebook
        self.my_dict["eval_embedding_type"]            = eval_embedding_type
        self.my_dict["eval_search_mode"]               = eval_search_mode
        self.my_dict["enable_rerank"]                  = enable_rerank
        self.my_dict["rerank_L"]                       = rerank_L
        self.my_dict["rerank_source"]                  = rerank_source
        self.my_dict["print_code_usage"]               = print_code_usage
        self.my_dict["load_model_train_flag"]          = load_model_train_flag
        self.my_dict["backbone_checkpoint"]            = backbone_checkpoint
        self.my_dict["freeze_backbone_epochs"]         = freeze_backbone_epochs
        self.my_dict["pdt_loss_start_epoch"]           = pdt_loss_start_epoch
        self.my_dict["pdt_loss_weight"]                = pdt_loss_weight
        self.my_dict["loss_recipe"]                    = loss_recipe
        self.my_dict["consistency_weight"]             = consistency_weight
        self.my_dict["quantized_metric_weight"]        = quantized_metric_weight
        self.my_dict["improved_qm_start_epoch"]        = improved_qm_start_epoch
        self.my_dict["improved_qm_warmup_epochs"]      = improved_qm_warmup_epochs
        self.my_dict["improved_qm_max_weight"]         = improved_qm_max_weight
        self.my_dict["improved_pairwise_weight"]       = improved_pairwise_weight
        self.my_dict["improved_entropy_weight"]        = improved_entropy_weight
        self.my_dict["improved_commit_weight"]         = improved_commit_weight
        self.my_dict["improved_uniform_weight"]        = improved_uniform_weight
        self.my_dict["improved_vq_adaptive_low_codebook"] = improved_vq_adaptive_low_codebook
        self.my_dict["pre_quant_bottleneck_enabled"]   = pre_quant_bottleneck_enabled
        self.my_dict["pre_quant_global_dim"]           = pre_quant_global_dim
        self.my_dict["pre_quant_local_dim"]            = pre_quant_local_dim
        self.my_dict["pre_quant_progress_dim"]         = pre_quant_progress_dim
        self.my_dict["pre_quant_use_motion_stats"]     = pre_quant_use_motion_stats
        self.my_dict["pre_quant_lambda_decor"]         = pre_quant_lambda_decor
        self.my_dict["pre_quant_lambda_stab"]          = pre_quant_lambda_stab
        self.my_dict["pre_quant_residual_alpha_init"]  = pre_quant_residual_alpha_init
        self.my_dict["pre_quant_lr_multiplier"]        = pre_quant_lr_multiplier
        self.my_dict["pre_quant_stab_late_epoch"]      = pre_quant_stab_late_epoch
        self.my_dict["pre_quant_stab_late_multiplier"] = pre_quant_stab_late_multiplier
        self.my_dict["pre_quant_refresh_start_epoch"]  = pre_quant_refresh_start_epoch
        self.my_dict["pre_quant_refresh_period"]       = pre_quant_refresh_period
        self.my_dict["pre_quant_refresh_end_epoch"]    = pre_quant_refresh_end_epoch
        self.my_dict["pre_quant_raw_metric_weight"]    = pre_quant_raw_metric_weight
        self.my_dict["pre_quant_neighbor_enabled"]     = pre_quant_neighbor_enabled
        self.my_dict["pre_quant_neighbor_use_btn"]     = pre_quant_neighbor_use_btn
        self.my_dict["pre_quant_neighbor_use_dec"]     = pre_quant_neighbor_use_dec
        self.my_dict["pre_quant_neighbor_topk"]        = pre_quant_neighbor_topk
        self.my_dict["pre_quant_neighbor_tau_btn"]     = pre_quant_neighbor_tau_btn
        self.my_dict["pre_quant_neighbor_tau_dec"]     = pre_quant_neighbor_tau_dec
        self.my_dict["pre_quant_neighbor_lambda_btn"]  = pre_quant_neighbor_lambda_btn
        self.my_dict["pre_quant_neighbor_lambda_dec"]  = pre_quant_neighbor_lambda_dec
        self.my_dict["pre_quant_neighbor_start_epoch"] = pre_quant_neighbor_start_epoch
        self.my_dict["pre_quant_neighbor_warmup_epochs"] = pre_quant_neighbor_warmup_epochs
        self.my_dict["pre_quant_neighbor_dec_stop_backbone"] = pre_quant_neighbor_dec_stop_backbone
        self.my_dict["pre_quant_neighbor_teacher_mode"] = pre_quant_neighbor_teacher_mode
        self.my_dict["pre_quant_neighbor_offline_path"] = pre_quant_neighbor_offline_path
        self.my_dict["pre_quant_landmark_enabled"]     = pre_quant_landmark_enabled
        self.my_dict["pre_quant_landmark_num"]         = pre_quant_landmark_num
        self.my_dict["pre_quant_landmark_select"]      = pre_quant_landmark_select
        self.my_dict["pre_quant_landmark_profile_transform"] = pre_quant_landmark_profile_transform
        self.my_dict["pre_quant_landmark_use_btn"]     = pre_quant_landmark_use_btn
        self.my_dict["pre_quant_landmark_use_dec"]     = pre_quant_landmark_use_dec
        self.my_dict["pre_quant_landmark_dec_ste_to_btn"] = pre_quant_landmark_dec_ste_to_btn
        self.my_dict["pre_quant_landmark_dec_bank_source"] = pre_quant_landmark_dec_bank_source
        self.my_dict["pre_quant_landmark_rank_weight"] = pre_quant_landmark_rank_weight
        self.my_dict["pre_quant_landmark_rank_tau"]    = pre_quant_landmark_rank_tau
        self.my_dict["pre_quant_landmark_lambda_btn"]  = pre_quant_landmark_lambda_btn
        self.my_dict["pre_quant_landmark_lambda_dec"]  = pre_quant_landmark_lambda_dec
        self.my_dict["pre_quant_landmark_start_epoch"] = pre_quant_landmark_start_epoch
        self.my_dict["pre_quant_landmark_warmup_epochs"] = pre_quant_landmark_warmup_epochs
        self.my_dict["pre_quant_landmark_bank_refresh_epochs"] = pre_quant_landmark_bank_refresh_epochs
        self.my_dict["pre_quant_landmark_artifact_path"] = pre_quant_landmark_artifact_path
        self.my_dict["pre_quant_landmark_fixed_bank_path"] = pre_quant_landmark_fixed_bank_path
        self.my_dict["pre_quant_landmark_teacher_checkpoint"] = pre_quant_landmark_teacher_checkpoint
        self.my_dict["decoded_ste_metric_enabled"]      = decoded_ste_metric_enabled
        self.my_dict["decoded_ste_metric_start_epoch"]  = decoded_ste_metric_start_epoch
        self.my_dict["decoded_ste_metric_warmup_epochs"] = decoded_ste_metric_warmup_epochs
        self.my_dict["decoded_ste_metric_max_weight"]   = decoded_ste_metric_max_weight
        self.my_dict["porto_opq_warmup_train_recon_path"] = porto_opq_warmup_train_recon_path
        self.my_dict["porto_opq_warmup_start_epoch"]   = porto_opq_warmup_start_epoch
        self.my_dict["porto_opq_warmup_end_epoch"]     = porto_opq_warmup_end_epoch
        self.my_dict["porto_opq_warmup_max_weight"]    = porto_opq_warmup_max_weight
        self.my_dict["porto_opq_teacher_rotated_train_path"] = porto_opq_teacher_rotated_train_path
        self.my_dict["porto_opq_teacher_codebook_path"] = porto_opq_teacher_codebook_path
        self.my_dict["porto_opq_teacher_start_epoch"]  = porto_opq_teacher_start_epoch
        self.my_dict["porto_opq_teacher_end_epoch"]    = porto_opq_teacher_end_epoch
        self.my_dict["porto_opq_teacher_z_weight"]     = porto_opq_teacher_z_weight
        self.my_dict["porto_opq_teacher_partition_weight"] = porto_opq_teacher_partition_weight
        self.my_dict["porto_opq_teacher_codebook_freeze_end_epoch"] = porto_opq_teacher_codebook_freeze_end_epoch
        self.my_dict["porto_opq_teacher_realign_codebook_on_unfreeze"] = porto_opq_teacher_realign_codebook_on_unfreeze
        self.my_dict["late_finetune_start_epoch"]      = late_finetune_start_epoch
        self.my_dict["late_finetune_main_lr_scale"]    = late_finetune_main_lr_scale
        self.my_dict["late_finetune_pre_quant_lr_scale"] = late_finetune_pre_quant_lr_scale
        self.my_dict["max_train_batches_per_epoch"]    = max_train_batches_per_epoch
        self.my_dict["triplet_pos_begin_pos"]          = triplet_pos_begin_pos
        self.my_dict["triplet_pos_end_pos"]            = triplet_pos_end_pos
        self.my_dict["triplet_neg_begin_pos"]          = triplet_neg_begin_pos
        self.my_dict["triplet_neg_end_pos"]            = triplet_neg_end_pos
        self.my_dict["train_ratio"]                    = train_ratio
        self.my_dict["mode"]                           = mode
        self.my_dict["test_epoch"]                     = test_epoch
        self.my_dict["print_epoch"]                    = print_epoch
        self.my_dict["save_model_epoch"]               = save_model_epoch
        self.my_dict["save_model"]                     = save_model
        self.my_dict["eval_save_epochs"]               = eval_save_epochs
        self.my_dict["save_feature_distance"]          = save_feature_distance
        self.my_dict["report_only_eval"]               = report_only_eval
        self.my_dict["save_model_path"]                = save_model_path
        self.my_dict["dist_type"]                      = dist_type
        self.my_dict["device"]                         = device
        self.my_dict["LDS"]                            = LDS
        self.my_dict["FDS"]                            = FDS
        self.my_dict["train_flag"]                     = train_flag
        self.my_dict["head_num"]                       = head_num
        self.my_dict["area_path"]                      = area_path
        self.my_dict["train_set"]                      = train_set
        self.my_dict["query_set"]                      = query_set
        self.my_dict["base_set"]                       = base_set
        self.write_config_to_file()
        
    def write_config_to_file(self):
        train_config_dir = os.path.join(self.my_dict["root_write_path"], "train_config")
        os.makedirs(train_config_dir, exist_ok=True)
        with open(os.path.join(train_config_dir, self.my_dict["train_flag"] + ".json"), "w", encoding = "utf-8") as f:
            json.dump(self.my_dict, f, ensure_ascii = False, cls = MyEncoder)
            

    def read_config_from_file(self):
        self.my_dict = {}
        with open(self.my_dict["root_write_path"] + "/train_config/" + self.my_dict["train_flag"] + ".json", "r", encoding = "utf-8") as f:
            self.my_dict = json.load(f)
