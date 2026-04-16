import os
import time
import pickle
import numpy as np
import sys
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PDT_ROOT = REPO_ROOT / "PDT_VQ"
for candidate in (REPO_ROOT, PDT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)
from mynn.transformer import Transformer
from mynn.model import (
    build_batch_neighbor_mask_from_global_knn,
    landmark_profile_loss,
    neighborhood_consistency_loss,
    neighborhood_consistency_loss_from_mask,
    normalize_landmark_profile,
    resolve_embedding_distance_fn,
)
from PDT_VQ.model import PDT
from PDT_VQ.train import get_args
import torch

from tools import sampling_methods
from tools import test_methods
from tools import function
from tools import torch_feature_distance
from tools import pre_rep
import json
from PDT_VQ.utils.engine import compute_fixed_codebooks, reconstruct_from_fixed_codebooks


class NeuTrajTrainer(object):
    def __init__(self, my_config):
        self.my_config = my_config
        self.pre_quant_neighbor_artifact = None
        self.pre_quant_landmark_artifact = None
        self.pre_quant_landmark_bank = None

    def _parse_epoch_list(self, value):
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {int(v) for v in value}
        text = str(value).strip()
        if not text:
            return set()
        result = set()
        for item in text.split(","):
            item = item.strip()
            if item:
                result.add(int(item))
        return result

    def _linear_warmup_weight(self, epoch, start_epoch, warmup_epochs, max_weight):
        if epoch < start_epoch:
            return 0.0
        if warmup_epochs <= 0:
            return max_weight
        progress = min(1.0, float(epoch - start_epoch + 1) / float(warmup_epochs))
        return max_weight * progress

    def _optimizer_phase(self, epoch):
        late_finetune_start_epoch = int(self.my_config.my_dict.get("late_finetune_start_epoch", -1))
        if late_finetune_start_epoch >= 0 and epoch >= late_finetune_start_epoch:
            return "late_finetune"
        return "base"

    def _optimizer_lr_scales(self, epoch):
        phase = self._optimizer_phase(epoch)
        main_lr_scale = 1.0
        pre_quant_lr_scale = 1.0
        if phase == "late_finetune":
            main_lr_scale = float(self.my_config.my_dict.get("late_finetune_main_lr_scale", 1.0))
            pre_quant_lr_scale = float(self.my_config.my_dict.get("late_finetune_pre_quant_lr_scale", 1.0))
        return phase, main_lr_scale, pre_quant_lr_scale

    def _build_optimizer(self, my_net, epoch=0, rebuild_reason=""):
        base_lr = float(self.my_config.my_dict["learning_rate"])
        weight_decay = 1e-4
        pre_quant_lr_multiplier = float(self.my_config.my_dict.get("pre_quant_lr_multiplier", 1.0))
        use_pre_quant_group = bool(getattr(my_net, "pre_quant_bottleneck_enabled", False))
        optimizer_phase, main_lr_scale, pre_quant_lr_scale = self._optimizer_lr_scales(int(epoch))
        main_lr = base_lr * main_lr_scale
        pre_quant_lr = base_lr * pre_quant_lr_multiplier * pre_quant_lr_scale

        main_params = []
        pre_quant_params = []
        for name, parameter in my_net.named_parameters():
            if not parameter.requires_grad:
                continue
            if use_pre_quant_group and name.startswith("pre_quant_bottleneck."):
                pre_quant_params.append(parameter)
            else:
                main_params.append(parameter)

        if not main_params and not pre_quant_params:
            raise ValueError("No trainable parameters are left for the optimizer.")
        optimizer_groups = []
        if main_params:
            optimizer_groups.append({
                "params": main_params,
                "lr": main_lr,
                "weight_decay": weight_decay,
            })
        if pre_quant_params:
            optimizer_groups.append({
                "params": pre_quant_params,
                "lr": pre_quant_lr,
                "weight_decay": weight_decay,
            })
        print("[Optimizer] epoch={} phase={} reason={} | groups: main={} @ {:.6f}, pre_quant={} @ {:.6f}".format(
            int(epoch),
            optimizer_phase,
            rebuild_reason or "default",
            len(main_params),
            main_lr,
            len(pre_quant_params),
            pre_quant_lr if pre_quant_params else 0.0,
        ))
        return torch.optim.Adam(optimizer_groups)

    def _set_backbone_trainable(self, my_net, freeze_backbone):
        trainable_names = []
        for name, parameter in my_net.named_parameters():
            is_pdt_param = name.startswith("PDT_model.")
            parameter.requires_grad = is_pdt_param or (not freeze_backbone)
            if parameter.requires_grad:
                trainable_names.append(name)
        stage_name = "pdt_only" if freeze_backbone else "joint"
        print("[TrainStage] {} | trainable params: {}".format(stage_name, len(trainable_names)))
        return stage_name

    def _collect_embedding_outputs(self,
                                   my_net,
                                   lon_tensor,
                                   lat_tensor,
                                   image_tensor,
                                   seq_len_list,
                                   test_batch,
                                   lon_grid_tensor=None,
                                   lat_grid_tensor=None,
                                   embedding_type="quantized",
                                   collect_code_usage=False,
                                   collect_transformed=False,
                                   collect_reconstructed=False,
                                   collect_raw_continuous=False):
        my_net.eval()
        outputs = {}
        continuous_list = []
        raw_continuous_list = []
        quantized_list = []
        codes_list = []
        transformed_list = []
        reconstructed_list = []
        start_time = time.time()
        total_num = lon_tensor.shape[0]
        with torch.no_grad():
            for i in range(0, total_num, test_batch):
                input_lon_onehot_tensor = lon_tensor[i: i + test_batch].to(self.my_config.my_dict["device"])
                input_lat_onehot_tensor = lat_tensor[i: i + test_batch].to(self.my_config.my_dict["device"])
                input_lon_lat_image_tensor = image_tensor[i: i + test_batch].to(self.my_config.my_dict["device"])
                input_seq_lengths = seq_len_list[i: i + test_batch]
                input_lon_grid_tensor = None if lon_grid_tensor is None else lon_grid_tensor[i: i + test_batch].to(self.my_config.my_dict["device"])
                input_lat_grid_tensor = None if lat_grid_tensor is None else lat_grid_tensor[i: i + test_batch].to(self.my_config.my_dict["device"])

                inference_outputs = my_net.inference_continuous(input_lon_onehot_tensor,
                                                                input_lat_onehot_tensor,
                                                                input_lon_lat_image_tensor,
                                                                seq_lengths=input_seq_lengths,
                                                                grid_lon=input_lon_grid_tensor,
                                                                grid_lat=input_lat_grid_tensor,
                                                                return_intermediate=True)
                continuous_embedding = inference_outputs["e_for_pdt"]
                if embedding_type in ["continuous", "both"] or collect_code_usage:
                    continuous_list.append(continuous_embedding.detach().cpu())
                if collect_raw_continuous:
                    raw_continuous_list.append(inference_outputs["e_cont"].detach().cpu())
                need_quantized_details = (
                    embedding_type in ["quantized", "both"]
                    or collect_code_usage
                    or collect_transformed
                    or collect_reconstructed
                )
                if need_quantized_details:
                    quantized_details = my_net.quantize_embeddings_with_details(continuous_embedding)
                    if embedding_type in ["quantized", "both"]:
                        quantized_list.append(quantized_details["e_hat"].detach().cpu())
                    if collect_transformed:
                        transformed_list.append(quantized_details["z"].detach().cpu())
                    if collect_code_usage:
                        codes_list.append(quantized_details["codes"].detach().cpu())
                    if collect_reconstructed:
                        reconstructed_list.append(quantized_details["z_hat"].detach().cpu())
        end_time = time.time()

        if continuous_list:
            outputs["continuous"] = torch.cat(continuous_list, dim=0)
        if raw_continuous_list:
            outputs["continuous_raw"] = torch.cat(raw_continuous_list, dim=0)
        if quantized_list:
            outputs["quantized"] = torch.cat(quantized_list, dim=0)
        if codes_list:
            outputs["codes"] = torch.cat(codes_list, dim=0)
        if transformed_list:
            outputs["transformed"] = torch.cat(transformed_list, dim=0)
        if reconstructed_list:
            outputs["reconstructed"] = torch.cat(reconstructed_list, dim=0)

        output_shapes = {key: tuple(value.shape) for key, value in outputs.items()}
        print("Embedding collection time: {:.4f}, outputs: {}".format(end_time - start_time, output_shapes))
        return outputs

    def _pad_eval_sequence_array(self, seq_list, max_length):
        if len(seq_list) == 0:
            return np.zeros((0, max_length, 1), dtype=np.float32)
        feature_dim = len(seq_list[0][0])
        padded = np.zeros((len(seq_list), max_length, feature_dim), dtype=np.float32)
        for idx, seq in enumerate(seq_list):
            seq_array = np.asarray(seq, dtype=np.float32)
            seq_len = seq_array.shape[0]
            if seq_len > max_length:
                raise ValueError("Sequence length {} is larger than max_length {}".format(seq_len, max_length))
            padded[idx, :seq_len] = seq_array
        return padded

    def _eval_preproc_cache_path(self, begin_pos, end_pos):
        cache_root = Path(self.my_config.my_dict["root_write_path"]) / "eval_preproc_cache"
        root_read_path = str(self.my_config.my_dict.get("root_read_path", ""))
        cache_signature = "|".join([
            str(self.my_config.my_dict.get("dataset", "")),
            root_read_path,
            str(self.my_config.my_dict.get("image_mode", "")),
            str(self.my_config.my_dict.get("lon_input_size", "")),
            str(self.my_config.my_dict.get("lat_input_size", "")),
            str(self.max_traj_length),
            str(len(self.traj_list)),
        ])
        cache_key = hashlib.sha1(cache_signature.encode("utf-8")).hexdigest()[:16]
        cache_dir = cache_root / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "chunk_{}_{}.npz".format(begin_pos, end_pos)

    def _load_eval_preproc_cache(self, begin_pos, end_pos):
        cache_path = self._eval_preproc_cache_path(begin_pos, end_pos)
        if not cache_path.exists():
            return None
        with np.load(cache_path, allow_pickle=False) as cache_data:
            print("[EvalCache] hit {}".format(cache_path))
            return (
                torch.from_numpy(cache_data["pad_lon"]),
                torch.from_numpy(cache_data["pad_lat"]),
                torch.from_numpy(cache_data["pad_img"]),
                cache_data["seq_len"].tolist(),
                torch.from_numpy(cache_data["pad_lon_grid"]),
                torch.from_numpy(cache_data["pad_lat_grid"]),
            )

    def _save_eval_preproc_cache(self, begin_pos, end_pos, tensors):
        cache_path = self._eval_preproc_cache_path(begin_pos, end_pos)
        if cache_path.exists():
            return
        pad_lon, pad_lat, pad_img, seq_len_list, pad_lon_grid, pad_lat_grid = tensors
        np.savez_compressed(
            cache_path,
            pad_lon=np.asarray(pad_lon, dtype=np.float32),
            pad_lat=np.asarray(pad_lat, dtype=np.float32),
            pad_img=np.asarray(pad_img, dtype=np.float32),
            seq_len=np.asarray(seq_len_list, dtype=np.int32),
            pad_lon_grid=np.asarray(pad_lon_grid, dtype=np.float32),
            pad_lat_grid=np.asarray(pad_lat_grid, dtype=np.float32),
        )
        print("[EvalCache] saved {}".format(cache_path))

    def _split_eval_array(self, values):
        train_set = self.my_config.my_dict["train_set"]
        query_set = self.my_config.my_dict["query_set"]
        query_values = values[train_set:train_set + query_set]
        base_values = values[train_set + query_set:]
        return query_values, base_values

    def _compute_adc_pred_knn(self, codebook, transformed_embeddings, codes, topk):
        if self.my_config.my_dict.get("pdt_vq_type", "dpq") != "dpq":
            raise ValueError("ADC evaluation is only supported for DPQ currently.")

        query_trans, _ = self._split_eval_array(transformed_embeddings)
        _, base_codes = self._split_eval_array(codes)

        query_trans = np.asarray(query_trans, dtype=np.float32)
        base_codes = np.asarray(base_codes, dtype=np.int32)
        codebook = np.asarray(codebook, dtype=np.float32)

        nq = query_trans.shape[0]
        nb = base_codes.shape[0]
        M, K, dsub = codebook.shape
        query_groups = query_trans.reshape(nq, M, dsub)
        adc_distance = np.zeros((nq, nb), dtype=np.float32)

        for group_idx in range(M):
            query_group = query_groups[:, group_idx, :]
            centroids = codebook[group_idx]
            query_norm = np.sum(query_group ** 2, axis=1, keepdims=True)
            centroid_norm = np.sum(centroids ** 2, axis=1, keepdims=True).T
            lookup = query_norm - 2.0 * np.matmul(query_group, centroids.T) + centroid_norm
            adc_distance += lookup[:, base_codes[:, group_idx]]

        pred_knn = np.argsort(adc_distance, axis=1)[:, :topk]
        return pred_knn

    def _compute_rerank_pred_knn(self, query_embeddings, base_embeddings, coarse_pred_knn, rerank_L):
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        base_embeddings = np.asarray(base_embeddings, dtype=np.float32)
        coarse_pred_knn = np.asarray(coarse_pred_knn, dtype=np.int32)
        effective_rerank_L = min(int(rerank_L), coarse_pred_knn.shape[1])
        shortlist = coarse_pred_knn[:, :effective_rerank_L]
        rerank_return_k = min(100, shortlist.shape[1])
        reranked_pred = np.empty((shortlist.shape[0], rerank_return_k), dtype=np.int32)
        batch_size = 64
        for begin in range(0, shortlist.shape[0], batch_size):
            end = min(begin + batch_size, shortlist.shape[0])
            batch_shortlist = shortlist[begin:end]
            candidate_embeddings = base_embeddings[batch_shortlist]
            batch_queries = query_embeddings[begin:end][:, None, :]
            distances = np.sum((candidate_embeddings - batch_queries) ** 2, axis=2)
            order = np.argsort(distances, axis=1)[:, :rerank_return_k]
            reranked_pred[begin:end] = np.take_along_axis(batch_shortlist, order, axis=1)
        return reranked_pred

    def _compute_pdtvq_coarse_pred_knn(self,
                                       transformed_embeddings,
                                       reconstructed_embeddings,
                                       codes,
                                       rerank_L,
                                       pdt_vq_type):
        query_transformed, base_transformed = self._split_eval_array(transformed_embeddings)
        _, base_reconstructed_model = self._split_eval_array(reconstructed_embeddings)
        _, base_codes = self._split_eval_array(codes)

        if pdt_vq_type == "qinco":
            train_set = self.my_config.my_dict["train_set"]
            train_transformed = np.asarray(transformed_embeddings[:train_set], dtype=np.float32)
            train_codes = np.asarray(codes[:train_set], dtype=np.int32)
            fixed_codebook = compute_fixed_codebooks(train_transformed, train_codes, k=int(self.my_config.my_dict.get("pdt_k", 256)))
            base_coarse = reconstruct_from_fixed_codebooks(np.asarray(base_codes, dtype=np.int32), fixed_codebook)
        else:
            base_coarse = np.asarray(base_reconstructed_model, dtype=np.float32)

        coarse_distance = test_methods.get_feature_distance(np.asarray(query_transformed, dtype=np.float32), base_coarse)
        coarse_pred_knn = np.argsort(coarse_distance, axis=1)[:, :max(100, rerank_L)]
        return coarse_pred_knn, np.asarray(base_reconstructed_model, dtype=np.float32)

    def _compute_pdtvq_rerank_pred_knn(self,
                                       query_embeddings,
                                       base_reconstructed_embeddings,
                                       coarse_pred_knn,
                                       decoder,
                                       rerank_L,
                                       decode_step,
                                       batch_size):
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        base_reconstructed_embeddings = np.asarray(base_reconstructed_embeddings, dtype=np.float32)
        coarse_pred_knn = np.asarray(coarse_pred_knn, dtype=np.int32)
        effective_rerank_L = min(int(rerank_L), coarse_pred_knn.shape[1])
        shortlist = coarse_pred_knn[:, :effective_rerank_L]
        rerank_return_k = min(100, shortlist.shape[1])
        reranked_pred = np.empty((shortlist.shape[0], rerank_return_k), dtype=np.int32)
        one_step_size = max(1, int(batch_size / max(1, effective_rerank_L)))
        device = self.my_config.my_dict["device"]

        for begin in range(0, shortlist.shape[0], one_step_size):
            end = min(begin + one_step_size, shortlist.shape[0])
            batch_shortlist = shortlist[begin:end]
            selected_queries = torch.from_numpy(query_embeddings[begin:end]).to(device)
            selected_reconstructed = torch.from_numpy(base_reconstructed_embeddings[batch_shortlist]).to(device)
            selected_reconstructed = selected_reconstructed.reshape(-1, selected_reconstructed.shape[-1])

            decoded_candidates = []
            with torch.no_grad():
                for offset in range(0, selected_reconstructed.shape[0], batch_size):
                    decoder_out = decoder(selected_reconstructed[offset: offset + batch_size], decode_step)
                    if isinstance(decoder_out, (list, tuple)):
                        decoder_out = decoder_out[-1]
                    decoded_candidates.append(decoder_out)
            decoded_candidates = torch.cat(decoded_candidates, dim=0)
            decoded_candidates = decoded_candidates.reshape(end - begin, effective_rerank_L, -1)
            dist = torch.norm(decoded_candidates - selected_queries[:, None, :], p=2, dim=2)
            order = torch.argsort(dist, dim=1)[:, :rerank_return_k].cpu().numpy()
            reranked_pred[begin:end] = np.take_along_axis(batch_shortlist, order, axis=1)

        return reranked_pred

    def extract_feature(self):
        return self.extract_feature_from_path(lon_grid_id_list=self.lon_grid_id_list,
                                              lat_grid_id_list=self.lat_grid_id_list)

    def _save_checkpoint(self, my_net, epoch_num):
        if self._report_only_eval():
            print("[Checkpoint] skip save at epoch {} (report_only_eval)".format(epoch_num))
            return None
        save_model_name = self._checkpoint_path(epoch_num, self.my_config.my_dict["train_flag"])
        torch.save(my_net.state_dict(), save_model_name)
        print("[Checkpoint] saved epoch {} to {}".format(epoch_num, save_model_name))
        return save_model_name

    def _summarize_code_usage(self, codes):
        codes_np = codes.cpu().numpy()
        pdt_k = int(self.my_config.my_dict.get("pdt_k", 256))
        unique_tuple_count = int(np.unique(codes_np, axis=0).shape[0])
        unique_per_group = [int(np.unique(codes_np[:, group_idx]).shape[0]) for group_idx in range(codes_np.shape[1])]
        usage_ratio_per_group = [float(unique_count) / float(pdt_k) for unique_count in unique_per_group]
        code_usage = {
            "num_samples": int(codes_np.shape[0]),
            "num_groups": int(codes_np.shape[1]),
            "codebook_size": pdt_k,
            "unique_tuple_count": unique_tuple_count,
            "unique_tuple_ratio": float(unique_tuple_count) / float(max(1, codes_np.shape[0])),
            "unique_codes_per_group": unique_per_group,
            "usage_ratio_per_group": usage_ratio_per_group,
        }
        print("[CodeUsage] unique tuples: {} / {}, per-group unique codes: {}".format(unique_tuple_count,
                                                                                       codes_np.shape[0],
                                                                                       unique_per_group))
        return code_usage

    def _initialize_pdt_codebook(self, my_net, init_limit=None):
        if not self.my_config.my_dict.get("pdt_init_codebook", True):
            print("[PDT] Skipping codebook initialization.")
            return
        print("[PDT] Initializing codebook from current train embeddings...")
        init_batch_size = min(16, int(self.my_config.my_dict.get("batch_size", 256)))
        init_limit = self.final_traj_train_num if init_limit is None else min(int(init_limit), self.final_traj_train_num)
        init_outputs = self._collect_embedding_outputs(my_net,
                                                       self.pad_total_train_lon_onehot[:init_limit],
                                                       self.pad_total_train_lat_onehot[:init_limit],
                                                       self.pad_total_train_lon_lat_image[:init_limit],
                                                       self.train_traj_length_list[:init_limit],
                                                       lon_grid_tensor=self.pad_total_train_lon_grid[:init_limit],
                                                       lat_grid_tensor=self.pad_total_train_lat_grid[:init_limit],
                                                       test_batch=init_batch_size,
                                                       embedding_type="continuous",
                                                       collect_code_usage=False)
        init_embeddings = init_outputs["continuous"].to(self.my_config.my_dict["device"])
        with torch.no_grad():
            my_net.PDT_model.init_codebook(init_embeddings, my_net.pdt_args.resume)
        print("[PDT] Codebook initialization finished with embeddings shape {}.".format(tuple(init_embeddings.shape)))

    def _reset_pdt_quantizer_temperatures(self, my_net):
        quantizer = getattr(getattr(my_net.PDT_model, "vq", None), "quantizer", None)
        if quantizer is not None and hasattr(quantizer, "log_temperatures"):
            with torch.no_grad():
                quantizer.log_temperatures.zero_()
            print("[PDT] Reset quantizer log temperatures to zero.")

    def _refresh_pdt_codebook(self, my_net, epoch):
        print("[PDT] Refreshing codebook from current train embeddings at epoch {}...".format(epoch))
        self._initialize_pdt_codebook(my_net)
        self._reset_pdt_quantizer_temperatures(my_net)
        print("[PDT] Refresh complete at epoch {}.".format(epoch))

    def _build_eval_tensors(self, begin_pos, end_pos):
        cached = self._load_eval_preproc_cache(begin_pos, end_pos)
        if cached is not None:
            return cached
        pad_total_lon_onehot = self._pad_eval_sequence_array(self.lon_onehot[begin_pos:end_pos], self.max_traj_length)
        pad_total_lat_onehot = self._pad_eval_sequence_array(self.lat_onehot[begin_pos:end_pos], self.max_traj_length)
        lon_grid_seq = [[[value] for value in traj] for traj in self.lon_grid_id_list[begin_pos:end_pos]]
        lat_grid_seq = [[[value] for value in traj] for traj in self.lat_grid_id_list[begin_pos:end_pos]]
        pad_total_lon_grid = self._pad_eval_sequence_array(lon_grid_seq, self.max_traj_length)
        pad_total_lat_grid = self._pad_eval_sequence_array(lat_grid_seq, self.max_traj_length)
        if self.my_config.my_dict.get("embedding_backbone", "msr") == "msr":
            pad_total_lon_lat_image = pre_rep.build_traj_image(
                self.lon_grid_id_list[begin_pos:end_pos],
                self.lat_grid_id_list[begin_pos:end_pos],
                self.my_config.my_dict["lon_input_size"],
                self.my_config.my_dict["lat_input_size"],
                image_mode=self.my_config.my_dict.get("image_mode", "binary"),
                traj_list=self.traj_list[begin_pos:end_pos],
            )
        else:
            pad_total_lon_lat_image = np.zeros((end_pos - begin_pos, 1, 1, 1), dtype=np.float32)
        seq_len_list = self.traj_length_list[begin_pos:end_pos]
        tensors = (
            torch.from_numpy(np.asarray(pad_total_lon_onehot, dtype=np.float32)),
            torch.from_numpy(np.asarray(pad_total_lat_onehot, dtype=np.float32)),
            torch.from_numpy(np.asarray(pad_total_lon_lat_image, dtype=np.float32)),
            seq_len_list,
            torch.from_numpy(np.asarray(pad_total_lon_grid, dtype=np.float32)),
            torch.from_numpy(np.asarray(pad_total_lat_grid, dtype=np.float32)),
        )
        self._save_eval_preproc_cache(begin_pos, end_pos, tensors)
        return tensors

    def _build_training_state(self):
        freeze_backbone_epochs = int(self.my_config.my_dict.get("freeze_backbone_epochs", 0))
        pdt_loss_start_epoch = int(self.my_config.my_dict.get("pdt_loss_start_epoch", 20))
        pdt_loss_weight = float(self.my_config.my_dict.get("pdt_loss_weight", 0.3))
        loss_recipe = self.my_config.my_dict.get("loss_recipe", "baseline")
        consistency_weight = float(self.my_config.my_dict.get("consistency_weight", 0.1))
        quantized_metric_weight = float(self.my_config.my_dict.get("quantized_metric_weight", 0.3))
        improved_qm_start_epoch = int(self.my_config.my_dict.get("improved_qm_start_epoch", 60))
        improved_qm_warmup_epochs = int(self.my_config.my_dict.get("improved_qm_warmup_epochs", 80))
        improved_qm_max_weight = float(self.my_config.my_dict.get("improved_qm_max_weight", 0.12))
        improved_pairwise_weight = float(self.my_config.my_dict.get("improved_pairwise_weight", 0.05))
        improved_entropy_weight = float(self.my_config.my_dict.get("improved_entropy_weight", 0.01))
        improved_commit_weight = float(self.my_config.my_dict.get("improved_commit_weight", 0.02))
        improved_uniform_weight = float(self.my_config.my_dict.get("improved_uniform_weight", 0.001))
        improved_vq_adaptive_low_codebook = bool(self.my_config.my_dict.get("improved_vq_adaptive_low_codebook", True))
        pdt_m = int(self.my_config.my_dict.get("pdt_m", 16))
        base_low_codebook_risk = max(0.0, (12.0 - float(pdt_m)) / 8.0) if pdt_m < 12 else 0.0
        adaptive_qm_start_epoch = improved_qm_start_epoch
        adaptive_qm_max_weight = improved_qm_max_weight
        adaptive_pairwise_weight = improved_pairwise_weight
        adaptive_entropy_weight = improved_entropy_weight
        adaptive_commit_weight = improved_commit_weight
        adaptive_uniform_weight = improved_uniform_weight
        low_codebook_risk = 0.0
        medium_codebook_mode = False
        high_capacity_mode = False
        effective_pdt_loss_start_epoch = pdt_loss_start_epoch
        effective_pdt_loss_weight = pdt_loss_weight
        high_capacity_reg_scale = 1.0
        low_m_refresh_epochs = set()
        if improved_vq_adaptive_low_codebook:
            low_codebook_risk = base_low_codebook_risk
            adaptive_qm_start_epoch = improved_qm_start_epoch + int(40 * low_codebook_risk)
            adaptive_qm_max_weight = improved_qm_max_weight * (1.0 - 0.75 * low_codebook_risk)
            adaptive_pairwise_weight = improved_pairwise_weight * (1.0 - 0.5 * low_codebook_risk)
            adaptive_entropy_weight = improved_entropy_weight * (1.0 + 3.0 * low_codebook_risk)
            adaptive_commit_weight = improved_commit_weight * (1.0 + 1.5 * low_codebook_risk)
            adaptive_uniform_weight = improved_uniform_weight * (1.0 + 5.0 * low_codebook_risk)
            medium_codebook_mode = (loss_recipe == "improved_vq" and pdt_m >= 8 and pdt_m < 12)
            high_capacity_mode = (loss_recipe == "improved_vq" and pdt_m >= 12)
            if medium_codebook_mode:
                effective_pdt_loss_start_epoch = min(pdt_loss_start_epoch, 20)
                effective_pdt_loss_weight = max(pdt_loss_weight, 0.3)
            elif high_capacity_mode:
                effective_pdt_loss_start_epoch = min(pdt_loss_start_epoch, 20)
                effective_pdt_loss_weight = max(pdt_loss_weight, 0.3)
                high_capacity_reg_scale = 0.25
            if loss_recipe == "improved_vq" and low_codebook_risk > 0.0:
                low_m_refresh_epochs = set(range(20, max(21, adaptive_qm_start_epoch + 1), 20))

        pre_quant_enabled = bool(self.my_config.my_dict.get("pre_quant_bottleneck_enabled", False))
        pre_quant_lambda_decor = float(self.my_config.my_dict.get("pre_quant_lambda_decor", 0.01))
        pre_quant_lambda_stab = float(self.my_config.my_dict.get("pre_quant_lambda_stab", 0.1))
        pre_quant_stab_late_epoch = int(self.my_config.my_dict.get("pre_quant_stab_late_epoch", 100))
        pre_quant_stab_late_multiplier = float(self.my_config.my_dict.get("pre_quant_stab_late_multiplier", 1.0))
        pre_quant_refresh_start_epoch = int(self.my_config.my_dict.get("pre_quant_refresh_start_epoch", 0))
        pre_quant_refresh_period = int(self.my_config.my_dict.get("pre_quant_refresh_period", 0))
        pre_quant_refresh_end_epoch = int(self.my_config.my_dict.get("pre_quant_refresh_end_epoch", 0))
        pre_quant_raw_metric_weight = float(self.my_config.my_dict.get("pre_quant_raw_metric_weight", 0.0))
        pre_quant_neighbor_use_btn = bool(self.my_config.my_dict.get("pre_quant_neighbor_use_btn", False))
        pre_quant_neighbor_use_dec = bool(self.my_config.my_dict.get("pre_quant_neighbor_use_dec", False))
        pre_quant_neighbor_enabled = bool(self.my_config.my_dict.get("pre_quant_neighbor_enabled", False))
        pre_quant_neighbor_enabled = pre_quant_enabled and pre_quant_neighbor_enabled and (pre_quant_neighbor_use_btn or pre_quant_neighbor_use_dec)
        pre_quant_neighbor_topk = int(self.my_config.my_dict.get("pre_quant_neighbor_topk", 10))
        pre_quant_neighbor_tau_btn = float(self.my_config.my_dict.get("pre_quant_neighbor_tau_btn", 0.07))
        pre_quant_neighbor_tau_dec = float(self.my_config.my_dict.get("pre_quant_neighbor_tau_dec", 0.07))
        pre_quant_neighbor_lambda_btn = float(self.my_config.my_dict.get("pre_quant_neighbor_lambda_btn", 0.05))
        pre_quant_neighbor_lambda_dec = float(self.my_config.my_dict.get("pre_quant_neighbor_lambda_dec", 0.10))
        pre_quant_neighbor_start_epoch = int(self.my_config.my_dict.get("pre_quant_neighbor_start_epoch", 0))
        pre_quant_neighbor_warmup_epochs = int(self.my_config.my_dict.get("pre_quant_neighbor_warmup_epochs", 0))
        pre_quant_neighbor_dec_stop_backbone = bool(self.my_config.my_dict.get("pre_quant_neighbor_dec_stop_backbone", False))
        pre_quant_neighbor_teacher_mode = str(self.my_config.my_dict.get("pre_quant_neighbor_teacher_mode", "batch_cont"))
        pre_quant_neighbor_offline_path = str(self.my_config.my_dict.get("pre_quant_neighbor_offline_path", "") or "")
        pre_quant_landmark_use_btn = bool(self.my_config.my_dict.get("pre_quant_landmark_use_btn", False))
        pre_quant_landmark_use_dec = bool(self.my_config.my_dict.get("pre_quant_landmark_use_dec", True))
        pre_quant_landmark_enabled = bool(self.my_config.my_dict.get("pre_quant_landmark_enabled", False))
        pre_quant_landmark_enabled = pre_quant_enabled and pre_quant_landmark_enabled and (pre_quant_landmark_use_btn or pre_quant_landmark_use_dec)
        pre_quant_landmark_num = int(self.my_config.my_dict.get("pre_quant_landmark_num", 64))
        pre_quant_landmark_select = str(self.my_config.my_dict.get("pre_quant_landmark_select", "fps"))
        pre_quant_landmark_profile_transform = str(self.my_config.my_dict.get("pre_quant_landmark_profile_transform", "log1p_zscore"))
        pre_quant_landmark_lambda_btn = float(self.my_config.my_dict.get("pre_quant_landmark_lambda_btn", 0.03))
        pre_quant_landmark_lambda_dec = float(self.my_config.my_dict.get("pre_quant_landmark_lambda_dec", 0.10))
        pre_quant_landmark_dec_ste_to_btn = bool(self.my_config.my_dict.get("pre_quant_landmark_dec_ste_to_btn", False))
        pre_quant_landmark_dec_bank_source = str(self.my_config.my_dict.get("pre_quant_landmark_dec_bank_source", "decoded"))
        pre_quant_landmark_rank_weight = float(self.my_config.my_dict.get("pre_quant_landmark_rank_weight", 0.0))
        pre_quant_landmark_rank_tau = float(self.my_config.my_dict.get("pre_quant_landmark_rank_tau", 0.5))
        pre_quant_landmark_start_epoch = int(self.my_config.my_dict.get("pre_quant_landmark_start_epoch", 100))
        pre_quant_landmark_warmup_epochs = int(self.my_config.my_dict.get("pre_quant_landmark_warmup_epochs", 20))
        pre_quant_landmark_bank_refresh_epochs = int(self.my_config.my_dict.get("pre_quant_landmark_bank_refresh_epochs", 10))
        pre_quant_landmark_artifact_path = str(self.my_config.my_dict.get("pre_quant_landmark_artifact_path", "") or "")
        pre_quant_landmark_fixed_bank_path = str(self.my_config.my_dict.get("pre_quant_landmark_fixed_bank_path", "") or "")
        pre_quant_landmark_teacher_checkpoint = str(self.my_config.my_dict.get("pre_quant_landmark_teacher_checkpoint", "") or "")
        decoded_ste_metric_enabled = bool(self.my_config.my_dict.get("decoded_ste_metric_enabled", False))
        decoded_ste_metric_start_epoch = int(self.my_config.my_dict.get("decoded_ste_metric_start_epoch", 80))
        decoded_ste_metric_warmup_epochs = int(self.my_config.my_dict.get("decoded_ste_metric_warmup_epochs", 20))
        decoded_ste_metric_max_weight = float(self.my_config.my_dict.get("decoded_ste_metric_max_weight", 0.03))
        effective_refresh_end_epoch = int(self.my_config.my_dict["epoch_num"])
        if pre_quant_refresh_end_epoch > 0:
            effective_refresh_end_epoch = min(pre_quant_refresh_end_epoch, effective_refresh_end_epoch)
        pre_quant_refresh_epochs = set()
        if pre_quant_enabled and pre_quant_refresh_period > 0 and effective_refresh_end_epoch > pre_quant_refresh_start_epoch:
            pre_quant_refresh_epochs = set(range(pre_quant_refresh_start_epoch,
                                                 effective_refresh_end_epoch,
                                                 pre_quant_refresh_period))
        all_refresh_epochs = low_m_refresh_epochs.union(pre_quant_refresh_epochs)
        late_finetune_start_epoch = int(self.my_config.my_dict.get("late_finetune_start_epoch", -1))
        late_finetune_main_lr_scale = float(self.my_config.my_dict.get("late_finetune_main_lr_scale", 1.0))
        late_finetune_pre_quant_lr_scale = float(self.my_config.my_dict.get("late_finetune_pre_quant_lr_scale", 1.0))
        late_finetune_enabled = late_finetune_start_epoch >= 0 and late_finetune_start_epoch < int(self.my_config.my_dict["epoch_num"])

        return {
            "freeze_backbone_epochs": freeze_backbone_epochs,
            "loss_recipe": loss_recipe,
            "consistency_weight": consistency_weight,
            "quantized_metric_weight": quantized_metric_weight,
            "improved_qm_warmup_epochs": improved_qm_warmup_epochs,
            "improved_entropy_weight": improved_entropy_weight,
            "improved_commit_weight": improved_commit_weight,
            "improved_uniform_weight": improved_uniform_weight,
            "improved_vq_adaptive_low_codebook": improved_vq_adaptive_low_codebook,
            "adaptive_qm_start_epoch": adaptive_qm_start_epoch,
            "adaptive_qm_max_weight": adaptive_qm_max_weight,
            "adaptive_pairwise_weight": adaptive_pairwise_weight,
            "adaptive_entropy_weight": adaptive_entropy_weight,
            "adaptive_commit_weight": adaptive_commit_weight,
            "adaptive_uniform_weight": adaptive_uniform_weight,
            "effective_pdt_loss_start_epoch": effective_pdt_loss_start_epoch,
            "effective_pdt_loss_weight": effective_pdt_loss_weight,
            "low_codebook_risk": low_codebook_risk,
            "low_m_refresh_epochs": low_m_refresh_epochs,
            "medium_codebook_mode": medium_codebook_mode,
            "high_capacity_mode": high_capacity_mode,
            "high_capacity_reg_scale": high_capacity_reg_scale,
            "pdt_m": pdt_m,
            "pre_quant_enabled": pre_quant_enabled,
            "pre_quant_lambda_decor": pre_quant_lambda_decor,
            "pre_quant_lambda_stab": pre_quant_lambda_stab,
            "pre_quant_stab_late_epoch": pre_quant_stab_late_epoch,
            "pre_quant_stab_late_multiplier": pre_quant_stab_late_multiplier,
            "pre_quant_refresh_end_epoch": effective_refresh_end_epoch,
            "pre_quant_refresh_epochs": pre_quant_refresh_epochs,
            "all_refresh_epochs": all_refresh_epochs,
            "pre_quant_raw_metric_weight": pre_quant_raw_metric_weight,
            "pre_quant_neighbor_enabled": pre_quant_neighbor_enabled,
            "pre_quant_neighbor_use_btn": pre_quant_neighbor_use_btn,
            "pre_quant_neighbor_use_dec": pre_quant_neighbor_use_dec,
            "pre_quant_neighbor_topk": pre_quant_neighbor_topk,
            "pre_quant_neighbor_tau_btn": pre_quant_neighbor_tau_btn,
            "pre_quant_neighbor_tau_dec": pre_quant_neighbor_tau_dec,
            "pre_quant_neighbor_lambda_btn": pre_quant_neighbor_lambda_btn,
            "pre_quant_neighbor_lambda_dec": pre_quant_neighbor_lambda_dec,
            "pre_quant_neighbor_start_epoch": pre_quant_neighbor_start_epoch,
            "pre_quant_neighbor_warmup_epochs": pre_quant_neighbor_warmup_epochs,
            "pre_quant_neighbor_dec_stop_backbone": pre_quant_neighbor_dec_stop_backbone,
            "pre_quant_neighbor_teacher_mode": pre_quant_neighbor_teacher_mode,
            "pre_quant_neighbor_offline_path": pre_quant_neighbor_offline_path,
            "pre_quant_landmark_enabled": pre_quant_landmark_enabled,
            "pre_quant_landmark_num": pre_quant_landmark_num,
            "pre_quant_landmark_select": pre_quant_landmark_select,
            "pre_quant_landmark_profile_transform": pre_quant_landmark_profile_transform,
            "pre_quant_landmark_use_btn": pre_quant_landmark_use_btn,
            "pre_quant_landmark_use_dec": pre_quant_landmark_use_dec,
            "pre_quant_landmark_dec_ste_to_btn": pre_quant_landmark_dec_ste_to_btn,
            "pre_quant_landmark_dec_bank_source": pre_quant_landmark_dec_bank_source,
            "pre_quant_landmark_rank_weight": pre_quant_landmark_rank_weight,
            "pre_quant_landmark_rank_tau": pre_quant_landmark_rank_tau,
            "pre_quant_landmark_lambda_btn": pre_quant_landmark_lambda_btn,
            "pre_quant_landmark_lambda_dec": pre_quant_landmark_lambda_dec,
            "pre_quant_landmark_start_epoch": pre_quant_landmark_start_epoch,
            "pre_quant_landmark_warmup_epochs": pre_quant_landmark_warmup_epochs,
            "pre_quant_landmark_bank_refresh_epochs": pre_quant_landmark_bank_refresh_epochs,
            "pre_quant_landmark_artifact_path": pre_quant_landmark_artifact_path,
            "pre_quant_landmark_fixed_bank_path": pre_quant_landmark_fixed_bank_path,
            "pre_quant_landmark_teacher_checkpoint": pre_quant_landmark_teacher_checkpoint,
            "decoded_ste_metric_enabled": decoded_ste_metric_enabled,
            "decoded_ste_metric_start_epoch": decoded_ste_metric_start_epoch,
            "decoded_ste_metric_warmup_epochs": decoded_ste_metric_warmup_epochs,
            "decoded_ste_metric_max_weight": decoded_ste_metric_max_weight,
            "late_finetune_enabled": late_finetune_enabled,
            "late_finetune_start_epoch": late_finetune_start_epoch,
            "late_finetune_main_lr_scale": late_finetune_main_lr_scale,
            "late_finetune_pre_quant_lr_scale": late_finetune_pre_quant_lr_scale,
        }

    def _get_pre_quant_reg_weights(self, epoch, training_state):
        lambda_decor = float(training_state.get("pre_quant_lambda_decor", 0.0))
        lambda_stab = float(training_state.get("pre_quant_lambda_stab", 0.0))
        if epoch >= int(training_state.get("pre_quant_stab_late_epoch", 10**9)):
            lambda_stab *= float(training_state.get("pre_quant_stab_late_multiplier", 1.0))
        return lambda_decor, lambda_stab

    def _compute_batch_loss(self, my_net, my_loss, batch, epoch, training_state):
        if len(batch) >= 8:
            inputs_lon_array, inputs_lat_array, inputs_lon_grid_array, inputs_lat_grid_array, inputs_lon_lat_image_array, inputs_length_array, distance_array, sample_index_array = batch[:8]
        elif len(batch) >= 6:
            inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array, inputs_length_array, distance_array, sample_index_array = batch[:6]
            inputs_lon_grid_array = None
            inputs_lat_grid_array = None
        else:
            inputs_lon_array, inputs_lat_array, inputs_lon_lat_image_array, inputs_length_array, distance_array = batch
            inputs_lon_grid_array = None
            inputs_lat_grid_array = None
            sample_index_array = None
        batch_outputs = my_net(inputs_lon_array,
                               inputs_lat_array,
                               inputs_lon_lat_image_array,
                               inputs_length_array=inputs_length_array,
                               inputs_lon_grid_array=inputs_lon_grid_array,
                               inputs_lat_grid_array=inputs_lat_grid_array,
                               return_intermediate=True)

        anchor_outputs = batch_outputs["anchor"]
        positive_outputs = batch_outputs["positive"]
        negative_outputs = batch_outputs["negative"]

        anchor_embedding = anchor_outputs["e_for_pdt"]
        positive_embedding = positive_outputs["e_for_pdt"]
        negative_embedding = negative_outputs["e_for_pdt"]
        v_a = anchor_outputs["quantized_embedding"]
        v_p = positive_outputs["quantized_embedding"]
        v_n = negative_outputs["quantized_embedding"]

        positive_distance_target = torch.tensor(distance_array[0]).to(self.my_config.my_dict["device"])
        negative_distance_target = torch.tensor(distance_array[1]).to(self.my_config.my_dict["device"])
        cross_distance_target = torch.tensor(distance_array[2]).to(self.my_config.my_dict["device"])

        positive_learning_distance, negative_learning_distance, cross_learning_distance = torch_feature_distance.all_feature_distance(
            self.my_config.my_dict["all_feature_distance_type"],
            anchor_embedding,
            positive_embedding,
            negative_embedding,
            self.my_config.my_dict["channel"],
        )

        if self.my_config.my_dict["loss_type"] != "triplet":
            raise ValueError("Loss Type Error")

        rank_loss, mse_loss, metric_loss = my_loss(
            self.my_config,
            epoch,
            positive_learning_distance,
            positive_distance_target,
            negative_learning_distance,
            negative_distance_target,
            cross_learning_distance,
            cross_distance_target,
        )

        raw_metric_loss = None
        raw_metric_weight = 0.0
        if getattr(my_net, "pre_quant_bottleneck_enabled", False):
            raw_metric_weight = float(training_state.get("pre_quant_raw_metric_weight", 0.0))
            if raw_metric_weight > 0.0:
                raw_positive_learning_distance, raw_negative_learning_distance, raw_cross_learning_distance = torch_feature_distance.all_feature_distance(
                    self.my_config.my_dict["all_feature_distance_type"],
                    anchor_outputs["e_cont"],
                    positive_outputs["e_cont"],
                    negative_outputs["e_cont"],
                    self.my_config.my_dict["channel"],
                )
                _, _, raw_metric_loss = my_loss(
                    self.my_config,
                    epoch,
                    raw_positive_learning_distance,
                    positive_distance_target,
                    raw_negative_learning_distance,
                    negative_distance_target,
                    raw_cross_learning_distance,
                    cross_distance_target,
                )

        pdt_loss = None
        use_pdt_loss = (epoch < training_state["freeze_backbone_epochs"]) or (epoch >= training_state["effective_pdt_loss_start_epoch"])
        if use_pdt_loss:
            pdt_loss = my_net.PDT_model(
                anchor_embedding,
                positive_embedding,
                negative_embedding,
                cross_distance_target,
                positive_distance_target,
                negative_distance_target,
            ).mean()

        if epoch < training_state["freeze_backbone_epochs"]:
            if pdt_loss is None:
                raise ValueError("PDT-only stage requires PDT loss to be enabled.")
            loss = training_state["effective_pdt_loss_weight"] * pdt_loss
        else:
            loss = metric_loss
            if pdt_loss is not None:
                loss = loss + training_state["effective_pdt_loss_weight"] * pdt_loss
            if raw_metric_loss is not None:
                loss = loss + raw_metric_weight * raw_metric_loss

        consistency_loss = None
        quantized_metric_loss = None
        pairwise_consistency_loss = None
        decoded_ste_metric_loss = None
        decoded_ste_metric_weight = 0.0
        decoded_ste_hard_soft_gap = None
        entropy_reg_loss = None
        commitment_reg_loss = None
        uniform_reg_loss = None
        quantized_metric_effective_weight = training_state["quantized_metric_weight"]

        if training_state["loss_recipe"] in ["consistency", "consistency_quantized_metric"]:
            consistency_loss = ((v_a - anchor_embedding.detach()) ** 2).mean() + \
                               ((v_p - positive_embedding.detach()) ** 2).mean() + \
                               ((v_n - negative_embedding.detach()) ** 2).mean()
            loss = loss + training_state["consistency_weight"] * consistency_loss

        if training_state["loss_recipe"] in ["quantized_metric", "consistency_quantized_metric"]:
            q_positive_learning_distance, q_negative_learning_distance, q_cross_learning_distance = torch_feature_distance.all_feature_distance(
                self.my_config.my_dict["all_feature_distance_type"],
                v_a,
                v_p,
                v_n,
                self.my_config.my_dict["channel"],
            )
            _, _, quantized_metric_loss = my_loss(
                self.my_config,
                epoch,
                q_positive_learning_distance,
                positive_distance_target,
                q_negative_learning_distance,
                negative_distance_target,
                q_cross_learning_distance,
                cross_distance_target,
            )
            loss = loss + training_state["quantized_metric_weight"] * quantized_metric_loss
        elif training_state["loss_recipe"] == "improved_vq":
            soft_v_a, aux_anchor = my_net.PDT_model.soft_quantize(anchor_embedding)
            soft_v_p, aux_positive = my_net.PDT_model.soft_quantize(positive_embedding)
            soft_v_n, aux_negative = my_net.PDT_model.soft_quantize(negative_embedding)

            q_positive_learning_distance, q_negative_learning_distance, q_cross_learning_distance = torch_feature_distance.all_feature_distance(
                self.my_config.my_dict["all_feature_distance_type"],
                soft_v_a,
                soft_v_p,
                soft_v_n,
                self.my_config.my_dict["channel"],
            )
            _, _, quantized_metric_loss = my_loss(
                self.my_config,
                epoch,
                q_positive_learning_distance,
                positive_distance_target,
                q_negative_learning_distance,
                negative_distance_target,
                q_cross_learning_distance,
                cross_distance_target,
            )

            pairwise_consistency_loss = torch.nn.functional.smooth_l1_loss(q_positive_learning_distance, positive_learning_distance.detach()) + \
                                        torch.nn.functional.smooth_l1_loss(q_negative_learning_distance, negative_learning_distance.detach()) + \
                                        torch.nn.functional.smooth_l1_loss(q_cross_learning_distance, cross_learning_distance.detach())
            decoded_ste_hard_soft_gap = (
                torch.nn.functional.smooth_l1_loss(v_a.detach(), soft_v_a) +
                torch.nn.functional.smooth_l1_loss(v_p.detach(), soft_v_p) +
                torch.nn.functional.smooth_l1_loss(v_n.detach(), soft_v_n)
            ) / 3.0
            commitment_reg_loss = (aux_anchor["commitment_loss"] + aux_positive["commitment_loss"] + aux_negative["commitment_loss"]) / 3.0
            entropy_reg_loss = (aux_anchor["entropy_loss"] + aux_positive["entropy_loss"] + aux_negative["entropy_loss"]) / 3.0
            uniform_reg_loss = (aux_anchor["uniform_loss"] + aux_positive["uniform_loss"] + aux_negative["uniform_loss"]) / 3.0

            if training_state.get("decoded_ste_metric_enabled", False):
                decoded_ste_metric_weight = self._linear_warmup_weight(
                    epoch,
                    training_state.get("decoded_ste_metric_start_epoch", 80),
                    training_state.get("decoded_ste_metric_warmup_epochs", 20),
                    training_state.get("decoded_ste_metric_max_weight", 0.03),
                )
                if decoded_ste_metric_weight > 0.0:
                    ste_v_a = soft_v_a + (v_a.detach() - soft_v_a).detach()
                    ste_v_p = soft_v_p + (v_p.detach() - soft_v_p).detach()
                    ste_v_n = soft_v_n + (v_n.detach() - soft_v_n).detach()
                    ste_positive_learning_distance, ste_negative_learning_distance, ste_cross_learning_distance = torch_feature_distance.all_feature_distance(
                        self.my_config.my_dict["all_feature_distance_type"],
                        ste_v_a,
                        ste_v_p,
                        ste_v_n,
                        self.my_config.my_dict["channel"],
                    )
                    _, _, decoded_ste_metric_loss = my_loss(
                        self.my_config,
                        epoch,
                        ste_positive_learning_distance,
                        positive_distance_target,
                        ste_negative_learning_distance,
                        negative_distance_target,
                        ste_cross_learning_distance,
                        cross_distance_target,
                    )
                    loss = loss + decoded_ste_metric_weight * decoded_ste_metric_loss

            if training_state["low_codebook_risk"] > 0.0:
                loss = loss + training_state["adaptive_commit_weight"] * commitment_reg_loss
                loss = loss + training_state["adaptive_entropy_weight"] * entropy_reg_loss
                loss = loss + training_state["adaptive_uniform_weight"] * uniform_reg_loss

            if training_state["medium_codebook_mode"]:
                quantized_metric_effective_weight = 0.0
                if epoch >= training_state["effective_pdt_loss_start_epoch"]:
                    loss = loss + training_state["adaptive_commit_weight"] * commitment_reg_loss
                    loss = loss + training_state["adaptive_entropy_weight"] * entropy_reg_loss
                    loss = loss + training_state["adaptive_uniform_weight"] * uniform_reg_loss
            elif training_state["high_capacity_mode"]:
                quantized_metric_effective_weight = 0.0
                if epoch >= training_state["effective_pdt_loss_start_epoch"]:
                    loss = loss + training_state["high_capacity_reg_scale"] * training_state["improved_commit_weight"] * commitment_reg_loss
                    loss = loss + training_state["high_capacity_reg_scale"] * training_state["improved_entropy_weight"] * entropy_reg_loss
                    loss = loss + training_state["high_capacity_reg_scale"] * training_state["improved_uniform_weight"] * uniform_reg_loss
            else:
                quantized_metric_effective_weight = self._linear_warmup_weight(
                    epoch,
                    training_state["adaptive_qm_start_epoch"],
                    training_state["improved_qm_warmup_epochs"],
                    training_state["adaptive_qm_max_weight"],
                )
                if quantized_metric_effective_weight > 0.0:
                    loss = loss + quantized_metric_effective_weight * quantized_metric_loss
                    loss = loss + training_state["adaptive_pairwise_weight"] * pairwise_consistency_loss
                    if training_state["low_codebook_risk"] <= 0.0:
                        loss = loss + training_state["adaptive_commit_weight"] * commitment_reg_loss
                        loss = loss + training_state["adaptive_entropy_weight"] * entropy_reg_loss
                        loss = loss + training_state["adaptive_uniform_weight"] * uniform_reg_loss

        pre_quant_stats = []
        if getattr(my_net, "pre_quant_bottleneck_enabled", False):
            lambda_decor, lambda_stab = self._get_pre_quant_reg_weights(epoch, training_state)
            for item in (anchor_outputs, positive_outputs, negative_outputs):
                pre_quant_stats.append(my_net.compute_pre_quant_regularizers(item,
                                                                            lambda_decor=lambda_decor,
                                                                            lambda_stab=lambda_stab))
            decor_loss = torch.stack([entry["decor_loss"] for entry in pre_quant_stats]).mean()
            stab_loss = torch.stack([entry["stab_loss"] for entry in pre_quant_stats]).mean()
            weighted_decor_loss = torch.stack([entry["weighted_decor_loss"] for entry in pre_quant_stats]).mean()
            weighted_stab_loss = torch.stack([entry["weighted_stab_loss"] for entry in pre_quant_stats]).mean()
            pre_quant_total_loss = torch.stack([entry["total_loss"] for entry in pre_quant_stats]).mean()
            loss = loss + pre_quant_total_loss
            aux_keys = [
                "u_global_mean",
                "u_global_std",
                "u_local_mean",
                "u_local_std",
                "u_progress_mean",
                "u_progress_std",
                "structured_delta_mean",
                "structured_delta_std",
                "e_bottleneck_mean",
                "e_bottleneck_std",
                "residual_alpha",
            ]
            bottleneck_aux = {}
            for key in aux_keys:
                bottleneck_aux[key] = float(np.mean([
                    item["bottleneck_outputs"]["aux"][key]
                    for item in (anchor_outputs, positive_outputs, negative_outputs)
                ]))
        else:
            zero = loss.new_tensor(0.0)
            decor_loss = zero
            stab_loss = zero
            weighted_decor_loss = zero
            weighted_stab_loss = zero
            pre_quant_total_loss = zero
            bottleneck_aux = {}
            lambda_decor = 0.0
            lambda_stab = 0.0

        zero = loss.new_tensor(0.0)
        neighbor_btn_loss = zero
        neighbor_dec_loss = zero
        neighbor_btn_total = zero
        neighbor_dec_total = zero
        neighbor_effective_topk = 0
        neighbor_tau_btn = float(training_state.get("pre_quant_neighbor_tau_btn", 0.0))
        neighbor_tau_dec = float(training_state.get("pre_quant_neighbor_tau_dec", 0.0))
        neighbor_btn_aux = {
            "avg_teacher_neighbor_distance": 0.0,
            "avg_student_neighbor_distance": 0.0,
            "avg_overlap_proxy": 0.0,
            "effective_topk": 0,
        }
        neighbor_dec_aux = {
            "avg_teacher_neighbor_distance": 0.0,
            "avg_student_neighbor_distance": 0.0,
            "avg_overlap_proxy": 0.0,
            "effective_topk": 0,
        }
        landmark_btn_loss = zero
        landmark_dec_loss = zero
        landmark_btn_total = zero
        landmark_dec_total = zero
        landmark_weight_scale = 0.0
        landmark_bank_epoch = -1
        landmark_effective_num = 0
        landmark_btn_aux = {
            "num_landmarks": 0,
            "avg_teacher_profile_distance": 0.0,
            "avg_student_profile_distance": 0.0,
            "avg_profile_cosine": 0.0,
        }
        landmark_dec_aux = {
            "num_landmarks": 0,
            "avg_teacher_profile_distance": 0.0,
            "avg_student_profile_distance": 0.0,
            "avg_profile_cosine": 0.0,
        }
        combined_e_cont = torch.cat((anchor_outputs["e_cont"], positive_outputs["e_cont"], negative_outputs["e_cont"]), dim=0)
        combined_e_for_pdt = torch.cat((anchor_outputs["e_for_pdt"], positive_outputs["e_for_pdt"], negative_outputs["e_for_pdt"]), dim=0)
        combined_e_hat = torch.cat((anchor_outputs["e_hat"], positive_outputs["e_hat"], negative_outputs["e_hat"]), dim=0)
        combined_sample_indices = None
        if sample_index_array is not None:
            combined_sample_indices = np.concatenate([
                np.asarray(sample_index_array[0], dtype=np.int64),
                np.asarray(sample_index_array[1], dtype=np.int64),
                np.asarray(sample_index_array[2], dtype=np.int64),
            ], axis=0)
        e_cont_norm = torch.norm(combined_e_cont, dim=1).mean()
        e_bottleneck_norm = torch.norm(combined_e_for_pdt, dim=1).mean()
        e_hat_norm = torch.norm(combined_e_hat, dim=1).mean()

        if training_state.get("pre_quant_neighbor_enabled", False):
            neighbor_weight_scale = self._linear_warmup_weight(
                epoch,
                training_state.get("pre_quant_neighbor_start_epoch", 0),
                training_state.get("pre_quant_neighbor_warmup_epochs", 0),
                1.0,
            )
            distance_fn = resolve_embedding_distance_fn(self.my_config.my_dict["all_feature_distance_type"])
            teacher_mode = training_state.get("pre_quant_neighbor_teacher_mode", "batch_cont")
            if teacher_mode == "offline_gt":
                if combined_sample_indices is None:
                    raise ValueError("Offline GT neighborhood mode requires sample indices in the training batch.")
                if self.pre_quant_neighbor_artifact is None:
                    raise ValueError("Offline GT neighborhood artifact is not loaded.")
                neighbor_mask, teacher_mask_aux = build_batch_neighbor_mask_from_global_knn(
                    sample_indices=combined_sample_indices,
                    global_neighbor_indices=self.pre_quant_neighbor_artifact["neighbor_indices"][:, :training_state["pre_quant_neighbor_topk"]],
                    global_neighbor_distances=self.pre_quant_neighbor_artifact["neighbor_distances"][:, :training_state["pre_quant_neighbor_topk"]],
                    device=combined_e_cont.device,
                )
                if training_state.get("pre_quant_neighbor_use_btn", False) and neighbor_weight_scale > 0.0:
                    neighbor_btn_loss, neighbor_btn_aux = neighborhood_consistency_loss_from_mask(
                        student_emb=combined_e_for_pdt,
                        distance_fn=distance_fn,
                        neighbor_mask=neighbor_mask,
                        tau=training_state["pre_quant_neighbor_tau_btn"],
                        teacher_aux=teacher_mask_aux,
                    )
                    neighbor_btn_total = (training_state["pre_quant_neighbor_lambda_btn"] * neighbor_weight_scale) * neighbor_btn_loss
                    loss = loss + neighbor_btn_total
                    neighbor_effective_topk = max(neighbor_effective_topk, int(neighbor_btn_aux["effective_topk"]))
                if training_state.get("pre_quant_neighbor_use_dec", False) and neighbor_weight_scale > 0.0:
                    neighbor_student_e_hat = combined_e_hat
                    if training_state.get("pre_quant_neighbor_dec_stop_backbone", False):
                        neighbor_student_e_hat = my_net.quantize_embeddings(combined_e_for_pdt.detach())
                    neighbor_dec_loss, neighbor_dec_aux = neighborhood_consistency_loss_from_mask(
                        student_emb=neighbor_student_e_hat,
                        distance_fn=distance_fn,
                        neighbor_mask=neighbor_mask,
                        tau=training_state["pre_quant_neighbor_tau_dec"],
                        teacher_aux=teacher_mask_aux,
                    )
                    neighbor_dec_total = (training_state["pre_quant_neighbor_lambda_dec"] * neighbor_weight_scale) * neighbor_dec_loss
                    loss = loss + neighbor_dec_total
                    neighbor_effective_topk = max(neighbor_effective_topk, int(neighbor_dec_aux["effective_topk"]))
            else:
                teacher_emb = combined_e_cont
                if training_state.get("pre_quant_neighbor_use_btn", False) and neighbor_weight_scale > 0.0:
                    neighbor_btn_loss, neighbor_btn_aux = neighborhood_consistency_loss(
                        teacher_emb=teacher_emb,
                        student_emb=combined_e_for_pdt,
                        distance_fn=distance_fn,
                        topk=training_state["pre_quant_neighbor_topk"],
                        tau=training_state["pre_quant_neighbor_tau_btn"],
                    )
                    neighbor_btn_total = (training_state["pre_quant_neighbor_lambda_btn"] * neighbor_weight_scale) * neighbor_btn_loss
                    loss = loss + neighbor_btn_total
                    neighbor_effective_topk = max(neighbor_effective_topk, int(neighbor_btn_aux["effective_topk"]))
                if training_state.get("pre_quant_neighbor_use_dec", False) and neighbor_weight_scale > 0.0:
                    neighbor_student_e_hat = combined_e_hat
                    if training_state.get("pre_quant_neighbor_dec_stop_backbone", False):
                        neighbor_student_e_hat = my_net.quantize_embeddings(combined_e_for_pdt.detach())
                    neighbor_dec_loss, neighbor_dec_aux = neighborhood_consistency_loss(
                        teacher_emb=teacher_emb,
                        student_emb=neighbor_student_e_hat,
                        distance_fn=distance_fn,
                        topk=training_state["pre_quant_neighbor_topk"],
                        tau=training_state["pre_quant_neighbor_tau_dec"],
                    )
                    neighbor_dec_total = (training_state["pre_quant_neighbor_lambda_dec"] * neighbor_weight_scale) * neighbor_dec_loss
                    loss = loss + neighbor_dec_total
                    neighbor_effective_topk = max(neighbor_effective_topk, int(neighbor_dec_aux["effective_topk"]))

        if training_state.get("pre_quant_landmark_enabled", False):
            if combined_sample_indices is None:
                raise ValueError("GT landmark profile mode requires sample indices in the training batch.")
            landmark_weight_scale = self._linear_warmup_weight(
                epoch,
                training_state.get("pre_quant_landmark_start_epoch", 0),
                training_state.get("pre_quant_landmark_warmup_epochs", 0),
                1.0,
            )
            if landmark_weight_scale > 0.0:
                if self.pre_quant_landmark_artifact is None:
                    raise ValueError("GT landmark profile artifact is not loaded.")
                if self.pre_quant_landmark_bank is None:
                    raise ValueError("GT landmark profile bank is not initialized.")
                landmark_bank_epoch = int(self.pre_quant_landmark_bank.get("epoch", -1))
                landmark_effective_num = int(self.pre_quant_landmark_bank["e_hat"].shape[0])
            if landmark_weight_scale > 0.0 and landmark_effective_num > 0:
                distance_fn = resolve_embedding_distance_fn(self.my_config.my_dict["all_feature_distance_type"])
                teacher_profile, teacher_profile_raw = self._gather_landmark_teacher_profiles(
                    combined_sample_indices,
                    combined_e_cont.device,
                )
                if training_state.get("pre_quant_landmark_use_btn", False):
                    landmark_btn_loss, landmark_btn_aux = landmark_profile_loss(
                        student_emb=combined_e_for_pdt,
                        landmark_bank=self.pre_quant_landmark_bank["e_for_pdt"].to(combined_e_for_pdt.device),
                        teacher_profile=teacher_profile,
                        teacher_profile_raw=teacher_profile_raw,
                        distance_fn=distance_fn,
                        profile_transform=training_state["pre_quant_landmark_profile_transform"],
                        rank_weight=training_state.get("pre_quant_landmark_rank_weight", 0.0),
                        rank_tau=training_state.get("pre_quant_landmark_rank_tau", 0.5),
                    )
                    landmark_btn_total = (training_state["pre_quant_landmark_lambda_btn"] * landmark_weight_scale) * landmark_btn_loss
                    loss = loss + landmark_btn_total
                if training_state.get("pre_quant_landmark_use_dec", False):
                    landmark_student_e_hat = combined_e_hat
                    if training_state.get("pre_quant_landmark_dec_ste_to_btn", False):
                        landmark_student_e_hat = combined_e_for_pdt + (combined_e_hat - combined_e_for_pdt).detach()
                    dec_bank_source = training_state.get("pre_quant_landmark_dec_bank_source", "decoded")
                    dec_landmark_bank = self.pre_quant_landmark_bank["e_hat"]
                    if dec_bank_source == "bottleneck":
                        dec_landmark_bank = self.pre_quant_landmark_bank["e_for_pdt"]
                    landmark_dec_loss, landmark_dec_aux = landmark_profile_loss(
                        student_emb=landmark_student_e_hat,
                        landmark_bank=dec_landmark_bank.to(combined_e_hat.device),
                        teacher_profile=teacher_profile,
                        teacher_profile_raw=teacher_profile_raw,
                        distance_fn=distance_fn,
                        profile_transform=training_state["pre_quant_landmark_profile_transform"],
                        rank_weight=training_state.get("pre_quant_landmark_rank_weight", 0.0),
                        rank_tau=training_state.get("pre_quant_landmark_rank_tau", 0.5),
                    )
                    landmark_dec_total = (training_state["pre_quant_landmark_lambda_dec"] * landmark_weight_scale) * landmark_dec_loss
                    loss = loss + landmark_dec_total

        return {
            "loss": loss,
            "rank_loss": rank_loss,
            "mse_loss": mse_loss,
            "metric_loss": metric_loss,
            "pdt_loss": pdt_loss,
            "consistency_loss": consistency_loss,
            "quantized_metric_loss": quantized_metric_loss,
            "pairwise_consistency_loss": pairwise_consistency_loss,
            "decoded_ste_metric_loss": decoded_ste_metric_loss,
            "decoded_ste_metric_weight": decoded_ste_metric_weight,
            "decoded_ste_hard_soft_gap": decoded_ste_hard_soft_gap,
            "entropy_reg_loss": entropy_reg_loss,
            "commitment_reg_loss": commitment_reg_loss,
            "uniform_reg_loss": uniform_reg_loss,
            "quantized_metric_effective_weight": quantized_metric_effective_weight,
            "raw_metric_loss": raw_metric_loss,
            "raw_metric_weight": raw_metric_weight,
            "pre_quant_decor_loss": decor_loss,
            "pre_quant_stab_loss": stab_loss,
            "pre_quant_weighted_decor_loss": weighted_decor_loss,
            "pre_quant_weighted_stab_loss": weighted_stab_loss,
            "pre_quant_total_loss": pre_quant_total_loss,
            "pre_quant_lambda_decor": lambda_decor,
            "pre_quant_lambda_stab": lambda_stab,
            "neighbor_btn_loss": neighbor_btn_loss,
            "neighbor_dec_loss": neighbor_dec_loss,
            "neighbor_btn_total": neighbor_btn_total,
            "neighbor_dec_total": neighbor_dec_total,
            "neighbor_effective_topk": neighbor_effective_topk,
            "neighbor_tau_btn": neighbor_tau_btn,
            "neighbor_tau_dec": neighbor_tau_dec,
            "neighbor_btn_aux": neighbor_btn_aux,
            "neighbor_dec_aux": neighbor_dec_aux,
            "landmark_btn_loss": landmark_btn_loss,
            "landmark_dec_loss": landmark_dec_loss,
            "landmark_btn_total": landmark_btn_total,
            "landmark_dec_total": landmark_dec_total,
            "landmark_weight_scale": landmark_weight_scale,
            "landmark_bank_epoch": landmark_bank_epoch,
            "landmark_effective_num": landmark_effective_num,
            "landmark_btn_aux": landmark_btn_aux,
            "landmark_dec_aux": landmark_dec_aux,
            "e_cont_norm": e_cont_norm,
            "e_bottleneck_norm": e_bottleneck_norm,
            "e_hat_norm": e_hat_norm,
            "bottleneck_aux": bottleneck_aux,
            "batch_outputs": batch_outputs,
        }

    def _scalar(self, value):
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return float(value.detach().item())
        return float(value)

    def _ensure_root_subdir(self, dirname):
        path = os.path.join(self.my_config.my_dict["root_write_path"], dirname)
        os.makedirs(path, exist_ok=True)
        return path

    def _artifact_prefix(self):
        return str(self.my_config.my_dict.get("artifact_prefix", "") or "").strip()

    def _use_standard_artifacts(self):
        return bool(self._artifact_prefix())

    def _report_only_eval(self):
        return bool(self.my_config.my_dict.get("report_only_eval", False))

    def _artifact_tag_for_epoch(self, epoch):
        prefix = self._artifact_prefix() or self.my_config.my_dict["train_flag"]
        if self._use_standard_artifacts():
            return "{}_e{}".format(prefix, int(epoch))
        return "{}_epochs_{}".format(self.my_config.my_dict["train_flag"], int(epoch))

    def _checkpoint_path(self, epoch_num, model_flag=None):
        model_flag = model_flag or self.my_config.my_dict["train_flag"]
        if self._use_standard_artifacts():
            checkpoint_dir = self._ensure_root_subdir("checkpoints")
            prefix = self._artifact_prefix() if model_flag == self.my_config.my_dict["train_flag"] else model_flag
            filename = "{}_e{}.pt".format(prefix or model_flag, int(epoch_num))
            return os.path.join(checkpoint_dir, filename)
        os.makedirs(self.my_config.my_dict["save_model_path"], exist_ok=True)
        return os.path.join(self.my_config.my_dict["save_model_path"], "{}_epochs_{}".format(model_flag, int(epoch_num)))

    def _metrics_path(self, metrics_flag):
        if self._use_standard_artifacts():
            metrics_dir = self._ensure_root_subdir("metrics")
            return os.path.join(metrics_dir, "{}.json".format(metrics_flag))
        feature_dir = self._ensure_root_subdir("feature_dir")
        return os.path.join(feature_dir, "metrics_{}.json".format(metrics_flag))

    def _resolve_pre_quant_neighbor_artifact_path(self, topk):
        configured_path = str(self.my_config.my_dict.get("pre_quant_neighbor_offline_path", "") or "").strip()
        if configured_path:
            return configured_path
        neighbor_dir = self._ensure_root_subdir("neighbor_dir")
        dataset_name = str(self.my_config.my_dict.get("dataset", "dataset"))
        dist_type = str(self.my_config.my_dict.get("dist_type", "dist"))
        filename = "gt_neighbors_{}_train{}_top{}.npz".format(
            "{}_{}".format(dataset_name, dist_type),
            int(self.final_traj_train_num),
            int(topk),
        )
        return os.path.join(neighbor_dir, filename)

    def _build_gt_neighbor_artifact(self, topk=None, force_rebuild=False):
        num_samples = int(self.final_traj_train_num)
        effective_topk = int(self.my_config.my_dict.get("pre_quant_neighbor_topk", 10)) if topk is None else int(topk)
        effective_topk = max(1, min(effective_topk, num_samples - 1))
        artifact_path = self._resolve_pre_quant_neighbor_artifact_path(effective_topk)

        if os.path.exists(artifact_path) and (not force_rebuild):
            print("[PreQuantNeighbor] Reusing existing GT neighbor artifact: {}".format(artifact_path))
            return artifact_path

        train_distance = np.asarray(self.train_distance_matrix[:num_samples, :num_samples], dtype=np.float32)
        sorted_indices = np.argsort(train_distance, axis=1)
        neighbor_indices = np.full((num_samples, effective_topk), -1, dtype=np.int32)
        neighbor_distances = np.full((num_samples, effective_topk), np.inf, dtype=np.float32)
        for row in range(num_samples):
            row_sorted = sorted_indices[row]
            row_sorted = row_sorted[row_sorted != row][:effective_topk]
            row_count = row_sorted.shape[0]
            neighbor_indices[row, :row_count] = row_sorted.astype(np.int32, copy=False)
            neighbor_distances[row, :row_count] = train_distance[row, row_sorted].astype(np.float32, copy=False)

        artifact_dir = os.path.dirname(artifact_path)
        if artifact_dir and (not os.path.exists(artifact_dir)):
            os.makedirs(artifact_dir)
        np.savez(
            artifact_path,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            topk=np.int32(effective_topk),
            num_samples=np.int32(num_samples),
            source=np.array(["train_distance_matrix"], dtype="U64"),
        )
        print("[PreQuantNeighbor] Built GT neighbor artifact: {}".format(artifact_path))
        return artifact_path

    def _load_pre_quant_neighbor_artifact(self, artifact_path):
        artifact = np.load(artifact_path, allow_pickle=False)
        loaded = {
            "artifact_path": artifact_path,
            "neighbor_indices": np.asarray(artifact["neighbor_indices"], dtype=np.int32),
            "neighbor_distances": np.asarray(artifact["neighbor_distances"], dtype=np.float32),
            "topk": int(np.asarray(artifact["topk"]).item()),
            "num_samples": int(np.asarray(artifact["num_samples"]).item()),
            "source": str(np.asarray(artifact["source"]).reshape(-1)[0]),
        }
        return loaded

    def _prepare_pre_quant_neighbor_teacher(self, training_state):
        self.pre_quant_neighbor_artifact = None
        if not training_state.get("pre_quant_neighbor_enabled", False):
            return training_state
        if training_state.get("pre_quant_neighbor_teacher_mode", "batch_cont") != "offline_gt":
            return training_state

        artifact_path = str(training_state.get("pre_quant_neighbor_offline_path", "") or "").strip()
        if (not artifact_path) or (not os.path.exists(artifact_path)):
            artifact_path = self._build_gt_neighbor_artifact(topk=training_state["pre_quant_neighbor_topk"])
        artifact = self._load_pre_quant_neighbor_artifact(artifact_path)
        if artifact["num_samples"] < int(self.final_traj_train_num):
            raise ValueError("Offline neighbor artifact only covers {} samples, but training needs {}.".format(
                artifact["num_samples"],
                int(self.final_traj_train_num),
            ))
        training_state["pre_quant_neighbor_offline_path"] = artifact_path
        training_state["pre_quant_neighbor_topk"] = min(
            int(training_state["pre_quant_neighbor_topk"]),
            int(artifact["topk"]),
        )
        self.my_config.my_dict["pre_quant_neighbor_offline_path"] = artifact_path
        self.my_config.my_dict["pre_quant_neighbor_topk"] = training_state["pre_quant_neighbor_topk"]
        self.my_config.write_config_to_file()
        self.pre_quant_neighbor_artifact = artifact
        print("[PreQuantNeighbor] Loaded offline teacher | source: {} | topk: {} | path: {}".format(
            artifact["source"],
            artifact["topk"],
            artifact_path,
        ))
        return training_state

    def build_gt_neighbor_teacher(self, topk=None, force_rebuild=False):
        artifact_path = self._build_gt_neighbor_artifact(topk=topk, force_rebuild=force_rebuild)
        artifact = self._load_pre_quant_neighbor_artifact(artifact_path)
        summary = {
            "artifact_path": artifact_path,
            "topk": int(artifact["topk"]),
            "num_samples": int(artifact["num_samples"]),
            "source": artifact["source"],
            "avg_neighbor_distance": float(np.mean(artifact["neighbor_distances"][artifact["neighbor_indices"] >= 0])),
        }
        summary_path = os.path.join(
            self._ensure_root_subdir("neighbor_dir"),
            "summary_{}.json".format(self.my_config.my_dict["train_flag"]),
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[PreQuantNeighbor] summary saved to {}".format(summary_path))
        return summary

    def _resolve_pre_quant_landmark_artifact_path(self, num_landmarks):
        configured_path = str(self.my_config.my_dict.get("pre_quant_landmark_artifact_path", "") or "").strip()
        if configured_path:
            return configured_path
        landmark_dir = self._ensure_root_subdir("landmark_dir")
        dataset_name = str(self.my_config.my_dict.get("dataset", "dataset"))
        dist_type = str(self.my_config.my_dict.get("dist_type", "dist"))
        filename = "gt_landmark_profile_{}_train{}_landmarks{}.npz".format(
            "{}_{}".format(dataset_name, dist_type),
            int(self.final_traj_train_num),
            int(num_landmarks),
        )
        return os.path.join(landmark_dir, filename)

    def _select_landmarks_fps(self, train_distance, num_landmarks):
        num_samples = int(train_distance.shape[0])
        effective_num = max(1, min(int(num_landmarks), num_samples))
        mean_distance = train_distance.mean(axis=1)
        first_index = int(np.argmax(mean_distance))
        selected = [first_index]
        min_distance_to_selected = train_distance[:, first_index].astype(np.float32, copy=True)
        min_distance_to_selected[first_index] = -np.inf
        while len(selected) < effective_num:
            next_index = int(np.argmax(min_distance_to_selected))
            if next_index in selected:
                break
            selected.append(next_index)
            min_distance_to_selected = np.minimum(min_distance_to_selected, train_distance[:, next_index])
            min_distance_to_selected[selected] = -np.inf
        return np.asarray(selected, dtype=np.int32)

    def _build_gt_landmark_artifact(self, num_landmarks=None, force_rebuild=False):
        num_samples = int(self.final_traj_train_num)
        requested_landmarks = int(self.my_config.my_dict.get("pre_quant_landmark_num", 64)) if num_landmarks is None else int(num_landmarks)
        requested_landmarks = max(1, min(requested_landmarks, num_samples))
        artifact_path = self._resolve_pre_quant_landmark_artifact_path(requested_landmarks)

        if os.path.exists(artifact_path) and (not force_rebuild):
            print("[PreQuantLandmark] Reusing existing GT landmark artifact: {}".format(artifact_path))
            return artifact_path

        train_distance = np.asarray(self.train_distance_matrix[:num_samples, :num_samples], dtype=np.float32)
        select_mode = str(self.my_config.my_dict.get("pre_quant_landmark_select", "fps"))
        if select_mode != "fps":
            raise ValueError("Unsupported landmark selection mode: {}".format(select_mode))
        landmark_indices = self._select_landmarks_fps(train_distance, requested_landmarks)
        landmark_distance_profiles = train_distance[:, landmark_indices].astype(np.float32, copy=False)
        gt_profile_tensor = torch.from_numpy(landmark_distance_profiles.copy())
        gt_profile_norm = normalize_landmark_profile(
            gt_profile_tensor,
            transform=str(self.my_config.my_dict.get("pre_quant_landmark_profile_transform", "log1p_zscore")),
        ).cpu().numpy().astype(np.float32, copy=False)

        artifact_dir = os.path.dirname(artifact_path)
        if artifact_dir and (not os.path.exists(artifact_dir)):
            os.makedirs(artifact_dir)
        np.savez(
            artifact_path,
            landmark_indices=landmark_indices,
            gt_profile_raw=landmark_distance_profiles,
            gt_profile_norm=gt_profile_norm,
            num_samples=np.int32(num_samples),
            num_landmarks=np.int32(landmark_indices.shape[0]),
            select_mode=np.array([select_mode], dtype="U32"),
            profile_transform=np.array([str(self.my_config.my_dict.get("pre_quant_landmark_profile_transform", "log1p_zscore"))], dtype="U32"),
            source=np.array(["train_distance_matrix"], dtype="U64"),
        )
        print("[PreQuantLandmark] Built GT landmark artifact: {}".format(artifact_path))
        return artifact_path

    def _load_pre_quant_landmark_artifact(self, artifact_path):
        artifact = np.load(artifact_path, allow_pickle=False)
        return {
            "artifact_path": artifact_path,
            "landmark_indices": np.asarray(artifact["landmark_indices"], dtype=np.int32),
            "gt_profile_raw": np.asarray(artifact["gt_profile_raw"], dtype=np.float32),
            "gt_profile_norm": np.asarray(artifact["gt_profile_norm"], dtype=np.float32),
            "num_samples": int(np.asarray(artifact["num_samples"]).item()),
            "num_landmarks": int(np.asarray(artifact["num_landmarks"]).item()),
            "select_mode": str(np.asarray(artifact["select_mode"]).reshape(-1)[0]),
            "profile_transform": str(np.asarray(artifact["profile_transform"]).reshape(-1)[0]),
            "source": str(np.asarray(artifact["source"]).reshape(-1)[0]),
        }

    def _coerce_config_value(self, value, reference):
        if isinstance(reference, bool):
            if isinstance(value, str):
                return value.strip().lower() in ["1", "true", "yes", "y", "on"]
            return bool(value)
        if isinstance(reference, int) and not isinstance(reference, bool):
            return int(value)
        if isinstance(reference, float):
            return float(value)
        return value

    def _infer_checkpoint_flag_epoch(self, checkpoint_path):
        checkpoint_name = os.path.basename(checkpoint_path)
        if "_epochs_" not in checkpoint_name:
            return checkpoint_name, -1
        train_flag, epoch_text = checkpoint_name.rsplit("_epochs_", 1)
        try:
            epoch = int(epoch_text)
        except ValueError:
            epoch = -1
        return train_flag, epoch

    def _resolve_checkpoint_model_config(self, checkpoint_path):
        resolved = dict(self.my_config.my_dict)
        teacher_flag, teacher_epoch = self._infer_checkpoint_flag_epoch(checkpoint_path)
        config_path = os.path.join(
            str(self.my_config.my_dict["root_write_path"]),
            "train_config",
            "{}.json".format(teacher_flag),
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
            typed_keys = [
                "lon_input_size",
                "lat_input_size",
                "target_size",
                "batch_size",
                "sampling_num",
                "channel",
                "head_num",
                "network_type",
                "image_mode",
                "pdt_m",
                "pdt_k",
                "pdt_vq_type",
                "pdt_codebook_init",
                "qinco_h",
                "qinco_L",
                "qinco_identity_init",
                "pre_quant_bottleneck_enabled",
                "pre_quant_global_dim",
                "pre_quant_local_dim",
                "pre_quant_progress_dim",
                "pre_quant_use_motion_stats",
                "pre_quant_lambda_decor",
                "pre_quant_lambda_stab",
                "pre_quant_residual_alpha_init",
            ]
            for key in typed_keys:
                if key in raw_config:
                    reference = resolved.get(key, raw_config[key])
                    resolved[key] = self._coerce_config_value(raw_config[key], reference)
        else:
            config_path = ""
        resolved["device"] = self.my_config.my_dict["device"]
        return resolved, teacher_flag, teacher_epoch, config_path

    def _resolve_pre_quant_landmark_fixed_bank_path(self, num_landmarks, checkpoint_path=None):
        configured_path = str(self.my_config.my_dict.get("pre_quant_landmark_fixed_bank_path", "") or "").strip()
        if configured_path:
            return configured_path
        landmark_dir = self._ensure_root_subdir("landmark_dir")
        checkpoint_tag = "teacher"
        if checkpoint_path:
            checkpoint_tag = os.path.basename(checkpoint_path)
        checkpoint_tag = checkpoint_tag.replace(os.sep, "_")
        filename = "fixed_landmark_bank_{}_train{}_landmarks{}.npz".format(
            checkpoint_tag,
            int(self.final_traj_train_num),
            int(num_landmarks),
        )
        return os.path.join(landmark_dir, filename)

    def _load_pre_quant_landmark_fixed_bank(self, bank_path):
        artifact = np.load(bank_path, allow_pickle=False)
        return {
            "bank_path": bank_path,
            "landmark_indices": np.asarray(artifact["landmark_indices"], dtype=np.int32),
            "e_for_pdt": np.asarray(artifact["e_for_pdt"], dtype=np.float32),
            "e_hat": np.asarray(artifact["e_hat"], dtype=np.float32),
            "num_landmarks": int(np.asarray(artifact["num_landmarks"]).item()),
            "source_checkpoint": str(np.asarray(artifact["source_checkpoint"]).reshape(-1)[0]),
            "source_train_flag": str(np.asarray(artifact["source_train_flag"]).reshape(-1)[0]),
            "source_epoch": int(np.asarray(artifact["source_epoch"]).item()),
            "source_config_path": str(np.asarray(artifact["source_config_path"]).reshape(-1)[0]),
        }

    def build_pre_quant_landmark_bank(self, checkpoint_path=None, num_landmarks=None, force_rebuild=False):
        checkpoint_path = str(checkpoint_path or self.my_config.my_dict.get("pre_quant_landmark_teacher_checkpoint", "") or "").strip()
        if (not checkpoint_path) or (not os.path.exists(checkpoint_path)):
            raise ValueError("A valid --pre_quant_landmark_teacher_checkpoint is required to build a fixed landmark bank.")

        gt_artifact_path = self._build_gt_landmark_artifact(num_landmarks=num_landmarks, force_rebuild=False)
        gt_artifact = self._load_pre_quant_landmark_artifact(gt_artifact_path)
        bank_path = self._resolve_pre_quant_landmark_fixed_bank_path(gt_artifact["num_landmarks"], checkpoint_path=checkpoint_path)
        if os.path.exists(bank_path) and (not force_rebuild):
            print("[PreQuantLandmark] Reusing existing fixed landmark bank: {}".format(bank_path))
            return bank_path

        teacher_config, teacher_flag, teacher_epoch, teacher_config_path = self._resolve_checkpoint_model_config(checkpoint_path)
        teacher_net = function.initialize_model(teacher_config, self.max_traj_length).to(self.my_config.my_dict["device"])
        teacher_net.load_state_dict(torch.load(checkpoint_path, map_location=self.my_config.my_dict["device"]))

        landmark_indices = gt_artifact["landmark_indices"]
        outputs = self._collect_embedding_outputs(
            teacher_net,
            self.pad_total_train_lon_onehot[landmark_indices],
            self.pad_total_train_lat_onehot[landmark_indices],
            self.pad_total_train_lon_lat_image[landmark_indices],
            [self.train_traj_length_list[idx] for idx in landmark_indices.tolist()],
            lon_grid_tensor=self.pad_total_train_lon_grid[landmark_indices],
            lat_grid_tensor=self.pad_total_train_lat_grid[landmark_indices],
            test_batch=min(128, max(1, int(teacher_config.get("batch_size", self.my_config.my_dict.get("batch_size", 128))))),
            embedding_type="both",
            collect_code_usage=False,
            collect_transformed=False,
            collect_reconstructed=False,
            collect_raw_continuous=False,
        )

        bank_dir = os.path.dirname(bank_path)
        if bank_dir and (not os.path.exists(bank_dir)):
            os.makedirs(bank_dir)
        np.savez(
            bank_path,
            landmark_indices=landmark_indices.astype(np.int32, copy=False),
            e_for_pdt=outputs["continuous"].cpu().numpy().astype(np.float32, copy=False),
            e_hat=outputs["quantized"].cpu().numpy().astype(np.float32, copy=False),
            num_landmarks=np.int32(landmark_indices.shape[0]),
            source_checkpoint=np.array([checkpoint_path], dtype="U512"),
            source_train_flag=np.array([teacher_flag], dtype="U128"),
            source_epoch=np.int32(teacher_epoch),
            source_config_path=np.array([teacher_config_path], dtype="U512"),
        )
        del teacher_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[PreQuantLandmark] Built fixed landmark bank: {}".format(bank_path))
        return bank_path

    def _prepare_pre_quant_landmark_teacher(self, training_state):
        self.pre_quant_landmark_artifact = None
        self.pre_quant_landmark_bank = None
        if not training_state.get("pre_quant_landmark_enabled", False):
            return training_state

        artifact_path = str(training_state.get("pre_quant_landmark_artifact_path", "") or "").strip()
        if (not artifact_path) or (not os.path.exists(artifact_path)):
            artifact_path = self._build_gt_landmark_artifact(num_landmarks=training_state["pre_quant_landmark_num"])
        artifact = self._load_pre_quant_landmark_artifact(artifact_path)
        if artifact["num_samples"] < int(self.final_traj_train_num):
            raise ValueError("Offline landmark artifact only covers {} samples, but training needs {}.".format(
                artifact["num_samples"],
                int(self.final_traj_train_num),
            ))
        training_state["pre_quant_landmark_artifact_path"] = artifact_path
        training_state["pre_quant_landmark_num"] = min(
            int(training_state["pre_quant_landmark_num"]),
            int(artifact["num_landmarks"]),
        )
        training_state["pre_quant_landmark_profile_transform"] = artifact["profile_transform"]
        self.my_config.my_dict["pre_quant_landmark_artifact_path"] = artifact_path
        self.my_config.my_dict["pre_quant_landmark_num"] = training_state["pre_quant_landmark_num"]
        self.my_config.my_dict["pre_quant_landmark_profile_transform"] = training_state["pre_quant_landmark_profile_transform"]
        self.my_config.write_config_to_file()
        self.pre_quant_landmark_artifact = artifact
        print("[PreQuantLandmark] Loaded offline teacher | source: {} | landmarks: {} | path: {}".format(
            artifact["source"],
            artifact["num_landmarks"],
            artifact_path,
        ))
        fixed_bank_path = str(training_state.get("pre_quant_landmark_fixed_bank_path", "") or "").strip()
        teacher_checkpoint = str(training_state.get("pre_quant_landmark_teacher_checkpoint", "") or "").strip()
        if teacher_checkpoint and ((not fixed_bank_path) or (not os.path.exists(fixed_bank_path))):
            fixed_bank_path = self.build_pre_quant_landmark_bank(
                checkpoint_path=teacher_checkpoint,
                num_landmarks=training_state["pre_quant_landmark_num"],
                force_rebuild=False,
            )
        if fixed_bank_path:
            fixed_bank = self._load_pre_quant_landmark_fixed_bank(fixed_bank_path)
            if fixed_bank["num_landmarks"] < training_state["pre_quant_landmark_num"]:
                raise ValueError("Fixed landmark bank only covers {} landmarks, but training expects {}.".format(
                    fixed_bank["num_landmarks"],
                    training_state["pre_quant_landmark_num"],
                ))
            expected_indices = artifact["landmark_indices"][:training_state["pre_quant_landmark_num"]]
            fixed_indices = fixed_bank["landmark_indices"][:training_state["pre_quant_landmark_num"]]
            if not np.array_equal(expected_indices, fixed_indices):
                raise ValueError("Fixed landmark bank indices do not match the GT landmark artifact.")
            self.pre_quant_landmark_bank = {
                "epoch": int(fixed_bank["source_epoch"]),
                "landmark_indices": fixed_bank["landmark_indices"][:training_state["pre_quant_landmark_num"]].copy(),
                "e_for_pdt": torch.from_numpy(fixed_bank["e_for_pdt"][:training_state["pre_quant_landmark_num"]].copy()),
                "e_hat": torch.from_numpy(fixed_bank["e_hat"][:training_state["pre_quant_landmark_num"]].copy()),
                "fixed": True,
                "bank_path": fixed_bank_path,
                "source_checkpoint": fixed_bank["source_checkpoint"],
                "source_train_flag": fixed_bank["source_train_flag"],
            }
            training_state["pre_quant_landmark_fixed_bank_path"] = fixed_bank_path
            training_state["pre_quant_landmark_bank_refresh_epochs"] = 0
            self.my_config.my_dict["pre_quant_landmark_fixed_bank_path"] = fixed_bank_path
            self.my_config.my_dict["pre_quant_landmark_bank_refresh_epochs"] = 0
            self.my_config.write_config_to_file()
            print("[PreQuantLandmark] Loaded fixed bank | source_train_flag: {} | source_epoch: {} | path: {}".format(
                fixed_bank["source_train_flag"],
                fixed_bank["source_epoch"],
                fixed_bank_path,
            ))
        return training_state

    def build_gt_landmark_profile(self, num_landmarks=None, force_rebuild=False):
        artifact_path = self._build_gt_landmark_artifact(num_landmarks=num_landmarks, force_rebuild=force_rebuild)
        artifact = self._load_pre_quant_landmark_artifact(artifact_path)
        summary = {
            "artifact_path": artifact_path,
            "num_samples": int(artifact["num_samples"]),
            "num_landmarks": int(artifact["num_landmarks"]),
            "select_mode": artifact["select_mode"],
            "profile_transform": artifact["profile_transform"],
            "source": artifact["source"],
            "avg_profile_distance": float(np.mean(artifact["gt_profile_raw"])),
        }
        summary_path = os.path.join(
            self._ensure_root_subdir("landmark_dir"),
            "summary_{}.json".format(self.my_config.my_dict["train_flag"]),
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[PreQuantLandmark] summary saved to {}".format(summary_path))
        return summary

    def _gather_landmark_teacher_profiles(self, sample_indices, device):
        if self.pre_quant_landmark_artifact is None:
            raise ValueError("GT landmark artifact is not loaded.")
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        gt_profile_norm = torch.from_numpy(self.pre_quant_landmark_artifact["gt_profile_norm"][sample_indices]).to(device)
        gt_profile_raw = torch.from_numpy(self.pre_quant_landmark_artifact["gt_profile_raw"][sample_indices]).to(device)
        return gt_profile_norm, gt_profile_raw

    def _refresh_pre_quant_landmark_bank(self, my_net, epoch):
        if self.pre_quant_landmark_artifact is None:
            raise ValueError("Cannot refresh landmark bank without a loaded GT landmark artifact.")
        landmark_indices = self.pre_quant_landmark_artifact["landmark_indices"]
        bank_batch = min(128, max(1, int(self.my_config.my_dict.get("batch_size", 256))))
        outputs = self._collect_embedding_outputs(
            my_net,
            self.pad_total_train_lon_onehot[landmark_indices],
            self.pad_total_train_lat_onehot[landmark_indices],
            self.pad_total_train_lon_lat_image[landmark_indices],
            [self.train_traj_length_list[idx] for idx in landmark_indices.tolist()],
            lon_grid_tensor=self.pad_total_train_lon_grid[landmark_indices],
            lat_grid_tensor=self.pad_total_train_lat_grid[landmark_indices],
            test_batch=bank_batch,
            embedding_type="both",
            collect_code_usage=False,
            collect_transformed=False,
            collect_reconstructed=False,
            collect_raw_continuous=False,
        )
        self.pre_quant_landmark_bank = {
            "epoch": int(epoch),
            "landmark_indices": landmark_indices.copy(),
            "e_for_pdt": outputs["continuous"].cpu(),
            "e_hat": outputs["quantized"].cpu(),
        }
        print("[PreQuantLandmark] Refreshed landmark bank at epoch {} with {} landmarks.".format(
            int(epoch),
            int(landmark_indices.shape[0]),
        ))

    def _maybe_refresh_pre_quant_landmark_bank(self, my_net, epoch, training_state):
        if not training_state.get("pre_quant_landmark_enabled", False):
            return
        if self.pre_quant_landmark_bank is not None and bool(self.pre_quant_landmark_bank.get("fixed", False)):
            return
        start_epoch = int(training_state.get("pre_quant_landmark_start_epoch", 0))
        if epoch < start_epoch:
            return
        refresh_period = int(training_state.get("pre_quant_landmark_bank_refresh_epochs", 10))
        current_bank_epoch = -10**9 if self.pre_quant_landmark_bank is None else int(self.pre_quant_landmark_bank.get("epoch", -10**9))
        should_refresh = self.pre_quant_landmark_bank is None
        if not should_refresh and refresh_period > 0:
            should_refresh = (epoch - current_bank_epoch) >= refresh_period
        if should_refresh:
            self._refresh_pre_quant_landmark_bank(my_net, epoch)

    def _metrics_section_key(self):
        rerank_source = self.my_config.my_dict.get("rerank_source", "decoded")
        rerank_L = int(self.my_config.my_dict.get("rerank_L", 100))
        return "rerank_{}_L{}".format(rerank_source, rerank_L)

    def _update_best_records(self, best_records, metrics_payload, ckpt_path, metrics_path, epoch):
        sections = {
            "continuous": metrics_payload.get("continuous"),
            "decoded": metrics_payload.get("decoded"),
            "strict_rerank": metrics_payload.get(self._metrics_section_key()),
        }
        for record_key, section in sections.items():
            if not isinstance(section, dict) or "top10_recall" not in section:
                continue
            score = float(section["top10_recall"])
            current_best = best_records.get(record_key)
            if current_best is None or score > float(current_best["top10_recall"]):
                best_records[record_key] = {
                    "epoch": int(epoch),
                    "top10_recall": score,
                    "metrics_path": metrics_path,
                    "checkpoint_path": ckpt_path,
                    "metrics": section,
                }

    def _save_training_artifacts(self, train_curve, best_records):
        if self._report_only_eval():
            print("[Artifacts] skip curve/summary save (report_only_eval)")
            return None, None
        if self._use_standard_artifacts():
            report_dir = self._ensure_root_subdir("reports")
            report_prefix = self._artifact_prefix() or self.my_config.my_dict["train_flag"]
            curve_path = os.path.join(report_dir, "{}_curve.json".format(report_prefix))
            summary_path = os.path.join(report_dir, "{}_summary.json".format(report_prefix))
        else:
            feature_dir = self._ensure_root_subdir("feature_dir")
            curve_path = os.path.join(feature_dir, "curve_{}.json".format(self.my_config.my_dict["train_flag"]))
            summary_path = os.path.join(feature_dir, "summary_{}.json".format(self.my_config.my_dict["train_flag"]))
        with open(curve_path, "w", encoding="utf-8") as f:
            json.dump(train_curve, f, ensure_ascii=False, indent=2)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "train_flag": self.my_config.my_dict["train_flag"],
                "best_records": best_records,
                "curve_path": curve_path,
            }, f, ensure_ascii=False, indent=2)
        print("[Artifacts] curve saved to {}".format(curve_path))
        print("[Artifacts] summary saved to {}".format(summary_path))
        return curve_path, summary_path

    def run_smoke_test(self):
        my_net = function.initialize_model(self.my_config.my_dict, self.max_traj_length).to(self.my_config.my_dict["device"])
        my_loss = function.initialize_loss(self.my_config.my_dict).to(self.my_config.my_dict["device"])
        training_state = self._build_training_state()
        training_state = self._prepare_pre_quant_neighbor_teacher(training_state)
        training_state = self._prepare_pre_quant_landmark_teacher(training_state)
        self._initialize_pdt_codebook(my_net, init_limit=min(16, self.final_traj_train_num))
        self._maybe_refresh_pre_quant_landmark_bank(my_net, epoch=0, training_state=training_state)
        optimizer = self._build_optimizer(my_net, epoch=0, rebuild_reason="smoke_test")

        my_net.train()
        my_loss.init_loss(0)
        my_net.PDT_model.set_epoch(0)

        smoke_batch_size = min(int(self.my_config.my_dict.get("batch_size", 8)), 8)
        smoke_batches = self.generate_train_data(
            self.final_traj_train_num,
            self.final_train_length_list,
            smoke_batch_size,
            self.train_distance_matrix,
            self.my_config.my_dict["sampling_num"],
            self.my_config.my_dict["sampling_type"],
            self.train_knn,
            self.pad_total_train_lon_onehot,
            self.pad_total_train_lat_onehot,
            self.pad_total_train_lon_grid,
            self.pad_total_train_lat_grid,
            self.pad_total_train_lon_lat_image,
            epoch=0,
        )
        if not smoke_batches:
            raise ValueError("Smoke test could not create any training batch.")

        batch_stats = self._compute_batch_loss(my_net, my_loss, smoke_batches[0], epoch=0, training_state=training_state)
        if torch.isnan(batch_stats["loss"]):
            raise ValueError("Smoke test loss is NaN before backward.")

        optimizer.zero_grad()
        batch_stats["loss"].backward()

        grad_summary = {}
        tracked_prefixes = list(getattr(my_net, "smoke_grad_prefixes", ["fc2.", "PDT_model."]))
        if getattr(my_net, "pre_quant_bottleneck_enabled", False) and "pre_quant_bottleneck." not in tracked_prefixes:
            tracked_prefixes.append("pre_quant_bottleneck.")
        for prefix in tracked_prefixes:
            grad_value = None
            for name, parameter in my_net.named_parameters():
                if name.startswith(prefix) and parameter.grad is not None:
                    grad_value = float(parameter.grad.detach().norm().item())
                    break
            grad_summary[prefix] = grad_value
            if grad_value is None or not np.isfinite(grad_value):
                raise ValueError("Smoke test detected missing or invalid gradients for prefix {}.".format(prefix))

        optimizer.step()
        ckpt_path = self._save_checkpoint(my_net, 0)
        metrics_path = self.extract_feature_from_path(
            lon_grid_id_list=self.lon_grid_id_list,
            lat_grid_id_list=self.lat_grid_id_list,
            model_flag=self.my_config.my_dict["train_flag"],
            model_epoch=0,
            metrics_flag="smoke_{}".format(self.my_config.my_dict["train_flag"]),
        )

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_payload = json.load(f)

        summary = {
            "loss": self._scalar(batch_stats["loss"]),
            "rank_loss": self._scalar(batch_stats["rank_loss"]),
            "mse_loss": self._scalar(batch_stats["mse_loss"]),
            "pdt_loss": self._scalar(batch_stats["pdt_loss"]),
            "pre_quant_decor_loss": self._scalar(batch_stats["pre_quant_decor_loss"]),
            "pre_quant_stab_loss": self._scalar(batch_stats["pre_quant_stab_loss"]),
            "neighbor_btn_loss": self._scalar(batch_stats["neighbor_btn_loss"]),
            "neighbor_dec_loss": self._scalar(batch_stats["neighbor_dec_loss"]),
            "neighbor_effective_topk": int(batch_stats["neighbor_effective_topk"]),
            "neighbor_tau_btn": self._scalar(batch_stats["neighbor_tau_btn"]),
            "neighbor_tau_dec": self._scalar(batch_stats["neighbor_tau_dec"]),
            "landmark_btn_loss": self._scalar(batch_stats["landmark_btn_loss"]),
            "landmark_dec_loss": self._scalar(batch_stats["landmark_dec_loss"]),
            "landmark_btn_total": self._scalar(batch_stats["landmark_btn_total"]),
            "landmark_dec_total": self._scalar(batch_stats["landmark_dec_total"]),
            "landmark_weight_scale": self._scalar(batch_stats["landmark_weight_scale"]),
            "landmark_bank_epoch": int(batch_stats["landmark_bank_epoch"]),
            "landmark_effective_num": int(batch_stats["landmark_effective_num"]),
            "decoded_ste_metric_loss": self._scalar(batch_stats["decoded_ste_metric_loss"]),
            "decoded_ste_metric_weight": self._scalar(batch_stats["decoded_ste_metric_weight"]),
            "decoded_ste_hard_soft_gap": self._scalar(batch_stats["decoded_ste_hard_soft_gap"]),
            "e_cont_norm": self._scalar(batch_stats["e_cont_norm"]),
            "e_bottleneck_norm": self._scalar(batch_stats["e_bottleneck_norm"]),
            "e_hat_norm": self._scalar(batch_stats["e_hat_norm"]),
            "grad_summary": grad_summary,
            "checkpoint_path": ckpt_path,
            "metrics_path": metrics_path,
            "metrics_payload": metrics_payload,
        }
        if self._use_standard_artifacts():
            smoke_dir = self._ensure_root_subdir("reports")
            summary_path = os.path.join(smoke_dir, "{}_smoke_summary.json".format(self._artifact_prefix() or self.my_config.my_dict["train_flag"]))
        else:
            smoke_dir = self._ensure_root_subdir("feature_dir")
            summary_path = os.path.join(smoke_dir, "smoke_summary_{}.json".format(self.my_config.my_dict["train_flag"]))
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[SmokeTest] summary saved to {}".format(summary_path))
        return summary
    def load_dtw_matrix_auto(self):
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
    def data_prepare(self,
                     traj_list,
                     train_dist_matrix,
                     test_dist_matrix,
                     lon_onehot = None,
                     lat_onehot = None,
                     lon_grid_id_list = None,
                     lat_grid_id_list = None):
        print("Start Data Prepare...")
        
        self.traj_list        = traj_list
        self.traj_length_list = [len(traj) for traj in self.traj_list]
        self.total_traj_num   = len(self.traj_list)
        self.max_traj_length  = max(self.traj_length_list)
        self.min_traj_length  = min(self.traj_length_list)

        print("Total Traj Number: {}".format(self.total_traj_num))
        print("Max Traj Length : {}".format(self.max_traj_length))
        print("Min Traj Length : {}".format(self.min_traj_length))

        self.lon_onehot = lon_onehot
        self.lat_onehot = lat_onehot
        self.lon_grid_id_list = lon_grid_id_list
        self.lat_grid_id_list = lat_grid_id_list
        '''
        self.lon_lat_image = np.memmap('/data3/menghaotian/Traj_sim/datasets/features_sampling/cache_data/Porto_base_images_1000000.npy',dtype='float32', mode='r', shape=(1000000, self.my_config.my_dict["lon_input_size"], self.my_config.my_dict["lat_input_size"]))

        self.pad_total_lon_onehot = np.memmap('/data3/menghaotian/Traj_sim/datasets/features_sampling/cache_data/Porto_base_lon_pad_1000000.npy',dtype='float32', mode='r', shape=(1000000, self.max_traj_length,1))
        self.pad_total_lat_onehot = np.memmap('/data3/menghaotian/Traj_sim/datasets/features_sampling/cache_data/Porto_base_lat_pad_1000000.npy',dtype='float32', mode='r', shape=(1000000, self.max_traj_length,1))
        self.pad_total_lon_lat_image = self.lon_lat_image

        self.pad_total_lon_onehot    = torch.tensor(self.pad_total_lon_onehot, dtype = torch.float32)
        self.pad_total_lat_onehot    = torch.tensor(self.pad_total_lat_onehot, dtype = torch.float32)
        self.pad_total_lon_lat_image = torch.tensor(self.pad_total_lon_lat_image, dtype = torch.float32)
        '''
        
        if self.my_config.my_dict.get("embedding_backbone", "msr") == "msr":
            self.lon_lat_image = pre_rep.build_traj_image(lon_grid_id_list[:self.my_config.my_dict["train_set"]],
                                                          lat_grid_id_list[:self.my_config.my_dict["train_set"]],
                                                          self.my_config.my_dict["lon_input_size"],
                                                          self.my_config.my_dict["lat_input_size"],
                                                          image_mode=self.my_config.my_dict.get("image_mode", "binary"),
                                                          traj_list=self.traj_list[:self.my_config.my_dict["train_set"]])
        else:
            self.lon_lat_image = np.zeros((self.my_config.my_dict["train_set"], 1, 1, 1), dtype=np.float32)
        
        
        self.pad_total_lon_onehot    = function.pad_traj_list(self.lon_onehot[:self.my_config.my_dict["train_set"]], self.max_traj_length, pad_value = 0.0)  
        self.pad_total_lat_onehot    = function.pad_traj_list(self.lat_onehot[:self.my_config.my_dict["train_set"]], self.max_traj_length, pad_value = 0.0)  
        train_lon_grid_seq = [[[value] for value in traj] for traj in self.lon_grid_id_list[:self.my_config.my_dict["train_set"]]]
        train_lat_grid_seq = [[[value] for value in traj] for traj in self.lat_grid_id_list[:self.my_config.my_dict["train_set"]]]
        self.pad_total_lon_grid      = function.pad_traj_list(train_lon_grid_seq, self.max_traj_length, pad_value = 0.0)
        self.pad_total_lat_grid      = function.pad_traj_list(train_lat_grid_seq, self.max_traj_length, pad_value = 0.0)
        self.pad_total_lon_lat_image = self.lon_lat_image


        self.pad_total_lon_onehot    = torch.tensor(self.pad_total_lon_onehot[:self.my_config.my_dict["train_set"]], dtype = torch.float32)
        self.pad_total_lat_onehot    = torch.tensor(self.pad_total_lat_onehot[:self.my_config.my_dict["train_set"]], dtype = torch.float32)
        self.pad_total_lon_grid      = torch.tensor(self.pad_total_lon_grid[:self.my_config.my_dict["train_set"]], dtype = torch.float32)
        self.pad_total_lat_grid      = torch.tensor(self.pad_total_lat_grid[:self.my_config.my_dict["train_set"]], dtype = torch.float32)
        self.pad_total_lon_lat_image = torch.tensor(self.pad_total_lon_lat_image[:self.my_config.my_dict["train_set"]], dtype = torch.float32)
        
        print("The Size Of Total Seq Pad List Is: ", self.pad_total_lon_onehot.size())

        self.total_train_lon_onehot = self.lon_onehot
        self.total_train_lat_onehot = self.lat_onehot
        self.pad_total_train_lon_onehot    = self.pad_total_lon_onehot[:self.my_config.my_dict["train_set"]]
        self.pad_total_train_lat_onehot    = self.pad_total_lat_onehot[:self.my_config.my_dict["train_set"]]
        self.pad_total_train_lon_grid      = self.pad_total_lon_grid[:self.my_config.my_dict["train_set"]]
        self.pad_total_train_lat_grid      = self.pad_total_lat_grid[:self.my_config.my_dict["train_set"]]
        self.pad_total_train_lon_lat_image = self.pad_total_lon_lat_image[:self.my_config.my_dict["train_set"]]
        self.train_traj_length_list = self.traj_length_list[:self.my_config.my_dict["train_set"]]


        '''
        self.total_train_lon_onehot = self.lon_onehot
        self.total_train_lat_onehot = self.lat_onehot
        '''



        self.train_distance_matrix = train_dist_matrix
        self.train_distance_matrix = self.train_distance_matrix[:self.my_config.my_dict["train_set"]]
        

        #if self.my_config.my_dict["dist_type"] == "dtw" or self.my_config.my_dict["dist_type"] == "edr":
        #    self.train_distance_matrix = self.train_distance_matrix / np.max(self.train_distance_matrix)
            
        
        self.avg_distance = np.mean(self.train_distance_matrix)
        self.max_distance = np.max(self.train_distance_matrix)
        print("Train Matrix size : {}".format(self.train_distance_matrix.shape))
        #print("Test  Matrix size : {}".format(self.test_distance_matrix.shape))
        print("Train Avg Distance: {}".format(self.avg_distance))
        print("Train Max Distance: {}".format(self.max_distance))

        self.final_traj_train_num = int(self.my_config.my_dict["train_set"] * self.my_config.my_dict["train_ratio"])
        print("Num Of Train Traj Is: {}".format(self.final_traj_train_num))
        print("Generating Train Traj Length List...")
        self.final_train_length_list = []
        for i in range(self.final_traj_train_num):
            self.final_train_length_list.append(self.traj_length_list[:self.final_traj_train_num])

        self.train_knn = np.empty(dtype=np.int32, shape=(self.final_traj_train_num, self.final_traj_train_num))
        for i in range(self.final_traj_train_num):
            self.train_knn[i] = np.argsort(self.train_distance_matrix[i][:self.final_traj_train_num])
        self.test_knn = test_dist_matrix
        #self.test_distance_matrix  = test_dist_matrix
        #if self.my_config.my_dict["dist_type"] == "dtw" or self.my_config.my_dict["dist_type"] == "edr":
            #self.test_distance_matrix  = self.test_distance_matrix / np.max(self.train_distance_matrix)
        '''
        self.test_knn = np.empty(dtype=np.int32, shape=(self.my_config.my_dict["query_set"], self.my_config.my_dict["base_set"]))
        for i in range(self.my_config.my_dict["query_set"]):
            self.test_knn[i] = np.argsort(self.test_distance_matrix[i])
        '''
        

        print("End Data Prepare !!!")
          
    def generate_train_data(self,
                            final_traj_train_num,
                            final_train_length_list,
                            batch_size,
                            train_distance_matrix,
                            sampling_num,
                            sampling_type,
                            test_knn,
                            lon_onehot,
                            lat_onehot,
                            lon_grid,
                            lat_grid,
                            lon_lat_image,
                            epoch):

        new_list = [[i, final_train_length_list[i]] for i in range(final_traj_train_num)]
        new_list = [x[0] for x in new_list]
        total_result = []
        for i in range(0, final_traj_train_num, batch_size):
            anchor_lon,         positive_lon,         negative_lon       = [], [], []
            anchor_lat,         positive_lat,         negative_lat       = [], [], []
            anchor_lon_grid,    positive_lon_grid,    negative_lon_grid  = [], [], []
            anchor_lat_grid,    positive_lat_grid,    negative_lat_grid  = [], [], []
            anchor_image,       positive_image,       negative_image     = [], [], []
            anchor_length,      positive_length,      negative_length    = [], [], []
            positive_distance,  negative_distance,    cross_distance     = [], [], []
            anchor_sample_idx,  positive_sample_idx,  negative_sample_idx = [], [], []
            for j in range(batch_size):
                if i + j >= final_traj_train_num:
                    break
                anchor_pos = new_list[(i + j)]
                
                positive_sampling_index_list, negative_sampling_index_list = sampling_methods.main_triplet_selection(sampling_type, sampling_num, test_knn[anchor_pos], train_distance_matrix[anchor_pos], anchor_pos, final_train_length_list[anchor_pos], final_train_length_list, epoch)

                # cross distance
                for k in range(len(positive_sampling_index_list)):
                    cross_distance.append(train_distance_matrix[positive_sampling_index_list[k]][negative_sampling_index_list[k]])

                # positive distance
                for positive_index in positive_sampling_index_list:
                    anchor_lon.append(lon_onehot[anchor_pos])
                    anchor_lat.append(lat_onehot[anchor_pos])
                    anchor_lon_grid.append(lon_grid[anchor_pos])
                    anchor_lat_grid.append(lat_grid[anchor_pos])
                    anchor_image.append(lon_lat_image[anchor_pos])
                    anchor_sample_idx.append(anchor_pos)
                    
                    positive_lon.append(lon_onehot[positive_index])
                    positive_lat.append(lat_onehot[positive_index])
                    positive_lon_grid.append(lon_grid[positive_index])
                    positive_lat_grid.append(lat_grid[positive_index])
                    positive_image.append(lon_lat_image[positive_index])
                    positive_sample_idx.append(positive_index)

                    anchor_length.append(self.traj_length_list[anchor_pos])
                    positive_length.append(self.traj_length_list[positive_index])

                    positive_distance.append(train_distance_matrix[anchor_pos][positive_index])
                
                # negative distance
                for negative_index in negative_sampling_index_list:
                    negative_lon.append(lon_onehot[negative_index])
                    negative_lat.append(lat_onehot[negative_index])
                    negative_lon_grid.append(lon_grid[negative_index])
                    negative_lat_grid.append(lat_grid[negative_index])
                    negative_image.append(lon_lat_image[negative_index])
                    negative_sample_idx.append(negative_index)
                    
                    negative_length.append(self.traj_length_list[negative_index])

                    negative_distance.append(train_distance_matrix[anchor_pos][negative_index])
            
            tem_batch = ([anchor_lon,        positive_lon,     negative_lon], 
                         [anchor_lat,        positive_lat,     negative_lat], 
                         [anchor_lon_grid,   positive_lon_grid, negative_lon_grid],
                         [anchor_lat_grid,   positive_lat_grid, negative_lat_grid],
                         [anchor_image,      positive_image,     negative_image], 
                         [anchor_length,     positive_length,    negative_length], 
                         [positive_distance, negative_distance,  cross_distance],
                         [anchor_sample_idx, positive_sample_idx, negative_sample_idx])
            total_result.append(tem_batch)
        return total_result


    def get_embeddings(self, my_net, test_batch, total_traj_num):
        outputs = self._collect_embedding_outputs(my_net,
                                                  self.pad_total_lon_onehot,
                                                  self.pad_total_lat_onehot,
                                                  self.pad_total_lon_lat_image,
                                                  self.traj_length_list[:self.pad_total_lon_onehot.shape[0]],
                                                  lon_grid_tensor=self.pad_total_lon_grid,
                                                  lat_grid_tensor=self.pad_total_lat_grid,
                                                  test_batch=test_batch,
                                                  embedding_type="quantized",
                                                  collect_code_usage=False)
        return outputs["quantized"].numpy()

    def train(self):
        my_net    = function.initialize_model(self.my_config.my_dict, self.max_traj_length).to(self.my_config.my_dict["device"])
        my_loss   = function.initialize_loss(self.my_config.my_dict).to(self.my_config.my_dict["device"])

        backbone_checkpoint = self.my_config.my_dict.get("backbone_checkpoint")
        if backbone_checkpoint:
            load_stats = my_net.load_backbone_checkpoint(backbone_checkpoint)
            print("[Checkpoint] Loaded MSR backbone from:", backbone_checkpoint)
            print("[Checkpoint] loaded keys:", load_stats["loaded_key_count"], "| skipped keys:", load_stats["skipped_key_count"])

        self._initialize_pdt_codebook(my_net)
        training_state = self._build_training_state()
        training_state = self._prepare_pre_quant_neighbor_teacher(training_state)
        training_state = self._prepare_pre_quant_landmark_teacher(training_state)
        if training_state["loss_recipe"] == "improved_vq" and not training_state["improved_vq_adaptive_low_codebook"]:
            print("[ImprovedVQ] explicit schedule mode | pdt_m: {} | pdt_start: {} | pdt_weight: {:.3f} | qm_start: {} | qm_max: {:.4f} | pairwise: {:.4f} | entropy: {:.4f} | commit: {:.4f} | uniform: {:.6f}".format(
                training_state["pdt_m"],
                training_state["effective_pdt_loss_start_epoch"],
                training_state["effective_pdt_loss_weight"],
                training_state["adaptive_qm_start_epoch"],
                training_state["adaptive_qm_max_weight"],
                training_state["adaptive_pairwise_weight"],
                training_state["adaptive_entropy_weight"],
                training_state["adaptive_commit_weight"],
                training_state["adaptive_uniform_weight"],
            ))
        if training_state["loss_recipe"] == "improved_vq" and training_state["low_codebook_risk"] > 0.0:
            print("[ImprovedVQ] adaptive stabilization | pdt_m: {} | risk: {:.3f} | qm_start: {} | qm_max: {:.4f} | pairwise: {:.4f} | entropy: {:.4f} | commit: {:.4f} | uniform: {:.6f} | refresh_epochs: {}".format(
                training_state["pdt_m"],
                training_state["low_codebook_risk"],
                training_state["adaptive_qm_start_epoch"],
                training_state["adaptive_qm_max_weight"],
                training_state["adaptive_pairwise_weight"],
                training_state["adaptive_entropy_weight"],
                training_state["adaptive_commit_weight"],
                training_state["adaptive_uniform_weight"],
                sorted(training_state["low_m_refresh_epochs"])
            ))
        if training_state["medium_codebook_mode"]:
            print("[ImprovedVQ] medium-codebook mode | pdt_m: {} | pdt_start: {} | pdt_weight: {:.3f} | soft_qm: disabled | pairwise: disabled".format(
                training_state["pdt_m"],
                training_state["effective_pdt_loss_start_epoch"],
                training_state["effective_pdt_loss_weight"]
            ))
        if training_state["high_capacity_mode"]:
            print("[ImprovedVQ] high-capacity mode | pdt_m: {} | pdt_start: {} | pdt_weight: {:.3f} | soft_qm: disabled | pairwise: disabled | reg_scale: {:.3f}".format(
                training_state["pdt_m"],
                training_state["effective_pdt_loss_start_epoch"],
                training_state["effective_pdt_loss_weight"],
                training_state["high_capacity_reg_scale"]
            ))
        if training_state["pre_quant_enabled"]:
            print("[PreQuant] lambda_decor: {:.4f} | lambda_stab: {:.4f} | late_epoch: {} | late_multiplier: {:.2f} | refresh_end: {} | refresh_epochs: {}".format(
                training_state["pre_quant_lambda_decor"],
                training_state["pre_quant_lambda_stab"],
                training_state["pre_quant_stab_late_epoch"],
                training_state["pre_quant_stab_late_multiplier"],
                training_state["pre_quant_refresh_end_epoch"],
                sorted(training_state["pre_quant_refresh_epochs"]),
            ))
        if training_state["pre_quant_neighbor_enabled"]:
            print("[PreQuantNeighbor] teacher_mode: {} | offline_path: {} | use_btn: {} | use_dec: {} | topk: {} | tau_btn: {:.4f} | tau_dec: {:.4f} | lambda_btn: {:.4f} | lambda_dec: {:.4f} | start_epoch: {} | warmup_epochs: {} | dec_stop_backbone: {}".format(
                training_state["pre_quant_neighbor_teacher_mode"],
                training_state["pre_quant_neighbor_offline_path"],
                training_state["pre_quant_neighbor_use_btn"],
                training_state["pre_quant_neighbor_use_dec"],
                training_state["pre_quant_neighbor_topk"],
                training_state["pre_quant_neighbor_tau_btn"],
                training_state["pre_quant_neighbor_tau_dec"],
                training_state["pre_quant_neighbor_lambda_btn"],
                training_state["pre_quant_neighbor_lambda_dec"],
                training_state["pre_quant_neighbor_start_epoch"],
                training_state["pre_quant_neighbor_warmup_epochs"],
                training_state["pre_quant_neighbor_dec_stop_backbone"],
            ))
        if training_state["pre_quant_landmark_enabled"]:
            print("[PreQuantLandmark] artifact_path: {} | fixed_bank_path: {} | teacher_checkpoint: {} | use_btn: {} | use_dec: {} | dec_ste_to_btn: {} | dec_bank_source: {} | rank_weight: {:.4f} | rank_tau: {:.4f} | landmarks: {} | select: {} | transform: {} | lambda_btn: {:.4f} | lambda_dec: {:.4f} | start_epoch: {} | warmup_epochs: {} | refresh_epochs: {}".format(
                training_state["pre_quant_landmark_artifact_path"],
                training_state["pre_quant_landmark_fixed_bank_path"],
                training_state["pre_quant_landmark_teacher_checkpoint"],
                training_state["pre_quant_landmark_use_btn"],
                training_state["pre_quant_landmark_use_dec"],
                training_state["pre_quant_landmark_dec_ste_to_btn"],
                training_state["pre_quant_landmark_dec_bank_source"],
                training_state["pre_quant_landmark_rank_weight"],
                training_state["pre_quant_landmark_rank_tau"],
                training_state["pre_quant_landmark_num"],
                training_state["pre_quant_landmark_select"],
                training_state["pre_quant_landmark_profile_transform"],
                training_state["pre_quant_landmark_lambda_btn"],
                training_state["pre_quant_landmark_lambda_dec"],
                training_state["pre_quant_landmark_start_epoch"],
                training_state["pre_quant_landmark_warmup_epochs"],
                training_state["pre_quant_landmark_bank_refresh_epochs"],
            ))
        if training_state["decoded_ste_metric_enabled"]:
            print("[DecodedSTEMetric] start_epoch: {} | warmup_epochs: {} | max_weight: {:.4f}".format(
                training_state["decoded_ste_metric_start_epoch"],
                training_state["decoded_ste_metric_warmup_epochs"],
                training_state["decoded_ste_metric_max_weight"],
            ))
        if training_state["late_finetune_enabled"]:
            print("[LateFinetune] start_epoch: {} | main_lr_scale: {:.4f} | pre_quant_lr_scale: {:.4f}".format(
                training_state["late_finetune_start_epoch"],
                training_state["late_finetune_main_lr_scale"],
                training_state["late_finetune_pre_quant_lr_scale"],
            ))
        current_freeze_state = None
        current_optimizer_phase = None
        optimizer = None
        milestone_epochs = self._parse_epoch_list(self.my_config.my_dict.get("eval_save_epochs", ""))
        if milestone_epochs:
            print("[MilestoneEval] epochs: {}".format(sorted(milestone_epochs)))
        max_train_batches_per_epoch = int(self.my_config.my_dict.get("max_train_batches_per_epoch", 0))
        train_curve = []
        best_records = {
            "continuous": None,
            "decoded": None,
            "strict_rerank": None,
        }

        for name,parameters in my_net.named_parameters():
            print(name,':',parameters.size())
        trainable_num = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
        print("Total Trainable Parameter Num:", trainable_num)

        for epoch in range(self.my_config.my_dict["epoch_num"]):
            start_time = time.time()

            freeze_backbone = epoch < training_state["freeze_backbone_epochs"]
            optimizer_phase = self._optimizer_phase(epoch)
            if current_freeze_state is None or current_freeze_state != freeze_backbone:
                current_freeze_state = freeze_backbone
                current_optimizer_phase = optimizer_phase
                stage_name = self._set_backbone_trainable(my_net, freeze_backbone)
                optimizer = self._build_optimizer(my_net, epoch=epoch, rebuild_reason="stage_change")
                trainable_num = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
                print("[TrainStage] epoch {} -> {} | trainable parameters: {}".format(epoch, stage_name, trainable_num))
            elif current_optimizer_phase != optimizer_phase:
                current_optimizer_phase = optimizer_phase
                optimizer = self._build_optimizer(my_net, epoch=epoch, rebuild_reason=optimizer_phase)
                print("[TrainStage] epoch {} -> optimizer rebuilt for {} phase".format(epoch, optimizer_phase))

            if epoch in training_state["all_refresh_epochs"]:
                self._refresh_pdt_codebook(my_net, epoch)
                optimizer = self._build_optimizer(my_net, epoch=epoch, rebuild_reason="codebook_refresh")
                print("[TrainStage] epoch {} -> optimizer rebuilt after codebook refresh".format(epoch))

            self._maybe_refresh_pre_quant_landmark_bank(my_net, epoch=epoch, training_state=training_state)

            my_net.train()
            my_loss.init_loss(epoch)
            my_net.PDT_model.set_epoch(epoch)
            
            train_data = self.generate_train_data(self.final_traj_train_num,
                                                  self.final_train_length_list,
                                                  self.my_config.my_dict["batch_size"],
                                                  self.train_distance_matrix,
                                                  self.my_config.my_dict["sampling_num"],
                                                  self.my_config.my_dict["sampling_type"],
                                                  self.train_knn,
                                                  self.pad_total_train_lon_onehot,
                                                  self.pad_total_train_lat_onehot,
                                                  self.pad_total_train_lon_grid,
                                                  self.pad_total_train_lat_grid,
                                                  self.pad_total_train_lon_lat_image,
                                                  epoch)
            if max_train_batches_per_epoch > 0:
                train_data = train_data[:max_train_batches_per_epoch]

            last_stats = None
            for i, batch in enumerate(train_data):
                last_stats = self._compute_batch_loss(my_net, my_loss, batch, epoch, training_state)
                optimizer.zero_grad()
                last_stats["loss"].backward()
                optimizer.step()
            end_time = time.time()

            if last_stats is None:
                raise ValueError("No training batch was processed at epoch {}.".format(epoch))

            train_curve.append({
                "epoch": epoch + 1,
                "stage": "pdt_only" if epoch < training_state["freeze_backbone_epochs"] else "joint",
                "optimizer_phase": current_optimizer_phase,
                "loss": self._scalar(last_stats["loss"]),
                "rank_loss": self._scalar(last_stats["rank_loss"]),
                "mse_loss": self._scalar(last_stats["mse_loss"]),
                "metric_loss": self._scalar(last_stats["metric_loss"]),
                "pdt_loss": self._scalar(last_stats["pdt_loss"]),
                "consistency_loss": self._scalar(last_stats["consistency_loss"]),
                "quantized_metric_loss": self._scalar(last_stats["quantized_metric_loss"]),
                "pairwise_consistency_loss": self._scalar(last_stats["pairwise_consistency_loss"]),
                "decoded_ste_metric_loss": self._scalar(last_stats["decoded_ste_metric_loss"]),
                "decoded_ste_metric_weight": self._scalar(last_stats["decoded_ste_metric_weight"]),
                "decoded_ste_hard_soft_gap": self._scalar(last_stats["decoded_ste_hard_soft_gap"]),
                "entropy_reg_loss": self._scalar(last_stats["entropy_reg_loss"]),
                "commitment_reg_loss": self._scalar(last_stats["commitment_reg_loss"]),
                "uniform_reg_loss": self._scalar(last_stats["uniform_reg_loss"]),
                "quantized_metric_effective_weight": self._scalar(last_stats["quantized_metric_effective_weight"]),
                "raw_metric_loss": self._scalar(last_stats["raw_metric_loss"]),
                "raw_metric_weight": self._scalar(last_stats["raw_metric_weight"]),
                "pre_quant_decor_loss": self._scalar(last_stats["pre_quant_decor_loss"]),
                "pre_quant_stab_loss": self._scalar(last_stats["pre_quant_stab_loss"]),
                "pre_quant_total_loss": self._scalar(last_stats["pre_quant_total_loss"]),
                "pre_quant_lambda_decor": self._scalar(last_stats["pre_quant_lambda_decor"]),
                "pre_quant_lambda_stab": self._scalar(last_stats["pre_quant_lambda_stab"]),
                "loss_nbr_btn": self._scalar(last_stats["neighbor_btn_loss"]),
                "loss_nbr_dec": self._scalar(last_stats["neighbor_dec_loss"]),
                "loss_nbr_btn_weighted": self._scalar(last_stats["neighbor_btn_total"]),
                "loss_nbr_dec_weighted": self._scalar(last_stats["neighbor_dec_total"]),
                "effective_topk": int(last_stats["neighbor_effective_topk"]),
                "tau_btn": self._scalar(last_stats["neighbor_tau_btn"]),
                "tau_dec": self._scalar(last_stats["neighbor_tau_dec"]),
                "neighbor_btn_aux": last_stats.get("neighbor_btn_aux", {}),
                "neighbor_dec_aux": last_stats.get("neighbor_dec_aux", {}),
                "loss_lm_btn": self._scalar(last_stats["landmark_btn_loss"]),
                "loss_lm_dec": self._scalar(last_stats["landmark_dec_loss"]),
                "loss_lm_btn_weighted": self._scalar(last_stats["landmark_btn_total"]),
                "loss_lm_dec_weighted": self._scalar(last_stats["landmark_dec_total"]),
                "landmark_weight_scale": self._scalar(last_stats["landmark_weight_scale"]),
                "landmark_bank_epoch": int(last_stats["landmark_bank_epoch"]),
                "landmark_effective_num": int(last_stats["landmark_effective_num"]),
                "landmark_btn_aux": last_stats.get("landmark_btn_aux", {}),
                "landmark_dec_aux": last_stats.get("landmark_dec_aux", {}),
                "e_cont_norm": self._scalar(last_stats["e_cont_norm"]),
                "e_bottleneck_norm": self._scalar(last_stats["e_bottleneck_norm"]),
                "e_hat_norm": self._scalar(last_stats["e_hat_norm"]),
                "time_sec": float(end_time - start_time),
                "bottleneck_aux": last_stats.get("bottleneck_aux", {}),
            })

            if (epoch + 1) % self.my_config.my_dict["print_epoch"] == 0:
                stage_name = "pdt_only" if epoch < training_state["freeze_backbone_epochs"] else "joint"
                print('Print Epoch: [{:3d}/{:3d}] Stage: {}, Recipe: {}, Rank Loss: {:.4f}, Mse Loss: {:.4f}, PDT Loss: {:.4f}, Consistency Loss: {:.4f}, Quantized Metric Loss: {:.4f}, Quantized Weight: {:.4f}, Raw Metric Loss: {:.4f}, Raw Metric Weight: {:.4f}, Pairwise Loss: {:.4f}, SteDec Loss: {:.4f}, SteDec Weight: {:.4f}, SteGap: {:.4f}, Entropy Loss: {:.4f}, Commit Loss: {:.4f}, Uniform Loss: {:.4f}, PreQuant Decor: {:.4f}, PreQuant Stab: {:.4f}, PreQuant Total: {:.4f}, PreQuant LambdaD: {:.4f}, PreQuant LambdaS: {:.4f}, NbrBtn: {:.4f}, NbrDec: {:.4f}, TopK: {}, TauBtn: {:.4f}, TauDec: {:.4f}, LmBtn: {:.4f}, LmDec: {:.4f}, LmScale: {:.4f}, LmBank: {}, LmNum: {}, EContNorm: {:.4f}, EBtnNorm: {:.4f}, EHatNorm: {:.4f}, Total Loss: {:.4f}, Time: {:.4f}'.format(
                    epoch,
                    self.my_config.my_dict["epoch_num"],
                    stage_name,
                    training_state["loss_recipe"],
                    self._scalar(last_stats["rank_loss"]),
                    self._scalar(last_stats["mse_loss"]),
                    self._scalar(last_stats["pdt_loss"]),
                    self._scalar(last_stats["consistency_loss"]),
                    self._scalar(last_stats["quantized_metric_loss"]),
                    self._scalar(last_stats["quantized_metric_effective_weight"]),
                    self._scalar(last_stats["raw_metric_loss"]),
                    self._scalar(last_stats["raw_metric_weight"]),
                    self._scalar(last_stats["pairwise_consistency_loss"]),
                    self._scalar(last_stats["decoded_ste_metric_loss"]),
                    self._scalar(last_stats["decoded_ste_metric_weight"]),
                    self._scalar(last_stats["decoded_ste_hard_soft_gap"]),
                    self._scalar(last_stats["entropy_reg_loss"]),
                    self._scalar(last_stats["commitment_reg_loss"]),
                    self._scalar(last_stats["uniform_reg_loss"]),
                    self._scalar(last_stats["pre_quant_decor_loss"]),
                    self._scalar(last_stats["pre_quant_stab_loss"]),
                    self._scalar(last_stats["pre_quant_total_loss"]),
                    self._scalar(last_stats["pre_quant_lambda_decor"]),
                    self._scalar(last_stats["pre_quant_lambda_stab"]),
                    self._scalar(last_stats["neighbor_btn_loss"]),
                    self._scalar(last_stats["neighbor_dec_loss"]),
                    int(last_stats["neighbor_effective_topk"]),
                    self._scalar(last_stats["neighbor_tau_btn"]),
                    self._scalar(last_stats["neighbor_tau_dec"]),
                    self._scalar(last_stats["landmark_btn_loss"]),
                    self._scalar(last_stats["landmark_dec_loss"]),
                    self._scalar(last_stats["landmark_weight_scale"]),
                    int(last_stats["landmark_bank_epoch"]),
                    int(last_stats["landmark_effective_num"]),
                    self._scalar(last_stats["e_cont_norm"]),
                    self._scalar(last_stats["e_bottleneck_norm"]),
                    self._scalar(last_stats["e_hat_norm"]),
                    self._scalar(last_stats["loss"]),
                    (end_time - start_time),
                ))
                

            current_epoch = epoch + 1
            test_due = (current_epoch % self.my_config.my_dict["test_epoch"] == 0)
            milestone_due = current_epoch in milestone_epochs
            should_eval = test_due or milestone_due or current_epoch == self.my_config.my_dict["epoch_num"]
            if should_eval:
                metrics_flag = self._artifact_tag_for_epoch(current_epoch)
                if self._report_only_eval():
                    save_model_name = None
                    metrics_payload, metrics_path = self.extract_feature_payload(
                        my_net=my_net,
                        model_epoch=current_epoch,
                        metrics_flag=metrics_flag,
                        save_metrics=False,
                    )
                else:
                    save_model_name = self._save_checkpoint(my_net, current_epoch)
                    metrics_payload, metrics_path = self.extract_feature_payload(
                        model_flag=self.my_config.my_dict["train_flag"],
                        model_epoch=current_epoch,
                        metrics_flag=metrics_flag,
                        save_metrics=True,
                    )
                self._update_best_records(best_records, metrics_payload, save_model_name, metrics_path, current_epoch)
                print("[RESULT] eval_epoch={}".format(current_epoch))
                if metrics_path is None:
                    print("[RESULT] metrics_path=not_saved(report_only_eval)")
                else:
                    print("[RESULT] metrics_path={}".format(metrics_path))
                print("[RESULT] ckpt_path={}".format(save_model_name))

        self._save_training_artifacts(train_curve, best_records)

                
  


    def extract_feature_payload(self,
                                my_net=None,
                                model_flag=None,
                                model_epoch=None,
                                metrics_flag=None,
                                save_metrics=True):
        model_flag = model_flag or self.my_config.my_dict.get("load_model_train_flag") or self.my_config.my_dict["train_flag"]
        model_epoch = self.my_config.my_dict["epoch_num"] if model_epoch is None else int(model_epoch)
        metrics_flag = metrics_flag or self.my_config.my_dict["train_flag"]
        if my_net is None:
            model_path = self._checkpoint_path(model_epoch, model_flag)
            my_net = function.initialize_model(self.my_config.my_dict, self.max_traj_length).to(self.my_config.my_dict["device"])
            my_net.load_state_dict(torch.load(model_path, map_location=self.my_config.my_dict["device"]))
        eval_embedding_type = self.my_config.my_dict.get("eval_embedding_type", "quantized")
        eval_search_mode = self.my_config.my_dict.get("eval_search_mode", "decoded")
        enable_rerank = bool(self.my_config.my_dict.get("enable_rerank", False))
        rerank_L = int(self.my_config.my_dict.get("rerank_L", 100))
        rerank_source = self.my_config.my_dict.get("rerank_source", "decoded")
        collect_code_usage = bool(self.my_config.my_dict.get("print_code_usage", True))
        need_continuous = eval_embedding_type in ["continuous", "both"] or enable_rerank
        need_quantized = eval_embedding_type in ["quantized", "both"] or eval_search_mode in ["decoded", "both"] or (enable_rerank and rerank_source == "decoded")
        need_transformed = False
        need_reconstructed = False
        collect_raw_continuous = bool(
            getattr(my_net, "pre_quant_bottleneck_enabled", False)
            and eval_embedding_type in ["continuous", "both"]
        )

        if need_continuous and need_quantized:
            collect_embedding_type = "both"
        elif need_continuous:
            collect_embedding_type = "continuous"
        else:
            collect_embedding_type = "quantized"

        if enable_rerank:
            need_transformed = True
            need_reconstructed = True

        eval_chunk_size = 10240
        eval_batch_size = 1024
        if bool(getattr(my_net, "pre_quant_bottleneck_enabled", False)) and self.my_config.my_dict.get("embedding_backbone", "msr") != "msr":
            eval_chunk_size = min(2048, len(self.traj_list))
            eval_batch_size = min(256, max(64, int(self.my_config.my_dict.get("batch_size", 256))))

        total_continuous_embeddings = []
        total_raw_continuous_embeddings = []
        total_quantized_embeddings = []
        total_codes = []
        total_transformed_embeddings = []
        total_reconstructed_embeddings = []
        begin_pos, end_pos = 0, min(eval_chunk_size, len(self.traj_list))
        total_time = 0
        while True:
            print(begin_pos, end_pos)
            self.pad_total_lon_onehot, self.pad_total_lat_onehot, self.pad_total_lon_lat_image, eval_seq_lengths, self.pad_total_lon_grid, self.pad_total_lat_grid = self._build_eval_tensors(begin_pos, end_pos)

            time1 = time.time()
            outputs = self._collect_embedding_outputs(my_net,
                                                      self.pad_total_lon_onehot,
                                                      self.pad_total_lat_onehot,
                                                      self.pad_total_lon_lat_image,
                                                      eval_seq_lengths,
                                                      lon_grid_tensor=self.pad_total_lon_grid,
                                                      lat_grid_tensor=self.pad_total_lat_grid,
                                                      test_batch=eval_batch_size,
                                                      embedding_type=collect_embedding_type,
                                                      collect_code_usage=collect_code_usage or need_transformed,
                                                      collect_transformed=need_transformed,
                                                      collect_reconstructed=need_reconstructed,
                                                      collect_raw_continuous=collect_raw_continuous)
            if "continuous" in outputs:
                total_continuous_embeddings.append(outputs["continuous"].numpy())
            if "continuous_raw" in outputs:
                total_raw_continuous_embeddings.append(outputs["continuous_raw"].numpy())
            if "quantized" in outputs:
                total_quantized_embeddings.append(outputs["quantized"].numpy())
            if "codes" in outputs:
                total_codes.append(outputs["codes"].numpy())
            if "transformed" in outputs:
                total_transformed_embeddings.append(outputs["transformed"].numpy())
            if "reconstructed" in outputs:
                total_reconstructed_embeddings.append(outputs["reconstructed"].numpy())
            time2 = time.time()
            total_time += time2 - time1
            if end_pos == len(self.traj_list):
                break
            begin_pos = end_pos
            end_pos += eval_chunk_size
            if end_pos > len(self.traj_list):
                end_pos = len(self.traj_list)

        time1 = time.time()
        if total_continuous_embeddings:
            total_continuous_embeddings = np.concatenate(total_continuous_embeddings, axis=0)
        if total_raw_continuous_embeddings:
            total_raw_continuous_embeddings = np.concatenate(total_raw_continuous_embeddings, axis=0)
        if total_quantized_embeddings:
            total_quantized_embeddings = np.concatenate(total_quantized_embeddings, axis=0)
        if total_codes:
            total_codes = np.concatenate(total_codes, axis=0)
        if total_transformed_embeddings:
            total_transformed_embeddings = np.concatenate(total_transformed_embeddings, axis=0)
        if total_reconstructed_embeddings:
            total_reconstructed_embeddings = np.concatenate(total_reconstructed_embeddings, axis=0)
        time2 = time.time()
        total_time += time2 - time1
        if isinstance(total_continuous_embeddings, np.ndarray):
            print("Continuous embedding shape:", total_continuous_embeddings.shape)
        if isinstance(total_quantized_embeddings, np.ndarray):
            print("Quantized embedding shape:", total_quantized_embeddings.shape)
        print("Total inference time: ", total_time)

        metrics_payload = {
            "evaluated_embedding_type": eval_embedding_type,
            "evaluated_search_mode": eval_search_mode,
            "rerank_enabled": enable_rerank,
            "rerank_L": rerank_L if enable_rerank else None,
            "rerank_source": rerank_source if enable_rerank else None,
            "rerank_protocol": "pdtvq_strict" if enable_rerank else None,
            "pdt_vq_type": self.my_config.my_dict.get("pdt_vq_type", "dpq"),
        }
        if isinstance(total_codes, np.ndarray) and collect_code_usage:
            metrics_payload["code_usage"] = self._summarize_code_usage(torch.from_numpy(total_codes))

        if eval_embedding_type in ["continuous", "both"]:
            continuous_metrics = test_methods.test_all_log(total_continuous_embeddings,
                                                           self.my_config,
                                                           "feature_continuous",
                                                           self.train_distance_matrix,
                                                           self.test_knn,
                                                           epoch=model_epoch)
            metrics_payload["continuous"] = continuous_metrics
            if isinstance(total_raw_continuous_embeddings, np.ndarray):
                metrics_payload["continuous_raw"] = test_methods.test_all_log(total_raw_continuous_embeddings,
                                                                               self.my_config,
                                                                               "feature_continuous_raw",
                                                                               self.train_distance_matrix,
                                                                               self.test_knn,
                                                                               epoch=model_epoch)

        decoded_pred_knn = None
        if need_quantized and eval_search_mode in ["decoded", "both"]:
            decoded_metrics = test_methods.test_all_log(total_quantized_embeddings,
                                                        self.my_config,
                                                        "feature_decoded",
                                                        self.train_distance_matrix,
                                                        self.test_knn,
                                                        epoch=model_epoch)
            metrics_payload["decoded"] = decoded_metrics
            decoded_query, decoded_base = self._split_eval_array(total_quantized_embeddings)
            decoded_distance = test_methods.get_feature_distance(decoded_query, decoded_base)
            decoded_pred_knn = np.argsort(decoded_distance, axis=1)[:, :max(100, rerank_L)]

        adc_pred_knn = None
        if eval_search_mode in ["adc", "both"]:
            if isinstance(total_continuous_embeddings, np.ndarray) and isinstance(total_quantized_embeddings, np.ndarray):
                adc_query, _ = self._split_eval_array(total_continuous_embeddings)
                _, adc_base = self._split_eval_array(total_quantized_embeddings)
                adc_distance = test_methods.get_feature_distance(adc_query, adc_base)
                adc_pred_knn = np.argsort(adc_distance, axis=1)[:, :max(100, rerank_L)]
                metrics_payload["adc"] = test_methods.metrics_from_pred_knn(self.test_knn, adc_pred_knn)
            else:
                metrics_payload["adc"] = {
                    "available": False,
                    "reason": "ADC-style asymmetric evaluation requires both continuous query and quantized base embeddings."
                }

        if enable_rerank:
            if rerank_source == "decoded":
                coarse_pred_knn, base_reconstructed_model = self._compute_pdtvq_coarse_pred_knn(total_transformed_embeddings,
                                                                                                total_reconstructed_embeddings,
                                                                                                total_codes,
                                                                                                rerank_L=rerank_L,
                                                                                                pdt_vq_type=self.my_config.my_dict.get("pdt_vq_type", "dpq"))
            elif rerank_source == "adc":
                coarse_pred_knn = adc_pred_knn
                base_reconstructed_model = None
            else:
                coarse_pred_knn = None
                base_reconstructed_model = None

            if coarse_pred_knn is not None:
                query_continuous, _ = self._split_eval_array(total_continuous_embeddings)
                if rerank_source == "decoded":
                    rerank_pred_knn = self._compute_pdtvq_rerank_pred_knn(query_continuous,
                                                                          base_reconstructed_model,
                                                                          coarse_pred_knn,
                                                                          my_net.PDT_model.decode,
                                                                          rerank_L=rerank_L,
                                                                          decode_step=my_net.pdt_args.steps,
                                                                          batch_size=1024)
                else:
                    _, base_continuous = self._split_eval_array(total_continuous_embeddings)
                    rerank_pred_knn = self._compute_rerank_pred_knn(query_continuous,
                                                                    base_continuous,
                                                                    coarse_pred_knn,
                                                                    rerank_L=rerank_L)
                metrics_payload["rerank_{}_L{}".format(rerank_source, rerank_L)] = test_methods.metrics_from_pred_knn(self.test_knn,
                                                                                                                        rerank_pred_knn)
            else:
                metrics_payload["rerank_{}_L{}".format(rerank_source, rerank_L)] = {
                    "available": False,
                    "reason": "Rerank source {} is unavailable.".format(rerank_source)
                }

        metrics_path = None
        if save_metrics and not self._report_only_eval():
            metrics_path = self._metrics_path(metrics_flag)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
            print("Metrics saved to:", metrics_path)
        else:
            print("Metrics not saved (report_only_eval).")
        return metrics_payload, metrics_path

    def extract_feature_from_path(self,
                                  lon_grid_id_list = None,
                                  lat_grid_id_list = None,
                                  model_flag = None,
                                  model_epoch = None,
                                  metrics_flag = None):
        _, metrics_path = self.extract_feature_payload(
            my_net=None,
            model_flag=model_flag,
            model_epoch=model_epoch,
            metrics_flag=metrics_flag,
            save_metrics=not self._report_only_eval(),
        )
        return metrics_path
