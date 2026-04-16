import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
PDT_ROOT = REPO_ROOT / "PDT_VQ"
for candidate in (REPO_ROOT, PDT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from mynn.model import (
    PreQuantTrajectoryBottleneck,
    compute_bottleneck_regularizers,
    summarize_motion_canvas_stats,
)
from mynn.transformer import Transformer
from PDT_VQ.model import PDT
from PDT_VQ.train import get_args
from PDT_VQ.utils.engine import inference_vetor

POOL = nn.MaxPool1d


def build_default_pdt_args(pdt_m=16, pdt_k=256, pdt_vq_type="dpq", pdt_codebook_init="uniform", qinco_h=256, qinco_L=1, qinco_identity_init=False):
    args = get_args([])
    args.M = int(pdt_m)
    args.K = int(pdt_k)
    args.vq_type = str(pdt_vq_type)
    args.codebook_init = str(pdt_codebook_init)
    args.qinco_h = int(qinco_h)
    args.qinco_L = int(qinco_L)
    args.qinco_identity_init = bool(qinco_identity_init)
    return args


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = torch.relu(out)
        out = out + x
        return out


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResBlock1D, self).__init__()
        padding = dilation * ((kernel_size - 1) // 2)
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              dilation=dilation,
                              bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = torch.relu(out)
        out = out + x
        return out


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = torch.relu(out)
        out = out + x
        return out


class ShortCutCNN(nn.Module):
    def __init__(self,
                 lon_input_size,
                 lat_input_size,
                 target_size,
                 batch_size,
                 sampling_num,
                 max_seq_length,
                 channel,
                 device,
                 head_num,
                 pdt_m=16,
                 pdt_k=256,
                 pdt_vq_type="dpq",
                 pdt_codebook_init="uniform",
                 qinco_h=256,
                 qinco_L=1,
                 qinco_identity_init=False,
                 pre_quant_bottleneck_enabled=False,
                 pre_quant_global_dim=48,
                 pre_quant_local_dim=48,
                 pre_quant_progress_dim=32,
                 pre_quant_use_motion_stats=False,
                 pre_quant_lambda_decor=0.01,
                 pre_quant_lambda_stab=0.1,
                 pre_quant_residual_alpha_init=0.15):
        super(ShortCutCNN, self).__init__()
        self.lon_input_size = lon_input_size
        self.lat_input_size = lat_input_size
        self.target_size = target_size
        self.max_seq_length = max_seq_length
        self.device = device
        self.channel = channel
        self.head_num = int(head_num)
        self.image_channels = 1
        self.smoke_grad_prefixes = ["fc2.", "PDT_model."]

        total_layers = int(math.log2(self.max_seq_length))

        self.MLP = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

        self.channel_1d = self.channel * 4
        self.conv_new = nn.Sequential(
            nn.Conv1d(16, self.channel_1d, 3, 1, padding=1, bias=False),
            nn.ReLU(),
            POOL(2),
        )
        for i in range(total_layers - 1):
            self.conv_new.add_module("res{}".format(i + 1), ResBlock(self.channel_1d, self.channel_1d))
            self.conv_new.add_module("pool{}".format(i + 1), POOL(2))

        self.channel_2d = int((self.channel / 2))
        self.conv_xy = nn.Sequential(
            nn.Conv2d(self.image_channels, self.channel_2d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(self.channel_2d, self.channel_2d),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(self.channel_2d, self.channel_2d),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(self.channel_2d, self.channel_2d),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.flat_size = self.channel_1d

        lon_length, lat_length = self.lon_input_size, self.lat_input_size
        for _ in range(4):
            lon_length = int(lon_length / 2)
            lat_length = int(lat_length / 2)

        self.image_feature_size = lon_length * lat_length * self.channel_2d
        self.total_size = self.flat_size + self.image_feature_size
        print("Total size: ", self.total_size)

        self.fc1 = nn.Linear(self.total_size, self.total_size)
        self.fc2 = nn.Linear(self.total_size, self.target_size)
        self.pdt_args = build_default_pdt_args(
            pdt_m=pdt_m,
            pdt_k=pdt_k,
            pdt_vq_type=pdt_vq_type,
            pdt_codebook_init=pdt_codebook_init,
            qinco_h=qinco_h,
            qinco_L=qinco_L,
            qinco_identity_init=qinco_identity_init,
        )
        self.PDT_model = PDT(self.target_size, self.pdt_args)
        self.norm = nn.LayerNorm(self.target_size)

        self.pre_quant_bottleneck_enabled = bool(pre_quant_bottleneck_enabled)
        self.pre_quant_use_motion_stats = bool(pre_quant_use_motion_stats)
        self.pre_quant_lambda_decor = float(pre_quant_lambda_decor)
        self.pre_quant_lambda_stab = float(pre_quant_lambda_stab)
        self.pre_quant_residual_alpha_init = float(pre_quant_residual_alpha_init)
        self.pre_quant_bottleneck = None
        if self.pre_quant_bottleneck_enabled:
            self.pre_quant_bottleneck = PreQuantTrajectoryBottleneck(
                seq_dim=self.flat_size,
                img_dim=self.image_feature_size,
                cont_dim=self.target_size,
                global_dim=pre_quant_global_dim,
                local_dim=pre_quant_local_dim,
                progress_dim=pre_quant_progress_dim,
                use_motion_stats=self.pre_quant_use_motion_stats,
                motion_stats_dim=6,
                residual_alpha_init=self.pre_quant_residual_alpha_init,
            )

        with torch.no_grad():
            self.PDT_model.init_transform(self.pdt_args.resume)

    def load_backbone_checkpoint(self, checkpoint_path):
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        model_state = self.state_dict()

        matched_state = {}
        skipped_keys = []
        for key, value in checkpoint_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                matched_state[key] = value
            else:
                skipped_keys.append(key)

        model_state.update(matched_state)
        self.load_state_dict(model_state)
        return {
            "loaded_key_count": len(matched_state),
            "skipped_key_count": len(skipped_keys),
            "loaded_keys": sorted(matched_state.keys()),
            "skipped_keys": sorted(skipped_keys),
        }

    def quantize_embeddings(self, embeddings):
        return self.quantize_embeddings_with_details(embeddings)["e_hat"]

    def quantize_embeddings_with_details(self, embeddings):
        z = self.PDT_model.encode(embeddings, out_step=self.pdt_args.steps)
        if isinstance(z, (list, tuple)):
            z = z[-1]
        codes = self.PDT_model.get_codes(z)
        z_hat = self.PDT_model.reconstruction(codes)
        e_hat = self.PDT_model.decode(z_hat, out_step=self.pdt_args.steps)
        if isinstance(e_hat, (list, tuple)):
            e_hat = e_hat[-1]
        return {
            "z": z,
            "codes": codes,
            "z_hat": z_hat,
            "e_hat": e_hat,
        }

    def get_quantized_codes(self, embeddings):
        return self.quantize_embeddings_with_details(embeddings)["codes"]

    def _encode_sequence_vec(self, x, y):
        seq_num = len(x)
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        x = torch.cat((x, y), dim=1)
        x = x.permute(0, 2, 1)
        x = self.MLP(x)
        x = x.permute(0, 2, 1)
        x = self.conv_new(x)
        x = x.view(seq_num, self.flat_size)
        return x

    def _encode_image_vec(self, xy):
        seq_num = len(xy)
        xy = xy.contiguous().view(seq_num, self.image_channels, self.lon_input_size, self.lat_input_size)
        xy = self.conv_xy(xy)
        xy = xy.view(seq_num, -1)
        return xy

    def _fuse_continuous_embedding(self, x_seq, x_img):
        cnn_feature = torch.cat((x_seq, x_img), dim=1)
        cnn_feature = self.fc1(cnn_feature)
        cnn_feature = torch.relu(cnn_feature)
        cnn_feature = self.fc2(cnn_feature)
        cnn_feature = self.norm(cnn_feature)
        return cnn_feature

    def _build_embedding_outputs(self, x_seq, x_img, e_cont, raw_xy=None):
        bottleneck_outputs = None
        motion_stats = None
        e_for_pdt = e_cont
        if self.pre_quant_bottleneck is not None:
            if self.pre_quant_use_motion_stats:
                motion_stats = summarize_motion_canvas_stats(raw_xy)
            bottleneck_outputs = self.pre_quant_bottleneck(
                x_seq=x_seq,
                x_img=x_img,
                e_cont=e_cont,
                extra_motion_stats=motion_stats,
            )
            e_for_pdt = bottleneck_outputs["e_bottleneck"]

        return {
            "x_seq": x_seq,
            "x_img": x_img,
            "e_cont": e_cont,
            "e_bottleneck": e_for_pdt if bottleneck_outputs is not None else None,
            "e_for_pdt": e_for_pdt,
            "motion_stats": motion_stats,
            "bottleneck_outputs": bottleneck_outputs,
        }

    def compute_pre_quant_regularizers(self, embedding_outputs, lambda_decor=None, lambda_stab=None):
        e_cont = embedding_outputs["e_cont"]
        zero = e_cont.new_tensor(0.0)
        if embedding_outputs.get("bottleneck_outputs") is None:
            return {
                "decor_loss": zero,
                "stab_loss": zero,
                "weighted_decor_loss": zero,
                "weighted_stab_loss": zero,
                "total_loss": zero,
            }

        lambda_decor = self.pre_quant_lambda_decor if lambda_decor is None else float(lambda_decor)
        lambda_stab = self.pre_quant_lambda_stab if lambda_stab is None else float(lambda_stab)
        reg = compute_bottleneck_regularizers(embedding_outputs["bottleneck_outputs"], e_cont)
        weighted_decor_loss = lambda_decor * reg["decor_loss"]
        weighted_stab_loss = lambda_stab * reg["stab_loss"]
        return {
            "decor_loss": reg["decor_loss"],
            "stab_loss": reg["stab_loss"],
            "weighted_decor_loss": weighted_decor_loss,
            "weighted_stab_loss": weighted_stab_loss,
            "total_loss": weighted_decor_loss + weighted_stab_loss,
        }

    def encode(self, x, y, xy, seq_lengths=None, grid_lon=None, grid_lat=None, return_intermediate=False, use_raw_continuous=False):
        x_seq = self._encode_sequence_vec(x, y)
        x_img = self._encode_image_vec(xy)
        e_cont = self._fuse_continuous_embedding(x_seq, x_img)
        outputs = self._build_embedding_outputs(x_seq, x_img, e_cont, raw_xy=xy)
        if return_intermediate:
            return outputs
        if use_raw_continuous:
            return outputs["e_cont"]
        return outputs["e_for_pdt"]

    def forward(self,
                inputs_lon_array,
                inputs_lat_array,
                inputs_lon_lat_image_array,
                inputs_length_array=None,
                inputs_lon_grid_array=None,
                inputs_lat_grid_array=None,
                return_intermediate=False):
        anchor_lon_list = torch.stack(inputs_lon_array[0]).to(self.device)
        positive_lon_list = torch.stack(inputs_lon_array[1]).to(self.device)
        negative_lon_list = torch.stack(inputs_lon_array[2]).to(self.device)

        anchor_lat_list = torch.stack(inputs_lat_array[0]).to(self.device)
        positive_lat_list = torch.stack(inputs_lat_array[1]).to(self.device)
        negative_lat_list = torch.stack(inputs_lat_array[2]).to(self.device)

        anchor_lon_lat_image_list = torch.stack(inputs_lon_lat_image_array[0]).to(self.device)
        positive_lon_lat_image_list = torch.stack(inputs_lon_lat_image_array[1]).to(self.device)
        negative_lon_lat_image_list = torch.stack(inputs_lon_lat_image_array[2]).to(self.device)

        anchor_outputs = self.encode(anchor_lon_list, anchor_lat_list, anchor_lon_lat_image_list, return_intermediate=True)
        positive_outputs = self.encode(positive_lon_list, positive_lat_list, positive_lon_lat_image_list, return_intermediate=True)
        negative_outputs = self.encode(negative_lon_list, negative_lat_list, negative_lon_lat_image_list, return_intermediate=True)

        anchor_quantized = self.quantize_embeddings_with_details(anchor_outputs["e_for_pdt"])
        positive_quantized = self.quantize_embeddings_with_details(positive_outputs["e_for_pdt"])
        negative_quantized = self.quantize_embeddings_with_details(negative_outputs["e_for_pdt"])

        if return_intermediate:
            anchor_outputs["z"] = anchor_quantized["z"]
            anchor_outputs["codes"] = anchor_quantized["codes"]
            anchor_outputs["z_hat"] = anchor_quantized["z_hat"]
            anchor_outputs["e_hat"] = anchor_quantized["e_hat"]
            anchor_outputs["quantized_embedding"] = anchor_quantized["e_hat"]
            positive_outputs["z"] = positive_quantized["z"]
            positive_outputs["codes"] = positive_quantized["codes"]
            positive_outputs["z_hat"] = positive_quantized["z_hat"]
            positive_outputs["e_hat"] = positive_quantized["e_hat"]
            positive_outputs["quantized_embedding"] = positive_quantized["e_hat"]
            negative_outputs["z"] = negative_quantized["z"]
            negative_outputs["codes"] = negative_quantized["codes"]
            negative_outputs["z_hat"] = negative_quantized["z_hat"]
            negative_outputs["e_hat"] = negative_quantized["e_hat"]
            negative_outputs["quantized_embedding"] = negative_quantized["e_hat"]
            return {
                "anchor": anchor_outputs,
                "positive": positive_outputs,
                "negative": negative_outputs,
            }

        return (
            anchor_outputs["e_for_pdt"],
            positive_outputs["e_for_pdt"],
            negative_outputs["e_for_pdt"],
            anchor_quantized["e_hat"],
            positive_quantized["e_hat"],
            negative_quantized["e_hat"],
        )

    def inference(self, x, y, xy, seq_lengths=None, grid_lon=None, grid_lat=None):
        return self.inference_quantized(x, y, xy, seq_lengths=seq_lengths, grid_lon=grid_lon, grid_lat=grid_lat)

    def inference_continuous(self, x, y, xy, seq_lengths=None, grid_lon=None, grid_lat=None, return_intermediate=False, use_raw_continuous=False):
        return self.encode(x, y, xy, seq_lengths=seq_lengths, grid_lon=grid_lon, grid_lat=grid_lat, return_intermediate=return_intermediate, use_raw_continuous=use_raw_continuous)

    def inference_quantized(self, x, y, xy, seq_lengths=None, grid_lon=None, grid_lat=None):
        embedding_outputs = self.inference_continuous(x, y, xy, seq_lengths=seq_lengths, grid_lon=grid_lon, grid_lat=grid_lat, return_intermediate=True)
        return self.quantize_embeddings(embedding_outputs["e_for_pdt"])


class ResidualMultiScaleMotionCanvasCNN(ShortCutCNN):
    def __init__(self,
                 lon_input_size,
                 lat_input_size,
                 target_size,
                 batch_size,
                 sampling_num,
                 max_seq_length,
                 channel,
                 device,
                 head_num,
                 image_channels=6,
                 pdt_m=16,
                 pdt_k=256,
                 pdt_vq_type="dpq",
                 pdt_codebook_init="uniform",
                 qinco_h=256,
                 qinco_L=1,
                 qinco_identity_init=False,
                 pre_quant_bottleneck_enabled=False,
                 pre_quant_global_dim=48,
                 pre_quant_local_dim=48,
                 pre_quant_progress_dim=32,
                 pre_quant_use_motion_stats=False,
                 pre_quant_lambda_decor=0.01,
                 pre_quant_lambda_stab=0.1,
                 pre_quant_residual_alpha_init=0.15):
        super(ResidualMultiScaleMotionCanvasCNN, self).__init__(
            lon_input_size,
            lat_input_size,
            target_size,
            batch_size,
            sampling_num,
            max_seq_length,
            channel,
            device,
            head_num,
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
            pre_quant_residual_alpha_init=pre_quant_residual_alpha_init,
        )
        self.image_channels = image_channels
        self.conv_xy = nn.Sequential(
            nn.Conv2d(self.image_channels, self.channel_2d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(self.channel_2d, self.channel_2d),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(self.channel_2d, self.channel_2d),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(self.channel_2d, self.channel_2d),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.c_aux = max(1, self.channel_1d // 2)
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(16, self.c_aux, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            ResBlock1D(self.c_aux, self.c_aux, kernel_size=3, dilation=1),
            nn.AdaptiveMaxPool1d(1),
        )
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(16, self.c_aux, kernel_size=5, padding=2, bias=False),
            nn.ReLU(),
            ResBlock1D(self.c_aux, self.c_aux, kernel_size=5, dilation=1),
            nn.AdaptiveMaxPool1d(1),
        )
        self.ms_proj = nn.Linear(self.c_aux * 2, self.channel_1d)
        self.ms_alpha_logit = nn.Parameter(torch.tensor(-1.7346))

    def _encode_1d_stem(self, x, y):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        x = torch.cat((x, y), dim=1)
        x = x.permute(0, 2, 1)
        x = self.MLP(x)
        x = x.permute(0, 2, 1)
        return x

    def _encode_1d_main(self, stem_feature):
        seq_num = len(stem_feature)
        x_main = self.conv_new(stem_feature)
        x_main = x_main.view(seq_num, self.flat_size)
        return x_main

    def _encode_1d_aux(self, stem_feature):
        seq_num = len(stem_feature)
        branch_k3 = self.branch_k3(stem_feature).view(seq_num, self.c_aux)
        branch_k5 = self.branch_k5(stem_feature).view(seq_num, self.c_aux)
        x_aux = torch.cat((branch_k3, branch_k5), dim=1)
        x_aux = self.ms_proj(x_aux)
        return x_aux

    def _encode_sequence_vec(self, x, y):
        stem_feature = self._encode_1d_stem(x, y)
        x_main = self._encode_1d_main(stem_feature)
        x_aux = self._encode_1d_aux(stem_feature)
        x_seq = x_main + torch.sigmoid(self.ms_alpha_logit) * x_aux
        return x_seq
