import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
PDT_ROOT = REPO_ROOT / "PDT_VQ"
for candidate in (REPO_ROOT, PDT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from PDT_VQ.model import PDT
from PDT_VQ.train import get_args
from mynn.model import PreQuantTrajectoryBottleneck, compute_bottleneck_regularizers


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


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        if self.batch_first:
            pe = pe.permute(1, 0, 2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        if self.batch_first:
            self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        else:
            self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    if pos_encoding == "fixed":
        return FixedPositionalEncoding
    raise NotImplementedError("Unsupported positional encoding: {}".format(pos_encoding))


def mean_pooling(x, padding_masks):
    padding_masks = padding_masks.to(dtype=x.dtype)
    x = x * padding_masks.unsqueeze(-1)
    denom = torch.clamp(torch.sum(padding_masks, dim=1, keepdim=True), min=1.0)
    return torch.sum(x, dim=1) / denom


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_out = nn.Linear(dim * 2, dim)

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(1)
        output = output.view(batch_size, 1, -1)
        input_size = context.size(1)
        attn = torch.bmm(output, context.transpose(1, 2))
        mask = attn.data == 0
        attn.data.masked_fill_(mask, -float("inf"))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        nan_mask = attn.data != attn.data
        attn.data.masked_fill_(nan_mask, 0.0)
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        out = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size)))
        out = out.view(batch_size, -1, hidden_size)
        out = torch.squeeze(out, 1)
        return out, attn


class SpatialExternalMemory(nn.Module):
    def __init__(self, n_grids_x, n_grids_y, hidden_size):
        super().__init__()
        self.n_grids_x = int(n_grids_x)
        self.n_grids_y = int(n_grids_y)
        self.hidden_size = int(hidden_size)
        self.register_buffer("memory", torch.zeros(self.n_grids_x, self.n_grids_y, self.hidden_size))
        self._offset_cache = {}

    def reset(self):
        self.memory.zero_()

    def _get_offsets(self, spatial_width, device):
        device_key = (int(spatial_width), str(device))
        if device_key not in self._offset_cache:
            offsets = torch.arange(-spatial_width, spatial_width + 1, device=device, dtype=torch.long)
            self._offset_cache[device_key] = offsets
        return self._offset_cache[device_key]

    def find_nearby_grids(self, grid_input, spatial_width):
        grid_x = torch.clamp(grid_input[:, 0].long(), 0, self.n_grids_x - 1)
        grid_y = torch.clamp(grid_input[:, 1].long(), 0, self.n_grids_y - 1)
        offsets = self._get_offsets(spatial_width, grid_input.device)
        batch_size = grid_x.size(0)
        n_offsets = offsets.numel()
        x_indices = torch.clamp(grid_x.unsqueeze(1) + offsets.unsqueeze(0), 0, self.n_grids_x - 1)
        y_indices = torch.clamp(grid_y.unsqueeze(1) + offsets.unsqueeze(0), 0, self.n_grids_y - 1)
        x_grid = x_indices.unsqueeze(2).expand(-1, -1, n_offsets)
        y_grid = y_indices.unsqueeze(1).expand(-1, n_offsets, -1)
        x_flat = x_grid.reshape(batch_size, -1)
        y_flat = y_grid.reshape(batch_size, -1)
        return self.memory[x_flat, y_flat, :]

    def read(self, grid_x, grid_y):
        grid_x = torch.clamp(grid_x.long(), 0, self.n_grids_x - 1)
        grid_y = torch.clamp(grid_y.long(), 0, self.n_grids_y - 1)
        return self.memory[grid_x, grid_y]

    def update(self, grid_x, grid_y, updates):
        grid_x = torch.clamp(grid_x.long(), 0, self.n_grids_x - 1)
        grid_y = torch.clamp(grid_y.long(), 0, self.n_grids_y - 1)
        self.memory[grid_x, grid_y] = updates


class SAMGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, grid_size, spatial_width=2, incell=True):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.incell = bool(incell)
        self.spatial_width = int(spatial_width)
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size - 2))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.spatial_embedding = SpatialExternalMemory(
            int(grid_size[0]) + 3 * self.spatial_width,
            int(grid_size[1]) + 3 * self.spatial_width,
            hidden_size,
        )
        self.atten = Attention(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_memory(self):
        self.spatial_embedding.reset()

    def forward(self, input_tensor, hidden):
        feature_tensor = input_tensor[:, :-2]
        grid_input = input_tensor[:, -2:].to(dtype=torch.long) + self.spatial_width
        gi = F.linear(feature_tensor, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden, self.weight_hh, self.bias_hh)
        i_r, i_i, i_n, i_s = gi.chunk(4, 1)
        h_r, h_i, h_n, h_s = gh.chunk(4, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_i + h_i)
        spatialgate = torch.sigmoid(i_s + h_s)
        newgate = torch.tanh(i_n + resetgate * h_n)
        context = self.spatial_embedding.find_nearby_grids(grid_input, self.spatial_width)
        atten_cs, _ = self.atten(newgate.detach(), context)
        curr_state = newgate + spatialgate * atten_cs
        output = curr_state + updategate * (hidden - curr_state)
        if self.incell:
            grid_x = grid_input[:, 0].data
            grid_y = grid_input[:, 1].data
            updates = spatialgate.data * self.spatial_embedding.read(grid_x, grid_y) + (1 - spatialgate.data) * output.data
            if self.training:
                self.spatial_embedding.update(grid_x, grid_y, updates)
        return output


class NeuTrajBackbone(nn.Module):
    def __init__(self,
                 hidden_size,
                 grid_size,
                 spatial_width=2,
                 incell=True,
                 use_standard_gru=False):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_standard_gru = bool(use_standard_gru)
        self.spatial_width = int(spatial_width)
        if self.use_standard_gru:
            self.cell = nn.GRU(2, self.hidden_size, batch_first=True)
        else:
            self.cell = SAMGRUCell(4, self.hidden_size, grid_size, spatial_width=self.spatial_width, incell=incell)

    def reset_memory(self):
        if hasattr(self.cell, "reset_memory"):
            self.cell.reset_memory()

    def forward(self, feature_tensor, seq_lengths):
        batch_size, _, _ = feature_tensor.shape
        step_indices = torch.as_tensor(seq_lengths, device=feature_tensor.device, dtype=torch.long).clamp_min(1) - 1
        if self.use_standard_gru:
            output, _ = self.cell(feature_tensor[:, :, :2])
            batch_indices = torch.arange(batch_size, device=feature_tensor.device, dtype=torch.long)
            return output[batch_indices, step_indices]

        hidden = feature_tensor.new_zeros((batch_size, self.hidden_size))
        last_valid_output = feature_tensor.new_zeros((batch_size, self.hidden_size))
        self.reset_memory()
        for t in range(feature_tensor.size(1)):
            hidden = self.cell(feature_tensor[:, t, :], hidden)
            mask = step_indices.eq(t).unsqueeze(1)
            last_valid_output = torch.where(mask, hidden, last_valid_output)
        return last_valid_output


class SIMformerBackbone(nn.Module):
    def __init__(self,
                 hidden_dim,
                 max_seq_len,
                 dimfeedforward=256,
                 n_heads=16,
                 num_layers=1,
                 pos_encoding="fixed"):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.project_inp = nn.Linear(2, self.hidden_dim)
        self.pos_enc = get_pos_encoder(pos_encoding)(self.hidden_dim, max_len=self.max_seq_len, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(n_heads),
            dim_feedforward=int(dimfeedforward),
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, int(num_layers))
        self.act = nn.ReLU()

    def forward(self, coord_tensor, seq_lengths):
        max_seq_len = coord_tensor.size(1)
        device = coord_tensor.device
        lengths = torch.as_tensor(seq_lengths, device=device, dtype=torch.long).clamp_min(1)
        padding_masks = torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        input_tensor = self.project_inp(coord_tensor)
        input_tensor = self.pos_enc(input_tensor)
        output = self.transformer_encoder(input_tensor, src_key_padding_mask=~padding_masks)
        output = mean_pooling(output, padding_masks)
        return self.act(output)


class ExternalSequenceBackbonePDTModel(nn.Module):
    def __init__(self,
                 backbone_kind,
                 lon_input_size,
                 lat_input_size,
                 target_size,
                 max_seq_length,
                 device,
                 pdt_m=16,
                 pdt_k=256,
                 pdt_vq_type="dpq",
                 pdt_codebook_init="uniform",
                 qinco_h=256,
                 qinco_L=1,
                 qinco_identity_init=False,
                 backbone_seq_max_length=200,
                 simformer_num_layers=1,
                 simformer_n_heads=16,
                 simformer_dimfeedforward=256,
                 simformer_pos_encoding="fixed",
                 neutraj_spatial_width=2,
                 neutraj_incell=True,
                 neutraj_use_standard_gru=False,
                 pre_quant_bottleneck_enabled=False,
                 pre_quant_global_dim=48,
                 pre_quant_local_dim=48,
                 pre_quant_progress_dim=32,
                 pre_quant_use_motion_stats=False,
                 pre_quant_lambda_decor=0.01,
                 pre_quant_lambda_stab=0.1,
                 pre_quant_residual_alpha_init=0.15):
        super().__init__()
        self.backbone_kind = str(backbone_kind)
        self.lon_input_size = int(lon_input_size)
        self.lat_input_size = int(lat_input_size)
        self.target_size = int(target_size)
        self.max_seq_length = int(max_seq_length)
        self.device = device
        self.pre_quant_bottleneck_enabled = bool(pre_quant_bottleneck_enabled)
        self.pre_quant_use_motion_stats = bool(pre_quant_use_motion_stats)
        self.pre_quant_lambda_decor = float(pre_quant_lambda_decor)
        self.pre_quant_lambda_stab = float(pre_quant_lambda_stab)
        self.pre_quant_residual_alpha_init = float(pre_quant_residual_alpha_init)
        self.pre_quant_bottleneck = None
        self.smoke_grad_prefixes = ["backbone.", "PDT_model."]
        self.backbone_seq_max_length = max(8, int(backbone_seq_max_length))
        self.sequence_global_summary_dim = 20
        self.sequence_local_summary_dim = 13

        if self.backbone_kind == "simformer":
            self.backbone = SIMformerBackbone(
                hidden_dim=self.target_size,
                max_seq_len=self.backbone_seq_max_length,
                dimfeedforward=simformer_dimfeedforward,
                n_heads=simformer_n_heads,
                num_layers=simformer_num_layers,
                pos_encoding=simformer_pos_encoding,
            )
        elif self.backbone_kind == "neutraj":
            self.backbone = NeuTrajBackbone(
                hidden_size=self.target_size,
                grid_size=(self.lon_input_size, self.lat_input_size),
                spatial_width=neutraj_spatial_width,
                incell=neutraj_incell,
                use_standard_gru=neutraj_use_standard_gru,
            )
        else:
            raise ValueError("Unsupported external backbone: {}".format(self.backbone_kind))

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
        if self.pre_quant_bottleneck_enabled:
            self.pre_quant_bottleneck = PreQuantTrajectoryBottleneck(
                seq_dim=self.sequence_local_summary_dim,
                img_dim=self.sequence_global_summary_dim,
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

    def _downsample_sequence(self, seq_tensor, seq_lengths, extra_tensor=None):
        batch_size, _, feat_dim = seq_tensor.shape
        target_max_len = min(int(max(seq_lengths)), self.backbone_seq_max_length)
        downsampled = seq_tensor.new_zeros((batch_size, target_max_len, feat_dim))
        extra_downsampled = None
        if extra_tensor is not None:
            extra_downsampled = extra_tensor.new_zeros((batch_size, target_max_len, extra_tensor.shape[2]))
        new_lengths = []
        for batch_idx, length_value in enumerate(seq_lengths):
            length_value = max(1, int(length_value))
            eff_len = min(length_value, self.backbone_seq_max_length)
            if length_value <= self.backbone_seq_max_length:
                select_idx = torch.arange(length_value, device=seq_tensor.device, dtype=torch.long)
            else:
                select_idx = torch.linspace(0, length_value - 1, steps=self.backbone_seq_max_length, device=seq_tensor.device)
                select_idx = torch.round(select_idx).long()
            downsampled[batch_idx, :eff_len] = seq_tensor[batch_idx, select_idx]
            if extra_downsampled is not None:
                extra_downsampled[batch_idx, :eff_len] = extra_tensor[batch_idx, select_idx]
            new_lengths.append(eff_len)
        return downsampled, new_lengths, extra_downsampled

    def _build_backbone_input(self, x, y, seq_lengths, grid_lon=None, grid_lat=None):
        coord_tensor = torch.cat((x, y), dim=2)
        if self.backbone_kind == "simformer":
            coord_tensor, seq_lengths, _ = self._downsample_sequence(coord_tensor, seq_lengths)
            return coord_tensor, seq_lengths, coord_tensor, None
        if grid_lon is None or grid_lat is None:
            raise ValueError("NeuTraj backbone requires grid-id sequences.")
        grid_tensor = torch.cat((grid_lon, grid_lat), dim=2)
        coord_tensor, seq_lengths, grid_tensor = self._downsample_sequence(coord_tensor, seq_lengths, extra_tensor=grid_tensor)
        feature_tensor = torch.cat((coord_tensor, grid_tensor), dim=2)
        return feature_tensor, seq_lengths, coord_tensor, grid_tensor

    def _masked_time_mean(self, values, mask):
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _masked_time_var(self, values, mask, mean):
        denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        centered = values - mean.unsqueeze(1)
        return (centered.pow(2) * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _safe_masked_min(self, values, mask):
        large = torch.full_like(values, float("inf"))
        return torch.where(mask.unsqueeze(-1) > 0, values, large).amin(dim=1)

    def _safe_masked_max(self, values, mask):
        small = torch.full_like(values, float("-inf"))
        return torch.where(mask.unsqueeze(-1) > 0, values, small).amax(dim=1)

    def _summarize_sequence_features(self, coord_tensor, seq_lengths, grid_tensor=None):
        batch_size, max_len, _ = coord_tensor.shape
        device = coord_tensor.device
        lengths = torch.as_tensor(seq_lengths, device=device, dtype=torch.long).clamp_min(1)
        mask = (torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)
        end_indices = lengths - 1

        start_xy = coord_tensor[:, 0]
        end_xy = coord_tensor[batch_indices, end_indices]
        mean_xy = self._masked_time_mean(coord_tensor, mask)
        var_xy = self._masked_time_var(coord_tensor, mask, mean_xy)
        std_xy = torch.sqrt(torch.clamp(var_xy, min=1e-8))
        min_xy = self._safe_masked_min(coord_tensor, mask)
        max_xy = self._safe_masked_max(coord_tensor, mask)
        span_xy = max_xy - min_xy
        displacement = end_xy - start_xy

        pair_mask = (mask[:, 1:] * mask[:, :-1]).float()
        deltas = coord_tensor[:, 1:] - coord_tensor[:, :-1]
        delta_mean = self._masked_time_mean(deltas, pair_mask)
        delta_var = self._masked_time_var(deltas, pair_mask, delta_mean)
        delta_std = torch.sqrt(torch.clamp(delta_var, min=1e-8))
        delta_abs_mean = self._masked_time_mean(deltas.abs(), pair_mask)

        step_norm = torch.linalg.norm(deltas, dim=2)
        step_mean = (step_norm * pair_mask).sum(dim=1, keepdim=True) / torch.clamp(pair_mask.sum(dim=1, keepdim=True), min=1.0)
        step_centered = step_norm - step_mean
        step_std = torch.sqrt(torch.clamp((step_centered.pow(2) * pair_mask).sum(dim=1, keepdim=True) / torch.clamp(pair_mask.sum(dim=1, keepdim=True), min=1.0), min=1e-8))
        total_step = (step_norm * pair_mask).sum(dim=1, keepdim=True)
        net_disp = torch.linalg.norm(displacement, dim=1, keepdim=True)
        straightness = net_disp / torch.clamp(total_step, min=1e-6)

        unit_delta = deltas / torch.clamp(step_norm.unsqueeze(-1), min=1e-6)
        turn_cos = (unit_delta[:, 1:] * unit_delta[:, :-1]).sum(dim=2)
        turn_mask = (pair_mask[:, 1:] * pair_mask[:, :-1]).float()
        turn_mean = ((1.0 - turn_cos) * turn_mask).sum(dim=1, keepdim=True) / torch.clamp(turn_mask.sum(dim=1, keepdim=True), min=1.0)
        turn_centered = (1.0 - turn_cos) - turn_mean
        turn_std = torch.sqrt(torch.clamp((turn_centered.pow(2) * turn_mask).sum(dim=1, keepdim=True) / torch.clamp(turn_mask.sum(dim=1, keepdim=True), min=1.0), min=1e-8))

        length_ratio = lengths.float().unsqueeze(1) / float(max(1, self.backbone_seq_max_length))
        bbox_area = (span_xy[:, :1] * span_xy[:, 1:2]).abs()

        if grid_tensor is not None:
            grid_mean = self._masked_time_mean(grid_tensor, mask)
            grid_var = self._masked_time_var(grid_tensor, mask, grid_mean)
            grid_std = torch.sqrt(torch.clamp(grid_var, min=1e-8))
            grid_min = self._safe_masked_min(grid_tensor, mask)
            grid_max = self._safe_masked_max(grid_tensor, mask)
            grid_span = grid_max - grid_min
        else:
            zeros = coord_tensor.new_zeros((batch_size, 2))
            grid_mean = zeros
            grid_std = zeros
            grid_span = zeros

        global_summary = torch.cat((
            start_xy,
            end_xy,
            displacement,
            mean_xy,
            std_xy,
            span_xy,
            length_ratio,
            bbox_area,
            grid_mean,
            grid_std,
            grid_span,
        ), dim=1)
        local_summary = torch.cat((
            delta_mean,
            delta_std,
            delta_abs_mean,
            step_mean,
            step_std,
            total_step,
            turn_mean,
            turn_std,
            straightness,
            net_disp,
        ), dim=1)
        motion_stats = torch.cat((
            step_mean,
            step_std,
            net_disp,
            delta_abs_mean[:, :1],
            delta_abs_mean[:, 1:2],
            turn_mean,
        ), dim=1)
        return global_summary, local_summary, motion_stats

    def _build_embedding_outputs(self, coord_tensor, grid_tensor, e_cont, seq_lengths):
        bottleneck_outputs = None
        motion_stats = None
        e_for_pdt = e_cont
        local_summary = None
        global_summary = None
        if self.pre_quant_bottleneck is not None:
            global_summary, local_summary, motion_stats = self._summarize_sequence_features(
                coord_tensor,
                seq_lengths,
                grid_tensor=grid_tensor,
            )
            bottleneck_outputs = self.pre_quant_bottleneck(
                x_seq=local_summary,
                x_img=global_summary,
                e_cont=e_cont,
                extra_motion_stats=motion_stats if self.pre_quant_use_motion_stats else None,
            )
            e_for_pdt = bottleneck_outputs["e_bottleneck"]

        return {
            "x_seq": local_summary,
            "x_img": global_summary,
            "e_cont": e_cont,
            "e_bottleneck": e_for_pdt if bottleneck_outputs is not None else None,
            "e_for_pdt": e_for_pdt,
            "motion_stats": motion_stats,
            "bottleneck_outputs": bottleneck_outputs,
        }

    def encode(self,
               x,
               y,
               xy=None,
               seq_lengths=None,
               grid_lon=None,
               grid_lat=None,
               return_intermediate=False,
               use_raw_continuous=False):
        if seq_lengths is None:
            raise ValueError("External backbones require sequence lengths.")
        backbone_input, seq_lengths, coord_tensor, grid_tensor = self._build_backbone_input(x, y, seq_lengths, grid_lon=grid_lon, grid_lat=grid_lat)
        e_cont = self.backbone(backbone_input, seq_lengths)
        outputs = self._build_embedding_outputs(coord_tensor, grid_tensor, e_cont, seq_lengths)
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
        del inputs_lon_lat_image_array
        if inputs_length_array is None:
            raise ValueError("External backbones require inputs_length_array.")
        anchor_lon_list = torch.stack(inputs_lon_array[0]).to(self.device)
        positive_lon_list = torch.stack(inputs_lon_array[1]).to(self.device)
        negative_lon_list = torch.stack(inputs_lon_array[2]).to(self.device)
        anchor_lat_list = torch.stack(inputs_lat_array[0]).to(self.device)
        positive_lat_list = torch.stack(inputs_lat_array[1]).to(self.device)
        negative_lat_list = torch.stack(inputs_lat_array[2]).to(self.device)
        anchor_lengths, positive_lengths, negative_lengths = inputs_length_array[:3]

        anchor_grid_lon = positive_grid_lon = negative_grid_lon = None
        anchor_grid_lat = positive_grid_lat = negative_grid_lat = None
        if inputs_lon_grid_array is not None and inputs_lat_grid_array is not None:
            anchor_grid_lon = torch.stack(inputs_lon_grid_array[0]).to(self.device)
            positive_grid_lon = torch.stack(inputs_lon_grid_array[1]).to(self.device)
            negative_grid_lon = torch.stack(inputs_lon_grid_array[2]).to(self.device)
            anchor_grid_lat = torch.stack(inputs_lat_grid_array[0]).to(self.device)
            positive_grid_lat = torch.stack(inputs_lat_grid_array[1]).to(self.device)
            negative_grid_lat = torch.stack(inputs_lat_grid_array[2]).to(self.device)

        anchor_outputs = self.encode(anchor_lon_list, anchor_lat_list, None, seq_lengths=anchor_lengths, grid_lon=anchor_grid_lon, grid_lat=anchor_grid_lat, return_intermediate=True)
        positive_outputs = self.encode(positive_lon_list, positive_lat_list, None, seq_lengths=positive_lengths, grid_lon=positive_grid_lon, grid_lat=positive_grid_lat, return_intermediate=True)
        negative_outputs = self.encode(negative_lon_list, negative_lat_list, None, seq_lengths=negative_lengths, grid_lon=negative_grid_lon, grid_lat=negative_grid_lat, return_intermediate=True)

        combined_embeddings = torch.cat(
            (anchor_outputs["e_for_pdt"], positive_outputs["e_for_pdt"], negative_outputs["e_for_pdt"]),
            dim=0,
        )
        combined_quantized = self.quantize_embeddings_with_details(combined_embeddings)
        anchor_size = anchor_outputs["e_for_pdt"].size(0)
        positive_size = positive_outputs["e_for_pdt"].size(0)
        negative_offset = anchor_size + positive_size

        def _slice_quantized(start, end):
            return {
                "z": combined_quantized["z"][start:end],
                "codes": combined_quantized["codes"][start:end],
                "z_hat": combined_quantized["z_hat"][start:end],
                "e_hat": combined_quantized["e_hat"][start:end],
            }

        anchor_quantized = _slice_quantized(0, anchor_size)
        positive_quantized = _slice_quantized(anchor_size, negative_offset)
        negative_quantized = _slice_quantized(negative_offset, negative_offset + negative_outputs["e_for_pdt"].size(0))

        if return_intermediate:
            for outputs, quantized in (
                (anchor_outputs, anchor_quantized),
                (positive_outputs, positive_quantized),
                (negative_outputs, negative_quantized),
            ):
                outputs["z"] = quantized["z"]
                outputs["codes"] = quantized["codes"]
                outputs["z_hat"] = quantized["z_hat"]
                outputs["e_hat"] = quantized["e_hat"]
                outputs["quantized_embedding"] = quantized["e_hat"]
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

    def inference(self, x, y, xy=None, seq_lengths=None, grid_lon=None, grid_lat=None):
        return self.inference_quantized(x, y, xy, seq_lengths=seq_lengths, grid_lon=grid_lon, grid_lat=grid_lat)

    def inference_continuous(self, x, y, xy=None, seq_lengths=None, grid_lon=None, grid_lat=None, return_intermediate=False, use_raw_continuous=False):
        return self.encode(x, y, xy, seq_lengths=seq_lengths, grid_lon=grid_lon, grid_lat=grid_lat, return_intermediate=return_intermediate, use_raw_continuous=use_raw_continuous)

    def inference_quantized(self, x, y, xy=None, seq_lengths=None, grid_lon=None, grid_lat=None):
        embedding_outputs = self.inference_continuous(x, y, xy, seq_lengths=seq_lengths, grid_lon=grid_lon, grid_lat=grid_lat, return_intermediate=True)
        return self.quantize_embeddings(embedding_outputs["e_for_pdt"])
