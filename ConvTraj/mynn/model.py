import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import feature_distance


def _safe_logit(value, eps=1e-4):
    value = float(value)
    value = min(max(value, eps), 1.0 - eps)
    tensor_value = torch.tensor(value, dtype=torch.float32)
    return torch.logit(tensor_value).item()


def summarize_motion_canvas_stats(motion_canvas):
    if motion_canvas is None or motion_canvas.dim() != 4 or motion_canvas.shape[1] < 6:
        return None

    occupancy = motion_canvas[:, 0]
    dir_x = motion_canvas[:, 1]
    dir_y = motion_canvas[:, 2]
    speed = motion_canvas[:, 3]
    progress = motion_canvas[:, 4]
    turning = motion_canvas[:, 5]

    direction_magnitude = torch.sqrt(torch.clamp(dir_x ** 2 + dir_y ** 2, min=0.0))
    coverage = (occupancy > 0).float()

    def spatial_mean(channel):
        return channel.mean(dim=(1, 2), keepdim=False).unsqueeze(1)

    stats = torch.cat([
        spatial_mean(occupancy),
        spatial_mean(progress),
        spatial_mean(turning),
        spatial_mean(direction_magnitude),
        spatial_mean(speed),
        spatial_mean(coverage),
    ], dim=1)
    return stats


class SmallProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        last_linear = self.net[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        return self.net(x)


class PreQuantTrajectoryBottleneck(nn.Module):
    def __init__(self,
                 seq_dim,
                 img_dim,
                 cont_dim,
                 global_dim=48,
                 local_dim=48,
                 progress_dim=32,
                 use_motion_stats=False,
                 motion_stats_dim=6,
                 residual_alpha_init=0.15,
                 learnable_alpha=True):
        super().__init__()
        total_dim = int(global_dim) + int(local_dim) + int(progress_dim)
        if total_dim != int(cont_dim):
            raise ValueError("PreQuantTrajectoryBottleneck output dim {} must match cont_dim {}.".format(total_dim, cont_dim))

        self.seq_dim = int(seq_dim)
        self.img_dim = int(img_dim)
        self.cont_dim = int(cont_dim)
        self.global_dim = int(global_dim)
        self.local_dim = int(local_dim)
        self.progress_dim = int(progress_dim)
        self.use_motion_stats = bool(use_motion_stats)
        self.motion_stats_dim = int(motion_stats_dim)
        self.residual_alpha_init = float(residual_alpha_init)
        self.learnable_alpha = bool(learnable_alpha)
        if self.learnable_alpha:
            self.residual_alpha_logit = nn.Parameter(torch.tensor(_safe_logit(self.residual_alpha_init), dtype=torch.float32))
        else:
            self.register_parameter("residual_alpha_logit", None)

        self.global_proj = SmallProjectionHead(self.img_dim + self.cont_dim, self.global_dim)
        self.local_proj = SmallProjectionHead(self.seq_dim + self.cont_dim, self.local_dim)
        progress_input_dim = self.cont_dim + (self.motion_stats_dim if self.use_motion_stats else 0)
        self.progress_proj = SmallProjectionHead(progress_input_dim, self.progress_dim)
        self.structured_norm = nn.LayerNorm(self.cont_dim)
        self.output_norm = nn.LayerNorm(self.cont_dim)

    def forward(self, x_seq, x_img, e_cont, extra_motion_stats=None):
        global_input = torch.cat((x_img, e_cont), dim=1)
        local_input = torch.cat((x_seq, e_cont), dim=1)

        progress_inputs = [e_cont]
        if self.use_motion_stats:
            if extra_motion_stats is None:
                extra_motion_stats = e_cont.new_zeros((e_cont.shape[0], self.motion_stats_dim))
            progress_inputs.append(extra_motion_stats)
        progress_input = torch.cat(progress_inputs, dim=1)

        u_global = self.global_proj(global_input)
        u_local = self.local_proj(local_input)
        u_progress = self.progress_proj(progress_input)
        structured_delta = torch.cat((u_global, u_local, u_progress), dim=1)
        base_global, base_local, base_progress = torch.split(
            e_cont,
            [self.global_dim, self.local_dim, self.progress_dim],
            dim=1,
        )
        if self.learnable_alpha:
            residual_alpha = torch.sigmoid(self.residual_alpha_logit)
            structured_feature = torch.cat((
                base_global + residual_alpha * u_global,
                base_local + residual_alpha * u_local,
                base_progress + residual_alpha * u_progress,
            ), dim=1)
            residual_alpha_value = float(residual_alpha.detach().cpu())
        else:
            residual_alpha = e_cont.new_tensor(self.residual_alpha_init)
            structured_feature = torch.cat((
                base_global + residual_alpha * u_global,
                base_local + residual_alpha * u_local,
                base_progress + residual_alpha * u_progress,
            ), dim=1)
            residual_alpha_value = float(residual_alpha.detach().cpu())
        structured_feature = self.structured_norm(structured_feature)
        e_bottleneck = self.output_norm(structured_feature)

        aux = {
            "motion_stats": extra_motion_stats,
            "u_global_mean": float(u_global.mean().detach().cpu()),
            "u_global_std": float(u_global.std(unbiased=False).detach().cpu()),
            "u_local_mean": float(u_local.mean().detach().cpu()),
            "u_local_std": float(u_local.std(unbiased=False).detach().cpu()),
            "u_progress_mean": float(u_progress.mean().detach().cpu()),
            "u_progress_std": float(u_progress.std(unbiased=False).detach().cpu()),
            "structured_delta_mean": float(structured_delta.mean().detach().cpu()),
            "structured_delta_std": float(structured_delta.std(unbiased=False).detach().cpu()),
            "structured_feature_mean": float(structured_feature.mean().detach().cpu()),
            "structured_feature_std": float(structured_feature.std(unbiased=False).detach().cpu()),
            "e_bottleneck_mean": float(e_bottleneck.mean().detach().cpu()),
            "e_bottleneck_std": float(e_bottleneck.std(unbiased=False).detach().cpu()),
            "residual_alpha": residual_alpha_value,
            "learnable_alpha": self.learnable_alpha,
        }
        return {
            "u_global": u_global,
            "u_local": u_local,
            "u_progress": u_progress,
            "structured_delta": structured_delta,
            "e_bottleneck": e_bottleneck,
            "aux": aux,
        }


def resolve_embedding_distance_fn(distance_type):
    if distance_type == "euclidean":
        return feature_distance.euclidean_torch
    raise ValueError("Unsupported embedding distance type for neighborhood consistency: {}".format(distance_type))


def compute_pairwise_distance_matrix(embeddings, distance_fn):
    if embeddings.dim() != 2:
        raise ValueError("Expected a 2D embedding tensor, got shape {}.".format(tuple(embeddings.shape)))
    return distance_fn(embeddings[:, None, :], embeddings[None, :, :])


def compute_cross_distance_matrix(query_embeddings, key_embeddings, distance_fn):
    if query_embeddings.dim() != 2 or key_embeddings.dim() != 2:
        raise ValueError("Expected [N, D] and [M, D] tensors, got {} and {}.".format(
            tuple(query_embeddings.shape),
            tuple(key_embeddings.shape),
        ))
    if query_embeddings.shape[1] != key_embeddings.shape[1]:
        raise ValueError("Embedding dims must match, got {} and {}.".format(
            int(query_embeddings.shape[1]),
            int(key_embeddings.shape[1]),
        ))
    return distance_fn(query_embeddings[:, None, :], key_embeddings[None, :, :])


def normalize_landmark_profile(profile, transform="log1p_zscore", eps=1e-6):
    if profile.dim() != 2:
        raise ValueError("Landmark profiles must be 2D, got {}.".format(tuple(profile.shape)))

    normalized = profile
    if transform == "none":
        return normalized
    if transform == "log1p_zscore":
        normalized = torch.log1p(torch.clamp(normalized, min=0.0))
    elif transform != "zscore":
        raise ValueError("Unsupported landmark profile transform: {}".format(transform))

    row_mean = normalized.mean(dim=1, keepdim=True)
    row_std = normalized.std(dim=1, unbiased=False, keepdim=True)
    return (normalized - row_mean) / (row_std + float(eps))


def landmark_profile_loss(student_emb,
                          landmark_bank,
                          teacher_profile,
                          distance_fn,
                          profile_transform="log1p_zscore",
                          teacher_profile_raw=None,
                          rank_weight=0.0,
                          rank_tau=0.5,
                          eps=1e-6):
    if student_emb.dim() != 2:
        raise ValueError("Expected student embeddings [B, D], got {}.".format(tuple(student_emb.shape)))
    if landmark_bank.dim() != 2:
        raise ValueError("Expected landmark bank [L, D], got {}.".format(tuple(landmark_bank.shape)))
    if teacher_profile.dim() != 2:
        raise ValueError("Expected teacher profile [B, L], got {}.".format(tuple(teacher_profile.shape)))
    if student_emb.shape[0] != teacher_profile.shape[0]:
        raise ValueError("Student batch/profile rows mismatch: {} vs {}.".format(
            int(student_emb.shape[0]),
            int(teacher_profile.shape[0]),
        ))
    if landmark_bank.shape[0] != teacher_profile.shape[1]:
        raise ValueError("Landmark bank/profile columns mismatch: {} vs {}.".format(
            int(landmark_bank.shape[0]),
            int(teacher_profile.shape[1]),
        ))
    if student_emb.shape[1] != landmark_bank.shape[1]:
        raise ValueError("Student/landmark dims mismatch: {} vs {}.".format(
            int(student_emb.shape[1]),
            int(landmark_bank.shape[1]),
        ))

    zero = student_emb.new_tensor(0.0)
    num_landmarks = int(landmark_bank.shape[0])
    default_aux = {
        "num_landmarks": num_landmarks,
        "avg_teacher_profile_distance": 0.0,
        "avg_student_profile_distance": 0.0,
        "avg_profile_cosine": 0.0,
        "rank_loss": 0.0,
        "rank_teacher_entropy": 0.0,
    }
    if student_emb.shape[0] <= 0 or num_landmarks <= 0:
        return zero, default_aux

    student_distance_profile = compute_cross_distance_matrix(student_emb, landmark_bank, distance_fn)
    normalized_student_profile = normalize_landmark_profile(student_distance_profile, transform=profile_transform, eps=eps)
    teacher_target = teacher_profile.detach()
    loss = F.smooth_l1_loss(normalized_student_profile, teacher_target)

    rank_loss = zero
    safe_rank_tau = max(float(rank_tau), float(eps))
    if float(rank_weight) > 0.0 and teacher_profile_raw is not None:
        teacher_rank_source = normalize_landmark_profile(
            teacher_profile_raw.detach(),
            transform=profile_transform,
            eps=eps,
        )
        student_rank_source = normalized_student_profile
        teacher_rank_prob = F.softmax((-teacher_rank_source) / safe_rank_tau, dim=1)
        student_rank_log_prob = F.log_softmax((-student_rank_source) / safe_rank_tau, dim=1)
        rank_loss = -(teacher_rank_prob * student_rank_log_prob).sum(dim=1).mean()
        loss = loss + (float(rank_weight) * rank_loss)

    with torch.no_grad():
        avg_teacher_profile_distance = 0.0
        if teacher_profile_raw is not None:
            avg_teacher_profile_distance = float(teacher_profile_raw.detach().mean().cpu())
        avg_student_profile_distance = float(student_distance_profile.detach().mean().cpu())
        avg_profile_cosine = float(F.cosine_similarity(
            normalized_student_profile.detach(),
            teacher_target,
            dim=1,
        ).mean().cpu())
        rank_teacher_entropy = 0.0
        if teacher_profile_raw is not None and float(rank_weight) > 0.0:
            teacher_rank_source = normalize_landmark_profile(
                teacher_profile_raw.detach(),
                transform=profile_transform,
                eps=eps,
            )
            teacher_rank_prob = F.softmax((-teacher_rank_source) / safe_rank_tau, dim=1)
            rank_teacher_entropy = float((-(teacher_rank_prob * torch.log(torch.clamp(teacher_rank_prob, min=float(eps)))).sum(dim=1)).mean().cpu())
        aux = {
            "num_landmarks": num_landmarks,
            "avg_teacher_profile_distance": avg_teacher_profile_distance,
            "avg_student_profile_distance": avg_student_profile_distance,
            "avg_profile_cosine": avg_profile_cosine,
            "rank_loss": float(rank_loss.detach().cpu()),
            "rank_teacher_entropy": rank_teacher_entropy,
        }
    return loss, aux


def neighborhood_consistency_loss(teacher_emb, student_emb, distance_fn, topk, tau, eps=1e-8):
    if teacher_emb.shape != student_emb.shape:
        raise ValueError("Teacher/student embedding shapes must match, got {} and {}.".format(
            tuple(teacher_emb.shape),
            tuple(student_emb.shape),
        ))
    if teacher_emb.dim() != 2:
        raise ValueError("Neighborhood consistency expects [B, D] tensors, got {}.".format(tuple(teacher_emb.shape)))

    zero = student_emb.new_tensor(0.0)
    batch_size = int(teacher_emb.shape[0])
    effective_topk = max(0, min(int(topk), batch_size - 1))
    safe_tau = max(float(tau), float(eps))
    default_aux = {
        "avg_teacher_neighbor_distance": 0.0,
        "avg_student_neighbor_distance": 0.0,
        "avg_overlap_proxy": 0.0,
        "effective_topk": effective_topk,
        "tau": safe_tau,
        "valid_query_count": 0,
    }
    if batch_size <= 1 or effective_topk <= 0:
        return zero, default_aux

    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=teacher_emb.device)

    with torch.no_grad():
        teacher_distance = compute_pairwise_distance_matrix(teacher_emb.detach(), distance_fn)
        teacher_distance = teacher_distance.masked_fill(diag_mask, float("inf"))
        teacher_neighbor_distance, teacher_neighbor_idx = torch.topk(
            teacher_distance,
            k=effective_topk,
            dim=1,
            largest=False,
        )

    student_distance = compute_pairwise_distance_matrix(student_emb, distance_fn)
    student_logits = -(student_distance / safe_tau)
    student_logits = student_logits.masked_fill(diag_mask, float("-inf"))

    neighbor_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=student_emb.device)
    neighbor_mask.scatter_(1, teacher_neighbor_idx, True)

    positive_logits = student_logits.masked_fill(~neighbor_mask, float("-inf"))
    positive_logsumexp = torch.logsumexp(positive_logits, dim=1)
    all_logsumexp = torch.logsumexp(student_logits, dim=1)
    valid_rows = torch.isfinite(positive_logsumexp) & torch.isfinite(all_logsumexp)
    if not bool(valid_rows.any()):
        return zero, default_aux

    loss = -(positive_logsumexp[valid_rows] - all_logsumexp[valid_rows]).mean()

    with torch.no_grad():
        student_distance_nodiag = student_distance.detach().masked_fill(diag_mask, float("inf"))
        student_teacher_neighbor_distance = student_distance_nodiag.gather(1, teacher_neighbor_idx)
        student_neighbor_idx = torch.topk(
            student_distance_nodiag,
            k=effective_topk,
            dim=1,
            largest=False,
        ).indices
        overlap_hits = (teacher_neighbor_idx.unsqueeze(2) == student_neighbor_idx.unsqueeze(1)).any(dim=2).float()
        aux = {
            "avg_teacher_neighbor_distance": float(teacher_neighbor_distance.mean().detach().cpu()),
            "avg_student_neighbor_distance": float(student_teacher_neighbor_distance.mean().detach().cpu()),
            "avg_overlap_proxy": float(overlap_hits.mean().detach().cpu()),
            "effective_topk": effective_topk,
            "tau": safe_tau,
            "valid_query_count": int(valid_rows.sum().item()),
        }
    return loss, aux


def build_batch_neighbor_mask_from_global_knn(sample_indices,
                                              global_neighbor_indices,
                                              global_neighbor_distances=None,
                                              device=None):
    if torch.is_tensor(sample_indices):
        sample_indices = sample_indices.detach().cpu().tolist()
    sample_indices = [int(idx) for idx in sample_indices]

    batch_size = len(sample_indices)
    if device is None:
        device = "cpu"

    neighbor_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    if batch_size <= 1:
        return neighbor_mask, {
            "avg_teacher_neighbor_distance": 0.0,
            "avg_batch_neighbor_count": 0.0,
            "effective_topk": 0,
            "valid_query_count": 0,
        }

    sample_to_batch_pos = {sample_id: batch_pos for batch_pos, sample_id in enumerate(sample_indices)}
    visible_counts = []
    visible_teacher_distances = []

    for row, sample_id in enumerate(sample_indices):
        if sample_id < 0 or sample_id >= len(global_neighbor_indices):
            visible_counts.append(0)
            continue
        teacher_neighbors = global_neighbor_indices[sample_id]
        teacher_distances = None
        if global_neighbor_distances is not None:
            teacher_distances = global_neighbor_distances[sample_id]

        row_visible_count = 0
        for neighbor_offset, neighbor_id in enumerate(teacher_neighbors):
            neighbor_id = int(neighbor_id)
            if neighbor_id < 0:
                continue
            col = sample_to_batch_pos.get(neighbor_id)
            if col is None or col == row:
                continue
            neighbor_mask[row, col] = True
            row_visible_count += 1
            if teacher_distances is not None:
                visible_teacher_distances.append(float(teacher_distances[neighbor_offset]))
        visible_counts.append(row_visible_count)

    valid_query_count = int(sum(1 for count in visible_counts if count > 0))
    effective_topk = int(max(visible_counts)) if visible_counts else 0
    avg_batch_neighbor_count = float(sum(visible_counts) / float(len(visible_counts))) if visible_counts else 0.0
    avg_teacher_neighbor_distance = (
        float(sum(visible_teacher_distances) / float(len(visible_teacher_distances)))
        if visible_teacher_distances else 0.0
    )
    return neighbor_mask, {
        "avg_teacher_neighbor_distance": avg_teacher_neighbor_distance,
        "avg_batch_neighbor_count": avg_batch_neighbor_count,
        "effective_topk": effective_topk,
        "valid_query_count": valid_query_count,
    }


def neighborhood_consistency_loss_from_mask(student_emb,
                                            distance_fn,
                                            neighbor_mask,
                                            tau,
                                            teacher_aux=None,
                                            eps=1e-8):
    if student_emb.dim() != 2:
        raise ValueError("Neighborhood consistency expects [B, D] tensors, got {}.".format(tuple(student_emb.shape)))
    if neighbor_mask.dim() != 2 or neighbor_mask.shape[0] != student_emb.shape[0] or neighbor_mask.shape[1] != student_emb.shape[0]:
        raise ValueError("Neighbor mask shape {} must match student batch size {}.".format(
            tuple(neighbor_mask.shape),
            int(student_emb.shape[0]),
        ))

    zero = student_emb.new_tensor(0.0)
    batch_size = int(student_emb.shape[0])
    safe_tau = max(float(tau), float(eps))
    default_aux = {
        "avg_teacher_neighbor_distance": 0.0,
        "avg_student_neighbor_distance": 0.0,
        "avg_overlap_proxy": 0.0,
        "avg_batch_neighbor_count": 0.0,
        "effective_topk": 0,
        "tau": safe_tau,
        "valid_query_count": 0,
    }
    if batch_size <= 1:
        return zero, default_aux

    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=student_emb.device)
    neighbor_mask = neighbor_mask.to(device=student_emb.device, dtype=torch.bool)
    neighbor_mask = neighbor_mask & (~diag_mask)
    row_neighbor_counts = neighbor_mask.sum(dim=1)
    valid_rows = row_neighbor_counts > 0
    if not bool(valid_rows.any()):
        if teacher_aux is not None:
            default_aux.update({
                "avg_teacher_neighbor_distance": float(teacher_aux.get("avg_teacher_neighbor_distance", 0.0)),
                "avg_batch_neighbor_count": float(teacher_aux.get("avg_batch_neighbor_count", 0.0)),
                "effective_topk": int(teacher_aux.get("effective_topk", 0)),
                "valid_query_count": int(teacher_aux.get("valid_query_count", 0)),
            })
        return zero, default_aux

    student_distance = compute_pairwise_distance_matrix(student_emb, distance_fn)
    student_logits = -(student_distance / safe_tau)
    student_logits = student_logits.masked_fill(diag_mask, float("-inf"))

    positive_logits = student_logits.masked_fill(~neighbor_mask, float("-inf"))
    positive_logsumexp = torch.logsumexp(positive_logits, dim=1)
    all_logsumexp = torch.logsumexp(student_logits, dim=1)
    valid_rows = valid_rows & torch.isfinite(positive_logsumexp) & torch.isfinite(all_logsumexp)
    if not bool(valid_rows.any()):
        if teacher_aux is not None:
            default_aux.update({
                "avg_teacher_neighbor_distance": float(teacher_aux.get("avg_teacher_neighbor_distance", 0.0)),
                "avg_batch_neighbor_count": float(teacher_aux.get("avg_batch_neighbor_count", 0.0)),
                "effective_topk": int(teacher_aux.get("effective_topk", 0)),
                "valid_query_count": int(teacher_aux.get("valid_query_count", 0)),
            })
        return zero, default_aux

    loss = -(positive_logsumexp[valid_rows] - all_logsumexp[valid_rows]).mean()

    with torch.no_grad():
        masked_student_distances = student_distance.detach()[neighbor_mask]
        avg_student_neighbor_distance = (
            float(masked_student_distances.mean().detach().cpu())
            if masked_student_distances.numel() > 0 else 0.0
        )
        visible_counts = row_neighbor_counts.detach().cpu().tolist()
        visible_topk = int(max(visible_counts)) if visible_counts else 0
        avg_batch_neighbor_count = float(sum(visible_counts) / float(len(visible_counts))) if visible_counts else 0.0

        avg_overlap_proxy = 0.0
        if visible_topk > 0:
            student_distance_nodiag = student_distance.detach().masked_fill(diag_mask, float("inf"))
            student_topk_idx = torch.topk(
                student_distance_nodiag,
                k=visible_topk,
                dim=1,
                largest=False,
            ).indices
            overlap_scores = []
            valid_row_indices = torch.nonzero(valid_rows, as_tuple=False).squeeze(1)
            for row in valid_row_indices.tolist():
                row_k = int(row_neighbor_counts[row].item())
                if row_k <= 0:
                    continue
                teacher_idx = torch.nonzero(neighbor_mask[row], as_tuple=False).squeeze(1)
                student_idx = student_topk_idx[row, :row_k]
                hits = (teacher_idx.unsqueeze(1) == student_idx.unsqueeze(0)).any(dim=1).float().mean()
                overlap_scores.append(float(hits.detach().cpu()))
            if overlap_scores:
                avg_overlap_proxy = float(sum(overlap_scores) / float(len(overlap_scores)))

        aux = {
            "avg_teacher_neighbor_distance": 0.0,
            "avg_student_neighbor_distance": avg_student_neighbor_distance,
            "avg_overlap_proxy": avg_overlap_proxy,
            "avg_batch_neighbor_count": avg_batch_neighbor_count,
            "effective_topk": visible_topk,
            "tau": safe_tau,
            "valid_query_count": int(valid_rows.sum().item()),
        }
        if teacher_aux is not None:
            aux["avg_teacher_neighbor_distance"] = float(teacher_aux.get("avg_teacher_neighbor_distance", 0.0))
            aux["avg_batch_neighbor_count"] = float(teacher_aux.get("avg_batch_neighbor_count", avg_batch_neighbor_count))
            aux["effective_topk"] = int(teacher_aux.get("effective_topk", visible_topk))
            aux["valid_query_count"] = int(teacher_aux.get("valid_query_count", int(valid_rows.sum().item())))

    return loss, aux


def centered_correlation_penalty(lhs, rhs, eps=1e-6):
    lhs = lhs - lhs.mean(dim=0, keepdim=True)
    rhs = rhs - rhs.mean(dim=0, keepdim=True)
    lhs = lhs / (lhs.std(dim=0, unbiased=False, keepdim=True) + eps)
    rhs = rhs / (rhs.std(dim=0, unbiased=False, keepdim=True) + eps)
    corr = torch.matmul(lhs.transpose(0, 1), rhs) / float(max(1, lhs.shape[0]))
    return corr.pow(2).mean()


def compute_subspace_decorrelation_loss(u_global, u_local, u_progress):
    return (
        centered_correlation_penalty(u_global, u_local) +
        centered_correlation_penalty(u_global, u_progress) +
        centered_correlation_penalty(u_local, u_progress)
    )


def compute_bottleneck_regularizers(bottleneck_outputs, e_cont):
    if bottleneck_outputs is None:
        zero = e_cont.new_tensor(0.0)
        return {
            "decor_loss": zero,
            "stab_loss": zero,
            "total_loss": zero,
        }

    decor_loss = compute_subspace_decorrelation_loss(
        bottleneck_outputs["u_global"],
        bottleneck_outputs["u_local"],
        bottleneck_outputs["u_progress"],
    )
    stab_loss = F.mse_loss(bottleneck_outputs["e_bottleneck"], e_cont.detach())
    total_loss = decor_loss + stab_loss
    return {
        "decor_loss": decor_loss,
        "stab_loss": stab_loss,
        "total_loss": total_loss,
    }
