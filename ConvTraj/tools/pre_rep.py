import hashlib
import os

import numpy as np
from scipy.ndimage import distance_transform_edt


_DISK_CACHED_IMAGE_MODES = frozenset([
    "dtw8",
    "motion6_pyr2",
    "shape5",
    "shape5_pyr2",
    "haus6",
    "dfd7",
])


def _use_disk_cache(image_mode):
    return image_mode in _DISK_CACHED_IMAGE_MODES


def is_disk_cached_image_mode(image_mode, disk_cache_enabled=True):
    return bool(disk_cache_enabled) and _use_disk_cache(image_mode)


def _cache_file_path(cache_dir, cache_key, suffix=".npy"):
    if not cache_dir or not cache_key:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "{}{}".format(str(cache_key), suffix))


def _load_cached_array(cache_dir, cache_key, cache_tag="SpecificCache"):
    cache_path = _cache_file_path(cache_dir, cache_key)
    if cache_path is None or not os.path.exists(cache_path):
        if cache_path is not None:
            print("[{}] miss {}".format(cache_tag, cache_path))
        return None
    print("[{}] hit {}".format(cache_tag, cache_path))
    return np.load(cache_path, allow_pickle=False)


def _save_cached_array(cache_dir, cache_key, array, cache_tag="SpecificCache"):
    cache_path = _cache_file_path(cache_dir, cache_key)
    if cache_path is None:
        return
    tmp_base = "{}.tmp-{}".format(cache_path[:-4], os.getpid())
    tmp_path = "{}.npy".format(tmp_base)
    np.save(tmp_base, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(tmp_path, cache_path)
    print("[{}] saved {}".format(cache_tag, cache_path))


def _haus6_dt_cache_key(x_ids, y_ids, x_num_classes, y_num_classes, grid_scale):
    digest = hashlib.sha1()
    digest.update(np.asarray(x_ids, dtype=np.int32).tobytes())
    digest.update(np.asarray(y_ids, dtype=np.int32).tobytes())
    digest.update("{}|{}|{}".format(int(x_num_classes), int(y_num_classes), int(grid_scale)).encode("utf-8"))
    return digest.hexdigest()


def _load_or_compute_haus6_distance(occ, cache_dir, cache_key):
    cached = _load_cached_array(cache_dir, cache_key, cache_tag="SpecificCache/haus6_dt")
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)

    dist = distance_transform_edt(1.0 - occ).astype(np.float32)
    max_dist = float(np.max(dist))
    if max_dist > 0.0:
        dist = np.log1p(dist) / np.log1p(max_dist)
    else:
        dist = np.zeros_like(dist, dtype=np.float32)
    _save_cached_array(cache_dir, cache_key, dist, cache_tag="SpecificCache/haus6_dt")
    return dist


def get_image_mode_channels(image_mode):
    if image_mode == "binary":
        return 1
    if image_mode == "motion6":
        return 6
    if image_mode == "dtw8":
        return 8
    if image_mode == "motion6_pyr2":
        return 12
    if image_mode == "multigrid3":
        return 3
    if image_mode == "shape5":
        return 5
    if image_mode == "shape5_pyr2":
        return 7
    if image_mode == "haus6":
        return 6
    if image_mode == "dfd7":
        return 7
    raise ValueError("Unsupported image_mode: {}".format(image_mode))

def onehot_encode(traj_list_onedim, num_classes):
    onehot_list = []
    for traj in traj_list_onedim:
        one_hot = np.zeros((len(traj), num_classes))
        one_hot[np.arange(len(traj)), traj] = 1.0
        onehot_list.append(one_hot.tolist())
    return onehot_list

def image_encode(x_list, y_list, x_num_classes, y_num_classes):
    image_list = []
    for i in range(len(x_list)):
        image = np.zeros((x_num_classes, y_num_classes))
        for j in range(len(x_list[i])):
            image[x_list[i][j]][y_list[i][j]] = 1.0
        image_list.append(image.tolist())
    return image_list


def _upsample_coarse_map(coarse_map, target_h, target_w, scale):
    expanded = np.repeat(np.repeat(coarse_map, scale, axis=0), scale, axis=1)
    return expanded[:target_h, :target_w]


def _scaled_grid_shape(x_num_classes, y_num_classes, scale):
    coarse_h = int(np.ceil(float(x_num_classes) / float(scale)))
    coarse_w = int(np.ceil(float(y_num_classes) / float(scale)))
    return coarse_h, coarse_w


def image_encode_multigrid_canvas(x_list, y_list, x_num_classes, y_num_classes, scale_factors=(1, 2, 4)):
    image_list = []
    for i in range(len(x_list)):
        canvas = np.zeros((len(scale_factors), x_num_classes, y_num_classes), dtype=np.float32)
        for scale_idx, scale in enumerate(scale_factors):
            coarse_h = int(np.ceil(float(x_num_classes) / float(scale)))
            coarse_w = int(np.ceil(float(y_num_classes) / float(scale)))
            coarse_map = np.zeros((coarse_h, coarse_w), dtype=np.float32)
            for j in range(len(x_list[i])):
                coarse_x = int(x_list[i][j]) // int(scale)
                coarse_y = int(y_list[i][j]) // int(scale)
                coarse_map[coarse_x, coarse_y] += 1.0

            max_value = float(np.max(coarse_map))
            if max_value > 0.0:
                coarse_map = coarse_map / max_value
            canvas[scale_idx] = _upsample_coarse_map(coarse_map, x_num_classes, y_num_classes, int(scale))
        image_list.append(canvas.tolist())
    return image_list


def _safe_average(sum_map, count_map):
    result = np.zeros_like(sum_map, dtype=np.float32)
    valid_mask = count_map > 0
    result[valid_mask] = sum_map[valid_mask] / count_map[valid_mask]
    return result


def _build_motion_canvas_single_scale(x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=1):
    coarse_h, coarse_w = _scaled_grid_shape(x_num_classes, y_num_classes, grid_scale)
    canvas = np.zeros((6, coarse_h, coarse_w), dtype=np.float32)
    hit_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    dx_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    dy_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    step_length_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    step_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    progress_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    progress_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    turn_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    turn_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)

    traj_length = min(len(x_ids), len(y_ids), len(traj))
    if traj_length == 0:
        return canvas

    lon_values = np.asarray([point[0] for point in traj[:traj_length]], dtype=np.float32)
    lat_values = np.asarray([point[1] for point in traj[:traj_length]], dtype=np.float32)
    max_step_length = 0.0
    scale = int(grid_scale)

    for j in range(traj_length):
        coarse_x = int(x_ids[j]) // scale
        coarse_y = int(y_ids[j]) // scale

        hit_count[coarse_x, coarse_y] += 1.0
        progress_count[coarse_x, coarse_y] += 1.0
        if traj_length > 1:
            progress_sum[coarse_x, coarse_y] += float(j) / float(traj_length - 1)

        if j == 0:
            continue

        dx = float(lon_values[j] - lon_values[j - 1])
        dy = float(lat_values[j] - lat_values[j - 1])
        step_length = float(np.hypot(dx, dy))
        max_step_length = max(max_step_length, step_length)

        if step_length > 0.0:
            dx_sum[coarse_x, coarse_y] += dx / step_length
            dy_sum[coarse_x, coarse_y] += dy / step_length

        step_length_sum[coarse_x, coarse_y] += step_length
        step_count[coarse_x, coarse_y] += 1.0

    if traj_length >= 3:
        for j in range(1, traj_length - 1):
            coarse_x = int(x_ids[j]) // scale
            coarse_y = int(y_ids[j]) // scale

            prev_dx = float(lon_values[j] - lon_values[j - 1])
            prev_dy = float(lat_values[j] - lat_values[j - 1])
            next_dx = float(lon_values[j + 1] - lon_values[j])
            next_dy = float(lat_values[j + 1] - lat_values[j])
            prev_norm = float(np.hypot(prev_dx, prev_dy))
            next_norm = float(np.hypot(next_dx, next_dy))

            if prev_norm > 0.0 and next_norm > 0.0:
                cross = prev_dx * next_dy - prev_dy * next_dx
                dot = prev_dx * next_dx + prev_dy * next_dy
                turn_value = abs(np.arctan2(cross, dot)) / np.pi
            else:
                turn_value = 0.0

            turn_sum[coarse_x, coarse_y] += turn_value
            turn_count[coarse_x, coarse_y] += 1.0

    max_hit_count = float(np.max(hit_count))
    if max_hit_count > 0.0:
        canvas[0] = np.clip(hit_count / max_hit_count, 0.0, 1.0)

    canvas[1] = np.clip(_safe_average(dx_sum, step_count), -1.0, 1.0)
    canvas[2] = np.clip(_safe_average(dy_sum, step_count), -1.0, 1.0)

    mean_step_length = _safe_average(step_length_sum, step_count)
    if max_step_length > 0.0:
        canvas[3] = np.clip(mean_step_length / max_step_length, 0.0, 1.0)

    canvas[4] = np.clip(_safe_average(progress_sum, progress_count), 0.0, 1.0)
    canvas[5] = np.clip(_safe_average(turn_sum, turn_count), 0.0, 1.0)
    return canvas


def _motion_canvas_to_target_scale(canvas, x_num_classes, y_num_classes, grid_scale):
    if int(grid_scale) == 1:
        return canvas
    target_canvas = np.zeros((canvas.shape[0], x_num_classes, y_num_classes), dtype=np.float32)
    for channel_idx in range(canvas.shape[0]):
        target_canvas[channel_idx] = _upsample_coarse_map(
            canvas[channel_idx],
            x_num_classes,
            y_num_classes,
            int(grid_scale),
        )
    return target_canvas


def _build_shape_dfd_stats_single_scale(x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=1):
    coarse_h, coarse_w = _scaled_grid_shape(x_num_classes, y_num_classes, grid_scale)
    hit_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    dx_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    dy_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    step_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    progress_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    progress_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    turn_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    signed_turn_sum = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    turn_count = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    start_marker = np.zeros((coarse_h, coarse_w), dtype=np.float32)
    end_marker = np.zeros((coarse_h, coarse_w), dtype=np.float32)

    traj_length = min(len(x_ids), len(y_ids), len(traj))
    if traj_length == 0:
        return {
            "hit_count": hit_count,
            "dx_sum": dx_sum,
            "dy_sum": dy_sum,
            "step_count": step_count,
            "progress_sum": progress_sum,
            "progress_count": progress_count,
            "turn_sum": turn_sum,
            "signed_turn_sum": signed_turn_sum,
            "turn_count": turn_count,
            "start_marker": start_marker,
            "end_marker": end_marker,
        }

    lon_values = np.asarray([point[0] for point in traj[:traj_length]], dtype=np.float32)
    lat_values = np.asarray([point[1] for point in traj[:traj_length]], dtype=np.float32)
    scale = int(grid_scale)

    cumulative_length = np.zeros((traj_length,), dtype=np.float32)
    step_lengths = np.zeros((max(traj_length - 1, 0),), dtype=np.float32)
    for j in range(1, traj_length):
        dx = float(lon_values[j] - lon_values[j - 1])
        dy = float(lat_values[j] - lat_values[j - 1])
        step_length = float(np.hypot(dx, dy))
        step_lengths[j - 1] = step_length
        cumulative_length[j] = cumulative_length[j - 1] + step_length
    total_length = float(cumulative_length[-1]) if traj_length > 0 else 0.0

    for j in range(traj_length):
        coarse_x = int(x_ids[j]) // scale
        coarse_y = int(y_ids[j]) // scale
        hit_count[coarse_x, coarse_y] += 1.0

        if j == 0:
            start_marker[coarse_x, coarse_y] = 1.0
        if j == traj_length - 1:
            end_marker[coarse_x, coarse_y] = 1.0

        progress_count[coarse_x, coarse_y] += 1.0
        if total_length > 0.0:
            progress_sum[coarse_x, coarse_y] += cumulative_length[j] / total_length
        elif traj_length > 1:
            progress_sum[coarse_x, coarse_y] += float(j) / float(traj_length - 1)

        if j == 0:
            continue

        dx = float(lon_values[j] - lon_values[j - 1])
        dy = float(lat_values[j] - lat_values[j - 1])
        step_length = float(step_lengths[j - 1])
        if step_length > 0.0:
            dx_sum[coarse_x, coarse_y] += dx / step_length
            dy_sum[coarse_x, coarse_y] += dy / step_length
            step_count[coarse_x, coarse_y] += 1.0

    if traj_length >= 3:
        for j in range(1, traj_length - 1):
            coarse_x = int(x_ids[j]) // scale
            coarse_y = int(y_ids[j]) // scale

            prev_dx = float(lon_values[j] - lon_values[j - 1])
            prev_dy = float(lat_values[j] - lat_values[j - 1])
            next_dx = float(lon_values[j + 1] - lon_values[j])
            next_dy = float(lat_values[j + 1] - lat_values[j])
            prev_norm = float(np.hypot(prev_dx, prev_dy))
            next_norm = float(np.hypot(next_dx, next_dy))

            if prev_norm > 0.0 and next_norm > 0.0:
                cross = prev_dx * next_dy - prev_dy * next_dx
                dot = prev_dx * next_dx + prev_dy * next_dy
                signed_turn_value = np.arctan2(cross, dot) / np.pi
                turn_value = abs(signed_turn_value)
            else:
                signed_turn_value = 0.0
                turn_value = 0.0

            turn_sum[coarse_x, coarse_y] += turn_value
            signed_turn_sum[coarse_x, coarse_y] += signed_turn_value
            turn_count[coarse_x, coarse_y] += 1.0

    return {
        "hit_count": hit_count,
        "dx_sum": dx_sum,
        "dy_sum": dy_sum,
        "step_count": step_count,
        "progress_sum": progress_sum,
        "progress_count": progress_count,
        "turn_sum": turn_sum,
        "signed_turn_sum": signed_turn_sum,
        "turn_count": turn_count,
        "start_marker": start_marker,
        "end_marker": end_marker,
    }


def _build_shape5_canvas_single_scale(x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=1):
    stats = _build_shape_dfd_stats_single_scale(
        x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=grid_scale
    )
    hit_count = stats["hit_count"]
    max_hit_count = float(np.max(hit_count))
    canvas = np.zeros((5,) + hit_count.shape, dtype=np.float32)
    canvas[0] = (hit_count > 0.0).astype(np.float32)
    if max_hit_count > 0.0:
        canvas[1] = np.clip(np.log1p(hit_count) / np.log1p(max_hit_count), 0.0, 1.0)
    canvas[2] = np.clip(_safe_average(stats["turn_sum"], stats["turn_count"]), 0.0, 1.0)
    canvas[3] = np.clip(stats["start_marker"], 0.0, 1.0)
    canvas[4] = np.clip(stats["end_marker"], 0.0, 1.0)
    return canvas


def _build_haus6_canvas_single_scale(
    x_ids,
    y_ids,
    traj,
    x_num_classes,
    y_num_classes,
    grid_scale=1,
    haus6_dt_cache_dir=None,
):
    stats = _build_shape_dfd_stats_single_scale(
        x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=grid_scale
    )
    hit_count = stats["hit_count"]
    max_hit_count = float(np.max(hit_count))
    occ = (hit_count > 0.0).astype(np.float32)

    if haus6_dt_cache_dir:
        dist = _load_or_compute_haus6_distance(
            occ,
            haus6_dt_cache_dir,
            _haus6_dt_cache_key(x_ids, y_ids, x_num_classes, y_num_classes, grid_scale),
        )
    else:
        dist = _load_or_compute_haus6_distance(occ, None, None)

    canvas = np.zeros((6,) + hit_count.shape, dtype=np.float32)
    canvas[0] = occ
    if max_hit_count > 0.0:
        canvas[1] = np.clip(np.log1p(hit_count) / np.log1p(max_hit_count), 0.0, 1.0)
    canvas[2] = np.clip(dist, 0.0, 1.0)
    canvas[3] = np.clip(_safe_average(stats["turn_sum"], stats["turn_count"]), 0.0, 1.0)
    canvas[4] = np.clip(stats["start_marker"], 0.0, 1.0)
    canvas[5] = np.clip(stats["end_marker"], 0.0, 1.0)
    return canvas


def _build_shape5_pyr2_canvas(x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=1):
    del grid_scale
    fine_canvas = _build_shape5_canvas_single_scale(
        x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=1
    )

    coarse_stats = _build_shape_dfd_stats_single_scale(
        x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=2
    )
    coarse_hit_count = coarse_stats["hit_count"]
    coarse_max_hit_count = float(np.max(coarse_hit_count))
    coarse_canvas = np.zeros((2,) + coarse_hit_count.shape, dtype=np.float32)
    coarse_canvas[0] = (coarse_hit_count > 0.0).astype(np.float32)
    if coarse_max_hit_count > 0.0:
        coarse_canvas[1] = np.clip(
            np.log1p(coarse_hit_count) / np.log1p(coarse_max_hit_count),
            0.0,
            1.0,
        )
    coarse_canvas = _motion_canvas_to_target_scale(
        coarse_canvas,
        x_num_classes,
        y_num_classes,
        grid_scale=2,
    )
    return np.concatenate((fine_canvas, coarse_canvas), axis=0)


def _build_dfd7_canvas_single_scale(x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=1):
    stats = _build_shape_dfd_stats_single_scale(
        x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=grid_scale
    )
    hit_count = stats["hit_count"]
    canvas = np.zeros((7,) + hit_count.shape, dtype=np.float32)
    canvas[0] = (hit_count > 0.0).astype(np.float32)
    canvas[1] = np.clip(_safe_average(stats["dx_sum"], stats["step_count"]), -1.0, 1.0)
    canvas[2] = np.clip(_safe_average(stats["dy_sum"], stats["step_count"]), -1.0, 1.0)
    canvas[3] = np.clip(_safe_average(stats["progress_sum"], stats["progress_count"]), 0.0, 1.0)
    canvas[4] = np.clip(_safe_average(stats["turn_sum"], stats["turn_count"]), 0.0, 1.0)
    canvas[5] = np.clip(stats["start_marker"], 0.0, 1.0)
    canvas[6] = np.clip(stats["end_marker"], 0.0, 1.0)
    return canvas


def _build_dtw8_canvas_single_scale(x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=1):
    stats = _build_shape_dfd_stats_single_scale(
        x_ids, y_ids, traj, x_num_classes, y_num_classes, grid_scale=grid_scale
    )
    hit_count = stats["hit_count"]
    max_hit_count = float(np.max(hit_count))
    canvas = np.zeros((8,) + hit_count.shape, dtype=np.float32)
    canvas[0] = (hit_count > 0.0).astype(np.float32)
    if max_hit_count > 0.0:
        canvas[1] = np.clip(np.log1p(hit_count) / np.log1p(max_hit_count), 0.0, 1.0)
    canvas[2] = np.clip(_safe_average(stats["dx_sum"], stats["step_count"]), -1.0, 1.0)
    canvas[3] = np.clip(_safe_average(stats["dy_sum"], stats["step_count"]), -1.0, 1.0)
    canvas[4] = np.clip(_safe_average(stats["progress_sum"], stats["progress_count"]), 0.0, 1.0)
    canvas[5] = np.clip(_safe_average(stats["signed_turn_sum"], stats["turn_count"]), -1.0, 1.0)
    canvas[6] = np.clip(stats["start_marker"], 0.0, 1.0)
    canvas[7] = np.clip(stats["end_marker"], 0.0, 1.0)
    return canvas


def image_encode_motion_canvas(x_list, y_list, x_num_classes, y_num_classes, traj_list, grid_scale=1):
    if traj_list is None:
        raise ValueError("traj_list is required when image_mode is motion6")
    if len(x_list) != len(y_list) or len(x_list) != len(traj_list):
        raise ValueError("x_list, y_list, and traj_list must have the same length")

    image_list = []
    for i in range(len(x_list)):
        motion_canvas = _build_motion_canvas_single_scale(
            x_list[i],
            y_list[i],
            traj_list[i],
            x_num_classes,
            y_num_classes,
            grid_scale=grid_scale,
        )
        motion_canvas = _motion_canvas_to_target_scale(
            motion_canvas,
            x_num_classes,
            y_num_classes,
            grid_scale=grid_scale,
        )
        image_list.append(np.asarray(motion_canvas, dtype=np.float32))

    return np.asarray(image_list, dtype=np.float32)


def image_encode_motion_canvas_pyramid(x_list, y_list, x_num_classes, y_num_classes, traj_list, scale_factors=(1, 2)):
    if traj_list is None:
        raise ValueError("traj_list is required when image_mode is motion6_pyr2")
    if len(x_list) != len(y_list) or len(x_list) != len(traj_list):
        raise ValueError("x_list, y_list, and traj_list must have the same length")

    image_list = []
    for i in range(len(x_list)):
        multi_scale_canvas = []
        for scale in scale_factors:
            motion_canvas = _build_motion_canvas_single_scale(
                x_list[i],
                y_list[i],
                traj_list[i],
                x_num_classes,
                y_num_classes,
                grid_scale=scale,
            )
            motion_canvas = _motion_canvas_to_target_scale(
                motion_canvas,
                x_num_classes,
                y_num_classes,
                grid_scale=scale,
            )
            multi_scale_canvas.append(motion_canvas)
        stacked_canvas = np.concatenate(multi_scale_canvas, axis=0)
        image_list.append(np.asarray(stacked_canvas, dtype=np.float32))
    return np.asarray(image_list, dtype=np.float32)


def _encode_special_canvas(
    x_list,
    y_list,
    x_num_classes,
    y_num_classes,
    traj_list,
    builder,
    image_mode,
    **builder_kwargs
):
    if traj_list is None:
        raise ValueError("traj_list is required when image_mode is {}".format(image_mode))
    if len(x_list) != len(y_list) or len(x_list) != len(traj_list):
        raise ValueError("x_list, y_list, and traj_list must have the same length")

    image_list = []
    for i in range(len(x_list)):
        canvas = builder(
            x_list[i],
            y_list[i],
            traj_list[i],
            x_num_classes,
            y_num_classes,
            grid_scale=1,
            **builder_kwargs
        )
        image_list.append(np.asarray(canvas, dtype=np.float32))
    return np.asarray(image_list, dtype=np.float32)


def image_encode_shape5(x_list, y_list, x_num_classes, y_num_classes, traj_list):
    return _encode_special_canvas(
        x_list, y_list, x_num_classes, y_num_classes, traj_list, _build_shape5_canvas_single_scale, "shape5"
    )


def image_encode_haus6(x_list, y_list, x_num_classes, y_num_classes, traj_list, haus6_dt_cache_dir=None):
    return _encode_special_canvas(
        x_list,
        y_list,
        x_num_classes,
        y_num_classes,
        traj_list,
        _build_haus6_canvas_single_scale,
        "haus6",
        haus6_dt_cache_dir=haus6_dt_cache_dir,
    )


def image_encode_shape5_pyr2(x_list, y_list, x_num_classes, y_num_classes, traj_list):
    return _encode_special_canvas(
        x_list, y_list, x_num_classes, y_num_classes, traj_list, _build_shape5_pyr2_canvas, "shape5_pyr2"
    )


def image_encode_dfd7(x_list, y_list, x_num_classes, y_num_classes, traj_list):
    return _encode_special_canvas(
        x_list, y_list, x_num_classes, y_num_classes, traj_list, _build_dfd7_canvas_single_scale, "dfd7"
    )


def image_encode_dtw8(x_list, y_list, x_num_classes, y_num_classes, traj_list):
    return _encode_special_canvas(
        x_list, y_list, x_num_classes, y_num_classes, traj_list, _build_dtw8_canvas_single_scale, "dtw8"
    )


def build_traj_image(x_list,
                     y_list,
                     x_num_classes,
                     y_num_classes,
                     image_mode="binary",
                     traj_list=None,
                     cache_dir=None,
                     cache_key=None,
                     haus6_dt_cache_dir=None):
    if is_disk_cached_image_mode(image_mode) and cache_dir and cache_key:
        cached = _load_cached_array(cache_dir, cache_key, cache_tag="SpecificCache/{}".format(image_mode))
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)

    if image_mode == "binary":
        images = image_encode(x_list, y_list, x_num_classes, y_num_classes)
    elif image_mode == "motion6":
        images = image_encode_motion_canvas(x_list, y_list, x_num_classes, y_num_classes, traj_list)
    elif image_mode == "dtw8":
        images = image_encode_dtw8(x_list, y_list, x_num_classes, y_num_classes, traj_list)
    elif image_mode == "motion6_pyr2":
        images = image_encode_motion_canvas_pyramid(
            x_list,
            y_list,
            x_num_classes,
            y_num_classes,
            traj_list,
            scale_factors=(1, 2),
        )
    elif image_mode == "multigrid3":
        images = image_encode_multigrid_canvas(x_list, y_list, x_num_classes, y_num_classes)
    elif image_mode == "shape5":
        images = image_encode_shape5(x_list, y_list, x_num_classes, y_num_classes, traj_list)
    elif image_mode == "haus6":
        images = image_encode_haus6(
            x_list,
            y_list,
            x_num_classes,
            y_num_classes,
            traj_list,
            haus6_dt_cache_dir=haus6_dt_cache_dir,
        )
    elif image_mode == "shape5_pyr2":
        images = image_encode_shape5_pyr2(x_list, y_list, x_num_classes, y_num_classes, traj_list)
    elif image_mode == "dfd7":
        images = image_encode_dfd7(x_list, y_list, x_num_classes, y_num_classes, traj_list)
    else:
        raise ValueError("Unsupported image_mode: {}".format(image_mode))

    images = np.asarray(images, dtype=np.float32)
    if is_disk_cached_image_mode(image_mode) and cache_dir and cache_key:
        _save_cached_array(cache_dir, cache_key, images, cache_tag="SpecificCache/{}".format(image_mode))
    return images
