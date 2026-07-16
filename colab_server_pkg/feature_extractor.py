import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from colab_server_pkg.config import device
from colab_server_pkg.models_state import models
from colab_server_pkg.image_utils import decode_base64_image


def get_dino_attn_map(frame: np.ndarray) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor_frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        features = models["dino"].forward_features(tensor_frame)
        cls_token = features[0, 0]
        patches = features[0, -196:]
        cls_token = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-8)
        patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-8)
        attn = torch.matmul(patches, cls_token.T).view(14, 14).cpu().numpy()
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    return attn_norm


def get_clip_cosine_similarity(text_prompt: str, pil_frame: Image):
    inputs_text = models["clip_processor"](
        text=[text_prompt], return_tensors="pt", padding=True
    ).to(device)
    inputs_vision = models["clip_processor"](images=pil_frame, return_tensors="pt").to(
        device
    )
    for k, v in inputs_text.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs_text[k] = v.to(torch.float16)
    for k, v in inputs_vision.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs_vision[k] = v.to(torch.float16)
    with torch.no_grad():
        text_feat_raw = models["clip"].get_text_features(**inputs_text)
        vision_out = models["clip"].vision_model(**inputs_vision)
        norm_states = models["clip"].vision_model.post_layernorm(
            vision_out.last_hidden_state
        )
        patches = norm_states[0, 1:]
        patches_projected = models["clip"].visual_projection(patches)

        text_feat = text_feat_raw / text_feat_raw.norm(dim=-1, keepdim=True)
        patches_projected = patches_projected / patches_projected.norm(
            dim=-1, keepdim=True
        )
        sim = torch.matmul(patches_projected, text_feat.T).view(14, 14).cpu().numpy()
        sim_norm = (sim.max() - sim) / (sim.max() - sim.min() + 1e-8)

    return sim_norm, text_feat_raw.squeeze(0).cpu()


def get_point_cloud(pil_frame: Image, frame: np.ndarray):
    h, w = pil_frame.height, pil_frame.width
    inputs_depth = models["depth_processor"](images=pil_frame, return_tensors="pt").to(
        device
    )
    for k, v in inputs_depth.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs_depth[k] = v.to(torch.float16)
    with torch.no_grad():
        outputs_depth = models["depth_model"](**inputs_depth)
        depth_map = (
            torch.nn.functional.interpolate(
                outputs_depth.predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    grid_x, grid_y = np.meshgrid(
        np.linspace(0, w - 1, 70).astype(int),
        np.linspace(0, h - 1, 70).astype(int),
    )
    xs = grid_x.flatten()
    ys = grid_y.flatten()
    zs = depth_map[ys, xs]

    focal_length = max(w, h)
    cx = w / 2.0
    cy = h / 2.0

    xs_proj = (xs - cx) * zs / focal_length
    ys_proj = (cy - ys) * zs / focal_length
    zs_proj = zs

    colors = frame[ys, xs]
    rs = colors[:, 0] / 255.0
    gs = colors[:, 1] / 255.0
    bs = colors[:, 2] / 255.0

    x_range = xs_proj.max() - xs_proj.min() if len(xs_proj) > 0 else 0
    y_range = ys_proj.max() - ys_proj.min() if len(ys_proj) > 0 else 0
    z_range = zs_proj.max() - zs_proj.min() if len(zs_proj) > 0 else 0
    max_range = max(x_range, y_range, z_range, 1e-8)

    xs_norm = (xs_proj - xs_proj.mean()) / max_range * 1.6
    ys_norm = (ys_proj - ys_proj.mean()) / max_range * 1.6
    zs_norm = (zs_proj - zs_proj.mean()) / max_range * 1.6

    point_cloud = np.stack([xs_norm, ys_norm, zs_norm, rs, gs, bs], axis=1)
    return point_cloud, depth_map, xs, ys, xs_proj, ys_proj, zs_proj, colors


def get_filtered_point_cloud(
    xs: np.ndarray,
    ys: np.ndarray,
    combined_mask: np.ndarray,
    xs_proj: np.ndarray,
    ys_proj: np.ndarray,
    zs_proj: np.ndarray,
    colors: np.ndarray,
):
    xs_14 = np.clip((xs / 224 * 14).astype(int), 0, 13)
    ys_14 = np.clip((ys / 224 * 14).astype(int), 0, 13)
    mask_filter = combined_mask[ys_14, xs_14] > 0.0

    pts_iso = np.stack([xs_proj, ys_proj, zs_proj], axis=1)
    max_range = max((pts_iso.max(axis=0) - pts_iso.min(axis=0)).max(), 1e-8)
    pts_norm = (pts_iso - pts_iso.mean(axis=0)) * (1.6 / max_range)
    xs_norm, ys_norm, zs_norm = pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2]
    colors_norm = colors / 255.0
    rs, gs, bs = colors_norm[:, 0], colors_norm[:, 1], colors_norm[:, 2]

    rs[~mask_filter] = 0.0
    gs[~mask_filter] = 0.0
    bs[~mask_filter] = 0.0

    pointnext_isolated = np.stack([xs_norm, ys_norm, zs_norm, rs, gs, bs], axis=1)
    return pointnext_isolated


def get_vggt_point_tracks_base(history_frames: list[str]) -> tuple:
    if len(history_frames) < 2:
        history_frames = [history_frames[0], history_frames[0]]
    while len(history_frames) > 2:
        history_frames.pop(0)
    frames_np = [decode_base64_image(f) for f in history_frames]

    seq_len = len(frames_np)
    h, w = frames_np[0].shape[:2]

    tensor_list = []
    for f in frames_np:
        t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        tensor_list.append(t)

    video_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        vggt_outputs = models["vggt"](video_tensor)
        wp_raw = vggt_outputs["world_points"].squeeze(0).cpu().numpy()
        wp_conf = vggt_outputs["world_points_conf"].squeeze(0).cpu().numpy()

    if wp_raw.shape[0] == seq_len:
        wp_clean_4d = np.transpose(wp_raw, (1, 2, 0, 3))
        wp_conf_clean = np.transpose(wp_conf, (1, 2, 0))
    else:
        wp_clean_4d = wp_raw
        wp_conf_clean = wp_conf

    subsample_step = 6
    wp_frame2 = wp_clean_4d[:, :, -1, :]

    y_idx, x_idx = np.indices((h, w))
    seeds_grid = np.stack((x_idx, y_idx), axis=-1).astype(np.float32)

    pt_sampled = wp_clean_4d[::subsample_step, ::subsample_step, :, :]
    seeds_sampled = seeds_grid[::subsample_step, ::subsample_step, :]
    conf_sampled = wp_conf_clean[::subsample_step, ::subsample_step, 0]

    pt_flat = pt_sampled.reshape(-1, seq_len, 3)
    seeds_flat = seeds_sampled.reshape(-1, 2)
    conf_flat = conf_sampled.flatten()

    ix = seeds_flat[:, 0].astype(np.int32)
    iy = seeds_flat[:, 1].astype(np.int32)

    rgb_values = frames_np[0][iy, ix]
    foreground_mask = np.any(rgb_values > 15, axis=1)
    confidence_mask = conf_flat > 0.3
    base_mask = foreground_mask & confidence_mask
    return pt_flat, seeds_flat, wp_frame2, base_mask, frames_np, seq_len, h, w


def get_vggt_2d_tracks_from_mask(
    pt_flat: np.ndarray,
    seeds_flat: np.ndarray,
    wp_frame2: np.ndarray,
    valid_mask: np.ndarray,
    seq_len: int,
    h: int,
    w: int,
) -> list:
    pt_flat_filtered = pt_flat[valid_mask]
    seeds_flat_filtered = seeds_flat[valid_mask]

    num_valid_seeds = seeds_flat_filtered.shape[0]
    if num_valid_seeds == 0:
        return []

    x_ui = np.zeros((num_valid_seeds, seq_len), dtype=np.float32)
    y_ui = np.zeros((num_valid_seeds, seq_len), dtype=np.float32)

    x_ui[:, 0] = seeds_flat_filtered[:, 0]
    y_ui[:, 0] = seeds_flat_filtered[:, 1]

    search_radius = 20

    for idx in range(num_valid_seeds):
        sx, sy = int(seeds_flat_filtered[idx, 0]), int(seeds_flat_filtered[idx, 1])
        p1_3d = pt_flat_filtered[idx, 0, :]

        x_start = max(0, sx - search_radius)
        x_end = min(w, sx + search_radius + 1)
        y_start = max(0, sy - search_radius)
        y_end = min(h, sy + search_radius + 1)

        local_wp_f2 = wp_frame2[y_start:y_end, x_start:x_end, :]
        local_dists = np.linalg.norm(local_wp_f2 - p1_3d, axis=2)
        best_local_idx = np.argmin(local_dists)

        local_h, local_w = local_dists.shape
        match_local_y = best_local_idx // local_w
        match_local_x = best_local_idx % local_w

        x_ui[idx, 1] = x_start + match_local_x
        y_ui[idx, 1] = y_start + match_local_y

    distances = np.hypot(x_ui[:, 1] - x_ui[:, 0], y_ui[:, 1] - y_ui[:, 0])
    motion_mask = (distances > 4.0) & (distances < float(search_radius))
    valid_indices = np.where(motion_mask)[0]

    tracks_224 = []
    for idx in valid_indices:
        x1_norm = float(x_ui[idx, 0] / w)
        y1_norm = float(y_ui[idx, 0] / h)
        x2_norm = float(x_ui[idx, 1] / w)
        y2_norm = float(y_ui[idx, 1] / h)
        tracks_224.append([x1_norm, y1_norm, x2_norm, y2_norm])

    return tracks_224


def get_segment_masks(annotations: dict, pil_frame: Image) -> tuple:
    segments = annotations.get("segments", [])
    click_pts = [[seg["x"], seg["y"]] for seg in segments]
    num_pts = len(click_pts)

    if len(segments) == 0:
        return np.zeros((14, 14), dtype=np.float32), np.zeros(
            (224, 224), dtype=np.float32
        )

    inputs = models["sam_processor"](
        images=[pil_frame] * num_pts,
        input_points=[[[pt]] for pt in click_pts],
        input_labels=[[[1]] for _ in range(num_pts)],
        return_tensors="pt",
    ).to(device)
    for k, v in inputs.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.float16)

    with torch.inference_mode():
        outputs = models["sam"](**inputs)
        pred_masks = outputs.pred_masks[:, 0, 0].unsqueeze(1)
        resized_masks = torch.nn.functional.interpolate(
            pred_masks, size=(14, 14), mode="bilinear", align_corners=False
        )
        resized_masks_224 = torch.nn.functional.interpolate(
            pred_masks, size=(224, 224), mode="bilinear", align_corners=False
        )

        sam_grid_masks = (
            (resized_masks.squeeze(1) > 0.0).cpu().numpy().astype(np.float32)
        )
        sam_combined_mask = np.maximum.reduce(sam_grid_masks, axis=0)

        sam_grid_masks_224 = (
            (resized_masks_224.squeeze(1) > 0.0).cpu().numpy().astype(np.float32)
        )
        sam_combined_mask_224 = np.maximum.reduce(sam_grid_masks_224, axis=0)

    return sam_combined_mask, sam_combined_mask_224


def extract_features_common(
    frame_str, history_frames, text_prompt, ui_annotations, view_name="world_center"
):
    frame = decode_base64_image(frame_str)
    pil_frame = Image.fromarray(frame)
    h, w, _ = frame.shape

    # Extract history frames specific to the active view_name from the dict history representation
    view_history = []
    for h_f in history_frames:
        if isinstance(h_f, dict):
            if view_name in h_f:
                view_history.append(h_f[view_name])
            elif "world_center" in h_f:
                view_history.append(h_f["world_center"])
            elif len(h_f) > 0:
                view_history.append(list(h_f.values())[0])
        else:
            view_history.append(h_f)

    dino_attn = get_dino_attn_map(frame)
    clip_sim, text_feat = get_clip_cosine_similarity(text_prompt, pil_frame)
    point_cloud, depth_map, xs, ys, xs_proj, ys_proj, zs_proj, colors = get_point_cloud(
        pil_frame, frame
    )
    pt_flat, seeds_flat, wp_frame2, base_mask, _, seq_len, _, _ = (
        get_vggt_point_tracks_base(view_history)
    )
    vggt_tracks = get_vggt_2d_tracks_from_mask(
        pt_flat, seeds_flat, wp_frame2, base_mask, seq_len, h, w
    )

    combined_mask = np.zeros((14, 14), dtype=np.float32)
    combined_mask_224 = np.zeros((224, 224), dtype=np.float32)
    sam_mask = np.zeros((14, 14), dtype=np.float32)
    sam_mask_224 = np.zeros((224, 224), dtype=np.float32)
    vggt_local = []
    dino_subspace = np.array([], dtype=np.float32)
    pointnext_isolated = []

    view_annos = (ui_annotations or {}).get(view_name, {})
    if view_annos and (view_annos.get("crops") or view_annos.get("segments")):
        for crop in view_annos.get("crops", []):
            x_start = int((crop["x"] / 224) * 14)
            y_start = int((crop["y"] / 224) * 14)
            x_end = int(((crop["x"] + crop["width"]) / 224) * 14)
            y_end = int(((crop["y"] + crop["height"]) / 224) * 14)
            combined_mask[y_start:y_end, x_start:x_end] = 1.0

            cx = int(crop["x"])
            cy = int(crop["y"])
            cw = int(crop["width"])
            ch_val = int(crop["height"])
            combined_mask_224[cy : cy + ch_val, cx : cx + cw] = 1.0

        sam_mask, sam_mask_224 = get_segment_masks(view_annos, pil_frame)
        combined_mask = np.maximum(combined_mask, sam_mask)
        combined_mask_224 = np.maximum(combined_mask_224, sam_mask_224)

        mx = np.clip(((seeds_flat[:, 0] / w) * 14).astype(np.int32), 0, 13)
        my = np.clip(((seeds_flat[:, 1] / h) * 14).astype(np.int32), 0, 13)
        workspace_mask = combined_mask[my, mx] > 0.0
        local_valid_mask = base_mask & workspace_mask
        vggt_local = get_vggt_2d_tracks_from_mask(
            pt_flat, seeds_flat, wp_frame2, local_valid_mask, seq_len, h, w
        )

        dino_subspace = dino_attn * combined_mask
        pointnext_isolated = get_filtered_point_cloud(
            xs, ys, combined_mask, xs_proj, ys_proj, zs_proj, colors
        )

    return {
        "dino_attn": dino_attn,
        "clip_sim": clip_sim,
        "text_feat": text_feat,
        "point_cloud": point_cloud,
        "vggt_tracks": vggt_tracks,
        "pil_frame": pil_frame,
        "task_isolated_features": {
            "dino_subspace": dino_subspace,
            "vggt_local": vggt_local,
            "pointnext_isolated": pointnext_isolated,
            "sam_mask": sam_mask,
            "sam_mask_224": sam_mask_224,
            "combined_mask_224": combined_mask_224,
        },
    }


def run_pointnext_model(point_cloud_np):
    """
    Run PointNeXt encoder on point cloud [NumPoints, >=3] to extract 384-dim feature token.
    """
    if models.get("pointnext") is None or len(point_cloud_np) == 0:
        fallback = torch.tensor(
            np.array(point_cloud_np).flatten()[:384], dtype=torch.float32
        )
        if len(fallback) < 384:
            fallback = torch.cat([fallback, torch.zeros(384 - len(fallback))])
        return fallback

    try:
        cloud_data = np.array(point_cloud_np)
        if cloud_data.shape[1] < 3:
            # Pad intensity/color with zeros if shape is [N, 2] or similar
            pad = np.zeros((cloud_data.shape[0], 3 - cloud_data.shape[1]))
            cloud_data = np.concatenate([cloud_data, pad], axis=1)
        else:
            cloud_data = cloud_data[:, :3]

        pc_t = torch.tensor(cloud_data, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                feat = models["pointnext"](pc_t)
            if feat.dim() > 2:
                feat = feat.mean(dim=-1)  # Global average pooling over points
            return feat.squeeze(0).cpu()
    except Exception as e:
        print(f"Error running PointNeXt model: {e}")
        fallback = torch.tensor(
            np.array(point_cloud_np).flatten()[:384], dtype=torch.float32
        )
        if len(fallback) < 384:
            fallback = torch.cat([fallback, torch.zeros(384 - len(fallback))])
        return fallback


def extract_single_view_stage3_obs_features(
    frame_str,
    history_frames,
    text_prompt,
    ui_annotations,
    tactile,
    proprioception,
    view_name="world_center",
):
    """
    Core low-level function that processes features for a single RGB view, maps all modality
    representations to PyTorch tensors on the target device, and returns an formatted obs_dict.
    """
    features = extract_features_common(
        frame_str,
        history_frames,
        text_prompt,
        ui_annotations,
        view_name=view_name,
    )

    dino_attn = features["dino_attn"]
    vision_feat = torch.tensor(dino_attn.flatten()[:384], dtype=torch.float32)
    if len(vision_feat) < 384:
        vision_feat = torch.cat([vision_feat, torch.zeros(384 - len(vision_feat))])

    pointnext_isolated = features["task_isolated_features"]["pointnext_isolated"]
    if len(pointnext_isolated) > 0:
        pt_feat = run_pointnext_model(pointnext_isolated)
    else:
        pt_feat = run_pointnext_model(features["point_cloud"])

    if len(pt_feat) < 384:
        pt_feat = torch.cat([pt_feat, torch.zeros(384 - len(pt_feat))])

    vggt_local = features["task_isolated_features"]["vggt_local"]
    if len(vggt_local) > 0:
        vggt_feat = torch.tensor(
            np.array(vggt_local).flatten()[:768], dtype=torch.float32
        )
    else:
        vggt_feat = torch.tensor(
            np.array(features["vggt_tracks"]).flatten()[:768], dtype=torch.float32
        )

    if len(vggt_feat) < 768:
        vggt_feat = torch.cat([vggt_feat, torch.zeros(768 - len(vggt_feat))])

    tactile_grid = torch.tensor(tactile, dtype=torch.float32)
    if tactile_grid.shape != (4, 4):
        padded_tac = torch.zeros(4, 4)
        h_tac = min(tactile_grid.shape[0], 4)
        w_tac = min(tactile_grid.shape[1], 4)
        padded_tac[:h_tac, :w_tac] = tactile_grid[:h_tac, :w_tac]
        tactile_grid = padded_tac

    proprio = torch.tensor(proprioception[:24], dtype=torch.float32)
    if len(proprio) < 24:
        proprio = torch.cat([proprio, torch.zeros(24 - len(proprio))])

    text_feat_raw = features["text_feat"][:512]
    if len(text_feat_raw) < 512:
        text_feat_raw = torch.cat(
            [text_feat_raw, torch.zeros(512 - len(text_feat_raw))]
        )

    obs_dict = {
        "features": features,
        "vision": vision_feat.unsqueeze(0).to(device),
        "pointnext": pt_feat.unsqueeze(0).to(device),
        "vggt": vggt_feat.unsqueeze(0).to(device),
        "tactile": tactile_grid.unsqueeze(0).to(device),
        "proprioception": proprio.unsqueeze(0).to(device),
        "text": text_feat_raw.unsqueeze(0).unsqueeze(0).to(device),
        "dino_attn": features["dino_attn"],
        "clip_sim": features["clip_sim"],
        "point_cloud": features["point_cloud"],
        "vggt_tracks": features["vggt_tracks"],
        "task_isolated_features": features["task_isolated_features"],
    }
    return obs_dict


def extract_stage3_obs_features(payload):
    # Handle multi-view frames
    frames_dict = (
        payload.frames
        if hasattr(payload, "frames")
        else {"world_center": payload.frame}
    )
    obs_dict = {}

    # Process each view and extract features
    for view_name, frame_str in frames_dict.items():
        obs_dict[view_name] = extract_single_view_stage3_obs_features(
            frame_str,
            payload.history_frames,
            payload.text_prompt,
            payload.ui_annotations,
            payload.tactile,
            payload.proprioception,
            view_name=view_name,
        )

    return obs_dict
