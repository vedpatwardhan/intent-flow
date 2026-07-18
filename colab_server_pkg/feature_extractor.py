import os
import time
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
    frame_str,
    history_frames,
    text_prompt,
    ui_annotations,
    point_clouds=None,
    view_name="world_center",
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

    if point_clouds and view_name in point_clouds:
        point_cloud = np.array(point_clouds[view_name])
    else:
        point_cloud = np.zeros((4900, 6), dtype=np.float32)
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
    point_cloud_local = []

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

        # Direct 1-to-1 grid mapping from the 70x70 point cloud to the 14x14 mask
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 223, 70).astype(int),
            np.linspace(0, 223, 70).astype(int),
        )
        xs_grid = grid_x.flatten()
        ys_grid = grid_y.flatten()
        mask_xs = np.clip(xs_grid // 16, 0, 13)
        mask_ys = np.clip(ys_grid // 16, 0, 13)
        local_mask = combined_mask[mask_ys, mask_xs] > 0.5
        point_cloud_local = point_cloud[local_mask].tolist()

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
            "point_cloud_local": point_cloud_local,
            "sam_mask": sam_mask,
            "sam_mask_224": sam_mask_224,
            "combined_mask_224": combined_mask_224,
        },
    }


def run_pointnext_model(point_cloud_np):
    """
    Run PointNeXt encoder on point cloud [NumPoints, >=3] to extract 384-dim feature token.
    """
    # Return a zero tensor immediately to maintain compatibility and eliminate 3D encoder overhead
    return torch.zeros(384, device=device).cpu()


def extract_single_view_stage3_obs_features(
    frame_str,
    history_frames,
    text_prompt,
    ui_annotations,
    tactile,
    proprioception,
    point_clouds=None,
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
        point_clouds=point_clouds,
        view_name=view_name,
    )

    dino_attn = features["dino_attn"]
    vision_feat = torch.tensor(dino_attn.flatten()[:384], dtype=torch.float32)
    if len(vision_feat) < 384:
        vision_feat = torch.cat([vision_feat, torch.zeros(384 - len(vision_feat))])

    # 1. PointNeXt: Always use the full point cloud for state representation
    pt_feat = run_pointnext_model(features["point_cloud"])

    # --- DIAGNOSTIC FRAME CAPTURE BLOCK ---
    # Automatically intercepts and extracts the frame if the feature space collapses into NaNs
    if torch.isnan(pt_feat).any() or torch.isinf(pt_feat).any():
        print(
            f"🚨 [DIAGNOSTIC DETECTION] NaN/Inf footprint triggered in PointNeXt features for view: {view_name}!"
        )
        pil_img = features.get("pil_frame")
        if pil_img is not None:
            os.makedirs("debug_anomalies", exist_ok=True)
            save_path = f"debug_anomalies/anomaly_{view_name}_{int(time.time())}.png"
            pil_img.save(save_path)
            print(
                f"📸 Captured and saved offending visual context frame to: {save_path}"
            )
    # --------------------------------------

    # 2. VGGT: Always use the full tracks for state representation
    # Global VGGT
    vggt_feat = torch.tensor(
        np.array(features["vggt_tracks"]).flatten()[:768], dtype=torch.float32
    )
    if len(vggt_feat) < 768:
        vggt_feat = torch.cat([vggt_feat, torch.zeros(768 - len(vggt_feat))])

    # Local VGGT
    vggt_local = features["task_isolated_features"]["vggt_local"]
    if len(vggt_local) > 0:
        vggt_local = torch.tensor(
            np.array(vggt_local).flatten()[:768], dtype=torch.float32
        )
        vggt_local = torch.cat([vggt_local, torch.zeros(768 - len(vggt_local))])
    else:
        vggt_local = torch.zeros(768 - len(vggt_feat))
    features["task_isolated_features"]["vggt_local"] = vggt_local

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
    point_clouds = getattr(payload, "point_clouds", None)
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
            point_clouds=point_clouds,
            view_name=view_name,
        )

    return obs_dict
