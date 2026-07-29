import cv2
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

        text_feat = torch.nn.functional.normalize(text_feat_raw, p=2, dim=-1)
        patches_projected = torch.nn.functional.normalize(
            patches_projected, p=2, dim=-1
        )
        sim = torch.matmul(patches_projected, text_feat.T).view(14, 14).cpu().numpy()
        sim_norm = (sim.max() - sim) / (sim.max() - sim.min() + 1e-8)

    return sim_norm, text_feat.squeeze(0)


def get_vggt_motion_field(frames: list[str]) -> tuple:
    if len(frames) < 2:
        raise IndexError("Motion vectors require atleast two frames of history.")
    frames = frames[-2:]
    frames = [decode_base64_image(frame) for frame in frames]
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )
    processed_tensors = (
        torch.stack([transform_pipeline(frame) for frame in frames], dim=0)
        .unsqueeze(0)
        .to(device)
    )

    # Execute model forward pass
    with torch.no_grad():
        outputs = models["vggt"](processed_tensors)
        # Shape: [2, 224, 224, 3]
        world_points = outputs["world_points"].squeeze(0).cpu().numpy()

    # Extract the absolute spatial positioning fields
    wp_t0 = world_points[0]  # Shape: [224, 224, 3]
    wp_t1 = world_points[1]  # Shape: [224, 224, 3]

    # Delta trajectories matrix = Point(t1) - Point(t0)
    dx = wp_t1[:, :, 0] - wp_t0[:, :, 0]  # [224, 224]
    dy = wp_t1[:, :, 1] - wp_t0[:, :, 1]  # [224, 224]

    # Magnitude = sqrt(dx^2 + dy^2)
    motion_magnitude = np.sqrt(dx**2 + dy**2)  # [224, 224]
    motion_min = motion_magnitude.min()
    motion_max = motion_magnitude.max()
    motion_norm = (motion_magnitude - motion_min) / (motion_max - motion_min + 1e-8)
    return motion_norm


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


def apply_70th_percentile_dino_thresholding(dino_attn: np.ndarray) -> np.ndarray:
    """Zeroes out the bottom 30% intensity values (below 70th percentile) of a DINO feature map."""
    if dino_attn.size == 0:
        return dino_attn
    p70 = np.percentile(dino_attn, 70)
    return np.where(dino_attn >= p70, dino_attn, 0.0)


def pad_features(feature, target_dim):
    if len(feature) < target_dim:
        feature = torch.cat(
            [feature, torch.zeros(target_dim - len(feature), device=feature.device)]
        )
    return feature


def extract_features_common(
    frame_str: str,
    history_frames: list[str],
    text_prompt: str,
    ui_annotations: dict[str, list],
    view_name: str = "world_center",
):
    frame = decode_base64_image(frame_str)
    pil_frame = Image.fromarray(frame)
    h, w, _ = frame.shape

    dino_attn = get_dino_attn_map(frame)  # [14, 14]
    dino_attn = apply_70th_percentile_dino_thresholding(dino_attn) * 0.7
    clip_sim, text_feat = get_clip_cosine_similarity(text_prompt, pil_frame)
    motion_field = get_vggt_motion_field(history_frames)  # [224, 224]

    combined_mask = np.zeros((14, 14), dtype=np.float32)
    combined_mask_224 = np.zeros((224, 224), dtype=np.float32)
    sam_mask = np.zeros((14, 14), dtype=np.float32)
    sam_mask_224 = np.zeros((224, 224), dtype=np.float32)
    dino_subspace = np.array([], dtype=np.float32)
    motion_field_subspace = np.array([], dtype=np.float32)

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
        dino_subspace = dino_attn * combined_mask
        motion_field_subspace = motion_field * combined_mask_224

    return {
        "dino_attn": dino_attn,
        "clip_sim": clip_sim,
        "text_feat": text_feat,
        "motion_field": motion_field,
        "pil_frame": pil_frame,
        "task_isolated_features": {
            "dino_subspace": dino_subspace,
            "motion_field_subspace": motion_field_subspace,
            "sam_mask": sam_mask,
            "sam_mask_224": sam_mask_224,
            "combined_mask_224": combined_mask_224,
        },
    }


def extract_single_view_stage3_obs_features(
    frame_str: str,
    history_frames: list[str],
    text_prompt: str,
    ui_annotations: dict[str, list],
    tactile: list[float],
    proprioception: list[list],
    view_name: str = "world_center",
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

    # Pad DINO to 384 dimensions
    dino_attn = features["dino_attn"]
    vision_feat = torch.tensor(dino_attn.flatten()[:384], dtype=torch.float32).to(
        device
    )
    vision_feat = pad_features(vision_feat, 384)
    dino_subspace = torch.tensor(
        features["task_isolated_features"]["dino_subspace"].flatten()[:384],
        dtype=torch.float32,
    ).to(device)
    dino_subspace = pad_features(dino_subspace, 384)
    features["task_isolated_features"]["dino_subspace"] = dino_subspace

    # Motion Field --> 224, 224
    vggt_feat = torch.tensor(features["motion_field"], dtype=torch.float32).to(device)
    vggt_subspace = torch.tensor(
        features["task_isolated_features"]["motion_field_subspace"],
        dtype=torch.float32,
    ).to(device)
    features["task_isolated_features"]["vggt_subspace"] = vggt_subspace

    # CLIP --> 384
    clip_feat = torch.tensor(
        features["clip_sim"].flatten()[:384], dtype=torch.float32
    ).to(device)
    clip_feat = pad_features(clip_feat, 384)

    # LEGACY: PointNeXt representation with zeros
    pt_feat = torch.zeros(384, device=device)

    # Pad Proprioception to 58 dimension
    proprioception = pad_features(torch.tensor(proprioception).to(device), 58)

    obs_dict = {
        "features": features,
        "vision": vision_feat.unsqueeze(0),  # [1, 384]
        "pointnext": pt_feat.unsqueeze(0),  # [1, 384]
        "vggt": vggt_feat.unsqueeze(0),  # [1, 224, 224]
        "tactile": torch.tensor(tactile).to(device).flatten().unsqueeze(0),  # [1, 16]
        "proprioception": proprioception.unsqueeze(0),  # [1, 58]
        # "text": features["text_feat"].unsqueeze(0),  # [1, 512]
        "text": clip_feat.unsqueeze(0),  # [1, 384]
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
        history_frames = [frames[view_name] for frames in payload.history_frames]
        obs_dict[view_name] = extract_single_view_stage3_obs_features(
            frame_str,  # str
            history_frames,  # list[str]
            payload.text_prompt,  # str
            payload.ui_annotations,  # dict[str, list]
            payload.tactile,  # list[float]
            payload.proprioception,  # list[list]
            view_name=view_name,  # str
        )

    any_view = next(iter(obs_dict.values()))
    return obs_dict, {
        "vision": torch.cat(
            [obs_dict[view]["vision"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "pointnext": torch.cat(
            [obs_dict[view]["pointnext"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "vggt": torch.cat(
            [obs_dict[view]["vggt"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "text": torch.cat(
            [obs_dict[view]["text"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "tactile": any_view["tactile"],
        "proprioception": any_view["proprioception"],
    }


def construct_stage3_latent_goal_features(obs_dict, ui_annotations):
    """
    Constructs on-manifold target goal representation (s_target) by applying post-extraction
    latent feature transformations across DINOv3, CLIP, and VGGT feature maps ONLY for views
    that have active UI annotations (crops and intent vectors).
    """
    # 1. Extract clean base features from unblurred original camera frames
    for view_name, view_annos in ui_annotations.items():
        # Retrieve annotations for the view
        crops = view_annos.get("crops", [])
        segments = view_annos.get("segments", [])
        vectors = view_annos.get("vectors", [])

        # Transform only if view has at least 2 active annotations and 1 vector
        active_annos = crops + segments
        if len(active_annos) < 2 or len(vectors) < 1:
            continue

        # Sample and process first two annotations and first vector
        p0_anno, p1_anno = active_annos[0], active_annos[1]
        vec = vectors[0]
        view_features = obs_dict[view_name]["features"]
        sam_mask_224 = view_features.get("task_isolated_features", {}).get(
            "sam_mask_224", None
        )

        # Helper to extract segment binary mask and SAM component centroid
        def extract_segment_mask_and_center(anno, sam_mask_224_in):
            is_crop_type = "width" in anno and "height" in anno
            if is_crop_type:
                cx = anno["x"] + anno["width"] / 2.0
                cy = anno["y"] + anno["height"] / 2.0
                mask_224 = np.zeros((224, 224), dtype=np.float32)
                x1 = min(223, max(0, int(anno["x"])))
                y1 = min(223, max(0, int(anno["y"])))
                x2 = min(224, max(x1 + 1, int(anno["x"] + anno["width"])))
                y2 = min(224, max(y1 + 1, int(anno["y"] + anno["height"])))
                mask_224[y1:y2, x1:x2] = 1.0
                return mask_224, cx, cy

            cx, cy = float(anno.get("x", 0)), float(anno.get("y", 0))
            if sam_mask_224_in is not None and np.sum(sam_mask_224_in) > 0:
                try:
                    mask_np_224 = (
                        np.array(sam_mask_224_in)
                        if isinstance(sam_mask_224_in, torch.Tensor)
                        else sam_mask_224_in
                    )
                    mask_uint8 = (mask_np_224 > 0).astype(np.uint8) * 255
                    num_labels, labels = cv2.connectedComponents(mask_uint8)

                    cx_scaled = min(223, max(0, int(cx)))
                    cy_scaled = min(223, max(0, int(cy)))
                    lbl = labels[cy_scaled, cx_scaled]

                    if lbl == 0:
                        window = labels[
                            max(0, cy_scaled - 5) : min(224, cy_scaled + 6),
                            max(0, cx_scaled - 5) : min(224, cx_scaled + 6),
                        ]
                        non_zero = window[window > 0]
                        if len(non_zero) > 0:
                            lbl = non_zero[0]

                    if lbl > 0:
                        segment_mask_224 = (labels == lbl).astype(np.float32)
                        indices = np.argwhere(segment_mask_224 > 0)
                        if len(indices) > 0:
                            y_min, x_min = indices.min(axis=0)
                            y_max, x_max = indices.max(axis=0)
                            cx = (x_min + x_max) / 2.0
                            cy = (y_min + y_max) / 2.0
                        return segment_mask_224, cx, cy
                except Exception as e:
                    print(f"Error extracting SAM component mask: {e}")

            mask_224 = np.zeros((224, 224), dtype=np.float32)
            px = min(223, max(0, int(cx)))
            py = min(223, max(0, int(cy)))
            mask_224[py, px] = 1.0
            return mask_224, cx, cy

        p0_mask_224, c0_x, c0_y = extract_segment_mask_and_center(p0_anno, sam_mask_224)
        p1_mask_224, c1_x, c1_y = extract_segment_mask_and_center(p1_anno, sam_mask_224)
        start_x, start_y = vec["start"][0], vec["start"][1]
        end_x, end_y = vec["end"][0], vec["end"][1]

        # Vector-to-Patch Distance Matching (Start vs End Swapping)
        # Ensure p0 is strictly closest to vector start (moving source: Hand)
        d0_start = (c0_x - start_x) ** 2 + (c0_y - start_y) ** 2
        d1_start = (c1_x - start_x) ** 2 + (c1_y - start_y) ** 2
        if d1_start < d0_start:
            p0_anno, p1_anno = p1_anno, p0_anno
            p0_mask_224, p1_mask_224 = p1_mask_224, p0_mask_224
            c0_x, c0_y, c1_x, c1_y = c1_x, c1_y, c0_x, c0_y

        # Normalized movement vector
        dx = end_x - start_x
        dy = end_y - start_y
        dist = np.sqrt(dx * dx + dy * dy) + 1e-8
        dir_x, dir_y = dx / dist, dy / dist

        # Convert 224x224 SAM centroids to 14x14 grid patch indices
        grid_h, grid_w = 14, 14
        h_start = min(13, max(0, int((c0_y / 224.0) * grid_h)))
        w_start = min(13, max(0, int((c0_x / 224.0) * grid_w)))
        h_end = min(13, max(0, int((c1_y / 224.0) * grid_h)))
        w_end = min(13, max(0, int((c1_x / 224.0) * grid_w)))

        # --- A. VGGT Latent Transformation (Trajectory Flow Heatmap Activation) ---
        # Activate coordinates along the path from Hand to Cube on the full motion field (no subspace masking)
        vggt_tensor = obs_dict[view_name]["vggt"].squeeze(0)  # [224, 224]
        num_vggt_steps = max(abs(int(c1_y - c0_y)), abs(int(c1_x - c0_x))) + 1
        if num_vggt_steps > 1:
            # Generate line coordinates and safely clamp within spatial bounds
            vggt_y_indices = np.linspace(c0_y, c1_y, num_vggt_steps).astype(int)
            vggt_x_indices = np.linspace(c0_x, c1_x, num_vggt_steps).astype(int)

            # Use the actual peak magnitude of the baseline motion field so normalization keeps both visible
            vggt_max = vggt_tensor.max().item()

            # Build a single-channel path anchor mask directly on your device
            path_mask = torch.zeros_like(vggt_tensor)
            path_mask[vggt_y_indices, vggt_x_indices] = 1.0

            # Dilate the line by 3 pixels in all directions using max_pool2d (7x7 window)
            dilated_mask = torch.nn.functional.max_pool2d(
                path_mask.unsqueeze(0).unsqueeze(0), kernel_size=7, stride=1, padding=3
            ).squeeze()

            # Apply elementwise maximum to preserve bound scale without inflating baseline motion
            vggt_tensor = torch.maximum(vggt_tensor, dilated_mask * vggt_max)

        obs_dict[view_name]["vggt"] = vggt_tensor.unsqueeze(0)

        # --- B. DINOv3 Latent Transformation (Linear Latent Arm Bridges) ---
        # Interpolate hand features into intermediate background patch tokens on the full unmasked DINO grid
        dino_grid = obs_dict[view_name]["vision"].squeeze(0)[:196].view(14, 14)
        num_bridge_steps = max(abs(h_end - h_start), abs(w_end - w_start)) + 1
        if num_bridge_steps > 1:
            r_indices = np.linspace(h_start, h_end, num_bridge_steps).astype(int)
            c_indices = np.linspace(w_start, w_end, num_bridge_steps).astype(int)
            dino_max = dino_grid.max().item()
            bridge_val = dino_max * (1.0 / 0.7)
            for step_i in range(num_bridge_steps):
                r = min(max(r_indices[step_i], 0), 13)
                c = min(max(c_indices[step_i], 0), 13)
                dino_grid[r, c] = bridge_val

        # Re-flatten and save transformed DINO features
        dino_transformed = dino_grid.flatten()[:384]
        dino_transformed = pad_features(dino_transformed, 384)
        obs_dict[view_name]["vision"] = dino_transformed.unsqueeze(0)

        # --- C. CLIP Latent Transformation (Segment Transfer) ---
        # Copy Segment 1 (Hand) features to the target position on the 14x14 text feature grid
        clip_sim_grid = obs_dict[view_name]["text"].squeeze(0)[:196].view(14, 14)
        clip_sim_transformed = clip_sim_grid.clone()

        # Interpolate hand mask to 14x14 grid size
        p0_mask_14 = (
            torch.nn.functional.interpolate(
                torch.tensor(p0_mask_224, dtype=torch.float32, device=device).view(
                    1, 1, 224, 224
                ),
                size=(14, 14),
                mode="nearest",
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        h_indices, w_indices = np.where(p0_mask_14 > 0)
        h_offset = h_end - h_start
        w_offset = w_end - w_start

        for r, c in zip(h_indices, w_indices):
            target_r = min(13, max(0, r + h_offset))
            target_c = min(13, max(0, c + w_offset))
            clip_sim_transformed[target_r, target_c] = clip_sim_grid[r, c]
            clip_sim_grid[r, c] = 0

        clip_sim_transformed = pad_features(clip_sim_transformed.flatten()[:384], 384)
        obs_dict[view_name]["text"] = clip_sim_transformed.unsqueeze(0)

    # Re-package encoded multi-view tuple
    any_view = next(iter(obs_dict.values()))
    combined_obs = {
        "vision": torch.cat(
            [obs_dict[view]["vision"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "pointnext": torch.cat(
            [obs_dict[view]["pointnext"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "vggt": torch.cat(
            [obs_dict[view]["vggt"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "text": torch.cat(
            [obs_dict[view]["text"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "tactile": any_view["tactile"],
        "proprioception": any_view["proprioception"],
    }

    return obs_dict, combined_obs
