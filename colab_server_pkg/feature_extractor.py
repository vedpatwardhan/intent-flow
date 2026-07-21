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
    return motion_magnitude


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
        "text": features["text_feat"].unsqueeze(0),  # [1, 512]
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
        "text": any_view["text"],
        "tactile": any_view["tactile"],
        "proprioception": any_view["proprioception"],
    }


def construct_stage3_latent_goal_features(payload):
    """
    Constructs on-manifold target goal representation (s_target) by applying post-extraction
    latent feature transformations across DINOv3, CLIP, and VGGT feature maps.
    All input camera frames remain 100% clean, unblurred, and uncorrupted.
    """
    # 1. Extract clean base features from unblurred original camera frames
    obs_dict, encoded_tuple = extract_stage3_obs_features(payload)

    ui_annotations = getattr(payload, "ui_annotations", {}) or {}
    crops = ui_annotations.get("crops", [])
    vectors = ui_annotations.get("vectors", [])

    # If annotations are available, apply post-extraction feature transformations
    if len(crops) >= 2 and len(vectors) >= 1:
        vec = vectors[0]
        start_x, start_y = vec["start"][0], vec["start"][1]
        end_x, end_y = vec["end"][0], vec["end"][1]

        # Normalized movement vector
        dx = end_x - start_x
        dy = end_y - start_y
        dist = np.sqrt(dx * dx + dy * dy) + 1e-8
        dir_x, dir_y = dx / dist, dy / dist

        # Extract 14x14 patch coordinates for Hand (Segment 0) and Target (Segment 1)
        grid_h, grid_w = 14, 14
        h_start, h_end = int(start_y * grid_h), int(end_y * grid_h)
        w_start, w_end = int(start_x * grid_w), int(end_x * grid_w)

        for view_name in obs_dict:
            view_features = obs_dict[view_name]["features"]
            sam_mask_14 = view_features["task_isolated_features"]["sam_mask"]
            combined_mask_224 = view_features["task_isolated_features"][
                "combined_mask_224"
            ]

            # --- A. VGGT Latent Transformation (Forward Trajectory Field Overwrite) ---
            # Overwrite velocity vectors for hand patch tokens in VGGT's 224x224 feature matrix
            vggt_tensor = obs_dict[view_name]["vggt"]  # [1, 224, 224]
            mask_224_tensor = torch.tensor(
                combined_mask_224, dtype=torch.float32, device=device
            )
            trajectory_field_x = torch.full_like(vggt_tensor, dir_x) * mask_224_tensor
            trajectory_field_y = torch.full_like(vggt_tensor, dir_y) * mask_224_tensor
            vggt_transformed = vggt_tensor + 0.5 * (
                trajectory_field_x + trajectory_field_y
            )
            obs_dict[view_name]["vggt"] = vggt_transformed

            # --- B. DINOv3 Latent Transformation (Linear Latent Arm Bridges) ---
            # Interpolate/blend hand feature tokens into intermediate background patch tokens
            dino_subspace = view_features["task_isolated_features"]["dino_subspace"]
            dino_grid = dino_subspace.view(14, 14).clone()

            # Interpolate along line segment from (h_start, w_start) to (h_end, w_end)
            num_bridge_steps = max(abs(h_end - h_start), abs(w_end - w_start)) + 1
            if num_bridge_steps > 1:
                r_indices = np.linspace(h_start, h_end, num_bridge_steps).astype(int)
                c_indices = np.linspace(w_start, w_end, num_bridge_steps).astype(int)
                hand_val = dino_grid[min(h_start, 13), min(w_start, 13)].item()

                for step_i in range(num_bridge_steps):
                    r = min(max(r_indices[step_i], 0), 13)
                    c = min(max(c_indices[step_i], 0), 13)
                    alpha = (step_i + 1) / float(num_bridge_steps)
                    dino_grid[r, c] = (1.0 - alpha) * hand_val + alpha * dino_grid[r, c]

            # Re-flatten and save transformed DINO subspace features
            dino_transformed_subspace = dino_grid.flatten()[:384]
            dino_transformed_subspace = pad_features(dino_transformed_subspace, 384)
            view_features["task_isolated_features"][
                "dino_subspace_transformed"
            ] = dino_transformed_subspace
            obs_dict[view_name]["vision"] = dino_transformed_subspace.unsqueeze(0)

    # Re-package encoded multi-view tuple
    any_view = next(iter(obs_dict.values()))
    encoded_tuple_transformed = {
        "vision": torch.cat(
            [obs_dict[view]["vision"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "pointnext": torch.cat(
            [obs_dict[view]["pointnext"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "vggt": torch.cat(
            [obs_dict[view]["vggt"].unsqueeze(0) for view in obs_dict], dim=1
        ),
        "text": any_view["text"],
        "tactile": any_view["tactile"],
        "proprioception": any_view["proprioception"],
    }

    return obs_dict, encoded_tuple_transformed
