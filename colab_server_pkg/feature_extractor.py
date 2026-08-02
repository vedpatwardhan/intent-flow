import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from colab_server_pkg.config import device
from colab_server_pkg.models_state import models
from colab_server_pkg.image_utils import decode_base64_image


def get_batch_dino_attn_maps(frames_list: list[np.ndarray]) -> torch.Tensor:
    """
    Batched DINO feature map extractor: Converts B frames to [B, 3, 224, 224] tensor,
    executes 1 batched forward pass through DINO, and returns normalized attention maps [B, 14, 14].
    """
    B = len(frames_list)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dino_tensors = torch.stack([transform(f) for f in frames_list], dim=0).to(device)

    with torch.no_grad():
        dino_feats = models["dino"].forward_features(
            dino_tensors
        )  # Shape [B, 197, 384]
        cls_tokens = dino_feats[:, 0]  # Shape [B, 384]
        cls_tokens = cls_tokens / (cls_tokens.norm(dim=-1, keepdim=True) + 1e-8)
        patches = dino_feats[:, -196:]  # Shape [B, 196, 384]
        patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-8)

        attn_batch = torch.bmm(patches, cls_tokens.unsqueeze(-1)).view(B, 14, 14)
        attn_mins = attn_batch.view(B, -1).min(dim=-1, keepdim=True)[0].view(B, 1, 1)
        attn_maxs = attn_batch.view(B, -1).max(dim=-1, keepdim=True)[0].view(B, 1, 1)
        attn_norm_batch = (attn_batch - attn_mins) / (attn_maxs - attn_mins + 1e-8)

        # Apply vectorized 70th percentile thresholding per batch item and scale by 0.8
        attn_flat = attn_norm_batch.view(B, -1).float()
        p70 = (
            torch.quantile(attn_flat, 0.70, dim=-1, keepdim=True)
            .unsqueeze(-1)
            .to(attn_norm_batch.dtype)
        )  # Shape [B, 1, 1]
        attn_norm_batch = (
            torch.where(
                attn_norm_batch >= p70,
                attn_norm_batch,
                torch.zeros_like(attn_norm_batch),
            )
            * 0.8
        )

    return attn_norm_batch


def get_batch_vggt_motion_fields(
    histories_list: list[list[np.ndarray]],
) -> torch.Tensor:
    """
    Batched VGGT motion vector field extractor: Converts B 4-frame history lists to [B, 4, 3, 224, 224] tensor,
    executes 1 batched forward pass through VGGT, and returns accumulated motion vector fields [B, 224, 224].
    """
    B = len(histories_list)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )
    vggt_tensors = torch.stack(
        [
            torch.stack([transform(f) for f in h_imgs], dim=0)
            for h_imgs in histories_list
        ],
        dim=0,
    ).to(device)

    with torch.no_grad():
        vggt_out = models["vggt"](vggt_tensors)
        world_points_batch = vggt_out["world_points"]  # Shape [B, 4, 224, 224, 3]

        motion_batch = torch.zeros(B, 224, 224, device=device)
        for t_idx in range(3):
            wp_curr = world_points_batch[:, t_idx]
            wp_next = world_points_batch[:, t_idx + 1]
            dx = wp_next[..., 0] - wp_curr[..., 0]
            dy = wp_next[..., 1] - wp_curr[..., 1]
            dz = wp_next[..., 2] - wp_curr[..., 2]
            motion_batch += torch.sqrt(dx**2 + dy**2 + dz**2)

        # Zero out sub-3mm micro noise floor (environment-agnostic physical threshold)
        motion_batch = torch.where(
            motion_batch < 0.003, torch.zeros_like(motion_batch), motion_batch
        )

        # Dynamic 95th Percentile Bounding per Payload (environment-agnostic)
        m_flat = motion_batch.view(B, -1)
        p95 = torch.quantile(m_flat.float(), 0.95, dim=-1, keepdim=True).to(device)
        # Avoid division by zero when entire motion is 0
        p95 = torch.where(p95 < 0.005, torch.tensor(0.005, device=device), p95).view(
            B, 1, 1
        )

        motion_norm_batch = torch.clamp(motion_batch / p95, max=1.0)

    return motion_norm_batch


def get_batch_clip_similarity(
    text_prompts: list[str], frames_list: list[np.ndarray]
) -> tuple:
    """
    Batched CLIP vision-text similarity extractor: Converts B frames to PIL images,
    processes text prompts and images in a single batched pass, and returns:
    - sim_norm_batch: Normalized vision-text similarity maps [B, 14, 14]
    - text_feat_batch: Project text feature vectors [B, 384]
    """
    B = len(frames_list)
    pil_frames = [Image.fromarray(f) for f in frames_list]

    inputs_text = models["clip_processor"](
        text=text_prompts, return_tensors="pt", padding=True
    ).to(device)
    inputs_vision = models["clip_processor"](images=pil_frames, return_tensors="pt").to(
        device
    )
    for k, v in inputs_text.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs_text[k] = v.to(torch.float16)
    for k, v in inputs_vision.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs_vision[k] = v.to(torch.float16)

    with torch.no_grad():
        text_feat_raw = models["clip"].get_text_features(
            **inputs_text
        )  # Shape [B, 512]
        vision_out = models["clip"].vision_model(**inputs_vision)
        norm_states = models["clip"].vision_model.post_layernorm(
            vision_out.last_hidden_state
        )  # Shape [B, 197, 768]
        patches = norm_states[:, 1:]  # Shape [B, 196, 768]
        patches_projected = models["clip"].visual_projection(
            patches
        )  # Shape [B, 196, 512]

        text_feat = torch.nn.functional.normalize(text_feat_raw, p=2, dim=-1)
        patches_projected = torch.nn.functional.normalize(
            patches_projected, p=2, dim=-1
        )
        # bmm: [B, 196, 512] x [B, 512, 1] -> [B, 196, 1] -> [B, 14, 14]
        sim_batch = torch.bmm(patches_projected, text_feat.unsqueeze(-1)).view(
            B, 14, 14
        )
        sim_maxs = sim_batch.view(B, -1).max(dim=-1, keepdim=True)[0].view(B, 1, 1)
        sim_mins = sim_batch.view(B, -1).min(dim=-1, keepdim=True)[0].view(B, 1, 1)
        sim_norm_batch = (sim_maxs - sim_batch) / (sim_maxs - sim_mins + 1e-8)

    return sim_norm_batch.cpu().numpy(), text_feat


def get_segment_masks(annotations: dict, pil_frame: Image) -> tuple:
    segments = (annotations or {}).get("segments", [])
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


def extract_batch_stage3_obs_features(payload_list: list):
    """
    Modular batched feature extractor that processes a list of B observation payloads in parallel.
    Executes single batched GPU forward passes for DINO, VGGT, CLIP, and SAM.
    """
    if len(payload_list) == 0:
        raise ValueError("Payload list for batched feature extraction cannot be empty.")

    B = len(payload_list)

    # 1. Decode RGB frames and history frames across all B payloads
    all_frames_dict = {cam: [] for cam in payload_list[0].frames.keys()}
    all_histories_dict = {cam: [] for cam in all_frames_dict.keys()}
    batch_obs_dicts = [{} for _ in range(B)]

    for payload in payload_list:
        for cam in all_frames_dict.keys():
            frame_str = payload.frames[cam]
            all_frames_dict[cam].append(decode_base64_image(frame_str))
            hist_strs = [f_dict[cam] for f_dict in payload.history_frames]
            hist_imgs = [decode_base64_image(s) for s in hist_strs[-4:]]
            all_histories_dict[cam].append(hist_imgs)

    for cam, frames_list in all_frames_dict.items():
        # Call modular batched helpers
        attn_norm_batch = get_batch_dino_attn_maps(frames_list)  # Shape [B, 14, 14]
        motion_norm_batch = get_batch_vggt_motion_fields(
            all_histories_dict[cam]
        )  # Shape [B, 224, 224]

        # Extract batched CLIP vision-text similarity
        text_prompts = [payload.text_prompt for payload in payload_list]
        clip_sim_batch, clip_text_feats = get_batch_clip_similarity(
            text_prompts, frames_list
        )

        attn_norm_np = attn_norm_batch.detach().cpu().numpy()
        motion_norm_np = motion_norm_batch.detach().cpu().numpy()

        # Assemble view obs dict for each batch item
        for b_idx in range(B):
            payload = payload_list[b_idx]
            pil_frame = Image.fromarray(frames_list[b_idx])
            dino_attn = attn_norm_np[b_idx]  # Shape [14, 14]
            motion_field = motion_norm_np[b_idx]  # Shape [224, 224]
            clip_sim = clip_sim_batch[b_idx]  # Shape [14, 14]
            text_feat = clip_text_feats[b_idx]

            # Compute combined crop and SAM segment masks
            combined_mask = np.zeros((14, 14), dtype=np.float32)
            combined_mask_224 = np.zeros((224, 224), dtype=np.float32)
            sam_mask = np.zeros((14, 14), dtype=np.float32)
            sam_mask_224 = np.zeros((224, 224), dtype=np.float32)
            dino_subspace = np.array([], dtype=np.float32)
            motion_field_subspace = np.array([], dtype=np.float32)

            ui_annotations = (
                payload.ui_annotations if hasattr(payload, "ui_annotations") else {}
            )
            view_annos = (ui_annotations or {}).get(cam, {})
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

            # Format raw features dictionary
            features_dict = {
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

            dino_flat = torch.tensor(
                dino_attn.flatten()[:384], dtype=torch.float32, device=device
            )
            vision_feat = pad_features(dino_flat, 384).unsqueeze(0)  # Shape [1, 384]

            vggt_feat = torch.tensor(
                motion_field, dtype=torch.float32, device=device
            ).unsqueeze(
                0
            )  # Shape [1, 224, 224]
            pt_feat = torch.zeros(1, 384, device=device)

            clip_flat = torch.tensor(
                clip_sim.flatten()[:384], dtype=torch.float32, device=device
            )
            clip_feat = pad_features(clip_flat, 384).unsqueeze(0)  # Shape [1, 384]

            tactile_feat = (
                torch.tensor(payload.tactile, device=device).flatten().unsqueeze(0)
                if hasattr(payload, "tactile")
                else torch.zeros(1, 16, device=device)
            )

            proprio = (
                payload.proprioception
                if hasattr(payload, "proprioception")
                else [0.0] * 58
            )
            proprio_tensor = torch.tensor(proprio, dtype=torch.float32, device=device)
            proprio_feat = pad_features(proprio_tensor, 58).unsqueeze(
                0
            )  # Shape [1, 58]

            batch_obs_dicts[b_idx][cam] = {
                "features": features_dict,
                "vision": vision_feat,
                "pointnext": pt_feat,
                "vggt": vggt_feat,
                "tactile": tactile_feat,
                "proprioception": proprio_feat,
                "text": clip_feat,
            }

    # Stack modalities across camera views into final combined_obs dictionary of shape [B, N_cam, ...]
    first_dict = batch_obs_dicts[0]
    any_view_name = next(iter(first_dict.keys()))

    combined_obs_batch = {
        "vision": torch.stack(
            [
                torch.cat([b_dict[cam]["vision"] for cam in b_dict], dim=0)
                for b_dict in batch_obs_dicts
            ],
            dim=0,
        ),  # Shape [B, N_cam, 384]
        "pointnext": torch.stack(
            [
                torch.cat([b_dict[cam]["pointnext"] for cam in b_dict], dim=0)
                for b_dict in batch_obs_dicts
            ],
            dim=0,
        ),  # Shape [B, N_cam, 384]
        "vggt": torch.stack(
            [
                torch.cat([b_dict[cam]["vggt"] for cam in b_dict], dim=0)
                for b_dict in batch_obs_dicts
            ],
            dim=0,
        ),  # Shape [B, N_cam, 224, 224]
        "text": torch.stack(
            [
                torch.cat([b_dict[cam]["text"] for cam in b_dict], dim=0)
                for b_dict in batch_obs_dicts
            ],
            dim=0,
        ),  # Shape [B, N_cam, 384]
        "tactile": torch.stack(
            [b_dict[any_view_name]["tactile"].squeeze(0) for b_dict in batch_obs_dicts],
            dim=0,
        ).unsqueeze(
            1
        ),  # Shape [B, 1, 16]
        "proprioception": torch.stack(
            [
                b_dict[any_view_name]["proprioception"].squeeze(0)
                for b_dict in batch_obs_dicts
            ],
            dim=0,
        ).unsqueeze(
            1
        ),  # Shape [B, 1, 58]
    }

    return batch_obs_dicts, combined_obs_batch


def extract_stage3_obs_features(payload):
    """
    Thin 1-line wrapper around extract_batch_stage3_obs_features for single payload processing.
    """
    obs_dict_batch, combined_obs_batch = extract_batch_stage3_obs_features([payload])
    single_combined_obs = {k: v.squeeze(0) for k, v in combined_obs_batch.items()}
    return obs_dict_batch[0], single_combined_obs


def extract_features_common(
    frame_str: str,
    history_frames: list[str],
    text_prompt: str,
    ui_annotations: dict[str, list],
    view_name: str = "world_center",
):
    """
    Restored single-frame feature extraction helper for the /process endpoint in main.py.
    Wraps parameters into a Stage3StepPayload and delegates to extract_batch_stage3_obs_features.
    """
    from colab_server_pkg.stage3_endpoints import Stage3StepPayload

    payload = Stage3StepPayload(
        frames={view_name: frame_str},
        history_frames=[{view_name: s} for s in history_frames],
        text_prompt=text_prompt,
        ui_annotations=ui_annotations or {},
        tactile=[[0.0] * 4 for _ in range(4)],
        proprioception=[0.0] * 58,
        pos_trajectories=[],
        episode_idx=0,
        step_idx=0,
        is_easy_task=True,
    )
    batch_obs_dicts, _ = extract_batch_stage3_obs_features([payload])
    return batch_obs_dicts[0][view_name]["features"]
