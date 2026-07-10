import os
import sys
import base64
import io
from time import perf_counter
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from torchvision import transforms
from vggt.models.vggt import VGGT

# Align paths to allow imports from repository root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from transformers import (
        pipeline,
        CLIPProcessor,
        CLIPModel,
        Sam2Model,
        Sam2Processor,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
    )
    from huggingface_hub import hf_hub_download

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


app = FastAPI(title="Latent-Flow Pretrained Encoder Server (Colab)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global models holders
models = {}


class FramePayload(BaseModel):
    frame: str  # Base64 encoded RGB frame
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    click_type: Optional[str] = None  # "original_click", "track_click", or "goal_click"
    text_prompt: Optional[str] = "cube block"
    text_modifier: Optional[str] = None  # Hierarchical text modifier
    ui_annotations: Optional[dict] = (
        None  # {"crops": [], "vectors": [], "segments": []}
    )
    history_frames: Optional[List[str]] = []  # Previous base64 frames for VGGT tracking


@app.on_event("startup")
def load_pretrained_models():
    print(f"Initializing models on device: {device}...")

    print("Loading DINOv3...")
    models["dino"] = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=True, num_classes=0
    ).to(device)
    models["dino"].eval()

    print("Loading CLIP...")
    models["clip"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(
        device
    )
    models["clip_processor"] = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch16"
    )
    models["clip"].eval()

    print("Loading SAM 2...")
    models["sam"] = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(device)
    models["sam_processor"] = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
    models["sam"].eval()
    models["sam_automatic_mask_generator"] = pipeline(
        task="mask-generation", model="facebook/sam2-hiera-large", device=device
    )

    print("Loading Depth-Anything V2...")
    models["depth_processor"] = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    models["depth_model"] = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device)
    models["depth_model"].eval()

    print("Loading VGGT...")
    vggt_model = VGGT()
    model_file_path = hf_hub_download(repo_id="facebook/VGGT-1B", filename="model.pt")
    checkpoint_state = torch.load(model_file_path, map_location=device)
    vggt_model.load_state_dict(checkpoint_state)
    vggt_model.to(device)
    vggt_model.eval()
    models["vggt"] = vggt_model

    print("All available models initialized successfully.")


def decode_base64_image(base64_str: str) -> np.ndarray:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return np.array(img)


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
    with torch.no_grad():
        text_feat = models["clip"].get_text_features(**inputs_text)
        vision_out = models["clip"].vision_model(**inputs_vision)
        norm_states = models["clip"].vision_model.post_layernorm(
            vision_out.last_hidden_state
        )
        patches = norm_states[0, 1:]
        patches_projected = models["clip"].visual_projection(patches)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        patches_projected = patches_projected / patches_projected.norm(
            dim=-1, keepdim=True
        )
        sim = torch.matmul(patches_projected, text_feat.T).view(14, 14).cpu().numpy()
        sim_norm = (sim.max() - sim) / (sim.max() - sim.min() + 1e-8)

    return sim_norm


def get_point_cloud(pil_frame: Image, frame: np.ndarray):
    h, w = pil_frame.height, pil_frame.width
    inputs_depth = models["depth_processor"](images=pil_frame, return_tensors="pt").to(
        device
    )
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

    # Scene-wide point cloud by sampling a grid (100 x 100 = 10000 points)
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, w - 1, 100).astype(int),
        np.linspace(0, h - 1, 100).astype(int),
    )
    xs = grid_x.flatten()
    ys = grid_y.flatten()

    # Extract raw predicted depth values directly (which is already in metric meters)
    zs = depth_map[ys, xs]

    # 3D pinhole projection perspective correction
    focal_length = max(w, h)
    cx = w / 2.0
    cy = h / 2.0

    # Normalize values for visualization
    xs_proj = (xs - cx) * zs / focal_length
    ys_proj = (cy - ys) * zs / focal_length
    zs_proj = zs

    # Convert colors to hex
    colors = frame[ys, xs]
    rs = colors[:, 0] / 255.0
    gs = colors[:, 1] / 255.0
    bs = colors[:, 2] / 255.0

    # Enforce aspect ratio preservation with global scaling normalization
    x_range = xs_proj.max() - xs_proj.min() if len(xs_proj) > 0 else 0
    y_range = ys_proj.max() - ys_proj.min() if len(ys_proj) > 0 else 0
    z_range = zs_proj.max() - zs_proj.min() if len(zs_proj) > 0 else 0
    max_range = max(x_range, y_range, z_range, 1e-8)

    xs_norm = (xs_proj - xs_proj.mean()) / max_range * 1.6
    ys_norm = (ys_proj - ys_proj.mean()) / max_range * 1.6
    zs_norm = (zs_proj - zs_proj.mean()) / max_range * 1.6

    # Format points: [x, y, z, r, g, b]
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

    # Use all points, but set colors to black for points outside mask
    pts_iso = np.stack([xs_proj, ys_proj, zs_proj], axis=1)

    max_range = max((pts_iso.max(axis=0) - pts_iso.min(axis=0)).max(), 1e-8)
    pts_norm = (pts_iso - pts_iso.mean(axis=0)) * (1.6 / max_range)
    xs_norm, ys_norm, zs_norm = pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2]
    colors_norm = colors / 255.0
    rs, gs, bs = colors_norm[:, 0], colors_norm[:, 1], colors_norm[:, 2]

    # Set colors to black for points outside mask
    rs[~mask_filter] = 0.0
    gs[~mask_filter] = 0.0
    bs[~mask_filter] = 0.0

    pointnext_isolated = np.stack([xs_norm, ys_norm, zs_norm, rs, gs, bs], axis=1)
    return pointnext_isolated


def get_vggt_point_tracks_base(history_frames: list[str]) -> tuple:
    """
    Runs the VGGT model on the historical sequence and returns intermediate outputs:
    (pt_flat, seeds_flat, wp_frame2, base_mask, frames_np, seq_len, h, w)
    """
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

    # Enforce standard spatial orientation: [H, W, T, 3]
    if wp_raw.shape[0] == seq_len:
        wp_clean_4d = np.transpose(wp_raw, (1, 2, 0, 3))
        wp_conf_clean = np.transpose(wp_conf, (1, 2, 0))
    else:
        wp_clean_4d = wp_raw
        wp_conf_clean = wp_conf

    subsample_step = 6
    wp_frame2 = wp_clean_4d[
        :, :, -1, :
    ]  # Support variable sequence length (last frame)

    # Generate uniform grid seeds matching temp_3.txt
    y_idx, x_idx = np.indices((h, w))
    seeds_grid = np.stack((x_idx, y_idx), axis=-1).astype(np.float32)

    # Subsample natively across the 4D matrix space
    pt_sampled = wp_clean_4d[::subsample_step, ::subsample_step, :, :]
    seeds_sampled = seeds_grid[::subsample_step, ::subsample_step, :]
    conf_sampled = wp_conf_clean[::subsample_step, ::subsample_step, 0]

    pt_flat = pt_sampled.reshape(-1, seq_len, 3)
    seeds_flat = seeds_sampled.reshape(-1, 2)
    conf_flat = conf_sampled.flatten()

    ix = seeds_flat[:, 0].astype(np.int32)
    iy = seeds_flat[:, 1].astype(np.int32)

    # Filter 1: Strip out empty black background regions
    rgb_values = frames_np[0][iy, ix]
    foreground_mask = np.any(rgb_values > 15, axis=1)

    # Filter 2: Ignore low-confidence depth regions
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
    """
    Computes local window feature matching from the filtered valid_mask
    and returns the 2D trajectory tracks.
    """
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
        p1_3d = pt_flat_filtered[idx, 0, :]  # Absolute 3D position in Frame 1

        # Define local search window boundaries inside Frame 2
        x_start = max(0, sx - search_radius)
        x_end = min(w, sx + search_radius + 1)
        y_start = max(0, sy - search_radius)
        y_end = min(h, sy + search_radius + 1)

        # Extract the local 3D neighborhood from Frame 2
        local_wp_f2 = wp_frame2[y_start:y_end, x_start:x_end, :]

        # Calculate 3D distances within this local window only
        local_dists = np.linalg.norm(local_wp_f2 - p1_3d, axis=2)
        best_local_idx = np.argmin(local_dists)

        # Map the match back to full canvas pixel coordinates
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


def get_segment_masks(annotations: dict, pil_frame: Image):
    # Process segments (click points for surfaces) using batched SAM 2 on the GPU
    segments = annotations.get("segments", [])
    click_pts = [[seg["x"], seg["y"]] for seg in segments]
    num_pts = len(click_pts)

    if len(segments) == 0:
        return np.zeros((14, 14), dtype=np.float32)

    # Prepare inputs
    inputs = models["sam_processor"](
        images=[pil_frame] * num_pts,
        input_points=[[[pt]] for pt in click_pts],
        input_labels=[[[1]] for _ in range(num_pts)],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = models["sam"](**inputs)

        # Downscale batched logits directly to 14x14 on the GPU
        pred_masks = outputs.pred_masks[:, 0, 0].unsqueeze(1)
        resized_masks = torch.nn.functional.interpolate(
            pred_masks, size=(14, 14), mode="bilinear", align_corners=False
        )

        # Get grid masks, reduce along the batch dimension
        sam_grid_masks = (
            (resized_masks.squeeze(1) > 0.0).cpu().numpy().astype(np.float32)
        )
        sam_combined_mask = np.maximum.reduce(sam_grid_masks, axis=0)

    return sam_combined_mask


@app.post("/process")
async def process_frame(payload: FramePayload):
    try:
        start = perf_counter()
        # Decode primary image frame
        frame = decode_base64_image(payload.frame)
        h, w, _ = frame.shape
        pil_frame = Image.fromarray(frame)

        # Output response dict
        response = {
            "dino_attn": [],
            "clip_sim": [],
            "sam_mask": "",
            "vggt_tracks": [],
            "point_cloud": [],
            "task_isolated_features": {
                "dino_subspace": [],
                "vggt_local": [],
                "pointnext_isolated": [],
                "tactile_active": [],
            },
        }

        # 1. DINOv3 Attention Map Extraction
        response["dino_attn"] = get_dino_attn_map(frame).tolist()
        print(f"DINO completed at {perf_counter() - start}")

        # 2. CLIP Cosine Similarity Map
        response["clip_sim"] = get_clip_cosine_similarity(
            payload.text_prompt, pil_frame
        ).tolist()
        print(f"CLIP completed at {perf_counter() - start}")

        # 3. Depth-Anything V2 & PointNeXt Segment Cloud
        point_cloud, depth_map, xs, ys, xs_proj, ys_proj, zs_proj, colors = (
            get_point_cloud(pil_frame, frame)
        )
        response["point_cloud"] = point_cloud.tolist()
        print(f"Point cloud completed at {perf_counter() - start}")

        # 4. VGGT Point Trajectory Tracks
        pt_flat, seeds_flat, wp_frame2, base_mask, frames_np, seq_len, h, w = (
            get_vggt_point_tracks_base(payload.history_frames)
        )
        response["vggt_tracks"] = get_vggt_2d_tracks_from_mask(
            pt_flat, seeds_flat, wp_frame2, base_mask, seq_len, h, w
        )
        print(f"VGGT completed at {perf_counter() - start}")

        # 5. Task-Isolated Feature Extraction (based on UI annotations and click points)
        annotations = payload.ui_annotations
        if annotations and (annotations.get("crops") or annotations.get("segments")):
            print("Annotations Found.")

            # Create binary mask from crops and segments for DINO/PointNeXt focusing
            combined_mask = np.zeros((14, 14), dtype=np.float32)

            # Process crops (patches)
            for crop in annotations.get("crops", []):
                # Map crop coordinates to 14x14 grid
                x_start = int((crop["x"] / 224) * 14)
                y_start = int((crop["y"] / 224) * 14)
                x_end = int(((crop["x"] + crop["width"]) / 224) * 14)
                y_end = int(((crop["y"] + crop["height"]) / 224) * 14)
                combined_mask[y_start:y_end, x_start:x_end] = 1.0

            sam_mask = get_segment_masks(annotations, pil_frame)
            combined_mask = np.maximum(combined_mask, sam_mask)
            response["task_isolated_features"]["sam_mask"] = sam_mask.tolist()

            # Process vectors (directions for explaining movement in isolated zone)
            mx = np.clip(((seeds_flat[:, 0] / w) * 14).astype(np.int32), 0, 13)
            my = np.clip(((seeds_flat[:, 1] / h) * 14).astype(np.int32), 0, 13)
            workspace_mask = combined_mask[my, mx] > 0.0
            local_valid_mask = base_mask & workspace_mask
            response["task_isolated_features"]["vggt_local"] = (
                get_vggt_2d_tracks_from_mask(
                    pt_flat, seeds_flat, wp_frame2, local_valid_mask, seq_len, h, w
                )
            )

            # DINO: Apply mask to get focused attention subspace
            dino_array = np.array(response["dino_attn"])
            masked_dino = dino_array * combined_mask
            response["task_isolated_features"]["dino_subspace"] = masked_dino.tolist()

            pointnext_isolated = get_filtered_point_cloud(
                xs, ys, combined_mask, xs_proj, ys_proj, zs_proj, colors
            )
            response["task_isolated_features"][
                "pointnext_isolated"
            ] = pointnext_isolated.tolist()

        print(f"Time taken {perf_counter() - start}")
        return response

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
