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

# Align paths to allow imports from repository root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vggt import VGGTEncoder

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
    models["vggt"] = VGGTEncoder().to(device)
    models["vggt"].eval()

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
    return point_cloud, xs, ys, xs_proj, ys_proj, zs_proj, colors


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

    xs_iso = xs_proj[mask_filter]
    ys_iso = ys_proj[mask_filter]
    zs_iso = zs_proj[mask_filter]
    pts_iso = np.stack([xs_iso, ys_iso, zs_iso], axis=1)

    max_range = max((pts_iso.max(axis=0) - pts_iso.min(axis=0)).max(), 1e-8)
    pts_norm = (pts_iso - pts_iso.mean(axis=0)) * (1.6 / max_range)
    xs_norm, ys_norm, zs_norm = pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2]
    colors_norm = colors[mask_filter] / 255.0
    rs, gs, bs = colors_norm[:, 0], colors_norm[:, 1], colors_norm[:, 2]

    pointnext_isolated = np.stack([xs_norm, ys_norm, zs_norm, rs, gs, bs], axis=1)
    return pointnext_isolated


def get_motion_tracks(history_frames: list[str]):
    history_imgs = [decode_base64_image(f) for f in history_frames]
    h, w = history_imgs[0].height, history_imgs[1].width
    gray_frames = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in history_imgs]
    tracks = []
    p0 = cv2.goodFeaturesToTrack(
        gray_frames[0], maxCorners=50, qualityLevel=0.02, minDistance=15
    )
    p_prev = p0.copy()
    for t_idx in range(1, len(gray_frames)):
        p_next, st, err = cv2.calcOpticalFlowPyrLK(
            gray_frames[t_idx - 1], gray_frames[t_idx], p_prev, None
        )
        p_prev = p_next.copy()

    for start_pt, end_pt in zip(p0, p_next):
        x_start = float(start_pt[0][0]) / w
        y_start = float(start_pt[0][1]) / h
        x_end = float(end_pt[0][0]) / w
        y_end = float(end_pt[0][1]) / h
        tracks.append([x_start, y_start, x_end, y_end])

    return tracks


def get_segment_masks(annotations: dict, pil_frame: Image):
    # Process segments (click points for surfaces) using batched SAM 2 on the GPU
    segments = annotations.get("segments", [])
    click_pts = [[seg["x"], seg["y"]] for seg in segments]
    num_pts = len(click_pts)

    # Single-image CPU preprocessing and GPU replicating optimization
    single_inputs = models["sam_processor"](
        images=pil_frame,
        input_points=[[[pt]] for pt in click_pts],
        input_labels=[[[1]] for _ in range(num_pts)],
        return_tensors="pt",
    )

    # Repeat them across all 5 points
    inputs = {
        "pixel_values": single_inputs["pixel_values"]
        .repeat(num_pts, 1, 1, 1)
        .to(device),
        "original_sizes": single_inputs["original_sizes"].repeat(num_pts, 1).to(device),
        "reshaped_input_sizes": single_inputs["reshaped_input_sizes"]
        .repeat(num_pts, 1)
        .to(device),
        "input_points": single_inputs["input_points"].to(device),
        "input_labels": single_inputs["input_labels"].to(device),
    }

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = models["sam"](**inputs)

        # Downscale batched logits directly to 14x14 on the GPU
        pred_masks = outputs.pred_masks[:, 0, 0].unsqueeze(1)
        resized_masks = torch.nn.functional.interpolate(
            pred_masks, size=(14, 14), mode="bilinear", align_corners=False
        )

        # Get grid masks and reduce to the combined mask
        sam_grid_masks = (
            (resized_masks.squeeze(1) > 0.0).cpu().numpy().astype(np.float32)
        )
    return sam_grid_masks


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
        point_cloud, xs, ys, xs_proj, ys_proj, zs_proj, colors = get_point_cloud(
            pil_frame, frame
        )
        response["point_cloud"] = point_cloud.tolist()
        print(f"Point cloud completed at {perf_counter() - start}")

        # 4. VGGT Point Trajectory Tracks
        # Uses optical flow to generate realistic point tracks across historical frames
        if len(payload.history_frames) >= 2:
            response["vggt_tracks"] = get_motion_tracks(payload.history_frames)
            print(f"VGGT completed at {perf_counter() - start}")

        # 5. Task-Isolated Feature Extraction (based on UI annotations and click points)
        if payload.ui_annotations:
            annotations = payload.ui_annotations

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

            # Get segment masks
            sam_grid_masks = get_segment_masks(annotations, pil_frame)
            combined_mask = np.maximum.reduce(sam_grid_masks, dtype=np.float32)
            response["task_isolated_features"]["sam_mask"] = sam_grid_masks.tolist()

            # Process vectors (directions for explaining movement)
            for vec in annotations.get("vectors", []):
                start_vec, end_vec = vec["start"], vec["end"]
                pass  # --> continue from here

            # DINO: Apply mask to get focused attention subspace
            dino_array = np.array(response["dino_attn"])
            masked_dino = dino_array * combined_mask
            response["task_isolated_features"]["dino_subspace"] = masked_dino.tolist()

            # PointNeXt: Filter and center 3D points directly using pre-computed grid variables and combined_mask
            pointnext_isolated = get_filtered_point_cloud(
                xs, ys, combined_mask, xs_proj, ys_proj, zs_proj, colors
            )
            response["task_isolated_features"][
                "pointnext_isolated"
            ] = pointnext_isolated.tolist()

            # VGGT: Return local tracks directly (independent of patches)
            if len(response["vggt_tracks"]) > 0:
                response["task_isolated_features"]["vggt_local"] = response[
                    "vggt_tracks"
                ][:10]

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
