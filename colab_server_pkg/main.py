import os
import sys
import base64
import io
from time import perf_counter
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Pre-trained encoders loaders
from colab_server_pkg.config import app, device
from colab_server_pkg.models_state import models
from colab_server_pkg.image_utils import decode_base64_image
from colab_server_pkg.feature_extractor import (
    get_dino_attn_map,
    get_clip_cosine_similarity,
    get_point_cloud,
    get_filtered_point_cloud,
    get_vggt_point_tracks_base,
    get_vggt_2d_tracks_from_mask,
    get_segment_masks,
)
from colab_server_pkg.stage3_endpoints import (
    Stage3StepPayload,
    Stage3CalibratePayload,
    Stage3DistillPayload,
    handle_stage3_step,
    handle_stage3_calibrate,
    handle_stage3_distill,
)

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

from vggt.models.vggt import VGGT


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


@app.post("/process")
async def process_frame(payload: FramePayload):
    try:
        start = perf_counter()
        frame = decode_base64_image(payload.frame)
        h, w, _ = frame.shape
        pil_frame = Image.fromarray(frame)

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

        response["dino_attn"] = get_dino_attn_map(frame).tolist()
        print(f"DINO completed at {perf_counter() - start}")

        response["clip_sim"] = get_clip_cosine_similarity(
            payload.text_prompt, pil_frame
        ).tolist()
        print(f"CLIP completed at {perf_counter() - start}")

        point_cloud, depth_map, xs, ys, xs_proj, ys_proj, zs_proj, colors = (
            get_point_cloud(pil_frame, frame)
        )
        response["point_cloud"] = point_cloud.tolist()
        print(f"Point cloud completed at {perf_counter() - start}")

        pt_flat, seeds_flat, wp_frame2, base_mask, frames_np, seq_len, h, w = (
            get_vggt_point_tracks_base(payload.history_frames)
        )
        response["vggt_tracks"] = get_vggt_2d_tracks_from_mask(
            pt_flat, seeds_flat, wp_frame2, base_mask, seq_len, h, w
        )
        print(f"VGGT completed at {perf_counter() - start}")

        annotations = payload.ui_annotations
        if annotations and (annotations.get("crops") or annotations.get("segments")):
            print("Annotations Found.")
            combined_mask = np.zeros((14, 14), dtype=np.float32)
            combined_mask_224 = np.zeros((224, 224), dtype=np.float32)

            for crop in annotations.get("crops", []):
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

            sam_mask, sam_mask_224 = get_segment_masks(annotations, pil_frame)
            combined_mask = np.maximum(combined_mask, sam_mask)
            combined_mask_224 = np.maximum(combined_mask_224, sam_mask_224)

            response["task_isolated_features"]["sam_mask"] = sam_mask.tolist()
            response["task_isolated_features"]["sam_mask_224"] = sam_mask_224.tolist()
            response["task_isolated_features"][
                "combined_mask_224"
            ] = combined_mask_224.tolist()

            mx = np.clip(((seeds_flat[:, 0] / w) * 14).astype(np.int32), 0, 13)
            my = np.clip(((seeds_flat[:, 1] / h) * 14).astype(np.int32), 0, 13)
            workspace_mask = combined_mask[my, mx] > 0.0
            local_valid_mask = base_mask & workspace_mask
            response["task_isolated_features"]["vggt_local"] = (
                get_vggt_2d_tracks_from_mask(
                    pt_flat, seeds_flat, wp_frame2, local_valid_mask, seq_len, h, w
                )
            )

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


@app.post("/stage3/step")
async def stage3_step(payload: Stage3StepPayload):
    return await handle_stage3_step(payload)


@app.post("/stage3/calibrate")
async def stage3_calibrate(payload: Stage3CalibratePayload):
    return await handle_stage3_calibrate(payload)


@app.post("/stage3/distill")
async def stage3_distill(payload: Stage3DistillPayload):
    return await handle_stage3_distill(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
