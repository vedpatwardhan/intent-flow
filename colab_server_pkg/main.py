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
    extract_features_common,
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
    models["clip"] = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch16", dtype=torch.float16
    ).to(device)
    models["clip_processor"] = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch16"
    )
    models["clip"].eval()

    print("Loading SAM 2...")
    models["sam"] = Sam2Model.from_pretrained(
        "facebook/sam2-hiera-large", dtype=torch.float16
    ).to(device)
    models["sam_processor"] = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
    models["sam"].eval()

    print("Loading Depth-Anything V2...")
    models["depth_processor"] = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    models["depth_model"] = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf", dtype=torch.float16
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

    print("Loading PointNeXt...")
    try:
        from openpoints.models import build_model_from_cfg
        from easydict import EasyDict

        cfg = EasyDict(
            {
                "model": {
                    "NAME": "BaseSeg",
                    "encoder_args": {
                        "NAME": "PointNextEncoder",
                        "blocks": [1, 1, 1, 1, 1, 1],
                        "strides": [1, 2, 2, 2, 2, 1],
                        "width": 32,
                        "in_channels": 3,  # x, y, z
                        "sa_layers": 3,
                        "sa_use_res": True,
                        "group_args": {
                            "NAME": "ballquery",
                            "radius": 0.15,
                            "nsample": 32,
                        },
                        "conv_args": {
                            "order": "conv-norm-act",
                        },
                        "norm_args": {
                            "norm": "bn",
                        },
                        "act_args": {
                            "act": "relu",
                        },
                    },
                    "conv_args": {
                        "order": "conv-norm-act",
                    },
                    "norm_args": {
                        "norm": "bn",
                    },
                    "act_args": {
                        "act": "relu",
                    },
                    "decoder_args": {"NAME": "PointNextDecoder"},
                    "cls_args": {
                        "NAME": "SegHead",
                        "num_classes": 384,
                    },
                }
            }
        )
        models["pointnext"] = build_model_from_cfg(cfg.model).to(device)
        models["pointnext"].eval()
    except Exception as e:
        print(f"Warning: Failed to load PointNeXt model ({e}). Using mock/fallback.")
        models["pointnext"] = None

    print("All available models initialized successfully.")


@app.post("/process")
async def process_frame(payload: FramePayload):
    try:
        start = perf_counter()
        features = extract_features_common(
            payload.frame,
            payload.history_frames,
            payload.text_prompt,
            payload.ui_annotations,
        )

        response = {
            "dino_attn": features["dino_attn"].tolist(),
            "clip_sim": features["clip_sim"].tolist(),
            "point_cloud": features["point_cloud"].tolist(),
            "vggt_tracks": features["vggt_tracks"],
            "sam_mask": "",
            "task_isolated_features": {
                "dino_subspace": features["task_isolated_features"][
                    "dino_subspace"
                ].tolist(),
                "vggt_local": features["task_isolated_features"]["vggt_local"],
                "point_cloud_local": np.array(
                    features["task_isolated_features"]["point_cloud_local"]
                ).tolist(),
                "sam_mask": np.array(
                    features["task_isolated_features"]["sam_mask"]
                ).tolist(),
                "sam_mask_224": np.array(
                    features["task_isolated_features"]["sam_mask_224"]
                ).tolist(),
                "combined_mask_224": np.array(
                    features["task_isolated_features"]["combined_mask_224"]
                ).tolist(),
                "tactile_active": [],
            },
        }

        print(f"Time taken {perf_counter() - start:.4f}s")
        return response

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}"
        )


@app.post("/stage3/step")
async def stage3_step(payload: Stage3StepPayload):
    # Called from the server.py on every step of every epoch
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
