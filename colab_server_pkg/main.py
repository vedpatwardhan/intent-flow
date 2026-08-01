import os
import asyncio
import pickle
from time import perf_counter
import torch
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Pre-trained encoders loaders
from colab_server_pkg.config import app, device
from colab_server_pkg.models_state import models
from colab_server_pkg.feature_extractor import extract_features_common
from colab_server_pkg.stage3_endpoints import (
    Stage3StepPayload,
    Stage3CalibratePayload,
    Stage3DistillPayload,
    handle_stage3_step,
    handle_stage3_calibrate,
    handle_stage3_distill,
    get_calibration_job_status,
    get_distill_job_status,
)

try:
    from transformers import (
        CLIPProcessor,
        CLIPModel,
        Sam2Model,
        Sam2Processor,
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

EXEMPLAR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "exemplars")
)
os.makedirs(EXEMPLAR_DIR, exist_ok=True)


class FramePayload(BaseModel):
    frame: str  # Base64 encoded RGB frame
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    click_type: Optional[str] = None  # "original_click", "track_click", or "goal_click"
    text_prompt: Optional[str] = "right hand to the red cube"
    text_modifier: Optional[str] = None  # Hierarchical text modifier
    ui_annotations: Optional[dict] = (
        None  # {"crops": [], "vectors": [], "segments": []}
    )
    history_frames: Optional[List[str]] = []  # Previous base64 frames for VGGT tracking
    view_name: str = "world_center"


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

    print("Loading VGGT...")
    vggt_model = VGGT()
    model_file_path = hf_hub_download(repo_id="facebook/VGGT-1B", filename="model.pt")
    checkpoint_state = torch.load(model_file_path, map_location=device)
    vggt_model.load_state_dict(checkpoint_state)
    vggt_model.to(device)
    vggt_model.eval()
    models["vggt"] = vggt_model

    # Point cloud encoder is decommissioned. Setting to None to trigger downstream mock bypass.
    models["pointnext"] = None

    print("All available models initialized successfully.")


@app.post("/process")
async def process_frame(payload: FramePayload):
    try:
        start = perf_counter()

        history_frames = payload.history_frames
        if len(history_frames) == 2:
            history_frames = [*history_frames, *history_frames]
        if len(history_frames) < 4:
            raise IndexError("Need atleast 2 frames in the history to operate")
        features = extract_features_common(
            payload.frame,  # str
            history_frames,  # list[str]
            payload.text_prompt,  # str
            payload.ui_annotations,  # dict
            payload.view_name,  # str
        )

        response = {
            "dino_attn": features["dino_attn"].tolist(),
            "clip_sim": features["clip_sim"].tolist(),
            "motion_field": features["motion_field"].tolist(),
            "sam_mask": "",
            "task_isolated_features": {
                "dino_subspace": features["task_isolated_features"][
                    "dino_subspace"
                ].tolist(),
                "motion_field_subspace": features["task_isolated_features"][
                    "motion_field_subspace"
                ].tolist(),
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


@app.get("/stage3/calibrate/status/{job_id}")
async def stage3_calibrate_status(job_id: str):
    return await get_calibration_job_status(job_id)


@app.post("/stage3/distill")
async def stage3_distill(payload: Stage3DistillPayload):
    return await handle_stage3_distill(payload)


@app.get("/stage3/distill/status/{job_id}")
async def stage3_distill_status(job_id: str):
    return await get_distill_job_status(job_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
