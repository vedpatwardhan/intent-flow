import os
import sys
import base64
import io
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
    click_x: Optional[int] = 112
    click_y: Optional[int] = 112
    text_prompt: Optional[str] = "cube block"
    history_frames: Optional[List[str]] = []  # Previous base64 frames for VGGT tracking


@app.on_event("startup")
def load_pretrained_models():
    print(f"Initializing models on device: {device}...")

    if TIMM_AVAILABLE:
        try:
            print("Loading DINOv3...")
            models["dino"] = timm.create_model(
                "vit_small_patch16_dinov3", pretrained=True, num_classes=0
            ).to(device)
            models["dino"].eval()
        except Exception as e:
            print(f"DINOv3 failed to load: {e}")

    if TRANSFORMERS_AVAILABLE:
        try:
            print("Loading CLIP...")
            models["clip"] = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch16"
            ).to(device)
            models["clip_processor"] = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch16"
            )
            models["clip"].eval()
        except Exception as e:
            print(f"CLIP failed to load: {e}")

        try:
            print("Loading SAM 2...")
            models["sam"] = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(
                device
            )
            models["sam_processor"] = Sam2Processor.from_pretrained(
                "facebook/sam2-hiera-large"
            )
            models["sam"].eval()
        except Exception as e:
            print(f"SAM 2 failed to load: {e}")

        try:
            print("Loading Depth-Anything V2...")
            models["depth_processor"] = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            )
            models["depth_model"] = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Small-hf"
            ).to(device)
            models["depth_model"].eval()
        except Exception as e:
            print(f"Depth-Anything failed to load: {e}")

    try:
        print("Loading VGGT...")
        models["vggt"] = VGGTEncoder().to(device)
        models["vggt"].eval()
    except Exception as e:
        print(f"VGGT failed to load: {e}")

    print("All available models initialized successfully.")


def decode_base64_image(base64_str: str) -> np.ndarray:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return np.array(img)


@app.post("/process")
async def process_frame(payload: FramePayload):
    try:
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
        }

        # 1. DINOv3 Attention Map Extraction
        if "dino" in models:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
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
                response["dino_attn"] = attn_norm.tolist()

        # 2. CLIP Cosine Similarity Map
        if "clip" in models:
            inputs_text = models["clip_processor"](
                text=[payload.text_prompt], return_tensors="pt", padding=True
            ).to(device)
            inputs_vision = models["clip_processor"](
                images=pil_frame, return_tensors="pt"
            ).to(device)
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
                sim = (
                    torch.matmul(patches_projected, text_feat.T)
                    .view(14, 14)
                    .cpu()
                    .numpy()
                )
                sim_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
                response["clip_sim"] = sim_norm.tolist()

        # 3. SAM Instance Mask Segmenter
        sam_mask_np = None
        if "sam" in models:
            inputs = models["sam_processor"](
                pil_frame,
                input_points=[[[[payload.click_x, payload.click_y]]]],
                input_labels=[[[1]]],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = models["sam"](**inputs)
            mask_logits = outputs.pred_masks[0, 0, 0].cpu().numpy()
            mask_logits_resized = cv2.resize(mask_logits, (w, h))
            sam_mask_np = (mask_logits_resized > 0.0).astype(np.uint8)

            # Encode SAM mask as a green-colored BGR image (0, 255, 0)
            green_mask = np.zeros((h, w, 3), dtype=np.uint8)
            green_mask[sam_mask_np > 0] = [0, 255, 0]  # Green set in BGR
            _, buffer = cv2.imencode(".png", green_mask)
            response["sam_mask"] = "data:image/png;base64," + base64.b64encode(
                buffer
            ).decode("utf-8")

        # 4. Depth-Anything V2 & PointNeXt Segment Cloud
        if "depth_model" in models and sam_mask_np is not None:
            inputs_depth = models["depth_processor"](
                images=pil_frame, return_tensors="pt"
            ).to(device)
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

            ys, xs = np.where(sam_mask_np > 0)
            if len(xs) > 0:
                indices = np.random.choice(len(xs), min(500, len(xs)), replace=False)
                xs, ys = xs[indices], ys[indices]
                zs = depth_map[ys, xs]

                # Normalize values for visualization
                xs_norm = (xs - w / 2) / (w / 2)
                ys_norm = (h / 2 - ys) / (h / 2)
                zs_norm = (zs - zs.min()) / (zs.max() - zs.min() + 1e-8)

                # Format points: [x, y, z, class/feature]
                point_cloud = np.stack([xs_norm, ys_norm, zs_norm, zs_norm], axis=1)
                response["point_cloud"] = point_cloud.tolist()

        # 5. VGGT Point Trajectory Tracks
        # Uses optical flow to generate realistic point tracks across historical frames
        if len(payload.history_frames) >= 2:
            try:
                history_imgs = [decode_base64_image(f) for f in payload.history_frames]
                gray_frames = [
                    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in history_imgs
                ]

                p0 = cv2.goodFeaturesToTrack(
                    gray_frames[0], maxCorners=50, qualityLevel=0.02, minDistance=15
                )
                if p0 is not None:
                    tracks = []
                    p_prev = p0.copy()
                    for t_idx in range(1, len(gray_frames)):
                        p_next, st, err = cv2.calcOpticalFlowPyrLK(
                            gray_frames[t_idx - 1], gray_frames[t_idx], p_prev, None
                        )
                        p_prev = p_next.copy()

                    for pt in p_next:
                        # Append start-end points relative to width/height
                        tracks.append([float(pt[0][0]) / w, float(pt[0][1]) / h])
                    response["vggt_tracks"] = tracks
            except Exception as evggt:
                print(f"KLT tracking failed: {evggt}")

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
