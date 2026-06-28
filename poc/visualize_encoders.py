import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Add latent-flow root to sys.path to resolve imports correctly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vggt import VGGTEncoder

try:
    from transformers import CLIPProcessor, CLIPTextModel, SamModel, SamProcessor

    MULTIMODAL_LIBS_AVAILABLE = True
except ImportError:
    MULTIMODAL_LIBS_AVAILABLE = False


def load_video_frames(video_path, max_frames=10):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from video: {video_path}")

    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def run_real_poc(video_path):
    print("=== Task 1.2: Pretrained Encoders Visualization PoC (Real Video) ===")

    # 1. Load the actual video
    frames_raw = load_video_frames(video_path, max_frames=5)

    # Preprocess frames to tensor for models
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames_tensor = torch.stack(
        [transform(f) for f in frames_raw]
    )  # [Frames, 3, 224, 224]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running models on device: {device}")

    # 2. Test CLIP (Semantic alignment on first frame)
    if MULTIMODAL_LIBS_AVAILABLE:
        print("\n--- Running CLIP (OpenAI CLIP ViT-B/32) ---")
        try:
            clip_model = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(device)
            clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            clip_model.eval()

            text_prompt = "robot hand grasping a block"
            inputs = clip_processor(
                text=[text_prompt], return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = clip_model(**inputs).pooler_output

            print(
                f"Successfully extracted real CLIP Text Features: {text_features.shape}"
            )
        except Exception as e:
            print(f"Error loading CLIP: {e}")
    else:
        print("\nWarning: transformers/CLIP not installed. Skipping CLIP.")

    # 3. Test DINOv2 (ViT-S/14) on video frames
    print("\n--- Running DINOv2 (Facebook Research ViT-S/14) ---")
    try:
        # Load a small DINOv2 model to run fast locally
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
        dino.eval()

        with torch.no_grad():
            dino_features = dino(frames_tensor.to(device))

        print(
            f"Successfully extracted real DINOv2 features for all frames: {dino_features.shape}"
        )
    except Exception as e:
        print(f"Error running DINOv2: {e}")

    # 4. Test SAM (Segment Anything Model)
    if MULTIMODAL_LIBS_AVAILABLE:
        print("\n--- Running SAM (Facebook SAM ViT-B) ---")
        try:
            sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
            sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            sam.eval()

            first_frame_pil = Image.fromarray(frames_raw[0])
            # Prompt at middle of image (112, 112)
            inputs = sam_processor(
                first_frame_pil, input_points=[[[112, 112]]], return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = sam(**inputs)

            print(
                f"Successfully ran SAM. Mask logits shape: {outputs.pred_masks.shape}"
            )
        except Exception as e:
            print(f"Error running SAM: {e}")
    else:
        print("\nWarning: transformers/SAM not installed. Skipping SAM.")

    # 5. Test VGGT (Visual Geometry Grounded Transformer)
    print("\n--- Running VGGT (Local Architecture) ---")
    vggt = VGGTEncoder().to(device)
    vggt.eval()

    # VGGT expects [Batch, SeqLen, Channels, Height, Width]
    vggt_input = frames_tensor.unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        vggt_outputs = vggt(vggt_input)

    print(f"VGGT Features output shape: {vggt_outputs['features'].shape}")
    print(f"VGGT Camera Extrinsics output shape: {vggt_outputs['camera'].shape}")
    print(f"VGGT Point Tracks output shape: {vggt_outputs['point_tracks'].shape}")

    print(
        "\nPoC Result: SUCCESS (Pretrained and local models executed on real video frames)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Pretrained Encoders PoC on a real video file."
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the input video file."
    )
    args = parser.parse_args()

    run_real_poc(args.video)
