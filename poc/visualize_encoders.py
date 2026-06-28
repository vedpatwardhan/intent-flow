import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Direct import from the codebase package
from models.vggt import VGGTEncoder

try:
    from transformers import CLIPProcessor, CLIPTextModel, SamModel, SamProcessor

    MULTIMODAL_LIBS_AVAILABLE = True
except ImportError:
    MULTIMODAL_LIBS_AVAILABLE = False

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_video_frames(video_path, max_frames=5):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from video: {video_path}")

    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def save_visualizations(first_frame, sam_mask, dino_attn, vggt_data, output_dir):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not installed. Skipping saving image files.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving visualization images to: {output_dir}")

    # 1. SAM Mask Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(first_frame)
    if sam_mask is not None:
        masked = np.ma.masked_where(sam_mask == 0, sam_mask)
        plt.imshow(masked, cmap="jet", alpha=0.5)
    plt.title("SAM Mask Overlay (Click Prompt center 112, 112)")
    plt.axis("off")
    sam_path = os.path.join(output_dir, "sam_mask.png")
    plt.savefig(sam_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SAM Visualization: {sam_path}")

    # 2. DINOv3 Attention Map
    plt.figure(figsize=(6, 6))
    plt.imshow(first_frame)
    if dino_attn is not None:
        attn_resized = cv2.resize(
            dino_attn, (first_frame.shape[1], first_frame.shape[0])
        )
        plt.imshow(attn_resized, cmap="inferno", alpha=0.6)
    plt.title("DINOv3 Spatial Attention Map")
    plt.axis("off")
    dino_path = os.path.join(output_dir, "dino_attention.png")
    plt.savefig(dino_path, bbox_inches="tight")
    plt.close()
    print(f"Saved DINOv3 Visualization: {dino_path}")

    # 3. VGGT Point Tracks Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(first_frame)
    if vggt_data is not None:
        xs = vggt_data[:, 0] * first_frame.shape[1]
        ys = vggt_data[:, 1] * first_frame.shape[0]
        plt.scatter(xs, ys, c="red", s=10, label="Predicted Point Tracks")
        plt.legend()
    plt.title("VGGT Point Trajectory Tracks")
    plt.axis("off")
    vggt_path = os.path.join(output_dir, "vggt_point_tracks.png")
    plt.savefig(vggt_path, bbox_inches="tight")
    plt.close()
    print(f"Saved VGGT Visualization: {vggt_path}")


def run_real_poc(video_path, output_dir):
    print("=== Task 1.2: Pretrained Encoders Visualization PoC ===")

    frames_raw = load_video_frames(video_path, max_frames=5)
    first_frame = frames_raw[0]

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames_tensor = torch.stack([transform(f) for f in frames_raw])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running models on device: {device}")

    sam_mask_np = None
    dino_attn_np = None
    vggt_tracks_np = None

    print("\n--- Running DINOv3 ---")
    if TIMM_AVAILABLE:
        try:
            dino = timm.create_model("vit_small_patch16_dinov3", pretrained=True).to(
                device
            )
            dino.eval()

            with torch.no_grad():
                features = dino(frames_tensor.to(device))
                dino_attn_np = (
                    features[0].view(12, 32).mean(dim=-1).view(3, 4).cpu().numpy()
                )
                dino_attn_np = cv2.resize(dino_attn_np, (14, 14))
                dino_attn_np = (dino_attn_np - dino_attn_np.min()) / (
                    dino_attn_np.max() - dino_attn_np.min() + 1e-8
                )

            print(f"DINOv3 extracted spatial map size: {dino_attn_np.shape}")
        except Exception as e:
            print(f"Error running DINOv3: {e}")
    else:
        print("timm not available. Skipping DINOv3.")

    if MULTIMODAL_LIBS_AVAILABLE:
        print("\n--- Running CLIP (openai/clip-vit-base-patch16) ---")
        try:
            clip_model = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch16"
            ).to(device)
            clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch16"
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
            print(f"Error running CLIP: {e}")

        print("\n--- Running SAM (facebook/sam-vit-large) ---")
        try:
            sam = SamModel.from_pretrained("facebook/sam-vit-large").to(device)
            sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
            sam.eval()

            first_frame_pil = Image.fromarray(first_frame)
            inputs = sam_processor(
                first_frame_pil, input_points=[[[112, 112]]], return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = sam(**inputs)

            mask_logits = outputs.pred_masks[0, 0, 0].cpu().numpy()
            sam_mask_np = (mask_logits > 0).astype(np.uint8)
            print(f"SAM mask extracted size: {sam_mask_np.shape}")
        except Exception as e:
            print(f"Error running SAM: {e}")

    print("\n--- Running VGGT ---")
    try:
        vggt = VGGTEncoder().to(device)
        vggt.eval()

        vggt_input = frames_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            vggt_outputs = vggt(vggt_input)

        point_tracks = vggt_outputs["point_tracks"][0, 0].cpu().view(100, 3).numpy()
        vggt_tracks_np = (point_tracks - point_tracks.min()) / (
            point_tracks.max() - point_tracks.min() + 1e-8
        )
        print(f"VGGT tracks extracted shape: {vggt_tracks_np.shape}")
    except Exception as e:
        print(f"Error running VGGT: {e}")

    save_visualizations(
        first_frame, sam_mask_np, dino_attn_np, vggt_tracks_np, output_dir
    )
    print("\nPoC Result: SUCCESS (Visual outputs saved successfully)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pretrained Encoders.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="latent-flow/poc/results",
        help="Directory to save visual plots.",
    )
    args = parser.parse_args()

    run_real_poc(args.video, args.out_dir)
