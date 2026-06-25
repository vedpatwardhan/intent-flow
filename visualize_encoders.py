import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# timm and Hugging Face imports
try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from transformers import (
        CLIPVisionModel,
        CLIPProcessor,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
    )
except ImportError:
    print("Please install transformers and timm: pip install transformers timm")
    raise


# Define a PointNeXt Block (Residual MLP with Inverted Bottleneck, GroupNorm, and GELU)
class PointNeXtBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 1)
        self.norm1 = nn.GroupNorm(8, out_channel)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(out_channel, out_channel, 1)
        self.norm2 = nn.GroupNorm(8, out_channel)
        self.act2 = nn.GELU()

        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 1), nn.GroupNorm(8, out_channel)
            )
            if in_channel != out_channel
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        out = self.act2(out)
        return out


class PointNeXtStage(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=2):
        super().__init__()
        self.sa_block = nn.Sequential(
            PointNeXtBlock(in_channel, out_channel),
            *[PointNeXtBlock(out_channel, out_channel) for _ in range(blocks - 1)],
        )

    def forward(self, xyz, features=None):
        if features is not None:
            x = torch.cat([xyz.transpose(1, 2), features], dim=1)
        else:
            x = xyz.transpose(1, 2)
        out = self.sa_block(x)
        return out.transpose(1, 2)


class PointNeXtEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = PointNeXtStage(in_channel=3, out_channel=64, blocks=2)
        self.stage2 = PointNeXtStage(in_channel=64 + 3, out_channel=128, blocks=2)

    def forward(self, xyz):
        # xyz shape: [B, N, 3]
        feat1 = self.stage1(xyz)
        feat2 = self.stage2(xyz, feat1.transpose(1, 2))
        return feat2  # returns shape [B, N, 128]


def generate_point_cloud_from_depth(
    rgb_img, depth_model, depth_processor, device, num_points=1024
):
    """
    Predicts a dense depth map from an RGB image using Depth Anything V2
    and unprojects it to a 3D point cloud of (x, y, z) coordinates.
    """
    inputs = depth_processor(images=rgb_img, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb_img.size[::-1],  # PIL size is (W, H)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    h, w = depth.shape

    # Grid sampling to extract points
    stride = int(np.sqrt((h * w) / num_points))
    if stride < 1:
        stride = 1

    # Standard virtual intrinsics
    fx = fy = 400.0
    cx, cy = w / 2.0, h / 2.0

    points = []
    colors = []

    rgb_arr = np.array(rgb_img)

    for v in range(0, h, stride):
        for u in range(0, w, stride):
            d = depth[v, u]
            if d <= 0.05:
                continue
            # Depth Anything outputs relative/scaled depth; we scale it for a reasonable 3D aspect
            z = d * 0.1
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

            # Extract RGB colors
            r, g, b = rgb_arr[v, u]
            colors.append([r, g, b])

    points = np.array(points)
    colors = np.array(colors)

    # Cap size to exactly num_points for PointNeXt input matching
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    elif len(points) < num_points and len(points) > 0:
        idx = np.random.choice(len(points), num_points, replace=True)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def project_points_to_3d_isometric(points, colors, width=320, height=320):
    """
    Projects 3D points using a clean isometric perspective (top-down side view)
    so the 3D structure is immediately recognizable.
    """
    # Map camera coords to world-like coords: X=x, Y=-y (up is positive), Z=z
    pts_world = np.stack([points[:, 0], -points[:, 1], points[:, 2]], axis=-1)

    # Define an isometric rotation matrix (pitch and yaw rotation)
    pitch = np.radians(20)  # Tilt down slightly to see height/depth
    yaw = np.radians(-45)  # Angle from side

    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Standard 3D rotation matrices
    R_yaw = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    R_pitch = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])

    # Apply rotation
    pts_rot = pts_world @ R_yaw @ R_pitch

    # Scale and center on the screen
    mean_depth = points[:, 2].mean() + 1e-8
    scale = 220.0 / mean_depth
    u = (pts_rot[:, 0] * scale + width / 2).astype(np.int32)
    v = (-pts_rot[:, 1] * scale + height / 2).astype(np.int32)

    # Keep only points within bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[valid], v[valid], colors[valid], valid


def process_video(video_path, output_path="encoder_visuals.mp4"):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10

    # Output panel dimensions (3 side-by-side plots of 320x320)
    panel_size = 320
    output_width = panel_size * 3
    output_height = panel_size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    print("Loading models...")
    # 1. Load DINOv3 via timm
    dino_model = timm.create_model(
        "vit_small_patch16_dinov3", pretrained=True, global_pool=""
    )
    if torch.cuda.is_available():
        dino_model = dino_model.cuda()
    dino_model.eval()

    # Hook to capture self-attention matrix in the last attention layer
    dino_attn_weights = []

    def get_dino_attn_hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, module.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        dino_attn_weights.append(attn.detach().cpu())

    dino_hook = dino_model.blocks[-1].attn.register_forward_hook(get_dino_attn_hook)

    # Get preprocessing transform for DINOv3
    dino_config = resolve_data_config({}, model=dino_model)
    dino_transform = create_transform(**dino_config, is_training=False)

    # 2. Load CLIP
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = (
        CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32", attn_implementation="eager"
        ).cuda()
        if torch.cuda.is_available()
        else CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32", attn_implementation="eager"
        )
    )
    clip_model.eval()

    # 3. Instantiate PointNeXt Encoder
    pointnet = PointNeXtEncoder()
    if torch.cuda.is_available():
        pointnet = pointnet.cuda()
    pointnet.eval()

    # 4. Load Depth Anything V2 for zero-shot depth estimation
    print("Loading Depth Anything V2...")
    depth_processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = depth_model.to(device)
    depth_model.eval()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx}...")

        # Convert CV2 frame (BGR) to PIL Image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Pre-resize original frame for overlays
        resized_frame = cv2.resize(frame, (panel_size, panel_size))

        # --- A. DINOv3 FEATURE VISUALIZATION (Spatial Attention Heatmap) ---
        dino_inputs = dino_transform(pil_img).unsqueeze(0)
        if torch.cuda.is_available():
            dino_inputs = dino_inputs.cuda()

        dino_attn_weights.clear()
        with torch.no_grad():
            _ = dino_model(dino_inputs)

        # Retrieve attention weights: (B, num_heads, N, N)
        attn = dino_attn_weights[0][0]
        num_prefix = getattr(dino_model, "num_prefix_tokens", 5)
        # CLS token self-attention to patch tokens (excluding CLS and registers)
        cls_attn = attn[:, 0, num_prefix:]
        # Average across all heads
        mean_attn = cls_attn.mean(dim=0).numpy()

        # Reshape and upscale attention map to panel dimensions
        grid_size = int(np.sqrt(mean_attn.shape[0]))
        mean_attn_grid = mean_attn.reshape(grid_size, grid_size)
        mean_attn_resized = cv2.resize(
            mean_attn_grid, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC
        )
        mean_attn_smoothed = cv2.GaussianBlur(mean_attn_resized, (9, 9), 0)

        # Normalize attention maps
        mean_attn_norm = (mean_attn_smoothed - mean_attn_smoothed.min()) / (
            mean_attn_smoothed.max() - mean_attn_smoothed.min() + 1e-8
        )

        # Apply Inferno colormap for the spatial-semantic attention prior
        dino_heatmap = cv2.applyColorMap(
            (mean_attn_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        dino_visual = cv2.addWeighted(resized_frame, 0.5, dino_heatmap, 0.5, 0)

        # --- B. CLIP ATTENTION HEATMAP VISUALIZATION ---
        clip_inputs = clip_processor(images=pil_img, return_tensors="pt")
        if torch.cuda.is_available():
            clip_inputs = {k: v.cuda() for k, v in clip_inputs.items()}

        with torch.no_grad():
            # Run CLIP vision model and output attentions
            clip_outputs = clip_model(**clip_inputs, output_attentions=True)
            # Take attention from the last layer, CLS token (index 0) attention weights
            # Shape of last attention layer: [batch, heads, num_patches, num_patches]
            last_attn = clip_outputs.attentions[-1].cpu().numpy().squeeze(0)
            # Average across heads: [num_patches, num_patches]
            avg_attn = last_attn.mean(axis=0)
            # Get CLS token's attention to all other patch tokens (excluding self-attention at 0)
            cls_attn = avg_attn[0, 1:]

        # Normalize attention
        cls_attn = (cls_attn - cls_attn.min()) / (
            cls_attn.max() - cls_attn.min() + 1e-8
        )
        # Reshape and resize to create a smooth overlay (CLIP ViT-B/32 has 7x7 patches)
        attn_grid_size = int(np.sqrt(cls_attn.shape[0]))
        attn_map = cls_attn.reshape(attn_grid_size, attn_grid_size)
        attn_map_resized = cv2.resize(attn_map, (panel_size, panel_size))

        # Overlay heatmap on original resized frame
        heatmap = cv2.applyColorMap(
            (attn_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        clip_visual = cv2.addWeighted(resized_frame, 0.6, heatmap, 0.4, 0)

        # --- C. POINTNEXT 3D GEOMETRIC ACTIVATIONS ---
        # Generate 3D point cloud from current frame depth using Depth Anything
        pts_3d, point_colors = generate_point_cloud_from_depth(
            pil_img, depth_model, depth_processor, device, num_points=1024
        )
        pts_tensor = torch.tensor(pts_3d, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            pts_tensor = pts_tensor.cuda()

        with torch.no_grad():
            # Get features from PointNeXt
            pn_feat = pointnet(pts_tensor).cpu().numpy().squeeze(0)  # [N, 128]

        # Standardize PointNeXt features and map to RGB using PCA to visualize what the model sees
        pn_feat_std = (pn_feat - pn_feat.mean(axis=0)) / (pn_feat.std(axis=0) + 1e-8)
        pca = PCA(n_components=3)
        pca_colors = pca.fit_transform(pn_feat_std)
        # Normalize to [0, 255]
        c_min, c_max = pca_colors.min(axis=0), pca_colors.max(axis=0)
        pca_colors = (pca_colors - c_min) / (c_max - c_min + 1e-8)
        pca_colors = (pca_colors * 255).astype(np.uint8)

        # Project 3D points to isometric 2D coordinates with PointNeXt PCA colors
        u, v, valid_colors, valid = project_points_to_3d_isometric(
            pts_3d, pca_colors, width=panel_size, height=panel_size
        )

        # Create PointNeXt visualization canvas (3D space representation on black background)
        pn_visual = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)

        # Draw projected 3D points with their PointNeXt feature colors
        for i in range(len(u)):
            r, g, b = valid_colors[i]
            # Convert RGB to BGR for CV2
            color = (int(b), int(g), int(r))
            cv2.circle(pn_visual, (u[i], v[i]), 2, color, -1)

        # --- ASSEMBLY ---
        # Add labels to the panels
        cv2.putText(
            dino_visual,
            "DINOv3 (Attn Map Overlay)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            clip_visual,
            "CLIP (Attn Heatmap)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            pn_visual,
            "PointNeXt (3D Point Cloud)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Stack horizontally
        combined_frame = np.hstack([dino_visual, clip_visual, pn_visual])
        out.write(combined_frame)

        frame_idx += 1

    dino_hook.remove()
    cap.release()
    out.release()
    print(f"Verification video successfully saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Encoder Latent Video Visualizer"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/Users/vedpatwardhan/Desktop/cortex-os/le-probe/datasets/gr1_pickup_grasp_2k/videos/observation.images.world_center/chunk-000/file-000.mp4",
        help="Path to input mp4 observation video",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/Users/vedpatwardhan/Desktop/cortex-os/lewm-flow/encoder_visuals.mp4",
        help="Path to save output diagnostic video",
    )
    args = parser.parse_args()

    process_video(args.video_path, args.output_path)
