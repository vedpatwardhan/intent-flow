import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Hugging Face imports
try:
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        CLIPVisionModel,
        CLIPProcessor,
    )
except ImportError:
    print("Please install transformers: pip install transformers")
    raise


# Define a standard PointNet++ Set Abstraction (SA) Layer for 3D geometric encoding
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points: sample points feature data, [B, S, D']
        """
        # For visualization, we keep it simple and process per-point features
        B, N, C = xyz.shape
        # Simple PointNet style mlp per point
        if points is not None:
            x = torch.cat([xyz, points], dim=-1)
        else:
            x = xyz

        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, C, N, 1]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = torch.relu(bn(conv(x)))
        return xyz, x.squeeze(-1).permute(0, 2, 1)


class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple hierarchical structure mirroring PointNet++
        self.sa1 = PointNetSetAbstraction(
            npoint=256,
            radius=0.1,
            nsample=32,
            in_channel=3,
            mlp=[32, 64],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.2,
            nsample=32,
            in_channel=64 + 3,
            mlp=[64, 128],
            group_all=False,
        )

    def forward(self, xyz):
        # xyz: [B, N, 3]
        xyz, feat1 = self.sa1(xyz, None)
        _, feat2 = self.sa2(xyz, feat1)
        return feat2  # Returns hierarchical local features


def generate_synthetic_tabletop_points(num_points=1024, frame_idx=0):
    """
    Generates a synthetic 3D point cloud of a tabletop workspace with a moving robot gripper
    and a static red cube.
    """
    points = []
    # 1. Table surface (plane at z = -0.2)
    table_pts = np.random.uniform(
        [-0.5, -0.5, -0.2], [0.5, 0.5, -0.2], (int(num_points * 0.6), 3)
    )
    points.append(table_pts)

    # 2. Static Red Cube (at x=0.0, y=0.0, z=-0.15)
    cube_pts = np.random.uniform(
        [-0.05, -0.05, -0.2], [0.05, 0.05, -0.1], (int(num_points * 0.15), 3)
    )
    points.append(cube_pts)

    # 3. Moving gripper (approaching cube from top-left)
    t = frame_idx / 30.0  # simulate progression
    gripper_center = np.array([-0.2 + 0.2 * t, -0.2 + 0.2 * t, 0.2 - 0.3 * t])
    gripper_pts = np.random.uniform(
        gripper_center - 0.04, gripper_center + 0.04, (int(num_points * 0.25), 3)
    )
    points.append(gripper_pts)

    return np.vstack(points)


def project_points_to_2d(points, width=320, height=320):
    """
    Projects 3D points to 2D image coordinates using a simple virtual camera.
    """
    # Virtual camera parameters
    focal_length = 300
    cx, cy = width / 2, height / 2

    # Simple perspective projection
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Shift along z-axis to place scene in front of camera
    z_cam = z + 1.0

    u = (x * focal_length / z_cam + cx).astype(np.int32)
    v = (y * focal_length / z_cam + cy).astype(np.int32)

    # Keep only points within image boundaries
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[valid], v[valid], z[valid], valid


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

    print("Loading models from Hugging Face...")
    # 1. Load DINOv2
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino_model = (
        AutoModel.from_pretrained("facebook/dinov2-small").cuda()
        if torch.cuda.is_available()
        else AutoModel.from_pretrained("facebook/dinov2-small")
    )
    dino_model.eval()

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

    # 3. Instantiate PointNet++ Encoder
    pointnet = PointNet2Encoder()
    if torch.cuda.is_available():
        pointnet = pointnet.cuda()
    pointnet.eval()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx}...")

        # Convert CV2 frame (BGR) to PIL Image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # --- A. DINOv2 FEATURE VISUALIZATION (PCA-3) ---
        dino_inputs = dino_processor(images=pil_img, return_tensors="pt")
        if torch.cuda.is_available():
            dino_inputs = {k: v.cuda() for k, v in dino_inputs.items()}

        with torch.no_grad():
            dino_outputs = dino_model(**dino_inputs)
            # Patch tokens shape: [1, num_patches, hidden_dim]
            patch_tokens = (
                dino_outputs.last_hidden_state[:, 1:, :].cpu().numpy().squeeze(0)
            )

        # Fit PCA to compress patch features down to 3 components (RGB channels)
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(patch_tokens)

        # Normalize PCA features to [0, 255]
        pca_features = (pca_features - pca_features.min()) / (
            pca_features.max() - pca_features.min() + 1e-8
        )
        pca_features = (pca_features * 255).astype(np.uint8)

        # Reshape to grid (dino-small has 14x14 patches for 224x224 input)
        grid_size = int(np.sqrt(patch_tokens.shape[0]))
        dino_visual = pca_features.reshape(grid_size, grid_size, 3)
        dino_visual = cv2.resize(
            dino_visual, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST
        )

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
        resized_frame = cv2.resize(frame, (panel_size, panel_size))
        heatmap = cv2.applyColorMap(
            (attn_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        clip_visual = cv2.addWeighted(resized_frame, 0.6, heatmap, 0.4, 0)

        # --- C. POINTNET++ 3D GEOMETRIC ACTIVATIONS ---
        # Generate 3D point cloud for this step
        pts_3d = generate_synthetic_tabletop_points(
            num_points=1024, frame_idx=frame_idx
        )
        pts_tensor = torch.tensor(pts_3d, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            pts_tensor = pts_tensor.cuda()

        with torch.no_grad():
            # Get features from PointNet++
            pn_feat = pointnet(pts_tensor).cpu().numpy().squeeze(0)  # [N, 128]
            # Calculate activation magnitude (L2 norm of features) for each point
            activation_magnitude = np.linalg.norm(pn_feat, axis=-1)

        # Project 3D points to 2D screen coordinate
        u, v, z_depth, valid = project_points_to_2d(
            pts_3d, width=panel_size, height=panel_size
        )
        act_valid = activation_magnitude[valid]

        # Create PointNet++ visualization canvas
        pn_visual = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)

        # Normalize activations for color mapping
        if len(act_valid) > 0:
            act_min = act_valid.min()
            act_norm = (act_valid - act_min) / (act_valid.max() - act_min + 1e-8)
            colors = plt.cm.plasma(act_norm)[:, :3] * 255  # Plasma colormap
            colors = colors.astype(np.uint8)

            # Draw points as small circles
            for i in range(len(u)):
                cv2.circle(
                    pn_visual,
                    (u[i], v[i]),
                    3,
                    (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])),
                    -1,
                )

        # --- ASSEMBLY ---
        # Add labels to the panels
        cv2.putText(
            dino_visual,
            "DINOv2 (PCA-3)",
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
            "PointNet++ (Activations)",
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
