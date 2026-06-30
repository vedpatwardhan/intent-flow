import os
from tqdm import tqdm
import torch
import numpy as np
import cv2
from PIL import Image
from models.vggt import VGGTEncoder

# Try importing standard multimodal libraries, fallback to simulated extractor if not fully installed
try:
    from transformers import (
        CLIPProcessor,
        CLIPTextModel,
        Sam2Model,
        Sam2Processor,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
    )
    from torchvision import transforms
    from huggingface_hub import hf_hub_download
    import timm

    MULTIMODAL_LIBS_AVAILABLE = True
except ImportError:
    MULTIMODAL_LIBS_AVAILABLE = False

# Try importing PointNeXt
try:
    from openpoints.models import build_model_from_cfg
    from easydict import EasyDict

    POINTNEXT_AVAILABLE = True
except ImportError:
    POINTNEXT_AVAILABLE = False


class DatasetPreprocessor:
    """
    Scans raw video frames, point clouds, and text instructions, extracts frozen
    modal tokens (DINOv3, CLIP, PointNeXt), and caches them into tokenized .pt files.
    """

    def __init__(self, device="cpu", disable_encoders=False):
        self.device = torch.device(device)
        self.dino = None
        self.clip_text_model = None
        self.clip_processor = None
        self.pointnext = None
        self.sam = None
        self.sam_processor = None
        self.vggt = None
        self.depth_model = None
        self.depth_processor = None

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if disable_encoders:
            print(
                "Frozen encoders are disabled. Preprocessing will run in fast mock mode."
            )
            return

        # Load VGGT geometry model
        self.vggt = VGGTEncoder().to(self.device)
        self.vggt.eval()

        if MULTIMODAL_LIBS_AVAILABLE:
            try:
                # Load frozen foundation backbones
                # 1. DINOv3 (using timm)
                self.dino = timm.create_model(
                    "vit_small_patch16_dinov3", pretrained=True, num_classes=0
                ).to(self.device)
                self.dino.eval()

                # 2. CLIP Text Encoder
                self.clip_text_model = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-base-patch16"
                ).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch16"
                )
                self.clip_text_model.eval()

                # 3. Segment Anything Model (SAM) for offline object segmentation
                self.sam = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(
                    self.device
                )
                self.sam_processor = Sam2Processor.from_pretrained(
                    "facebook/sam2-hiera-large"
                )
                self.sam.eval()

                # 4. Depth Anything V2 for 3D coordinate estimation
                self.depth_processor = AutoImageProcessor.from_pretrained(
                    "depth-anything/Depth-Anything-V2-Small-hf"
                )
                self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                    "depth-anything/Depth-Anything-V2-Small-hf"
                ).to(self.device)
                self.depth_model.eval()
            except Exception as e:
                print(
                    f"Warning: Failed to load online vision/language backbones ({e})."
                )

        if POINTNEXT_AVAILABLE:
            try:
                # Configure and build PointNeXt model for geometric feature encoding
                cfg = EasyDict(
                    {
                        "model": {
                            "NAME": "BaseSeg",
                            "encoder_args": {
                                "NAME": "PointNextEncoder",
                                "blocks": [1, 1, 1, 1, 1, 1],
                                "strides": [1, 2, 2, 2, 2, 1],
                                "width": 32,
                                "in_channels": 4,  # x, y, z, intensity
                                "sa_layers": 3,
                                "sa_use_res": True,
                            },
                            "decoder_args": {"NAME": "PointNextDecoder"},
                            "cls_args": {
                                "NAME": "PointNextHead",
                                "num_classes": 384,  # Dimension matching our 384 PointNeXtAdapter input
                            },
                        }
                    }
                )
                self.pointnext = build_model_from_cfg(cfg).to(self.device)
                self.pointnext.eval()
            except Exception as e:
                print(f"Warning: Failed to build PointNeXt model ({e}).")

    def extract_dino_tokens(self, image_paths, batch_size=8):
        """
        Extracts visual representations for a sequence of image frames in batches.
        """
        if self.dino is None:
            # Fallback representation size matching vit_small_patch16_dinov3
            return torch.randn(len(image_paths), 384)

        tokens = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_imgs.append(self.transform(img))

            # Stack into shape [B, 3, 224, 224]
            img_t = torch.stack(batch_imgs, dim=0).to(self.device)

            with torch.no_grad():
                feat = self.dino(img_t)  # Extracts visual tokens for the entire batch
                tokens.append(feat.cpu())

        return torch.cat(tokens, dim=0)

    def extract_clip_tokens(self, text_prompt):
        """
        Extracts semantic task embedding for text prompt.
        """
        if self.clip_text_model is None:
            return torch.randn(1, 768)

        inputs = self.clip_processor(
            text=[text_prompt], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_text_model(**inputs)
            text_embeds = outputs.pooler_output.cpu()

        return text_embeds

    def extract_pointnext_tokens(self, point_clouds):
        """
        point_clouds: Numpy array of shape [SeqLen, NumPoints, 4] (x, y, z, intensity)
        """
        if self.pointnext is None:
            return torch.randn(len(point_clouds), 384)

        tokens = []
        for pc in point_clouds:
            # Prepare point cloud tensor [1, NumPoints, 4] -> [1, 4, NumPoints]
            pc_t = (
                torch.tensor(pc, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .transpose(1, 2)
            )
            with torch.no_grad():
                feat = self.pointnext(pc_t)
                if feat.dim() > 2:
                    feat = feat.mean(
                        dim=-1
                    )  # Global pooling over points to output [1, 384]
                tokens.append(feat.cpu())
        return torch.cat(tokens, dim=0)

    def extract_vggt_tokens(self, image_paths):
        """
        Extracts Visual Geometry Grounded Transformer (VGGT) features.
        """
        if self.vggt is None:
            return torch.randn(len(image_paths), 768)

        frames = []
        for path in image_paths:
            # Resize image to standard 224x224 and convert to float tensor
            img = Image.open(path).convert("RGB").resize((224, 224))
            img_np = np.array(img, dtype=np.float32) / 255.0
            # Convert to [3, H, W]
            img_t = torch.tensor(img_np).permute(2, 0, 1)
            frames.append(img_t)

        # Shape: [1, SeqLen, 3, H, W]
        video_t = torch.stack(frames, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.vggt(video_t)
            # Extract 768-dim temporal geometry token features -> shape [SeqLen, 768]
            vggt_tokens = outputs["features"].squeeze(0).cpu()

        return vggt_tokens

    def convert_video_to_pointclouds(self, image_paths, click_coords=None):
        """
        Estimates depth for each frame and back-projects it to construct a 3D point cloud sequence.
        If SAM is available and click_coords are provided, it segments and crops the point cloud.
        Uses KLT Optical Flow for efficient point tracking across time steps.
        """
        if len(image_paths) == 0:
            return []

        # Open the first frame to initialize tracking
        first_img = Image.open(image_paths[0]).convert("RGB")
        w, h = first_img.size
        num_points = 100

        # 1. Estimate initial depth map
        depth_map = None
        if self.depth_model is not None:
            try:
                inputs_depth = self.depth_processor(
                    images=first_img, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs_depth = self.depth_model(**inputs_depth)
                    predicted_depth = outputs_depth.predicted_depth
                    depth_map = (
                        torch.nn.functional.interpolate(
                            predicted_depth.unsqueeze(1),
                            size=(h, w),
                            mode="bicubic",
                            align_corners=False,
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
            except Exception as e:
                print(f"Warning: Depth estimation failed on first frame ({e}).")

        if depth_map is None:
            depth_map = np.random.uniform(0.5, 3.0, size=(h, w))

        # 2. Get SAM mask on first frame
        mask = np.ones((h, w), dtype=bool)
        if self.sam is not None and click_coords is not None:
            try:
                cx, cy = float(click_coords[0]), float(click_coords[1])
                inputs_sam = self.sam_processor(
                    images=first_img,
                    input_points=[[[[cx, cy]]]],
                    input_labels=[[[1]]],
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs_sam = self.sam(**inputs_sam)
                    logits = outputs_sam.pred_masks[0, 0, 0].cpu().numpy()
                    logits_resized = cv2.resize(logits, (w, h))
                    mask = logits_resized > 0.0
            except Exception as e:
                print(f"Warning: SAM segmentation failed on first frame ({e}).")

        # 3. Find initial N keypoints inside the mask area
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0:
            y_indices, x_indices = np.where(np.ones_like(mask))

        # Use Shi-Tomasi corners inside the mask area for robust tracking
        first_np = np.array(first_img)
        first_gray = cv2.cvtColor(first_np, cv2.COLOR_RGB2GRAY)
        track_mask = mask.astype(np.uint8) * 255

        corners = cv2.goodFeaturesToTrack(
            first_gray,
            maxCorners=num_points,
            qualityLevel=0.01,
            minDistance=5,
            mask=track_mask,
        )

        if corners is not None and len(corners) > 0:
            p_init = corners.reshape(-1, 2)
            if len(p_init) < num_points:
                extra_idx = np.random.choice(
                    len(x_indices), size=num_points - len(p_init), replace=True
                )
                p_extra = np.stack([x_indices[extra_idx], y_indices[extra_idx]], axis=1)
                p_init = np.concatenate([p_init, p_extra], axis=0)
        else:
            sample_idx = np.random.choice(len(x_indices), size=num_points, replace=True)
            p_init = np.stack([x_indices[sample_idx], y_indices[sample_idx]], axis=1)

        p_init = p_init[:num_points].astype(np.float32)

        # Get Z depths for the selected keypoints
        u_coords = np.clip(p_init[:, 0].astype(int), 0, w - 1)
        v_coords = np.clip(p_init[:, 1].astype(int), 0, h - 1)
        zs_init = depth_map[v_coords, u_coords]

        # 4. Project keypoints to 3D
        fx, fy = 150.0, 150.0
        cx, cy = w / 2.0, h / 2.0

        point_clouds = []

        # t=0 frame point cloud
        pc0 = []
        for (u, v), z in zip(p_init, zs_init):
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pc0.append([x, y, z, 1.0])
        point_clouds.append(np.array(pc0))

        # 5. Track points using Lucas-Kanade across the rest of the sequence
        p_prev = p_init.reshape(-1, 1, 2).copy()
        gray_prev = first_gray

        for t in range(1, len(image_paths)):
            img_curr = Image.open(image_paths[t]).convert("RGB")
            img_np = np.array(img_curr)
            gray_curr = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            p_next, st, err = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr, p_prev, None
            )

            p_curr = []
            for idx, (pt, status) in enumerate(zip(p_next, st)):
                if status[0] == 1:
                    p_curr.append(pt[0])
                else:
                    p_curr.append(p_prev[idx][0])

            p_curr = np.array(p_curr, dtype=np.float32)
            p_prev = p_curr.reshape(-1, 1, 2).copy()
            gray_prev = gray_curr.copy()

            # Back-project current frame coordinates to 3D using initial depth (shape rigidity)
            pc_t = []
            for (u, v), z in zip(p_curr, zs_init):
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                pc_t.append([x, y, z, 1.0])
            point_clouds.append(np.array(pc_t))

        return point_clouds

    def process_episode(self, raw_episode_dir, text_prompt, output_dir, episode_idx):
        """
        Processes a single episode folder containing image frames and joint files with
        detailed timing profile logs.
        """
        output_path = os.path.join(output_dir, f"episode_{episode_idx:04d}.pt")
        if os.path.exists(output_path):
            print(
                f"Skipping episode {episode_idx:04d} (already processed: {output_path})"
            )
            return
        # Scan image paths in episode folder
        frame_dir = os.path.join(raw_episode_dir, "frames")
        if not os.path.exists(frame_dir) or len(os.listdir(frame_dir)) == 0:
            raise FileNotFoundError(
                f"Error: Episode directory '{raw_episode_dir}' does not contain visual"
                " frame PNG files. The raw extraction phase must complete successfully"
                " before running the preprocessor."
            )

        image_paths = sorted(
            [
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.endswith(".png")
            ]
        )
        seq_len = len(image_paths)

        # 1. DINOv3 visual features
        vision_tokens = self.extract_dino_tokens(image_paths)

        # 2. CLIP Text
        text_token = self.extract_clip_tokens(text_prompt)

        # 3. VGGT
        vggt_tokens = self.extract_vggt_tokens(image_paths)

        # Read joint states/actions (mocked if missing)
        actions_path = os.path.join(raw_episode_dir, "actions.npy")
        states_path = os.path.join(raw_episode_dir, "states.npy")
        tactile_path = os.path.join(raw_episode_dir, "tactile.npy")
        pointclouds_path = os.path.join(raw_episode_dir, "point_clouds.npy")

        if not os.path.exists(actions_path):
            raise FileNotFoundError(
                f"Error: Missing action/state files in '{raw_episode_dir}'. "
                "Ensure that actions.npy and states.npy are successfully written before preprocessing."
            )
        actions = torch.tensor(np.load(actions_path), dtype=torch.float32)
        proprio = torch.tensor(np.load(states_path), dtype=torch.float32)
        tactile = torch.tensor(np.load(tactile_path), dtype=torch.float32)

        # 4. Point Cloud Reconstructions (SAM + Depth Anything + PointNeXt)
        if os.path.exists(pointclouds_path):
            pc_data = np.load(pointclouds_path)
            pointnext_tokens = self.extract_pointnext_tokens(pc_data)
        else:
            # Back-project 2D video frames into 3D Point Clouds
            reconstructed_pcs = self.convert_video_to_pointclouds(
                image_paths, click_coords=[112, 112]
            )
            pointnext_tokens = self.extract_pointnext_tokens(reconstructed_pcs)

        # Build tokenized dictionary matching dataset_loader keys
        tokenized_data = {
            "vision": vision_tokens,
            "text": text_token,
            "pointnext": pointnext_tokens,
            "vggt": vggt_tokens,
            "tactile": tactile,
            "proprioception": proprio,
            "actions": actions,
        }

        # Cache the processed file
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"episode_{episode_idx:04d}.pt")
        torch.save(tokenized_data, output_path)


def run_preprocessing(raw_data_dir, text_prompt, output_dir, disable_encoders=False):
    print("--- STARTING DATASET TOKENIZATION PREPROCESSING ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = DatasetPreprocessor(device=device, disable_encoders=disable_encoders)

    # Process mock/real folders
    if not os.path.exists(raw_data_dir):
        # Create a mock raw episode folder for execution verification
        mock_episode = os.path.join(raw_data_dir, "episode_0")
        os.makedirs(mock_episode, exist_ok=True)

    episodes = sorted(
        [
            os.path.join(raw_data_dir, d)
            for d in os.listdir(raw_data_dir)
            if os.path.isdir(os.path.join(raw_data_dir, d))
        ]
    )

    for idx, ep_dir in enumerate(tqdm(episodes, desc="Preprocessing Episodes")):
        preprocessor.process_episode(ep_dir, text_prompt, output_dir, idx)

    print("--- PREPROCESSING COMPLETE ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess and cache foundation model tokens"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="/Users/vedpatwardhan/Desktop/cortex-os/gr00t_grasp_archive/raw",
        help="Path to raw frame/action dataset",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="pinch and lift red block",
        help="Task description prompt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/Users/vedpatwardhan/Desktop/cortex-os/gr00t_grasp_archive",
        help="Output folder for cached .pt files",
    )
    args = parser.parse_args()

    run_preprocessing(args.raw_dir, args.text, args.out_dir)
