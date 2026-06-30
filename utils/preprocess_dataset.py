import os
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

    def extract_dino_tokens(self, image_paths, batch_size=16):
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
        """
        num_frames = len(image_paths)
        num_points = 100  # Subsample 100 points for computational efficiency
        batch_size = 16

        # Pre-compute depth maps and SAM masks in batches for massive GPU speedup
        depth_maps = []
        masks = []

        try:
            for i in range(0, num_frames, batch_size):
                batch_paths = image_paths[i : i + batch_size]
                batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
                w, h = batch_imgs[0].size

                # 1. Batched Depth Anything V2
                if self.depth_model is not None:
                    inputs_depth = self.depth_processor(
                        images=batch_imgs, return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        outputs_depth = self.depth_model(**inputs_depth)
                        predicted_depth = outputs_depth.predicted_depth
                        depth_batch = (
                            torch.nn.functional.interpolate(
                                predicted_depth.unsqueeze(1),
                                size=(h, w),
                                mode="bicubic",
                                align_corners=False,
                            )
                            .squeeze(1)
                            .cpu()
                            .numpy()
                        )
                        if depth_batch.ndim == 2:
                            depth_batch = np.expand_dims(depth_batch, axis=0)
                        depth_maps.extend(list(depth_batch))
                else:
                    depth_maps.extend(
                        [np.random.uniform(0.5, 3.0, size=(h, w)) for _ in batch_imgs]
                    )

                # 2. Batched SAM 2
                if self.sam is not None and click_coords is not None:
                    cx, cy = float(click_coords[0]), float(click_coords[1])
                    inputs_sam = self.sam_processor(
                        images=batch_imgs,
                        input_points=[[[[cx, cy]]]] * len(batch_imgs),
                        input_labels=[[[1]]] * len(batch_imgs),
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        outputs_sam = self.sam(**inputs_sam)
                        pred_masks = (
                            outputs_sam.pred_masks.squeeze(1)[:, 0].cpu().numpy()
                        )  # [B, 256, 256]
                        for logits in pred_masks:
                            logits_resized = cv2.resize(logits, (w, h))
                            masks.append(logits_resized > 0.0)
                else:
                    masks.extend([np.ones((h, w), dtype=bool) for _ in batch_imgs])
        except Exception as e:
            print(
                f"Warning: Batched depth/SAM preprocessing failed ({e}). Falling back to frame-by-frame heuristics."
            )
            depth_maps = []
            masks = []

        point_clouds = []
        for idx, path in enumerate(image_paths):
            img = Image.open(path).convert("RGB")
            w, h = img.size

            if idx < len(depth_maps):
                depth_map = depth_maps[idx]
                mask = masks[idx]
            else:
                depth_map = np.random.uniform(0.5, 3.0, size=(h, w))
                mask = np.ones((h, w), dtype=bool)

            # 3. Assume default pinhole camera intrinsics
            fx, fy = 150.0, 150.0
            cx, cy = w / 2.0, h / 2.0

            # 4. Filter pixels inside SAM mask
            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0:
                # Fallback to full image if mask is empty
                y_indices, x_indices = np.where(np.ones_like(mask))

            # Randomly subsample points from the segmented area
            sample_idx = np.random.choice(
                len(x_indices), size=min(num_points, len(x_indices)), replace=True
            )
            u_coords = x_indices[sample_idx]
            v_coords = y_indices[sample_idx]

            pc_points = []
            for u, v in zip(u_coords, v_coords):
                d = depth_map[v, u]
                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d
                intensity = 1.0  # Default intensity/feature channel
                pc_points.append([x, y, z, intensity])

            point_clouds.append(np.array(pc_points))

        return point_clouds

    def process_episode(self, raw_episode_dir, text_prompt, output_dir, episode_idx):
        """
        Processes a single episode folder containing image frames and joint files.
        """
        # Scan image paths in episode folder
        frame_dir = os.path.join(raw_episode_dir, "frames")
        if not os.path.exists(frame_dir) or len(os.listdir(frame_dir)) == 0:
            os.makedirs(frame_dir, exist_ok=True)
            # Create mock frame files if directory is empty to verify process runs
            for i in range(8):
                Image.new("RGB", (224, 224), color=(73, 109, 137)).save(
                    os.path.join(frame_dir, f"frame_{i:04d}.png")
                )

        image_paths = sorted(
            [
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.endswith(".png")
            ]
        )
        seq_len = len(image_paths)

        # Extract vision, text, and VGGT tokens
        vision_tokens = self.extract_dino_tokens(image_paths)
        text_token = self.extract_clip_tokens(text_prompt)
        vggt_tokens = self.extract_vggt_tokens(image_paths)

        # Read joint states/actions (mocked if missing)
        actions_path = os.path.join(raw_episode_dir, "actions.npy")
        states_path = os.path.join(raw_episode_dir, "states.npy")
        tactile_path = os.path.join(raw_episode_dir, "tactile.npy")
        pointclouds_path = os.path.join(raw_episode_dir, "point_clouds.npy")

        def load_and_clean_array(path):
            arr = np.load(path, allow_pickle=True)
            if arr.dtype == object:
                # Convert strings/objects to floats, defaulting to 0.0 for conversion errors
                def clean_val(x):
                    try:
                        return float(x)
                    except (ValueError, TypeError):
                        return 0.0

                arr = np.vectorize(clean_val)(arr)
            return torch.tensor(arr.astype(np.float32))

        if os.path.exists(actions_path):
            actions = load_and_clean_array(actions_path)
            proprio = load_and_clean_array(states_path)
            tactile = load_and_clean_array(tactile_path)
        else:
            actions = torch.randn(seq_len, 12)
            proprio = torch.randn(seq_len, 24)
            tactile = torch.randn(seq_len, 4, 4)

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
        print(f"Processed and cached: {output_path}")


def run_preprocessing(raw_data_dir, text_prompt, output_dir, disable_encoders=False):
    print("--- STARTING DATASET TOKENIZATION PREPROCESSING ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = DatasetPreprocessor(device=device, disable_encoders=disable_encoders)

    # Process mock/real folders
    if not os.path.exists(raw_data_dir):
        # Create a mock raw episode folder for execution verification
        mock_episode = os.path.join(raw_data_dir, "episode_0")
        os.makedirs(mock_episode, exist_ok=True)

    episodes = [os.path.join(raw_data_dir, d) for d in os.listdir(raw_data_dir)]
    for idx, ep_dir in enumerate(episodes):
        if os.path.isdir(ep_dir):
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
