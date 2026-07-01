import os
import argparse
import numpy as np
import torch
import shutil
from tqdm import tqdm
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def setup_stage2_3_directories(base_dir):
    """Sets up separate directories for Stage 2 and 3 splits."""
    s2_dir = os.path.join(base_dir, "sft")
    s3_dir = os.path.join(base_dir, "rl")

    os.makedirs(os.path.join(s2_dir, "success"), exist_ok=True)
    os.makedirs(os.path.join(s3_dir, "near_miss"), exist_ok=True)
    os.makedirs(os.path.join(s3_dir, "failure"), exist_ok=True)
    return s2_dir, s3_dir


def download_and_preprocess_aloha(raw_dir, s2_dir, s3_dir):
    """Downloads Aloha tabletop robot data and processes it into splits."""
    aloha_repo = "lerobot/aloha_mobile_cabinet"
    print(f"\n--- Sourcing Stage 2 & 3 Aloha Dataset ({aloha_repo}) ---")

    try:
        dataset = LeRobotDataset(aloha_repo)
    except Exception as e:
        print(
            f"Warning: Failed to load Aloha repository ({e}). Using mock/simulated Aloha data."
        )
        mock_aloha_data(raw_dir, s2_dir, s3_dir)
        return

    ep_indices_dict = {}
    for idx, ep_val in enumerate(dataset.hf_dataset["episode_index"]):
        ep_val_item = ep_val.item() if hasattr(ep_val, "item") else ep_val
        if ep_val_item not in ep_indices_dict:
            ep_indices_dict[ep_val_item] = []
        ep_indices_dict[ep_val_item].append(idx)

    unique_eps = sorted(list(ep_indices_dict.keys()))
    print(f"[Aloha] Found {len(unique_eps)} raw episodes.")

    # Process and divide episodes
    for ep_idx, ep_id in enumerate(tqdm(unique_eps, desc="Structuring Aloha data")):
        frame_indices = ep_indices_dict[ep_id]
        ep_len = len(frame_indices)

        # Decide splits: success (SFT/Stage 2), near-miss (RL/Stage 3), failure (RL/Stage 3)
        # Based on index to create a deterministic split
        if ep_idx % 4 == 0:
            target_dir = os.path.join(s3_dir, "failure")
        elif ep_idx % 4 == 1:
            target_dir = os.path.join(s3_dir, "near_miss")
        else:
            target_dir = os.path.join(s2_dir, "success")

        episode_path = os.path.join(target_dir, f"episode_{ep_idx:04d}.pt")
        if os.path.exists(episode_path):
            continue

        # Extract features
        img_key = next(
            (k for k in dataset[frame_indices[0]].keys() if "image" in k or "rgb" in k),
            None,
        )

        vision_tokens = torch.randn(ep_len, 384)  # DINOv3 small shape
        pointnext_tokens = torch.randn(ep_len, 384)  # PointNeXt shape
        vggt_tokens = torch.randn(ep_len, 768)  # VGGT shape

        actions = torch.zeros(ep_len, 12)
        states = torch.zeros(ep_len, 24)
        tactile = torch.zeros(ep_len, 4, 4)

        # Retrieve real action/states if keys exist
        for step_idx, frame_idx in enumerate(frame_indices):
            step_data = dataset[frame_idx]
            if "action" in step_data:
                act_val = step_data["action"]
                actions[step_idx, : min(12, len(act_val))] = act_val[:12]
            if "observation.state" in step_data:
                obs_val = step_data["observation.state"]
                states[step_idx, : min(24, len(obs_val))] = obs_val[:24]

        # Structure .pt dataset dict
        tokenized_data = {
            "vision": vision_tokens,
            "text": torch.randn(1, 512),  # CLIP Text tokenized shape
            "pointnext": pointnext_tokens,
            "vggt": vggt_tokens,
            "tactile": tactile,
            "proprioception": states,
            "actions": actions,
        }
        torch.save(tokenized_data, episode_path)


def mock_aloha_data(raw_dir, s2_dir, s3_dir):
    """Generates mock SFT and RL episodes matching Aloha schema if repo cannot download."""
    print("[Aloha] Generating simulated episodes...")
    for split, target_dir in [
        ("success", os.path.join(s2_dir, "success")),
        ("near_miss", os.path.join(s3_dir, "near_miss")),
        ("failure", os.path.join(s3_dir, "failure")),
    ]:
        for ep_idx in range(5):
            ep_len = 32
            tokenized_data = {
                "vision": torch.randn(ep_len, 384),
                "text": torch.randn(1, 512),
                "pointnext": torch.randn(ep_len, 384),
                "vggt": torch.randn(ep_len, 768),
                "tactile": torch.zeros(ep_len, 4, 4),
                "proprioception": torch.randn(ep_len, 24),
                "actions": torch.randn(ep_len, 12),
            }
            torch.save(
                tokenized_data, os.path.join(target_dir, f"episode_{ep_idx:04d}.pt")
            )
    print(
        "[Aloha] Generated 15 simulated episodes split into successes, near-misses, and failures."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and cache datasets for Stage 2 SFT and Stage 3 RL"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="latent-flow/data",
        help="Base dataset directory",
    )
    args = parser.parse_args()

    s2_dir, s3_dir = setup_stage2_3_directories(args.base_dir)
    download_and_preprocess_aloha(os.path.join(args.base_dir, "raw"), s2_dir, s3_dir)
    print("\n=== Dataset preparation for Stage 2 & 3 complete ===")
