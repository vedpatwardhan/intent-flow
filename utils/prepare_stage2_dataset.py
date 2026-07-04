import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from utils.preprocess_dataset import DatasetPreprocessor


def prepare_aloha_dataset(raw_dir, use_subset=False, target_ratio=0.70):
    """Downloads and structures the ALOHA mobile cabinet dataset (70% of mix)."""
    print("\n--- Preparing ALOHA Dataset (70% of mix) ---")
    aloha_dir = os.path.join(raw_dir, "aloha")
    os.makedirs(aloha_dir, exist_ok=True)

    # Check if already processed
    if os.path.exists(os.path.join(aloha_dir, "actions.npy")):
        print("[ALOHA] Already prepared.")
        return aloha_dir

    print("[ALOHA] Loading from lerobot/aloha_static_coffee_new...")
    dataset = LeRobotDataset("lerobot/aloha_static_coffee_new")
    views = [
        key for key in dataset.features.keys() if key.startswith("observation.images")
    ]
    print(f"[ALOHA] Features: {list(dataset.features.keys())}")
    print(f"[ALOHA] Num Episodes: {dataset.num_episodes}")
    print(f"[ALOHA] Views: {views}")
    print(f"[ALOHA] Meta: {dataset.meta}")

    # Assuming total target frames ~20k, ALOHA should contribute ~70k frames
    target_frames = int((20000 if not use_subset else 2000) * target_ratio)
    print(
        f"[ALOHA] Processing episodes for {target_ratio*100}%"
        f" of mixture = {target_frames} frames..."
    )

    # Extract actual episodes
    with tqdm(total=target_frames, desc="Processing ALOHA", unit="frame") as pbar:
        for ep_idx in range(dataset.num_episodes):
            episode_meta = dataset.meta.episodes[ep_idx]
            episode_idx = episode_meta["episode_index"]

            episode_dir = os.path.join(aloha_dir, f"episode_{episode_idx:02d}")
            frame_dir = os.path.join(episode_dir, "frames")
            os.makedirs(frame_dir, exist_ok=True)

            # Support resume check
            if os.path.exists(os.path.join(episode_dir, "actions.npy")):
                continue

            # Pre-create view directories outside the frame loop to avoid filesystem check overhead
            for view in views:
                os.makedirs(os.path.join(frame_dir, view), exist_ok=True)

            episode_data = Subset(
                dataset,
                range(
                    episode_meta["dataset_from_index"], episode_meta["dataset_to_index"]
                ),
            )
            curr_len = len(episode_data)

            actions = []
            states = []
            for frame_idx, frame in enumerate(episode_data):
                for view in views:
                    view_dir = os.path.join(frame_dir, view)
                    img_t = frame[view]
                    img_np = (
                        (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        if img_t.dtype == torch.float32
                        else img_t.permute(1, 2, 0).numpy().astype(np.uint8)
                    )
                    cv2.imwrite(
                        os.path.join(view_dir, f"frame_{frame_idx:04d}.png"),
                        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                    )

                actions.append(frame["action"].numpy())
                states.append(frame["observation.state"].numpy())

            # Save arrays
            np.save(
                os.path.join(episode_dir, "actions.npy"),
                np.stack(actions).astype(np.float32),
            )
            np.save(
                os.path.join(episode_dir, "states.npy"),
                np.stack(states).astype(np.float32),
            )
            # ALOHA does not have tactile data, set to zeros
            np.save(
                os.path.join(episode_dir, "tactile.npy"),
                np.zeros((curr_len, 4, 4), dtype=np.float32),
            )

            target_frames -= curr_len
            pbar.update(curr_len)
            if target_frames <= 0:
                break

    print(f"[ALOHA] Successfully prepared {ep_idx} episodes.")
    return aloha_dir


def prepare_trex_dataset(raw_dir, use_subset=False, target_ratio=0.30):
    """Prepares the T-REX tactile-rich dataset (30% of mix) with streaming."""
    print("\n--- Preparing T-REX Tactile Dataset (30% of mix) ---")
    trex_dir = os.path.join(raw_dir, "trex")
    os.makedirs(trex_dir, exist_ok=True)

    # Check if already processed
    if os.path.exists(os.path.join(trex_dir, "actions.npy")):
        print("[T-REX] Already prepared.")
        return trex_dir

    target_frames = int((20000 if not use_subset else 2000) * target_ratio)
    print(
        f"[T-REX] Processing episodes for {target_ratio*100}% of mixture = {target_frames} frames..."
    )

    # Use StreamingLeRobotDataset to avoid downloading terabyte-scale dataset
    dataset = StreamingLeRobotDataset(
        "zekaiwang/trex_dataset",
        streaming=True,
        buffer_size=1,
    )
    print(f"[T-REX] Features: {list(dataset.meta.features.keys())}")
    print(f"[T-REX] Streaming from: {dataset.repo_id}")
    print(f"[T-REX] Streaming enabled: {dataset.streaming}")

    def save_episode(ep_data, ep_idx):
        """Helper function to save episode data."""
        ep_dir = os.path.join(trex_dir, f"episode_{ep_idx:02d}")
        frame_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        if os.path.exists(os.path.join(ep_dir, "actions.npy")):
            return 0

        curr_len = len(ep_data)
        actions = []
        states = []
        tactile = []

        img_keys = [k for k in ep_data[0].keys() if "image" in k]
        tactile_keys = [k for k in ep_data[0].keys() if "tactile" in k.lower()]

        for step_idx, row in ep_data:
            if img_keys:
                img_t = row[img_keys[0]]
                img_np = (
                    (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if img_t.dtype == torch.float32
                    else img_t.permute(1, 2, 0).numpy().astype(np.uint8)
                )
                cv2.imwrite(
                    os.path.join(frame_dir, f"frame_{step_idx:04d}.png"),
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                )

            actions.append(row["action"].numpy())
            states.append(row["observation.state"].numpy())

            if tactile_keys:
                tactile_data = row[tactile_keys[0]].numpy()
                if tactile_data.shape[-1] == 16:
                    tactile.append(tactile_data.reshape(4, 4))
                elif tactile_data.size >= 16:
                    tactile.append(tactile_data.flatten()[:16].reshape(4, 4))
                else:
                    tactile.append(np.zeros((4, 4), dtype=np.float32))
            else:
                tactile.append(np.zeros((4, 4), dtype=np.float32))

        np.save(
            os.path.join(ep_dir, "actions.npy"), np.stack(actions).astype(np.float32)
        )
        np.save(os.path.join(ep_dir, "states.npy"), np.stack(states).astype(np.float32))
        np.save(
            os.path.join(ep_dir, "tactile.npy"), np.stack(tactile).astype(np.float32)
        )

        print(f"[T-REX] Processed episode {ep_idx}: {curr_len} frames")
        return curr_len

    # Iterate directly through streaming dataset without pre-fetching all indices
    ep_idx = 0
    current_ep_data = []
    current_ep_index = None

    with tqdm(total=target_frames, desc="Processing T-REX", unit="frame") as pbar:
        for item in dataset:
            if target_frames <= 0:
                break

            item_ep_index = item.get("episode_index", current_ep_index)
            print(
                f"Item: Episode {item.get('episode_index')}, "
                f"Frame {item.get('frame_index')}"
            )

            # Start new episode if episode index changes
            if current_ep_index is None or item_ep_index != current_ep_index:
                # Save previous episode if exists
                if current_ep_data and current_ep_index is not None:
                    curr_len = save_episode(current_ep_data, ep_idx)
                    target_frames -= curr_len
                    ep_idx += 1
                    pbar.update(curr_len)

                # Start new episode
                current_ep_index = item_ep_index
                current_ep_data = [item]

            else:
                # Add to current episode
                current_ep_data.append(item)

    print(f"[T-REX] Successfully prepared {ep_idx} episodes.")
    return trex_dir


def prepare_stage2_dataset(
    raw_dir, processed_dir, use_subset=False, disable_encoders=True
):
    """Prepares Stage 2 SFT dataset mixture: ALOHA (70%), T-REX (30%)."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("=== Stage 2 SFT Dataset Mixture Builder ===")
    print("Target mixture: ALOHA (70%), T-REX (30%)")

    # 1. Prepare ALOHA (70% of mix)
    # aloha_raw = prepare_aloha_dataset(raw_dir, use_subset=use_subset, target_ratio=0.70)

    # 2. Prepare T-REX (30% of mix)
    trex_raw = prepare_trex_dataset(raw_dir, use_subset=use_subset, target_ratio=0.30)

    # 4. Preprocess all directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = DatasetPreprocessor(device=device, disable_encoders=disable_encoders)

    # Process ALOHA, T-REX, and Fourier ActionNet into the processed directory
    all_episodes = []

    # for root_dir in [aloha_raw, trex_raw]:
    for root_dir in [trex_raw]:
        episodes = sorted(
            [
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        all_episodes.extend(
            [(ep_dir, os.path.basename(root_dir)) for ep_dir in episodes]
        )

    print(f"\n[Dataset Prep] Total episodes to process: {len(all_episodes)}")

    for idx, (ep_dir, dataset_name) in enumerate(
        tqdm(all_episodes, desc="Processing episodes")
    ):
        # Build unique output name to prevent collisions
        out_name = f"{dataset_name}_ep_{idx:03d}"
        preprocessor.process_episode(
            ep_dir,
            text_prompt="perform coordinated robotic contact manipulation",
            output_dir=processed_dir,
            episode_idx=idx,
        )
        # Rename episode file to include the dataset source to avoid overwriting
        old_file = os.path.join(processed_dir, f"episode_{idx:04d}.pt")
        new_file = os.path.join(processed_dir, f"{out_name}.pt")
        if os.path.exists(old_file):
            os.rename(old_file, new_file)

    print("\n[Dataset Prep] Stage 2 Data Preparation Complete.")
    print(f"[Dataset Prep] Processed {len(all_episodes)} episodes from 3 datasets.")
    print(f"[Dataset Prep] Mixture: ALOHA (70%), T-REX (30%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Stage 2 SFT Datasets")
    parser.add_argument(
        "--use_subset", action="store_true", help="Use a tiny subset of data"
    )
    parser.add_argument(
        "--enable_encoders", action="store_true", help="Run with active visual encoders"
    )
    args = parser.parse_args()

    raw_dir = "latent-flow/data/raw/stage2"
    processed_dir = "latent-flow/data/processed/sft/success"

    prepare_stage2_dataset(
        raw_dir,
        processed_dir,
        use_subset=args.use_subset,
        disable_encoders=not args.enable_encoders,
    )
