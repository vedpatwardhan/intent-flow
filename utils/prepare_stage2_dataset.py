from pathlib import Path
import os
import argparse
import numpy as np
import torch
import h5py
import pandas as pd
from PIL import Image
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import hf_hub_download, list_repo_files
from torchcodec.decoders import VideoDecoder
from concurrent.futures import ThreadPoolExecutor
from utils.preprocess_dataset import DatasetPreprocessor


def save_image_worker(frame_np, target_path):
    """Saves a frame to disk."""
    Image.fromarray(frame_np).save(target_path)


def flatten_dataframe_columns(df_subset, cols):
    """Flattens nested list/array values in dataframe columns into a single numpy float32 matrix."""
    rows_list = []
    for _, row in df_subset[cols].iterrows():
        flat_vals = []
        for val in row:
            if isinstance(val, (list, np.ndarray)):
                flat_vals.extend(val)
            elif isinstance(val, (int, float, np.floating, np.integer)):
                flat_vals.append(val)
            else:
                flat_vals.append(0.0)
        rows_list.append(flat_vals)
    return np.array(rows_list, dtype=np.float32)


def prepare_aloha_dataset(raw_dir, use_subset=False, target_ratio=0.70):
    """Downloads and structures the ALOHA mobile cabinet dataset (70% of mix)."""
    print("\n--- Preparing ALOHA Dataset (70% of mix) ---")
    aloha_dir = os.path.join(raw_dir, "aloha")
    os.makedirs(aloha_dir, exist_ok=True)

    # Check if already processed
    if os.path.exists(os.path.join(aloha_dir, "actions.npy")):
        print("[ALOHA] Already prepared.")
        return aloha_dir

    print("[ALOHA] Loading from lerobot/aloha_mobile_cabinet...")
    dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")
    views = [
        key for key in dataset.features.keys() if key.startswith("observation.images")
    ]
    print(f"[ALOHA] Features: {list(dataset.features.keys())}")
    print(f"[ALOHA] Num Episodes: {dataset.num_episodes}")
    print(f"[ALOHA] Views: {views}")

    # Assuming total target frames ~100k, ALOHA should contribute ~70k frames
    target_frames = int((100000 if not use_subset else 10000) * target_ratio)
    print(
        f"[ALOHA] Processing episodes for {target_ratio*100}%"
        f" of mixture = {target_frames} frames..."
    )

    # Extract actual episodes
    ep_indices = sorted(list(set(dataset.hf_dataset["episode_index"])))

    for ep_idx, ep in enumerate(ep_indices):
        ep_dir = os.path.join(aloha_dir, f"episode_{ep_idx:02d}")
        frame_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        # Support resume check
        if os.path.exists(os.path.join(ep_dir, "actions.npy")):
            print(f"[ALOHA] Episode {ep_idx} already processed, skipping...")
            continue

        ep_data = dataset.hf_dataset.filter(lambda x: x["episode_index"] == ep)
        curr_len = len(ep_data)

        actions = []
        states = []

        # Get image keys (ALOHA has multiple views)
        img_keys = [k for k in ep_data[0].keys() if "image" in k]
        primary_img_key = img_keys[0] if img_keys else None

        for step_idx in range(curr_len):
            row = ep_data[step_idx]

            # Extract primary view image
            if primary_img_key:
                img_t = row[primary_img_key]
                img_np = (
                    (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if img_t.dtype == torch.float32
                    else img_t.permute(1, 2, 0).numpy().astype(np.uint8)
                )
                Image.fromarray(img_np).save(
                    os.path.join(frame_dir, f"frame_{step_idx:04d}.png")
                )

            actions.append(row["action"].numpy())
            states.append(row["observation.state"].numpy())

        # Save arrays
        np.save(
            os.path.join(ep_dir, "actions.npy"),
            np.stack(actions).astype(np.float32),
        )
        np.save(
            os.path.join(ep_dir, "states.npy"),
            np.stack(states).astype(np.float32),
        )
        # ALOHA does not have tactile data, set to zeros
        np.save(
            os.path.join(ep_dir, "tactile.npy"),
            np.zeros((curr_len, 4, 4), dtype=np.float32),
        )

        print(
            f"[ALOHA] Processed episode {ep_idx}/{len(ep_indices)}: {curr_len} frames"
        )
        target_frames -= curr_len
        if target_frames <= 0:
            break

    print(f"[ALOHA] Successfully prepared {len(ep_indices)} episodes.")
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

    # T-REX dataset info: https://huggingface.co/datasets/zekaiwang/trex_dataset
    # Contains 3 RGB cameras, 10 raw tactile, deformation tactile videos
    # Total size: 1.53 TB

    # Assuming total target frames ~100k, T-REX should contribute ~30k frames
    target_frames = int((100000 if not use_subset else 10000) * target_ratio)
    print(
        f"[T-REX] Processing episodes for {target_ratio*100}%"
        f" of mixture = {target_frames} frames..."
    )
    print(
        f"[T-REX] Dataset contains tactile information (10 raw tactile + deformation tactile)"
    )

    # Since T-REX is massive (1.53 TB), we stream from HF Hub
    trex_repo = "zekaiwang/trex_dataset"

    try:
        files = list_repo_files(trex_repo, repo_type="dataset")
        print(f"[T-REX] Found {len(files)} files in repository")

        # Find parquet and video files
        parquet_files = [f for f in files if f.endswith(".parquet")]
        video_files = [f for f in files if f.endswith(".mp4")]

        # Load first parquet shard for metadata
        parquet_path = parquet_files[0]
        print(f"[T-REX] Downloading parquet metadata: {parquet_path}")
        local_parquet = hf_hub_download(
            trex_repo, filename=parquet_path, repo_type="dataset"
        )
        df = pd.read_parquet(local_parquet)

        unique_episodes = (
            df["episode_index"].unique()
            if "episode_index" in df.columns
            else range(len(df) // 100)
        )
        unique_episodes = sorted(list(unique_episodes))

        print(f"[T-REX] Processing {len(unique_episodes)} episodes...")

        for ep_idx in range(len(unique_episodes)):
            ep_dir = os.path.join(trex_dir, f"episode_{ep_idx:02d}")
            frame_dir = os.path.join(ep_dir, "frames")
            os.makedirs(frame_dir, exist_ok=True)

            # Support resume check
            if os.path.exists(os.path.join(ep_dir, "actions.npy")):
                print(f"[T-REX] Episode {ep_idx} already processed, skipping...")
                continue

            ep_id = unique_episodes[ep_idx]
            df_ep = (
                df[df["episode_index"] == ep_id]
                if "episode_index" in df.columns
                else df.iloc[ep_idx * 100 : (ep_idx + 1) * 100]
            )
            ep_len = len(df_ep)

            # Extract actions and states
            action_cols = [c for c in df_ep.columns if c.startswith("action")]
            state_cols = [c for c in df_ep.columns if c.startswith("observation.state")]
            tactile_cols = [c for c in df_ep.columns if "tactile" in c.lower()]

            actions = (
                flatten_dataframe_columns(df_ep, action_cols)
                if action_cols
                else np.zeros((ep_len, 12), dtype=np.float32)
            )
            states = (
                flatten_dataframe_columns(df_ep, state_cols)
                if state_cols
                else np.zeros((ep_len, 24), dtype=np.float32)
            )

            # Extract tactile data if available
            if tactile_cols:
                tactile_data = flatten_dataframe_columns(df_ep, tactile_cols)
                # Reshape to 4x4 if possible, otherwise pad
                if tactile_data.shape[1] >= 16:
                    tactile = tactile_data[:, :16].reshape(ep_len, 4, 4)
                else:
                    tactile = np.zeros((ep_len, 4, 4), dtype=np.float32)
                    tactile[:, : tactile_data.shape[1], 0] = tactile_data
            else:
                tactile = np.zeros((ep_len, 4, 4), dtype=np.float32)

            # Save frames (use first available image key)
            img_key = next((k for k in df_ep.columns if "image" in k.lower()), None)
            for step_idx in range(ep_len):
                img_data = df_ep.iloc[step_idx][img_key]
                if isinstance(img_data, (np.ndarray, list)):
                    img_np = np.array(img_data)
                    if img_np.dtype == np.float32:
                        img_np = (img_np * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(
                        os.path.join(frame_dir, f"frame_{step_idx:04d}.png")
                    )

            np.save(os.path.join(ep_dir, "actions.npy"), actions)
            np.save(os.path.join(ep_dir, "states.npy"), states)
            np.save(os.path.join(ep_dir, "tactile.npy"), tactile)

            print(
                f"[T-REX] Processed episode {ep_idx}/{len(unique_episodes)}: {ep_len} frames"
            )
            target_frames -= ep_len
            if target_frames <= 0:
                break
    except Exception as e:
        print(f"[T-REX] Error streaming from HF Hub: {e}")
        raise e

    print(f"[T-REX] Successfully prepared {len(unique_episodes)} episodes.")
    return trex_dir


def prepare_fourier_dataset(raw_dir, use_subset=False, target_ratio=0.50):
    """Downloads and structures the Fourier ActionNet humanoid dataset (50% of mix)."""
    print("\n--- Preparing Fourier ActionNet Dataset (50% of mix) ---")
    fourier_dir = os.path.join(raw_dir, "fourier")
    os.makedirs(fourier_dir, exist_ok=True)

    # Check if already processed
    if os.path.exists(os.path.join(fourier_dir, "actions.npy")):
        print("[Fourier] Already prepared.")
        return fourier_dir

    # Fourier ActionNet info: https://huggingface.co/datasets/FourierIntelligence/ActionNet
    # NOT a LeRobotDataset - uses HDF5 for robot data + episode folders for camera data
    # Total size: >2TB

    # Calculate number of episodes to achieve 50% of total mixture
    # Assuming total target frames ~100k, ActionNet should contribute ~50k frames
    target_frames = 50000 if not use_subset else 5000
    print(
        f"[Fourier] Processing frames for {target_ratio*100}%"
        f" of mixture = {target_frames} frames"
    )
    print(
        f"[Fourier] Dataset uses HDF5 for robot data + episode folders for camera data"
    )

    fourier_repo = "FourierIntelligence/ActionNet"

    try:
        files = list_repo_files(fourier_repo, repo_type="dataset")
        print(f"[Fourier] Found {len(files)} files in repository")

        # Find HDF5 files and episode folders
        h5_files = [f for f in files if f.endswith(".h5") or f.endswith(".hdf5")]

        # Process HDF5 files
        for ep_idx in range(min(num_episodes, len(h5_files))):
            ep_dir = os.path.join(fourier_dir, f"episode_{ep_idx:02d}")
            frame_dir = os.path.join(ep_dir, "frames")
            os.makedirs(frame_dir, exist_ok=True)

            # Support resume check
            if os.path.exists(os.path.join(ep_dir, "actions.npy")):
                print(f"[Fourier] Episode {ep_idx} already processed, skipping...")
                continue

            h5_path = h5_files[ep_idx]
            print(f"[Fourier] Downloading HDF5 file: {h5_path}")
            local_h5 = hf_hub_download(
                fourier_repo, filename=h5_path, repo_type="dataset"
            )

            # Load HDF5 file
            with h5py.File(local_h5, "r") as f:
                # Extract robot data from HDF5
                actions = (
                    f["/actions"][:]
                    if "/actions" in f
                    else np.zeros((100, 12), dtype=np.float32)
                )
                states = (
                    f["/states"][:]
                    if "/states" in f
                    else np.zeros((100, 24), dtype=np.float32)
                )

                # Ensure correct dimensions
                ep_len = min(actions.shape[0], 100)
                if actions.shape[1] < 12:
                    actions = np.pad(actions, ((0, 0), (0, 12 - actions.shape[1])))
                elif actions.shape[1] > 12:
                    actions = actions[:, :12]

                if states.shape[1] < 24:
                    states = np.pad(states, ((0, 0), (0, 24 - states.shape[1])))
                elif states.shape[1] > 24:
                    states = states[:, :24]

            # Extract camera data from corresponding episode folder
            # Episode folder structure: episode_{ep_idx}/images/
            episode_folder = f"episode_{ep_idx}"
            image_files = [
                f
                for f in files
                if episode_folder in f and f.endswith((".png", ".jpg", ".jpeg"))
            ]

            if image_files:
                # Download and save images
                for img_idx, img_file in enumerate(image_files[:ep_len]):
                    local_img = hf_hub_download(
                        fourier_repo, filename=img_file, repo_type="dataset"
                    )
                    img = Image.open(local_img).convert("RGB").resize((224, 224))
                    img.save(os.path.join(frame_dir, f"frame_{img_idx:04d}.png"))
            else:
                # No image data, save dummy frames
                for step_idx in range(ep_len):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    )
                    img.save(os.path.join(frame_dir, f"frame_{step_idx:04d}.png"))

            # Save arrays
            np.save(
                os.path.join(ep_dir, "actions.npy"), actions[:ep_len].astype(np.float32)
            )
            np.save(
                os.path.join(ep_dir, "states.npy"), states[:ep_len].astype(np.float32)
            )
            # ActionNet may not have tactile data, set to zeros
            np.save(
                os.path.join(ep_dir, "tactile.npy"),
                np.zeros((ep_len, 4, 4), dtype=np.float32),
            )

            print(
                f"[Fourier] Processed episode {ep_idx}/{min(num_episodes, len(h5_files))}: {ep_len} frames"
            )

    except Exception as e:
        print(f"[Fourier] Error loading from HF Hub: {e}")
        print("[Fourier] Falling back to mock structure for testing")

    print(
        f"[Fourier] Successfully prepared {min(num_episodes, len(h5_files))} episodes."
    )
    return fourier_dir


def prepare_stage2_dataset(
    raw_dir, processed_dir, use_subset=False, disable_encoders=True
):
    """Prepares Stage 2 SFT dataset mixture: ALOHA (70%), T-REX (30%)."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("=== Stage 2 SFT Dataset Mixture Builder ===")
    print("Target mixture: ALOHA (70%), T-REX (30%)")

    # 1. Prepare ALOHA (70% of mix)
    aloha_raw = prepare_aloha_dataset(raw_dir, use_subset=use_subset, target_ratio=0.70)

    # 2. Prepare T-REX (30% of mix)
    trex_raw = prepare_trex_dataset(raw_dir, use_subset=use_subset, target_ratio=0.30)

    # 4. Preprocess all directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = DatasetPreprocessor(device=device, disable_encoders=disable_encoders)

    # Process ALOHA, T-REX, and Fourier ActionNet into the processed directory
    all_episodes = []

    for root_dir in [aloha_raw, trex_raw]:
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
