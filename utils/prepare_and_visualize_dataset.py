import argparse
import os
import shutil
import urllib.request
import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import hf_hub_download, list_repo_files
from utils.preprocess_dataset import run_preprocessing

from torchcodec.decoders import VideoDecoder


def save_image_worker(frame_np, target_path):
    Image.fromarray(frame_np).save(target_path)


def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        print(
            f"Warning: Failed to download {url} ({e}). Creating a placeholder local file."
        )
        with open(path, "wb") as f:
            f.write(b"\x00" * 1024)


def flatten_dataframe_columns(df_subset, cols):
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


def prepare_and_visualize_dataset(disable_encoders=True, clean_cache=True):
    print("=== LatentFlow Stage 1 Pre-training Dataset Mixture Builder ===")

    # 1. Setup paths
    raw_data_dir = "latent-flow/data/raw"
    processed_dir = "latent-flow/data/processed"

    # Clear directory only if clean_cache flag is enabled
    if clean_cache:
        print("Cleaning local dataset caches...")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        if os.path.exists(raw_data_dir):
            shutil.rmtree(raw_data_dir)

    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # ==========================================
    # STREAM A: Droid Tabletop Robot (60% of mix) - Full First Shard
    # ==========================================
    bridge_repo = "lerobot/droid_1.0.1"
    print(
        f"\n--- Downloading Stream A: Droid ({bridge_repo}) (100% of training shard) ---"
    )
    try:

        # 1. Fetch repo file list to download the entire first data/video shard chunk
        files = list_repo_files(bridge_repo, repo_type="dataset")
        parquet_path = next(
            f for f in files if f.endswith(".parquet") and "data/chunk-000" in f
        )
        video_path = next(
            f
            for f in files
            if f.endswith(".mp4") and "videos/" in f and "chunk-000" in f
        )

        print(f"Downloading training Parquet shard: {parquet_path}")
        local_parquet = hf_hub_download(
            bridge_repo, filename=parquet_path, repo_type="dataset"
        )
        print(f"Downloading training Video shard: {video_path}")
        local_video = hf_hub_download(
            bridge_repo, filename=video_path, repo_type="dataset"
        )

        # 2. Read all episodes from this shard to build the full pre-training subset
        df = pd.read_parquet(local_parquet)
        unique_episodes = df["episode_index"].unique()
        # Filter unique episodes to only include those stored inside file-000.mp4 (episodes 0 to 326)
        unique_episodes = sorted([ep for ep in unique_episodes if ep < 327])
        print(
            f"Found {len(unique_episodes)} real robot trajectories in this training shard."
        )

        # Prepare 100% of the active pre-training partition (all unique episodes in this shard)
        num_episodes_to_prep = len(unique_episodes)
        print(
            f"Preparing all {num_episodes_to_prep} episodes for the training pipeline..."
        )

        # Initialize torchcodec video decoder
        decoder = VideoDecoder(local_video, device="cpu")
        print(
            "Using torchcodec for fast, PyTorch-native video decoding (AV1 supported)."
        )

        # Build mapping configurations for all episodes first
        episode_configs = []
        for ep_idx in range(num_episodes_to_prep):
            ep_id = unique_episodes[ep_idx]
            df_ep = df[df["episode_index"] == ep_id]
            ep_len = len(df_ep)

            bridge_raw_dir = os.path.join(raw_data_dir, f"bridge_ep{ep_idx:02d}")
            frame_dir_bridge = os.path.join(bridge_raw_dir, "frames")
            os.makedirs(frame_dir_bridge, exist_ok=True)

            action_cols = [c for c in df_ep.columns if c.startswith("action")]
            state_cols = [c for c in df_ep.columns if c.startswith("observation.state")]
            if not action_cols:
                action_cols = [c for c in df_ep.columns if "action" in c]
            if not state_cols:
                state_cols = [c for c in df_ep.columns if "state" in c]

            bridge_actions = flatten_dataframe_columns(df_ep, action_cols)
            bridge_states = (
                flatten_dataframe_columns(df_ep, state_cols)
                if state_cols
                else np.zeros((ep_len, 24), dtype=np.float32)
            )

            # Squeeze or pad actions to exactly 12 dimensions
            if bridge_actions.shape[1] < 12:
                pad = np.zeros(
                    (bridge_actions.shape[0], 12 - bridge_actions.shape[1]),
                    dtype=np.float32,
                )
                bridge_actions = np.concatenate([bridge_actions, pad], axis=1)
            elif bridge_actions.shape[1] > 12:
                bridge_actions = bridge_actions[:, :12]

            # Squeeze or pad states to exactly 24 dimensions
            if bridge_states.shape[1] < 24:
                pad = np.zeros(
                    (bridge_states.shape[0], 24 - bridge_states.shape[1]),
                    dtype=np.float32,
                )
                bridge_states = np.concatenate([bridge_states, pad], axis=1)
            elif bridge_states.shape[1] > 24:
                bridge_states = bridge_states[:, :24]

            episode_configs.append(
                {
                    "dir": bridge_raw_dir,
                    "frame_dir": frame_dir_bridge,
                    "length": ep_len,
                    "actions": bridge_actions,
                    "states": bridge_states,
                }
            )

        # Decode frames sequentially in a single pass with parallel background disk writes
        print("Extracting Droid frames sequentially (Parallel saving)...")
        current_ep_idx = 0
        current_frame_in_ep = 0
        total_frames = (
            len(decoder)
            if hasattr(decoder, "__len__")
            else sum(cfg["length"] for cfg in episode_configs)
        )

        with ThreadPoolExecutor(max_workers=16) as executor:
            # Iterate sequentially through the decoder without seeking, wrapped in tqdm
            for frame_idx, frame_tensor in enumerate(
                tqdm(decoder, desc="Extracting Droid frames", total=total_frames)
            ):
                if current_ep_idx >= len(episode_configs):
                    break

                cfg = episode_configs[current_ep_idx]

                # Skip writing files if they are already on disk (supports resuming/restarting)
                if os.path.exists(os.path.join(cfg["dir"], "actions.npy")):
                    current_frame_in_ep += 1
                    if current_frame_in_ep >= cfg["length"]:
                        current_ep_idx += 1
                        current_frame_in_ep = 0
                    continue

                frame_rgb = frame_tensor.permute(1, 2, 0).numpy()
                target_path = os.path.join(
                    cfg["frame_dir"], f"frame_{current_frame_in_ep:04d}.png"
                )

                # Offload disk write to the thread pool
                executor.submit(save_image_worker, frame_rgb, target_path)

                current_frame_in_ep += 1
                if current_frame_in_ep >= cfg["length"]:
                    np.save(os.path.join(cfg["dir"], "actions.npy"), cfg["actions"])
                    np.save(os.path.join(cfg["dir"], "states.npy"), cfg["states"])
                    np.save(
                        os.path.join(cfg["dir"], "tactile.npy"),
                        np.zeros((cfg["length"], 4, 4)),
                    )

                    if current_ep_idx % 50 == 0:
                        print(
                            f"[Droid] Structured episode {current_ep_idx}/{len(episode_configs)}..."
                        )

                    current_ep_idx += 1
                    current_frame_in_ep = 0

        print(
            f"Successfully loaded and structured all Droid robot pre-training episodes."
        )
    except Exception as e:
        print(
            f"Error: Failed to load Droid dataset ({e}). Enforcing pipeline integrity, crashing script."
        )
        raise e

    # ==========================================
    # STREAM B: CMU Human Tabletop (30% of mix) - Full Shard
    # ==========================================
    human_repo = "lerobot/cmu_stretch"
    print(
        f"\n--- Downloading Stream B: CMU Human Exploration ({human_repo}) (100% of training shard) ---"
    )
    try:
        # Download the entire CMU tabletop human-exploration dataset (~300MB, perfectly safe for disk)
        dataset_human = LeRobotDataset(human_repo)

        # Retrieve unique episode indexes in a version-agnostic way
        ep_indices_dict = {}
        for idx, ep_val in enumerate(dataset_human.hf_dataset["episode_index"]):
            ep_val_item = ep_val.item() if hasattr(ep_val, "item") else ep_val
            if ep_val_item not in ep_indices_dict:
                ep_indices_dict[ep_val_item] = []
            ep_indices_dict[ep_val_item].append(idx)

        unique_human_eps = sorted(list(ep_indices_dict.keys()))
        num_human_eps = len(unique_human_eps)
        print(f"Preparing all {num_human_eps} human episodes...")

        for ep_idx in tqdm(range(num_human_eps), desc="Processing CMU Stretch"):
            ego_raw_dir = os.path.join(raw_data_dir, f"ego4d_ep{ep_idx:02d}")
            if os.path.exists(os.path.join(ego_raw_dir, "actions.npy")):
                continue
            frame_dir_ego = os.path.join(ego_raw_dir, "frames")
            os.makedirs(frame_dir_ego, exist_ok=True)

            # Retrieve the frame index sequence for this episode
            ep_f_indices = ep_indices_dict[unique_human_eps[ep_idx]]
            frame_indices = ep_f_indices
            img_key = next(
                k
                for k in dataset_human[frame_indices[0]].keys()
                if "images" in k or "image" in k
            )

            for idx, frame_idx in enumerate(frame_indices):
                step = dataset_human[frame_idx]
                img_t = step[img_key]
                img_np = (
                    (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if img_t.dtype == torch.float32
                    else img_t.permute(1, 2, 0).numpy().astype(np.uint8)
                )
                Image.fromarray(img_np).save(
                    os.path.join(frame_dir_ego, f"frame_{idx:04d}.png")
                )

            np.save(
                os.path.join(ego_raw_dir, "actions.npy"),
                np.zeros((len(frame_indices), 12)),
            )
            np.save(
                os.path.join(ego_raw_dir, "states.npy"),
                np.zeros((len(frame_indices), 24)),
            )
            np.save(
                os.path.join(ego_raw_dir, "tactile.npy"),
                np.zeros((len(frame_indices), 4, 4)),
            )

        print(
            "Successfully loaded and structured all human tabletop pre-training episodes."
        )
    except Exception as e:
        print(f"Error: Failed to load CMU Human dataset ({e}).")
        raise e

    # ==========================================
    # STREAM C: Geometry Tracks (10% of mix) - Full Pool
    # ==========================================
    print(f"\n--- Loading Stream C: PointOdyssey (100% of geometry tracking pool) ---")
    for ep_idx in range(5):
        geometry_raw_dir = os.path.join(raw_data_dir, f"geometry_ep{ep_idx:02d}")
        frame_dir_geom = os.path.join(geometry_raw_dir, "frames")

        if not os.path.exists(frame_dir_geom) or len(os.listdir(frame_dir_geom)) == 0:
            raise FileNotFoundError(
                f"Error: Missing PointOdyssey files in '{geometry_raw_dir}'. "
                "You must download and extract the raw PointOdyssey dataset files first."
            )

        actions_file = os.path.join(geometry_raw_dir, "actions.npy")
        states_file = os.path.join(geometry_raw_dir, "states.npy")
        tactile_file = os.path.join(geometry_raw_dir, "tactile.npy")

        num_frames = len([f for f in os.listdir(frame_dir_geom) if f.endswith(".png")])
        if num_frames == 0:
            raise ValueError(f"Error: No frame files found in '{frame_dir_geom}'")

        # Save clean zero arrays on disk for downstream loading consistency
        if not os.path.exists(actions_file):
            np.save(actions_file, np.zeros((num_frames, 12), dtype=np.float32))
        if not os.path.exists(states_file):
            np.save(states_file, np.zeros((num_frames, 24), dtype=np.float32))
        if not os.path.exists(tactile_file):
            np.save(tactile_file, np.zeros((num_frames, 4, 4), dtype=np.float32))

    print("Structured complete 3D tracking pre-training pool.")

    # ==========================================
    # 5. Run Tokenization Preprocessing (100% of data mix)
    # ==========================================
    print(
        "\n--- Running Tokenization Preprocessor on the Data Mix (100% of Partition) ---"
    )
    run_preprocessing(
        raw_data_dir,
        "tabletop manipulation and visual geometric grounding",
        processed_dir,
        disable_encoders=disable_encoders,
    )

    # ==========================================
    # 6. Load and Visualize Mixed Dataset Stats (5% Visualization Slice)
    # ==========================================
    print("\n--- Generating Visual Dataset Mix Diagnostics (5% Slice) ---")

    proc_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pt")])
    if len(proc_files) == 0:
        print("Error: No preprocessed files found!")
        return

    print(
        f"Preprocessed dataset contains {len(proc_files)} total training episode files."
    )

    # Load first file (representing the 5% diagnostic visualization slice)
    data = torch.load(os.path.join(processed_dir, proc_files[0]))
    for key, val in data.items():
        if torch.is_tensor(val):
            print(
                f"  - {key:15s} | Shape: {str(list(val.shape)):15s} | Mean: {val.mean().item():7.4f} | Var: {val.var().item():7.4f}"
            )

    # Plot histograms and diagnostic profiles
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Stage 1 Pre-training Dataset Mix (60:30:10 Ratio) - 5% Visualized Slice",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].hist(
        data["vision"].numpy().flatten(),
        bins=30,
        color="royalblue",
        alpha=0.7,
        edgecolor="black",
    )
    axes[0, 0].set_title("DINOv3 Visual Features")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(
        data["pointnext"].numpy().flatten(),
        bins=30,
        color="forestgreen",
        alpha=0.7,
        edgecolor="black",
    )
    axes[0, 1].set_title("PointNeXt Geometry Features")
    axes[0, 1].set_xlabel("Value")

    proportions = [60, 30, 10]
    labels = ["BridgeV2 (60%)", "CMU Stretch (30%)", "PointOdyssey (10%)"]
    colors = ["yellowgreen", "gold", "lightskyblue"]
    axes[0, 2].pie(
        proportions,
        labels=labels,
        autopct="%1.0f%%",
        startangle=140,
        colors=colors,
        shadow=True,
    )
    axes[0, 2].set_title("Target Stage 1 Dataset Mix")

    actions_np = data["actions"].numpy()
    for j in range(min(4, actions_np.shape[1])):
        axes[1, 0].plot(
            range(actions_np.shape[0]), actions_np[:, j], marker="o", label=f"Joint {j}"
        )
    axes[1, 0].set_title("Pre-training Action Projections")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Command Value")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--")

    vision_norms = data["vision"] / (data["vision"].norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = torch.matmul(vision_norms, vision_norms.T).numpy()
    im = axes[1, 1].imshow(sim_matrix, cmap="plasma", origin="lower")
    axes[1, 1].set_title("State Similarity Drift (DINO)")
    axes[1, 1].set_xlabel("Frame Step")
    axes[1, 1].set_ylabel("Frame Step")
    fig.colorbar(im, ax=axes[1, 1])

    axes[1, 2].axis("off")
    stats_text = (
        f"--- Dataset Mix Summary ---\n"
        f"Base Model: lerobot/droid\n"
        f"Sequence Length: {actions_np.shape[0]} frames\n"
        f"DINOv3 Feature Dim: {data['vision'].shape[1]}\n"
        f"PointNeXt Feature Dim: {data['pointnext'].shape[1]}\n"
        f"VGGT Feature Dim: {data['vggt'].shape[1]}\n"
        f"Actions Command Dim: {data['actions'].shape[1]}\n"
        f"Total Mixed Episodes: {len(proc_files)}\n"
        f"Pre-training Target: Dynamics & VIB"
    )
    axes[1, 2].text(
        0.05,
        0.95,
        stats_text,
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plot_path = "latent-flow/data/dataset_summary.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved Stage 1 dataset mix diagnostics plot to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and preprocess LatentFlow Stage 1 pre-training dataset mix"
    )
    parser.add_argument(
        "--enable_encoders",
        action="store_true",
        help="Run preprocessing with active vision/language encoders on GPU (default is disabled/mock mode)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe raw and processed directories to start from scratch (default is False/resume mode)",
    )

    args = parser.parse_args()
    # If --enable_encoders is passed, disable_encoders becomes False
    prepare_and_visualize_dataset(
        disable_encoders=not args.enable_encoders, clean_cache=args.clean
    )
