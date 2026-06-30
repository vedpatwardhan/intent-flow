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
import tarfile
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import hf_hub_download, list_repo_files
from utils.preprocess_dataset import run_preprocessing
from torchcodec.decoders import VideoDecoder


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


def setup_directories(raw_data_dir, processed_dir, clean_cache):
    """Handles directory creation and caches clean-up."""
    if clean_cache:
        print("[Dataset Setup] Cleaning local dataset caches...")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        if os.path.exists(raw_data_dir):
            shutil.rmtree(raw_data_dir)

    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)


def prepare_stream_a_droid(raw_data_dir):
    """Downloads, decodes, and structures the Droid tabletop robot stream (60% of mix)."""
    bridge_repo = "lerobot/droid_1.0.1"
    print(f"\n--- Loading Stream A: Droid ({bridge_repo}) ---")

    # Step 1: Download shard metadata and raw MP4 video
    files = list_repo_files(bridge_repo, repo_type="dataset")
    parquet_path = next(
        f for f in files if f.endswith(".parquet") and "data/chunk-000" in f
    )
    video_path = next(
        f for f in files if f.endswith(".mp4") and "videos/" in f and "chunk-000" in f
    )

    print(f"[Droid] Downloading training Parquet shard: {parquet_path}")
    local_parquet = hf_hub_download(
        bridge_repo, filename=parquet_path, repo_type="dataset"
    )
    print(f"[Droid] Downloading training Video shard: {video_path}")
    local_video = hf_hub_download(bridge_repo, filename=video_path, repo_type="dataset")

    # Step 2: Read episode definitions
    df = pd.read_parquet(local_parquet)
    unique_episodes = df["episode_index"].unique()
    unique_episodes = sorted([ep for ep in unique_episodes if ep < 327])
    print(
        f"[Droid] Found {len(unique_episodes)} real robot trajectories in this shard."
    )

    # Step 3: Initialize fast PyTorch-native video decoder
    decoder = VideoDecoder(local_video, device="cpu")

    # Step 4: Map episode boundaries and build flat joint configurations
    episode_configs = []
    for ep_idx in range(len(unique_episodes)):
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

        # Enforce exact dimension limits
        if bridge_actions.shape[1] < 12:
            pad = np.zeros(
                (bridge_actions.shape[0], 12 - bridge_actions.shape[1]),
                dtype=np.float32,
            )
            bridge_actions = np.concatenate([bridge_actions, pad], axis=1)
        elif bridge_actions.shape[1] > 12:
            bridge_actions = bridge_actions[:, :12]

        if bridge_states.shape[1] < 24:
            pad = np.zeros(
                (bridge_states.shape[0], 24 - bridge_states.shape[1]), dtype=np.float32
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

    # Step 5: Save frames sequentially using a ThreadPoolExecutor for background disk writes
    print("[Droid] Extracting frames (Parallel background disk writes)...")
    current_ep_idx = 0
    current_frame_in_ep = 0
    total_frames = (
        len(decoder)
        if hasattr(decoder, "__len__")
        else sum(cfg["length"] for cfg in episode_configs)
    )

    with ThreadPoolExecutor(max_workers=16) as executor:
        for _, frame_tensor in enumerate(
            tqdm(decoder, desc="Extracting Droid frames", total=total_frames)
        ):
            if current_ep_idx >= len(episode_configs):
                break

            cfg = episode_configs[current_ep_idx]

            # Support resume: if action file exists, skip this entire segment
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
            executor.submit(save_image_worker, frame_rgb, target_path)

            current_frame_in_ep += 1
            if current_frame_in_ep >= cfg["length"]:
                np.save(os.path.join(cfg["dir"], "actions.npy"), cfg["actions"])
                np.save(os.path.join(cfg["dir"], "states.npy"), cfg["states"])
                np.save(
                    os.path.join(cfg["dir"], "tactile.npy"),
                    np.zeros((cfg["length"], 4, 4), dtype=np.float32),
                )

                if current_ep_idx % 50 == 0:
                    print(
                        f"  - [Droid] Structured episode {current_ep_idx}/{len(episode_configs)}"
                    )

                current_ep_idx += 1
                current_frame_in_ep = 0

    print("[Droid] Successfully loaded and structured Droid stream.")


def prepare_stream_b_cmu(raw_data_dir):
    """Downloads, decodes, and structures the CMU human tabletop exploration stream (30% of mix)."""
    human_repo = "lerobot/cmu_stretch"
    print(f"\n--- Loading Stream B: CMU Human Exploration ({human_repo}) ---")

    # Step 1: Download raw CMU HuggingFace dataset
    dataset_human = LeRobotDataset(human_repo)

    # Step 2: Read episode groups
    ep_indices_dict = {}
    for idx, ep_val in enumerate(dataset_human.hf_dataset["episode_index"]):
        ep_val_item = ep_val.item() if hasattr(ep_val, "item") else ep_val
        if ep_val_item not in ep_indices_dict:
            ep_indices_dict[ep_val_item] = []
        ep_indices_dict[ep_val_item].append(idx)

    unique_human_eps = sorted(list(ep_indices_dict.keys()))
    num_human_eps = len(unique_human_eps)
    print(f"[CMU] Preparing all {num_human_eps} human episodes...")

    # Step 3: Extract frames and zero-masked kinetics
    for ep_idx in tqdm(range(num_human_eps), desc="Processing CMU Stretch"):
        ego_raw_dir = os.path.join(raw_data_dir, f"ego4d_ep{ep_idx:02d}")

        # Support resume check
        if os.path.exists(os.path.join(ego_raw_dir, "actions.npy")):
            continue

        frame_dir_ego = os.path.join(ego_raw_dir, "frames")
        os.makedirs(frame_dir_ego, exist_ok=True)

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

        # Save arrays
        np.save(
            os.path.join(ego_raw_dir, "actions.npy"),
            np.zeros((len(frame_indices), 12), dtype=np.float32),
        )
        np.save(
            os.path.join(ego_raw_dir, "states.npy"),
            np.zeros((len(frame_indices), 24), dtype=np.float32),
        )
        np.save(
            os.path.join(ego_raw_dir, "tactile.npy"),
            np.zeros((len(frame_indices), 4, 4), dtype=np.float32),
        )

    print("[CMU] Successfully loaded and structured CMU stream.")


def prepare_stream_c_pointodyssey(raw_data_dir):
    """Downloads the PointOdyssey sample zip, extracts the frames + true 3D point tracks (10% of mix)."""
    print(f"\n--- Loading Stream C: PointOdyssey ---")

    # Audit existing geometry folders
    geometry_ep_dirs = sorted(
        [
            d
            for d in os.listdir(raw_data_dir)
            if d.startswith("geometry_ep")
            and os.path.isdir(os.path.join(raw_data_dir, d))
        ]
    )

    is_complete = len(geometry_ep_dirs) > 0 and all(
        os.path.exists(os.path.join(raw_data_dir, d, "point_clouds.npy"))
        for d in geometry_ep_dirs
    )

    if not is_complete:
        print(
            "[PointOdyssey] Local files missing/incomplete. Re-downloading 3.32 GB sample..."
        )

        # Clean incomplete directories first
        for d in geometry_ep_dirs:
            shutil.rmtree(os.path.join(raw_data_dir, d), ignore_errors=True)

        # Step 1: Download from HF Hub
        try:
            hf_hub_download(
                repo_id="aharley/pointodyssey",
                filename="sample.tar.gz",
                repo_type="dataset",
                local_dir="latent-flow/data/",
            )
        except Exception as e:
            raise RuntimeError(f"[PointOdyssey] Failed to download sample.tar.gz ({e})")

        # Step 2: Extract tar.gz archive
        sample_tar = "latent-flow/data/sample.tar.gz"
        print(f"[PointOdyssey] Extracting {sample_tar}...")
        try:
            with tarfile.open(sample_tar, "r:gz") as tar:
                tar.extractall(path="latent-flow/data/")
        except Exception as e:
            raise RuntimeError(f"[PointOdyssey] Failed to extract archive ({e})")

        # Step 3: Locate unpacked annotation and image files
        annos = sorted(glob.glob("latent-flow/data/**/anno.npz", recursive=True))
        if not annos:
            raise FileNotFoundError(
                "[PointOdyssey] Extraction succeeded, but no anno.npz files were found."
            )

        num_geom_episodes = len(annos)
        print(
            f"[PointOdyssey] Found {num_geom_episodes} sequences. Restructuring all..."
        )

        # Step 4: Map coordinates and extract RGB frames
        for ep_idx, anno_path in enumerate(annos):
            ep_dir = os.path.dirname(anno_path)
            target_ep_dir = os.path.join(raw_data_dir, f"geometry_ep{ep_idx:02d}")
            target_frame_dir = os.path.join(target_ep_dir, "frames")
            os.makedirs(target_frame_dir, exist_ok=True)

            # Load true 3D point cloud tracks
            anno_data = np.load(anno_path)
            if "trajs_3d" not in anno_data:
                raise KeyError(
                    f"[PointOdyssey] Annotation missing 'trajs_3d' key. Found: {list(anno_data.keys())}"
                )

            trajs_3d = anno_data["trajs_3d"]  # Shape: [Frames, NumPoints, 3]
            num_frames = trajs_3d.shape[0]
            num_pts_source = trajs_3d.shape[1]

            # Subsample 100 points and pad with 1.0 intensity channel to match PointNeXt
            sampled_indices = np.random.choice(
                num_pts_source, min(100, num_pts_source), replace=False
            )
            trajs_sampled = trajs_3d[:, sampled_indices, :]
            intensity = np.ones((num_frames, 100, 1), dtype=np.float32)
            pts3d_padded = np.concatenate([trajs_sampled, intensity], axis=2)

            np.save(os.path.join(target_ep_dir, "point_clouds.npy"), pts3d_padded)

            # Find and copy RGB frame images case-insensitively
            rgb_files = sorted(
                [
                    f
                    for f in glob.glob(os.path.join(ep_dir, "**", "*"), recursive=True)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")) and "rgbs" in f
                ]
            )

            for i, img_path in enumerate(rgb_files):
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                img.save(os.path.join(target_frame_dir, f"frame_{i:04d}.png"))

            # Write zero-masked action, state, and tactile files
            np.save(
                os.path.join(target_ep_dir, "actions.npy"),
                np.zeros((num_frames, 12), dtype=np.float32),
            )
            np.save(
                os.path.join(target_ep_dir, "states.npy"),
                np.zeros((num_frames, 24), dtype=np.float32),
            )
            np.save(
                os.path.join(target_ep_dir, "tactile.npy"),
                np.zeros((num_frames, 4, 4), dtype=np.float32),
            )

        # Step 5: Clean up temp extraction folder
        print("[PointOdyssey] Cleaning up raw unpacked sample folder...")
        shutil.rmtree("latent-flow/data/sample", ignore_errors=True)
    else:
        num_geom_episodes = len(geometry_ep_dirs)

    # Validate output folder contents
    for ep_idx in range(num_geom_episodes):
        geometry_raw_dir = os.path.join(raw_data_dir, f"geometry_ep{ep_idx:02d}")
        frame_dir_geom = os.path.join(geometry_raw_dir, "frames")

        actions_file = os.path.join(geometry_raw_dir, "actions.npy")
        states_file = os.path.join(geometry_raw_dir, "states.npy")
        tactile_file = os.path.join(geometry_raw_dir, "tactile.npy")
        pointclouds_file = os.path.join(geometry_raw_dir, "point_clouds.npy")

        if not os.path.exists(frame_dir_geom) or len(os.listdir(frame_dir_geom)) == 0:
            raise FileNotFoundError(
                f"[PointOdyssey] Missing frames in '{geometry_raw_dir}'"
            )
        if not os.path.exists(pointclouds_file):
            raise FileNotFoundError(
                f"[PointOdyssey] Missing point_clouds.npy in '{geometry_raw_dir}'"
            )
        if (
            not os.path.exists(actions_file)
            or not os.path.exists(states_file)
            or not os.path.exists(tactile_file)
        ):
            raise FileNotFoundError(
                f"[PointOdyssey] Missing kinematics files in '{geometry_raw_dir}'"
            )

    print(
        f"[PointOdyssey] Structured PointOdyssey stream ({num_geom_episodes} episodes)."
    )


def prepare_and_visualize_dataset(disable_encoders=True, clean_cache=False):
    """Coordinates directory setup, stream downloads, preprocessing, and diagnostics plot rendering."""
    print("=== LatentFlow Stage 1 Pre-training Dataset Mixture Builder ===")

    raw_data_dir = "latent-flow/data/raw"
    processed_dir = "latent-flow/data/processed"

    # Setup directories
    setup_directories(raw_data_dir, processed_dir, clean_cache)

    # Load Stream A (Droid - Robot Actions)
    prepare_stream_a_droid(raw_data_dir)

    # Load Stream B (CMU Human Tabletop)
    prepare_stream_b_cmu(raw_data_dir)

    # Load Stream C (PointOdyssey - Visual 3D Geometry)
    prepare_stream_c_pointodyssey(raw_data_dir)

    # Run preprocessor
    print("\n--- Running Tokenization Preprocessor on the Data Mix ---")
    run_preprocessing(
        raw_data_dir,
        "tabletop manipulation and visual geometric grounding",
        processed_dir,
        disable_encoders=disable_encoders,
    )

    # Generate Diagnostics Visualizations
    print("\n--- Generating Visual Dataset Mix Diagnostics (5% Slice) ---")
    proc_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pt")])
    if len(proc_files) == 0:
        print("Error: No preprocessed files found!")
        return

    print(
        f"Preprocessed dataset contains {len(proc_files)} total training episode files."
    )

    data = torch.load(os.path.join(processed_dir, proc_files[0]))
    for key, val in data.items():
        if torch.is_tensor(val):
            print(
                f"  - {key:15s} | Shape: {str(list(val.shape)):15s} | Mean: {val.mean().item():7.4f} | Var: {val.var().item():7.4f}"
            )

    # Plot visual histograms
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
        help="Run preprocessing with active vision/language encoders on GPU",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe raw and processed directories to start from scratch",
    )

    args = parser.parse_args()
    prepare_and_visualize_dataset(
        disable_encoders=not args.enable_encoders, clean_cache=args.clean
    )
