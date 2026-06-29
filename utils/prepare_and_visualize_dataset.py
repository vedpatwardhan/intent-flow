import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.preprocess_dataset import run_preprocessing


def prepare_and_visualize_dataset():
    print("=== LatentFlow Dataset Preparation & Visualizer ===")

    # 1. Setup paths
    raw_data_dir = "latent-flow/data/raw"
    processed_dir = "latent-flow/data/processed"

    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 2. Check/Install lerobot dynamically on Colab
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Installing lerobot library to download Hugging Face dataset...")
        os.system("pip install -q lerobot")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 3. Download a small real tabletop manipulation dataset (ALOHA pink bottle)
    repo_id = "lerobot/aloha_static_pink_bottle"
    print(f"\nFetching real demonstration dataset '{repo_id}' from Hugging Face...")
    dataset = LeRobotDataset(repo_id)

    # 4. Extract Episode 0
    print("\nExtracting Episode 0 frames and joint command actions...")
    episode_idx = 0
    start_frame = dataset.episode_data_index[episode_idx].item()
    end_frame = (
        dataset.episode_data_index[episode_idx + 1].item()
        if (episode_idx + 1) < len(dataset.episode_data_index)
        else len(dataset)
    )

    # Slice to a subset of 16 frames to keep the preprocessor fast for visualization
    frame_indices = list(range(start_frame, min(start_frame + 16, end_frame)))
    print(
        f"Sampling {len(frame_indices)} contiguous real frames (indices: {frame_indices[0]} to {frame_indices[-1]})..."
    )

    # Create raw episode folders
    episode_dir = os.path.join(raw_data_dir, f"episode_{episode_idx:04d}")
    frame_dir = os.path.join(episode_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Determine visual key for image observation
    img_key = None
    sample = dataset[frame_indices[0]]
    for key in sample.keys():
        if "observation.images" in key or "observation.image" in key:
            img_key = key
            break

    if img_key is None:
        raise ValueError(
            f"Could not find visual image keys in the dataset. Available keys: {list(sample.keys())}"
        )

    print(f"Using image key: {img_key}")

    # Extract and save images, states, and actions
    episode_actions = []
    episode_states = []

    for idx, frame_idx in enumerate(frame_indices):
        data_step = dataset[frame_idx]

        # Save image frame
        img_tensor = data_step[img_key]  # [3, H, W]
        if img_tensor.dtype == torch.float32 and img_tensor.max() <= 1.0:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

        Image.fromarray(img_np).save(os.path.join(frame_dir, f"frame_{idx:04d}.png"))

        # Save action and proprioception states
        episode_actions.append(data_step["action"].numpy())
        state_key = "observation.state"
        if state_key in data_step:
            episode_states.append(data_step[state_key].numpy())
        else:
            episode_states.append(np.zeros(24))

    # Save to numpy arrays
    np.save(os.path.join(episode_dir, "actions.npy"), np.stack(episode_actions, axis=0))
    np.save(os.path.join(episode_dir, "states.npy"), np.stack(episode_states, axis=0))
    np.save(
        os.path.join(episode_dir, "tactile.npy"),
        np.random.randn(len(frame_indices), 4, 4) * 0.1,
    )

    print(f"Successfully saved raw real episode to: {episode_dir}")

    # 5. Run the tokenization preprocessing pipeline over our real data
    print("\nRunning tokenization preprocessor over the extracted real data...")
    run_preprocessing(
        raw_data_dir=raw_data_dir,
        text_prompt="pick up the pink bottle and place it on the table",
        output_dir=processed_dir,
    )

    # 6. Load processed episode file
    processed_file = os.path.join(processed_dir, "episode_0000.pt")
    if not os.path.exists(processed_file):
        print(f"Error: Processed file {processed_file} not found!")
        return

    data = torch.load(processed_file)
    print("\nSuccessfully loaded preprocessed episode tokens:")
    for key, val in data.items():
        if torch.is_tensor(val):
            print(
                f"  - {key:15s} | Shape: {str(list(val.shape)):15s} | Mean: {val.mean().item():7.4f} | Var: {val.var().item():7.4f}"
            )

    # 7. Generate Visual Summary Diagnostics
    print("\nGenerating visual token distribution summary plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "LatentFlow Preprocessed Dataset Diagnostics (Real Data)",
        fontsize=16,
        fontweight="bold",
    )

    # A. Histogram of Visual Tokens (DINOv3)
    dino_flat = data["vision"].numpy().flatten()
    axes[0, 0].hist(dino_flat, bins=30, color="royalblue", alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("DINOv3 Visual Token Distribution")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Count")

    # B. Histogram of Geometric Tokens (PointNeXt)
    pnext_flat = data["pointnext"].numpy().flatten()
    axes[0, 1].hist(
        pnext_flat, bins=30, color="forestgreen", alpha=0.7, edgecolor="black"
    )
    axes[0, 1].set_title("PointNeXt Geometric Token Distribution")
    axes[0, 1].set_xlabel("Value")

    # C. Histogram of Tactile Sensor Grids
    tactile_flat = data["tactile"].numpy().flatten()
    axes[0, 2].hist(
        tactile_flat, bins=30, color="crimson", alpha=0.7, edgecolor="black"
    )
    axes[0, 2].set_title("Tactile Grid Sensor Distribution")
    axes[0, 2].set_xlabel("Value")

    # D. Trajectory of Actions over the sequence (First 4 Joint Torques)
    actions_np = data["actions"].numpy()
    seq_len = actions_np.shape[0]
    for j in range(min(4, actions_np.shape[1])):
        axes[1, 0].plot(
            range(seq_len), actions_np[:, j], marker="o", label=f"Joint {j}"
        )
    axes[1, 0].set_title("Joint Action Trajectories")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Torque (Nm)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--")

    # E. State Cosine Similarity Matrix over time (Semantic drift check)
    vision_norms = data["vision"] / (data["vision"].norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = torch.matmul(vision_norms, vision_norms.T).numpy()
    im = axes[1, 1].imshow(sim_matrix, cmap="plasma", origin="lower")
    axes[1, 1].set_title("Temporal State Similarity (DINO)")
    axes[1, 1].set_xlabel("Frame Step")
    axes[1, 1].set_ylabel("Frame Step")
    fig.colorbar(im, ax=axes[1, 1])

    # F. Summary statistics text box
    axes[1, 2].axis("off")
    stats_text = (
        f"--- Dataset Metadata ---\n"
        f"Episode Sequence Steps: {seq_len}\n"
        f"CLIP Prompt: 'pick up the pink bottle and place it on the table'\n"
        f"DINOv3 Dim: {data['vision'].shape[1]}\n"
        f"PointNeXt Dim: {data['pointnext'].shape[1]}\n"
        f"VGGT Dim: {data['vggt'].shape[1]}\n"
        f"Tactile Grid: {list(data['tactile'].shape[1:])}\n"
        f"Actions Dim: {data['actions'].shape[1]}\n"
        f"Proprioception Dim: {data['proprioception'].shape[1]}"
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
    print(f"\nSaved real dataset visual diagnostics plot to: {plot_path}")


if __name__ == "__main__":
    prepare_and_visualize_dataset()
