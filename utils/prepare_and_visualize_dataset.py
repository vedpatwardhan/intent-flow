import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess_dataset import run_preprocessing


def prepare_and_visualize_dataset():
    print("=== LatentFlow Dataset Preparation & Visualizer ===")

    # 1. Setup paths
    raw_data_dir = "latent-flow/data/raw"
    processed_dir = "latent-flow/data/processed"

    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Create a mock raw episode structure if none exists
    raw_episode = os.path.join(raw_data_dir, "episode_0000")
    if not os.path.exists(raw_episode):
        print("Creating mock raw episode data for dry-run visualization...")
        os.makedirs(os.path.join(raw_episode, "frames"), exist_ok=True)
        # Create mock actions, states, tactile grids
        np.save(os.path.join(raw_episode, "actions.npy"), np.random.randn(8, 12) * 5.0)
        np.save(os.path.join(raw_episode, "states.npy"), np.random.randn(8, 24) * 2.0)
        np.save(
            os.path.join(raw_episode, "tactile.npy"), np.random.randn(8, 4, 4) * 1.5
        )
        np.save(
            os.path.join(raw_episode, "point_clouds.npy"), np.random.randn(8, 100, 4)
        )

    # 2. Run preprocessing pipeline
    print("\nRunning dataset preprocessor...")
    run_preprocessing(
        raw_data_dir=raw_data_dir,
        text_prompt="pinch the red cube's corners",
        output_dir=processed_dir,
    )

    # 3. Load processed episode file
    processed_file = os.path.join(processed_dir, "episode_0000.pt")
    if not os.path.exists(processed_file):
        print(f"Error: Processed file {processed_file} not found!")
        return

    data = torch.load(processed_file)
    print("\nSuccessfully loaded preprocessed episode tokens:")
    for key, val in data.items():
        if torch.is_tensor(val):
            print(
                f"  - {key:15s} | Shape: {list(val.shape):15s} | Mean: {val.mean().item():7.4f} | Var: {val.var().item():7.4f}"
            )

    # 4. Generate Visual Summary Diagnostics (to verify scaling, normalization, and semantic flow)
    print("\nGenerating visual token distribution summary plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "LatentFlow Preprocessed Dataset Diagnostics", fontsize=16, fontweight="bold"
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
        f"CLIP Prompt: 'pinch the red cube's corners'\n"
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
    print(f"\nSaved visual diagnostics plot to: {plot_path}")


if __name__ == "__main__":
    prepare_and_visualize_dataset()
