import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Path stabilization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainers.stage1_pretrain import Stage1PretrainModule
from utils.dataset_loader import PretrainingDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 Latent Manifold Harvesting and Visualization"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Stage 1 checkpoint file (.ckpt or .pt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to processed .pt dataset directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="latent-flow/data/stage1_manifold_analysis.png",
        help="Path to save output plot",
    )
    return parser.parse_args()


def collect_latent_trajectories(model, data_dir):
    # Initialize the dataset
    dataset = PretrainingDataset(
        data_dir=data_dir, window_size=32, mask_ratio=0.0, use_subset=False
    )

    # We want to identify the source datasets: Droid (Index <= 326), CMU (327 <= Index <= 461), Odyssey (Index >= 462)
    categories = {"Droid (Robot)": [], "CMU (Human)": [], "PointOdyssey (Geometry)": []}

    # Track which file belongs to which category
    for fname in sorted(os.listdir(dataset.data_dir)):
        if not fname.endswith(".pt"):
            continue
        try:
            ep_idx = int(fname.split("_")[1].split(".")[0])
        except Exception:
            continue

        file_path = os.path.join(dataset.data_dir, fname)
        data = torch.load(file_path, map_location="cpu")

        # Squeeze dimensions for mapping through adapters & MSAT
        vision = data["vision"]
        text = data["text"]
        pointnext = data["pointnext"]
        vggt = data["vggt"]

        seq_len = vision.shape[0]
        latents_list = []

        for t in range(seq_len):
            with torch.no_grad():
                vis_tok = model.vis_adapter(vision[t : t + 1].to(model.device))
                txt_tok = model.txt_adapter(text.squeeze(1).to(model.device))
                pt_tok = model.pt_adapter(pointnext[t : t + 1].to(model.device))
                vggt_tok = model.vggt_adapter(vggt[t : t + 1].to(model.device))

                # Mock tactile mask
                tactile_mask = torch.zeros(1, model.latent_dim, device=model.device)

                modality_dict = {
                    "vision": vis_tok,
                    "text": txt_tok,
                    "pointnext": pt_tok,
                    "vggt": vggt_tok,
                    "tactile": tactile_mask,
                }
                s_t = model.msat(modality_dict)
                latents_list.append(s_t.cpu().squeeze(0).numpy())

        latents = np.array(latents_list)

        if ep_idx <= 326:
            categories["Droid (Robot)"].append(latents)
        elif 327 <= ep_idx <= 461:
            categories["CMU (Human)"].append(latents)
        else:
            categories["PointOdyssey (Geometry)"].append(latents)

    # Return sampled trajectories (limit to first 3 episodes from each category for clear plotting)
    sampled_data = {}
    for key, episodes in categories.items():
        if len(episodes) > 0:
            sampled_data[key] = episodes[:3]
    return sampled_data


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model configuration & weights
    print(f"Loading checkpoint from: {args.checkpoint}")
    # Load default Hydra-style parameters
    config = {
        "model": {
            "latent_dim": 512,
            "num_heads": 8,
            "num_layers": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "vggt_dim": 768,
            "action_dim": 12,
            "state_dim": 24,
            "horizon": 8,
            "bottleneck_dim": 16,
        },
        "stage1": {
            "lr": 0.0005,
            "weight_decay": 0.0001,
        },
    }

    model = Stage1PretrainModule(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Collect latent trajectories
    print("Harvesting latent trajectories from preprocessed files...")
    sampled_data = collect_latent_trajectories(model, args.data_dir)

    # Initialize plotting layout (9 plots: 3 rows for datasets, 3 columns for PCA, t-SNE, UMAP)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        "Stage 1 Latent Manifold Projection Analysis", fontsize=18, fontweight="bold"
    )

    row_mapping = {"Droid (Robot)": 0, "CMU (Human)": 1, "PointOdyssey (Geometry)": 2}

    for dataset_name, episodes in sampled_data.items():
        row_idx = row_mapping[dataset_name]

        # Flatten all points in these episodes to run dimension reduction together
        flat_latents = np.concatenate(episodes, axis=0)  # [TotalPoints, 512]

        # Track index bounds for color gradient representation
        flat_frame_indices = []
        for ep in episodes:
            flat_frame_indices.extend(list(range(ep.shape[0])))
        flat_frame_indices = np.array(flat_frame_indices)

        # Run dimension reductions
        print(f"Running dimension reductions for {dataset_name}...")
        pca_proj = PCA(n_components=2).fit_transform(flat_latents)
        tsne_proj = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(
            flat_latents
        )
        umap_proj = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
        ).fit_transform(flat_latents)

        # Plots for PCA (Col 0)
        ax_pca = axes[row_idx, 0]
        sc_pca = ax_pca.scatter(
            pca_proj[:, 0],
            pca_proj[:, 1],
            c=flat_frame_indices,
            cmap="plasma",
            edgecolors="none",
            alpha=0.8,
            s=15,
        )
        ax_pca.set_title(f"{dataset_name} - PCA")
        ax_pca.set_xlabel("PC 1")
        ax_pca.set_ylabel("PC 2")
        fig.colorbar(sc_pca, ax=ax_pca, label="Frame Index")

        # Plots for t-SNE (Col 1)
        ax_tsne = axes[row_idx, 1]
        sc_tsne = ax_tsne.scatter(
            tsne_proj[:, 0],
            tsne_proj[:, 1],
            c=flat_frame_indices,
            cmap="plasma",
            edgecolors="none",
            alpha=0.8,
            s=15,
        )
        ax_tsne.set_title(f"{dataset_name} - t-SNE")
        ax_tsne.set_xlabel("Dimension 1")
        ax_tsne.set_ylabel("Dimension 2")
        fig.colorbar(sc_tsne, ax=ax_tsne, label="Frame Index")

        # Plots for UMAP (Col 2)
        ax_umap = axes[row_idx, 2]
        sc_umap = ax_umap.scatter(
            umap_proj[:, 0],
            umap_proj[:, 1],
            c=flat_frame_indices,
            cmap="plasma",
            edgecolors="none",
            alpha=0.8,
            s=15,
        )
        ax_umap.set_title(f"{dataset_name} - UMAP")
        ax_umap.set_xlabel("Dimension 1")
        ax_umap.set_ylabel("Dimension 2")
        fig.colorbar(sc_umap, ax=ax_umap, label="Frame Index")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    print(f"Saved manifold projection plot successfully to: {args.output_path}")


if __name__ == "__main__":
    main()
