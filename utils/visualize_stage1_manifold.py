import os
import sys
import argparse
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Path stabilization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainers.stage1_pretrain import Stage1PretrainModule
from utils.dataset_loader import PretrainingDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 Latent Manifold Harvesting and Visualization via Plotly"
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
        default="latent-flow/data/stage1_manifold_analysis.html",
        help="Path to save output HTML report",
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

    # Initialize Plotly Grid (3x3 Subplots)
    print("Initializing interactive Plotly canvas...")
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            "Droid - PCA",
            "Droid - t-SNE",
            "Droid - UMAP",
            "CMU - PCA",
            "CMU - t-SNE",
            "CMU - UMAP",
            "PointOdyssey - PCA",
            "PointOdyssey - t-SNE",
            "PointOdyssey - UMAP",
        ],
        horizontal_spacing=0.07,
        vertical_spacing=0.08,
    )

    row_mapping = {
        "Droid (Robot)": 1,
        "CMU (Human)": 2,
        "PointOdyssey (Geometry)": 3,
    }

    for dataset_name, episodes in sampled_data.items():
        row_idx = row_mapping[dataset_name]

        flat_latents = np.concatenate(episodes, axis=0)

        flat_frame_indices = []
        hover_labels = []
        for ep_idx, ep in enumerate(episodes):
            for f_idx in range(ep.shape[0]):
                flat_frame_indices.append(f_idx)
                hover_labels.append(f"Episode {ep_idx} | Frame {f_idx}")
        flat_frame_indices = np.array(flat_frame_indices)

        print(f"Running dimension reductions for {dataset_name}...")
        pca_proj = PCA(n_components=2).fit_transform(flat_latents)
        tsne_proj = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(
            flat_latents
        )
        umap_proj = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
        ).fit_transform(flat_latents)

        # Plot PCA
        fig.add_trace(
            go.Scatter(
                x=pca_proj[:, 0],
                y=pca_proj[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=flat_frame_indices,
                    colorscale="Plasma",
                    showscale=(row_idx == 1),
                ),
                text=hover_labels,
                hoverinfo="text",
                name=f"{dataset_name} PCA",
            ),
            row=row_idx,
            col=1,
        )

        # Plot t-SNE
        fig.add_trace(
            go.Scatter(
                x=tsne_proj[:, 0],
                y=tsne_proj[:, 1],
                mode="markers",
                marker=dict(size=6, color=flat_frame_indices, colorscale="Plasma"),
                text=hover_labels,
                hoverinfo="text",
                name=f"{dataset_name} t-SNE",
            ),
            row=row_idx,
            col=2,
        )

        # Plot UMAP
        fig.add_trace(
            go.Scatter(
                x=umap_proj[:, 0],
                y=umap_proj[:, 1],
                mode="markers",
                marker=dict(size=6, color=flat_frame_indices, colorscale="Plasma"),
                text=hover_labels,
                hoverinfo="text",
                name=f"{dataset_name} UMAP",
            ),
            row=row_idx,
            col=3,
        )

    fig.update_layout(
        title="Stage 1 Latent Manifold Projection Analysis",
        title_x=0.5,
        width=1200,
        height=1200,
        showlegend=False,
        template="plotly_dark",
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fig.write_html(args.output_path)
    print(f"Saved interactive Plotly manifold analysis to: {args.output_path}")


if __name__ == "__main__":
    main()
