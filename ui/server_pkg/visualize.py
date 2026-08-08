import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_energy_landscape(landscape_data: dict, epoch: int, output_dir: str):
    """
    Renders an interactive HTML Energy Landscape plot (Plotly).
    Supports PCA 2D Latent Space Map (with positive/negative anchor markers)
    alongside Trajectory Pull Ratio distributions.
    """
    if (
        "mean_pos_dist_per_track" not in landscape_data
        or "mean_neg_dist_per_track" not in landscape_data
    ):
        return

    mean_pos = np.array(landscape_data["mean_pos_dist_per_track"])
    mean_neg = np.array(landscape_data["mean_neg_dist_per_track"])
    ratios = np.array(landscape_data["pos_neg_ratio"])
    metadata = landscape_data.get("trajectory_metadata", [])

    has_pca = "pca_rollout_coords" in landscape_data

    # Extract metadata labels for interactive hover tooltips
    hover_texts = []
    for idx in range(len(mean_pos)):
        meta = metadata[idx] if idx < len(metadata) else {}
        ep = meta.get("episode_idx", epoch)
        st = meta.get("step_idx", 0)
        cand = meta.get("candidate_idx", idx)
        text = (
            f"<b>Trajectory #{idx + 1}</b><br>"
            f"Episode: {ep} | Step: {st} | Candidate: {cand}<br>"
            f"D+ (Pos Dist): {mean_pos[idx]:.4f}<br>"
            f"D- (Neg Dist): {mean_neg[idx]:.4f}<br>"
            f"Pull Ratio (D+/D-): {ratios[idx]:.4f}"
        )
        hover_texts.append(text)

    # Render interactive Plotly HTML plot
    left_title = (
        f"Epoch {epoch}: PCA 2D Latent Space Map"
        if has_pca
        else f"Epoch {epoch}: Latent Energy Landscape (D+ vs D-)"
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            left_title,
            f"Epoch {epoch}: Trajectory Pull Distribution",
        ),
    )

    if has_pca:
        pca_rollouts = np.array(landscape_data["pca_rollout_coords"])
        pca_pos = np.array(landscape_data.get("pca_pos_coords", []))
        pca_neg = np.array(landscape_data.get("pca_neg_coords", []))

        # Subplot 1: PCA Rollouts
        fig.add_trace(
            go.Scatter(
                x=pca_rollouts[:, 0],
                y=pca_rollouts[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=ratios,
                    colorscale="rdylbu",
                    showscale=True,
                    colorbar=dict(title="Pull Ratio (D+/D-)", x=0.45),
                    line=dict(width=1, color="black"),
                ),
                text=hover_texts,
                hoverinfo="text",
                name="Rollouts",
            ),
            row=1,
            col=1,
        )

        # Positive Anchors (Green Stars)
        if len(pca_pos) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pca_pos[:, 0],
                    y=pca_pos[:, 1],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="green",
                        symbol="star",
                        line=dict(width=1, color="black"),
                    ),
                    name="Positive Anchors (Expert)",
                ),
                row=1,
                col=1,
            )

        # Negative Anchors (Red Crosses)
        if len(pca_neg) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pca_neg[:, 0],
                    y=pca_neg[:, 1],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="red",
                        symbol="x",
                        line=dict(width=2, color="black"),
                    ),
                    name="Negative Anchors (Failure)",
                ),
                row=1,
                col=1,
            )

        fig.update_xaxes(title_text="PCA Component 1", row=1, col=1)
        fig.update_yaxes(title_text="PCA Component 2", row=1, col=1)
    else:
        # Subplot 1: D+ vs D- Scatter
        fig.add_trace(
            go.Scatter(
                x=mean_pos,
                y=mean_neg,
                mode="markers",
                marker=dict(
                    size=10,
                    color=ratios,
                    colorscale="rdylbu",
                    showscale=True,
                    colorbar=dict(title="Pull Ratio (D+/D-)", x=0.45),
                    line=dict(width=1, color="black"),
                ),
                text=hover_texts,
                hoverinfo="text",
                name="Rollouts",
            ),
            row=1,
            col=1,
        )

        # Parity Line (D+ = D-)
        max_val = max(float(mean_pos.max()), float(mean_neg.max()), 1.0)
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Parity Line (D+ = D-)",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Mean Normalized Cosine Distance to Positive Anchors (D+)",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Mean Normalized Cosine Distance to Negative Anchors (D-)",
            row=1,
            col=1,
        )

    # Subplot 2: Histogram
    fig.add_trace(
        go.Histogram(
            x=ratios,
            nbinsx=20,
            marker=dict(color="teal", line=dict(color="black", width=1)),
            name="Pull Ratios",
        ),
        row=1,
        col=2,
    )

    # Parity Threshold line
    fig.add_vline(x=1.0, line_width=2, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_xaxes(title_text="Pull Ratio (D+ / D-)", row=1, col=2)
    fig.update_yaxes(title_text="Trajectory Count", row=1, col=2)

    fig.update_layout(
        title_text=f"Stage 3 Epoch {epoch} Latent Energy Landscape Diagnostics",
        width=1400,
        height=600,
        template="plotly_white",
    )

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"epoch_{epoch:02d}_energy_landscape.html")
    fig.write_html(html_path)
    print(f"📊 [Landscape Analytics] Saved interactive HTML plot to: {html_path}")
