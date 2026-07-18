import torch
import torch.nn as nn


class VisualAdapter(nn.Module):
    """
    Projects dense DINOv3 visual tokens (d_in=1024) to the shared latent dimension (d_out=512).
    """

    def __init__(self, d_in=1024, d_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)


class TextAdapter(nn.Module):
    """
    Projects CLIP text embeddings (d_in=768) to the shared latent dimension (d_out=512).
    """

    def __init__(self, d_in=768, d_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)


class PointNeXtAdapter(nn.Module):
    """
    Projects PointNeXt geometric point cloud tokens (d_in=384) to the shared latent dimension (d_out=512).
    """

    def __init__(self, d_in=384, d_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)


class TactileAdapter(nn.Module):
    """
    Compresses a spatial touch grid (e.g., 4x4 fingers grid) and projects it to d_out=512.
    """

    def __init__(self, grid_shape=(4, 4), d_out=512):
        super().__init__()
        flat_dim = grid_shape[0] * grid_shape[1]
        self.net = nn.Sequential(
            nn.Linear(flat_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        # Flatten touch grid input [Batch, GridH, GridW] -> [Batch, Flat]
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)


class ActionAdapter(nn.Module):
    """
    Projects continuous joint control actions (d_in=12) into the shared latent space.
    """

    def __init__(self, d_in=12, d_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)


class VGGTAdapter(nn.Module):
    """
    Projects VGGT visual geometry features (d_in=768) to the shared latent dimension (d_out=512).
    """

    def __init__(self, d_in=768, d_out=512):
        super().__init__()

        # Downsamples the 224x224 canvas to a 16x16 grid (224 / 14 = 16)
        # This preserves the geometric structure of the entire image loop
        self.spatial_pool = nn.AvgPool2d(kernel_size=14, stride=14)

        # Maps the 256 spatial grid points perfectly up to the 768 token space
        # required by your existing Stage 1 / Stage 2 checkpoint configurations
        self.token_expansion = nn.Linear(16 * 16, d_in)

        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        """
        Args:
            x: Raw global or local heatmap tensor of shape [Batch, 1, 224, 224]
        """
        # Ensure the tensor has the correct channel/spatial dimensions for pooling
        if x.dim() == 3:  # If passed as [Batch, 224, 224]
            x = x.unsqueeze(1)

        # Step 1: Downsample from [Batch, 1, 224, 224] -> [Batch, 1, 16, 16]
        x = self.spatial_pool(x)

        # Step 2: Flatten spatial topology -> [Batch, 256]
        x = x.flatten(1)

        # Step 3: Project up to internal token dimensions -> [Batch, 768]
        x = self.token_expansion(x)

        # Step 4: Adapt to the shared embedding space -> [Batch, 512]
        return self.net(x)
