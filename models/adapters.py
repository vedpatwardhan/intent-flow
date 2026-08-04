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
            x: Raw global or local heatmap tensor of shape [Batch, Channels/Views, 224, 224] or [Batch, 224, 224]
        """
        # Handle multi-view input [Batch, Views, 224, 224] where Views > 1
        is_multi_view = x.dim() == 4 and x.shape[1] > 1

        if is_multi_view:
            B, V, H, W = x.shape
            x = x.view(B * V, 1, H, W)
        elif x.dim() == 3:  # If passed as [Batch, 224, 224]
            x = x.unsqueeze(1)

        # Step 1: Downsample to 16x16 grid
        x = self.spatial_pool(x)

        # Step 2: Flatten spatial topology -> [Batch (* Views), 256]
        x = x.flatten(1)

        # Step 3: Project up to internal token dimensions -> [Batch (* Views), 768]
        x = self.token_expansion(x)

        # Step 4: Adapt to the shared embedding space -> [Batch (* Views), 512]
        x = self.net(x)

        if is_multi_view:
            # Restore batch and view dimensions
            x = x.view(B, V, -1)

        return x


class EdgeAdapter(nn.Module):
    """
    Projects 1-channel Sobel edge-gradient maps (d_in=384) to shared latent dimension (d_out=512).
    Converts 224x224 edge maps into a 14x14 spatial patch grid (196 tokens per view).
    """

    def __init__(self, d_in=384, d_out=512):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, d_in, kernel_size=16, stride=16)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        """
        Args:
            x: Raw edge-gradient tensor of shape [B, 1, H, W], [B, V, 1, H, W], or [1, H, W]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1, 1, H, W]

        is_multi_view = x.dim() == 5
        if is_multi_view:
            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)
        else:
            B, C, H, W = x.shape
            V = 1

        # 1. Patchify: [B*V, 1, 224, 224] -> [B*V, 384, 14, 14]
        x = self.patch_embed(x)
        # 2. Flatten spatial dimensions: [B*V, 384, 196] -> [B*V, 196, 384]
        x = x.flatten(2).transpose(1, 2)
        # 3. Adapt: [B*V, 196, 512]
        x = self.net(x)

        if is_multi_view:
            # Reshape back to [B, V * 196, 512]
            x = x.view(B, V * 196, -1)

        return x
