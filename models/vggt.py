import torch
import torch.nn as nn


class VGGTEncoder(nn.Module):
    """
    Core Visual Geometry Grounded Transformer (VGGT) architecture.
    Processes video frames to extract dense spatial representations and temporal tracks.
    """

    def __init__(self, in_channels=3, feature_dim=768, num_layers=4, num_heads=8):
        super().__init__()
        # 1. Spatial visual feature extractor (representing a lightweight ViT/CNN patch encoder)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # 64 patches
        )
        self.proj = nn.Linear(256, feature_dim)

        # 2. Spatio-Temporal Sequence Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 3. Geometry Heads
        # A. Camera parameters head (extrinsic Rotation R and Translation T)
        self.camera_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 9 + 3),  # 9-dim rotation matrix + 3-dim translation vector
        )

        # B. 3D Point track predictor
        self.point_track_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(
                256, 100 * 3
            ),  # Tracks 100 point coordinates (X, Y, Z) across time
        )

        # C. Depth estimation projection head
        self.depth_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),  # Mean depth scalar per patch
        )

    def forward(self, x):
        """
        x: Video batch of frames [Batch, SeqLen, Channels, Height, Width]
        """
        batch_size, seq_len, c, h, w = x.size()

        # Flatten time and batch dimensions to run visual feature extraction
        x_flat = x.view(batch_size * seq_len, c, h, w)
        features = self.patch_embed(x_flat)  # [Batch*SeqLen, 256, 8, 8]
        features = features.flatten(2).transpose(1, 2)  # [Batch*SeqLen, 64, 256]
        features = self.proj(features)  # [Batch*SeqLen, 64, FeatureDim]

        # Average pool spatial tokens to get sequence representation per frame
        features = features.mean(dim=1)  # [Batch*SeqLen, FeatureDim]
        features = features.view(batch_size, seq_len, -1)  # [Batch, SeqLen, FeatureDim]

        # Apply spatio-temporal self-attention across frames
        temporal_features = self.temporal_transformer(
            features
        )  # [Batch, SeqLen, FeatureDim]

        # Compute output geometry states
        camera_matrices = self.camera_head(temporal_features)  # [Batch, SeqLen, 12]
        point_tracks = self.point_track_head(temporal_features)  # [Batch, SeqLen, 300]
        depth_predictions = self.depth_head(temporal_features)  # [Batch, SeqLen, 1]

        return {
            "features": temporal_features,  # [Batch, SeqLen, 768] (Fuses temporal-geometry tokens)
            "camera": camera_matrices,  # Camera rotation and position vectors
            "point_tracks": point_tracks,  # 3D points tracks
            "depth": depth_predictions,  # Estimated frame depths
        }
