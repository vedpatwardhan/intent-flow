import torch
import torch.nn as nn


class MultiStreamActionTransformer(nn.Module):
    """
    Fuses multiple pre-aligned modality tokens using a Multi-Stream Transformer architecture.
    """

    def __init__(self, latent_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        # Learnable modality indicators/embeddings
        self.modality_embeddings = nn.ParameterDict(
            {
                "text": nn.Parameter(torch.randn(1, 1, latent_dim)),
                "vision": nn.Parameter(torch.randn(1, 1, latent_dim)),
                "pointnext": nn.Parameter(torch.randn(1, 1, latent_dim)),
                "tactile": nn.Parameter(torch.randn(1, 1, latent_dim)),
                "proprioception": nn.Parameter(torch.randn(1, 1, latent_dim)),
            }
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(latent_dim)

    def forward(self, modality_dict):
        """
        modality_dict: Dict of Tensors with shape [Batch, SequenceLen, LatentDim]
        """
        batch_size = next(iter(modality_dict.values())).size(0)
        tokens_list = []

        for key, tokens in modality_dict.items():
            if tokens is None:
                continue
            # Ensure tokens are 3D: [Batch, Len, Dim]
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(1)

            # Add modality specific indicator bias
            mod_indicator = self.modality_embeddings[key].expand(
                batch_size, tokens.size(1), -1
            )
            tokens = tokens + mod_indicator
            tokens_list.append(tokens)

        # Concatenate all streams along the sequence dimension
        fused_sequence = torch.cat(
            tokens_list, dim=1
        )  # [Batch, TotalSeqLen, LatentDim]

        # Process through transformer
        transformed = self.transformer(fused_sequence)

        # Pool to construct single state representation (e.g. mean pooling over sequence)
        state_representation = self.ln_out(transformed.mean(dim=1))

        return state_representation
