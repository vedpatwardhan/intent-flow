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
                "vggt": nn.Parameter(torch.randn(1, 1, latent_dim)),
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
        active_keys = ["vision", "vggt", "proprioception"]

        for key in active_keys:
            if key not in modality_dict or modality_dict[key] is None:
                continue
            tokens = modality_dict[key]  # [Batch, Len, Dim]

            # Unified Modality Bounding System
            if key == "vggt":
                v_norm = tokens.abs().mean(dim=-1, keepdim=True) + 1e-8
                tokens = tokens * torch.clamp(0.85 / v_norm, min=1.0)

            if key == "proprioception":
                p_norm = tokens.abs().mean(dim=-1, keepdim=True) + 1e-8
                tokens = tokens * torch.clamp(0.65 / p_norm, min=1.0)

            if key == "vision":
                d_norm = tokens.abs().mean(dim=-1, keepdim=True) + 1e-8
                tokens = tokens * torch.clamp(0.60 / d_norm, max=1.0)

            # Add modality specific indicator bias
            mod_indicator = self.modality_embeddings[key].expand(
                batch_size, tokens.size(1), -1
            )
            print(
                f"Tokens shape: {tokens.shape}, "
                f"modality_emb shape: {self.modality_embeddings[key].shape}, "
                f"modality indicator shape: {mod_indicator.shape}"
            )
            tokens = tokens + mod_indicator
            tokens_list.append(tokens)

        # Concatenate all streams along the sequence dimension
        # [Batch, TotalSeqLen, LatentDim]
        fused_sequence = torch.cat(tokens_list, dim=1)
        print(f"fused_sequence shape: {fused_sequence.shape}")

        # Process through transformer
        transformed = self.transformer(fused_sequence)
        print(f"transformed shape: {transformed.shape}")

        # --- NEW REGISTRATION BLOCK ---
        with torch.no_grad():
            profiles = {}
            current_idx = 0
            for key in active_keys:
                if key not in modality_dict or modality_dict[key] is None:
                    continue
                tokens = modality_dict[key]
                seq_len = tokens.size(1) if tokens.dim() == 3 else 1
                segment_states = transformed[:, current_idx : current_idx + seq_len, :]
                profiles[f"modality_weight/{key}"] = segment_states.abs().mean().item()
                current_idx += seq_len
            # Cache directly on the module instance
            self.last_modality_profile = profiles
        # ------------------------------

        # Pool to construct single state representation (e.g. mean pooling over sequence)
        state_representation = self.ln_out(transformed.mean(dim=1))

        return state_representation
