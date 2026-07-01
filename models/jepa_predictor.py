import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LatentActionEncoder(nn.Module):
    """
    Temporary training-only helper (h_ψ) that extracts motion latents from
    state transitions (s_t, s_t+1) during Stage 1 pre-training.
    """

    def __init__(self, state_dim=512, bottleneck_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, bottleneck_dim),
        )

    def forward(self, s_t, s_next):
        # Concatenate current and next state features along channel dimension
        x = torch.cat([s_t, s_next], dim=-1)
        return self.net(x)


# --- Transformer Components matching le-probe architecture ---


def modulate(x, shift, scale):
    """AdaLN-zero modulation helper"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking support"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning mapping actions"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # c is the action conditioning embedding: [B, T, D] or [B, D]
        # Match dimensions for sequence-level AdaLN
        if c.dim() == 2:
            c = c.unsqueeze(1)

        # Split modulation parameters along last dimension
        mods = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods

        # Perform self-attention and MLP with AdaLN modulations
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# --- 6-Layer JepaPredictor Transformer ---


class JepaPredictor(nn.Module):
    """
    Dynamics Predictor (g_θ) utilizing a 6-layer Conditional Transformer
    with AdaLN-zero conditioning (matches le-probe capacity).
    """

    def __init__(
        self,
        state_dim=512,
        action_dim=16,
        hidden_dim=512,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Action Projection to map arbitrary action dimensionalities (e.g. 16 or 512) to condition dimension
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        # Positional Embeddings for sequence horizon reasoning
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, hidden_dim))

        # 6 layers of ConditionalBlocks
        self.layers = nn.ModuleList(
            [
                ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, s_t, action):
        """
        s_t: [Batch, StateDim] or [Batch, Horizon, StateDim]
        action: [Batch, ActionDim] or [Batch, Horizon, ActionDim]
        """
        # Ensure x is sequence-shaped: [B, T, D]
        is_2d_state = s_t.dim() == 2
        if is_2d_state:
            s_t = s_t.unsqueeze(1)

        is_2d_action = action.dim() == 2
        if is_2d_action:
            action = action.unsqueeze(1)

        B, T, D = s_t.shape

        # 1. Project action features
        c = self.action_proj(action)  # [B, T, hidden_dim]

        # 2. Add position embedding
        x = s_t + self.pos_embedding[:, :T]

        # 3. Propagate through 6 Transformer Layers
        for layer in self.layers:
            x = layer(x, c)

        x = self.norm(x)
        out = self.output_proj(x)

        if is_2d_state:
            out = out.squeeze(1)

        return out

    def rollout(self, s_t, actions_seq):
        """
        Autoregressive multi-step prediction over horizon H.
        actions_seq: [Batch, Horizon, ActionDim]
        """
        horizon = actions_seq.size(1)
        predictions = []
        curr_state = s_t

        for k in range(horizon):
            action_k = actions_seq[:, k, :]
            next_state_pred = self.forward(curr_state, action_k)
            predictions.append(next_state_pred)
            curr_state = next_state_pred

        # Stack along horizon dimension -> [Batch, Horizon, StateDim]
        return torch.stack(predictions, dim=1)
