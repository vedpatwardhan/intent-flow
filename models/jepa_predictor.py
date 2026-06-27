import torch
import torch.nn as nn


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


class JepaPredictor(nn.Module):
    """
    Dynamics Predictor (g_θ) utilizing Bilinear Gating to prevent predictor collapse
    by forcing mathematical dependency on the action input channel.
    """

    def __init__(self, state_dim=512, action_dim=512, hidden_dim=512):
        super().__init__()
        # State path projection
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        # Action path projection
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, state_dim), nn.LayerNorm(state_dim)
        )

    def forward(self, s_t, action):
        """
        s_t: [Batch, StateDim]
        action: [Batch, ActionDim] (can be either latent action z or physical joint action)
        """
        h_s = self.state_proj(s_t)
        h_a = self.action_proj(action)

        # Bilinear gating via Hadamard product
        gated = h_s * h_a

        return self.output_proj(gated)

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
