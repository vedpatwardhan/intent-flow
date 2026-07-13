import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboStocTimeEmbedding(nn.Module):
    """Strict 3D grid time embedding layer with no fallback shapes."""

    def __init__(self, action_dim, time_dim):
        super().__init__()
        self.action_dim = action_dim
        self.time_dim = time_dim

        self.vectorized_proj1 = nn.Conv1d(
            in_channels=action_dim,
            out_channels=action_dim * (time_dim // 2),
            kernel_size=1,
            groups=action_dim,
        )
        self.act = nn.GELU()
        self.vectorized_proj2 = nn.Conv1d(
            in_channels=action_dim * (time_dim // 2),
            out_channels=action_dim * time_dim,
            kernel_size=1,
            groups=action_dim,
        )

    def forward(self, t):
        # Strict Input Contract: [B, H, action_dim]
        B, H, D = t.shape

        # Flatten sequence length into batch dimension to process via Conv1D channels
        t_input = t.reshape(B * H, D).unsqueeze(-1)  # [B*H, D, 1]

        h = self.vectorized_proj1(t_input)
        h = self.act(h)
        h = self.vectorized_proj2(h)  # [B*H, D * TimeDim, 1]

        # Reshape back to separate sequence and time features natively
        time_features = h.view(B, H, self.action_dim, self.time_dim)
        return time_features  # Output: [B, H, action_dim, time_dim]


class SpatialTemporalDiTBlock(nn.Module):
    """Strict hierarchical sequence block executing over explicit 3D layouts."""

    def __init__(self, joint_dim, cond_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(joint_dim, hidden_dim)
        self.mod_layer = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, joint_dim)

    def forward(self, x, cond):
        # x: [B, H, joint_dim], cond: [B, cond_dim]
        B, H, _ = x.shape

        h_x = self.in_proj(x)  # [B, H, hidden_dim]

        # Explicit unsqueeze and broadcast across sequence steps without generic dimensions
        modulation = self.mod_layer(cond).view(B, 1, -1).expand(-1, H, -1)

        h_modulated = self.norm1(h_x + modulation)
        attn_out, _ = self.temporal_attn(h_modulated, h_modulated, h_modulated)

        h_x = h_x + attn_out
        h_out = self.norm2(h_x + modulation)
        return x + self.out_proj(h_out)


class ActionVelocityField(nn.Module):
    """Continuous flow field network with deterministic sequence dimensions."""

    def __init__(
        self, action_dim=58, state_dim=512, time_dim=16, hidden_dim=256, config=None
    ):
        super().__init__()
        self.action_dim = action_dim
        self.time_mlp = ComboStocTimeEmbedding(action_dim=action_dim, time_dim=time_dim)

        self.embodiment_embedding = nn.Embedding(num_embeddings=3, embedding_dim=32)
        cond_dim = state_dim * 2 + action_dim * time_dim + 32

        self.block1 = SpatialTemporalDiTBlock(
            joint_dim=action_dim, cond_dim=cond_dim, hidden_dim=hidden_dim
        )
        self.block2 = SpatialTemporalDiTBlock(
            joint_dim=action_dim, cond_dim=cond_dim, hidden_dim=hidden_dim
        )
        self.block3 = SpatialTemporalDiTBlock(
            joint_dim=action_dim, cond_dim=cond_dim, hidden_dim=hidden_dim
        )

        self.out_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t, t, s_t, s_target, embodiment_id):
        B, H, _ = x_t.shape

        t_embed = self.time_mlp(t)  # [B, H, action_dim, time_dim]

        # Mean pool temporal coordinates across sequence window to unify conditioning context
        t_flat = t_embed.mean(dim=1).view(B, -1)  # [B, action_dim * time_dim]

        emb_feat = self.embodiment_embedding(embodiment_id)  # [B, 32]
        cond = torch.cat([s_t, s_target, t_flat, emb_feat], dim=-1)

        macro_anchors = self.block1(x_t, cond)
        motion_primitives = self.block2(macro_anchors, cond)
        joint_trajectory = self.block3(motion_primitives, cond)

        return self.out_net(joint_trajectory)


class CLAPFlowMatcher(nn.Module):
    def __init__(self, action_dim=58, state_dim=512, hidden_dim=256, config=None):
        super().__init__()
        self.velocity_field = ActionVelocityField(
            action_dim, state_dim, hidden_dim=hidden_dim, config=config
        )
        self.action_dim = action_dim

    def get_cfm_loss(self, x_1, s_t, s_target, embodiment_id):
        B, H, D = x_1.shape
        x_0 = torch.randn_like(x_1)

        # ComboStoc: Strict 3D noise time allocation matching targets exactly
        t = torch.rand(B, H, D, device=x_1.device)

        x_t = t * x_1 + (1.0 - t) * x_0
        target_velocity = x_1 - x_0

        pred_velocity = self.velocity_field(x_t, t, s_t, s_target, embodiment_id)
        return torch.mean((pred_velocity - target_velocity) ** 2)

    @torch.no_grad()
    def sample(self, s_t, s_target, embodiment_id=None, horizon=7, num_steps=10):
        B = s_t.size(0)
        if embodiment_id is None:
            embodiment_id = torch.ones(B, dtype=torch.long, device=s_t.device)
        x_t = torch.randn(B, horizon, self.action_dim, device=s_t.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((B, horizon, self.action_dim), t_val, device=s_t.device)
            v_t = self.velocity_field(x_t, t, s_t, s_target, embodiment_id)
            x_t = x_t + v_t * dt

        return x_t
