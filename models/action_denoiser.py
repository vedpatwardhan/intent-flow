import torch
import torch.nn as nn


class ComboStocTimeEmbedding(nn.Module):
    """
    Vectorized multi-dimensional time embedding layer that preserves individual
    joint timeline steps without structural python loop execution bottlenecks.
    """

    def __init__(self, action_dim, time_dim):
        super().__init__()
        self.action_dim = action_dim
        self.time_dim = time_dim

        # Parallel group projection: maps action_dim channels independently
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

        # Combined timing representation reduction layer
        self.reduction = nn.Linear(action_dim * time_dim, time_dim)

    def forward(self, t):
        # 1. Device and shape alignment wrapper
        if t.size(-1) == 1:
            t = t.repeat(1, self.action_dim)

        # Ensure correct shape configuration for grouped Conv1D processing: [B, ActionDim, 1]
        t_input = t.unsqueeze(-1)

        # 2. Fully vectorized parallel channel computation
        h = self.vectorized_proj1(t_input)
        h = self.act(h)
        h = self.vectorized_proj2(h)  # [B, ActionDim * TimeDim, 1]

        # Reshape and flatten channel features securely
        flat_embed = h.squeeze(-1)
        return self.reduction(flat_embed)


class HierarchicalDiTBlock(nn.Module):
    """
    A true hierarchical block that refines the continuous trajectory state
    conditioned on cross-modal tokens and multi-timeline coordinates.
    """

    def __init__(self, in_dim, cond_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.mod_layer = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, cond):
        h_x = self.in_proj(x)
        modulation = self.mod_layer(cond)
        h = self.norm(h_x + modulation)
        return x + self.out_proj(h)


class ActionVelocityField(nn.Module):
    """
    Hierarchical Block Diffusion Network with explicit sequential block chaining:
    - Block 1 (Macro Anchors) ──► Block 2 (Primitives) ──► Block 3 (Trajectories)
    """

    def __init__(
        self, action_dim=12, state_dim=512, time_dim=64, hidden_dim=256, config=None
    ):
        super().__init__()
        dit_config = config.get("stage2", {}) if config else {}
        h_dim_1 = dit_config.get("dit_hidden_dim_1", hidden_dim)
        h_dim_2 = dit_config.get("dit_hidden_dim_2", hidden_dim)
        h_dim_3 = dit_config.get("dit_hidden_dim_3", hidden_dim)

        self.time_mlp = ComboStocTimeEmbedding(action_dim=action_dim, time_dim=time_dim)
        cond_dim = state_dim * 2 + time_dim

        # Structural blocks mapping to EAR / IAR pipeline paradigms
        self.block1 = HierarchicalDiTBlock(
            in_dim=action_dim, cond_dim=cond_dim, hidden_dim=h_dim_1
        )
        self.block2 = HierarchicalDiTBlock(
            in_dim=action_dim, cond_dim=cond_dim, hidden_dim=h_dim_2
        )
        self.block3 = HierarchicalDiTBlock(
            in_dim=action_dim, cond_dim=cond_dim, hidden_dim=h_dim_3
        )

        self.out_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t, t, s_t, s_target):
        t_embed = self.time_mlp(t)
        cond = torch.cat([s_t, s_target, t_embed], dim=-1)

        # Autoregressive Chaining refinement loop
        macro_anchors = self.block1(x_t, cond)
        motion_primitives = self.block2(macro_anchors, cond)
        joint_trajectory = self.block3(motion_primitives, cond)

        return self.out_net(joint_trajectory)


class CLAPFlowMatcher(nn.Module):
    def __init__(self, action_dim=12, state_dim=512, hidden_dim=256, config=None):
        super().__init__()
        self.velocity_field = ActionVelocityField(
            action_dim, state_dim, hidden_dim=hidden_dim, config=config
        )
        self.action_dim = action_dim

    def get_cfm_loss(self, x_1, s_t, s_target):
        batch_size = x_1.size(0)
        x_0 = torch.randn_like(x_1)

        # ComboStoc independent sampling target layout
        t = torch.rand(batch_size, self.action_dim, device=x_1.device)

        # Asynchronous multi-timeline path interpolation
        x_t = t * x_1 + (1.0 - t) * x_0
        target_velocity = x_1 - x_0

        pred_velocity = self.velocity_field(x_t, t, s_t, s_target)
        return torch.mean((pred_velocity - target_velocity) ** 2)

    @torch.no_grad()
    def sample(self, s_t, s_target, num_steps=10):
        batch_size = s_t.size(0)
        # Ensure base noise starts on identical hardware device target
        x_t = torch.randn(batch_size, self.action_dim, device=s_t.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            # Safety Anchoring: Ensure time vectors explicitly lock onto identical active hardware devices
            t = torch.full((batch_size, 1), t_val, device=s_t.device)
            v_t = self.velocity_field(x_t, t, s_t, s_target)
            x_t = x_t + v_t * dt

        return x_t
