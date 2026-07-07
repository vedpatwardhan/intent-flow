import torch
import torch.nn as nn


class DiTBlock(nn.Module):
    """
    A simple Diffusion Transformer-style (DiT) block used as a component of
    our Hierarchical Block Diffusion network. Optionally accepts a side condition
    to allow sequential block dependency.
    """

    def __init__(self, in_dim, cond_dim, hidden_dim, side_cond_dim=None):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.side_cond_proj = None
        if side_cond_dim is not None:
            self.side_cond_proj = nn.Linear(side_cond_dim, hidden_dim)

        self.mod_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, cond, side_cond=None):
        h_x = self.in_proj(x)
        h_cond = self.cond_proj(cond)

        if self.side_cond_proj is not None and side_cond is not None:
            h_side = self.side_cond_proj(side_cond)
            h_cond = h_cond + h_side

        # Adaptive modulation
        modulation = self.mod_layer(h_cond)
        h = self.norm(h_x + h_cond) * (1.0 + modulation)
        return x + self.out_proj(h)


class ActionVelocityField(nn.Module):
    """
    Hierarchical Block Diffusion network with sequential autoregressive dependency across blocks:
    - Block 1 (High-Level): Macro subgoal conditioning
    - Block 2 (Mid-Level): Action primitives (conditioned on Block 1 output)
    - Block 3 (Low-Level): Continuous joint trajectory generation (conditioned on Block 2 output)
    """

    def __init__(
        self, action_dim=12, state_dim=512, time_dim=64, hidden_dim=256, config=None
    ):
        super().__init__()
        # Configure block sizes from config if provided, otherwise default
        dit_config = config.get("stage2", {}) if config else {}
        h_dim_1 = dit_config.get("dit_hidden_dim_1", hidden_dim)
        h_dim_2 = dit_config.get("dit_hidden_dim_2", hidden_dim)
        h_dim_3 = dit_config.get("dit_hidden_dim_3", hidden_dim)

        # Sinusoidal/Linear time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        # ComboStoc: timeline projection layer initialized to average timesteps at step 0
        self.combostoc_time_proj = nn.Linear(action_dim, 1)
        nn.init.constant_(self.combostoc_time_proj.weight, 1.0 / action_dim)
        nn.init.constant_(self.combostoc_time_proj.bias, 0.0)

        # Combined conditioning dimension: s_t (state_dim) + s_target (state_dim) + time (time_dim)
        cond_dim = state_dim * 2 + time_dim

        # Hierarchical Block 1 (High-Level)
        self.block1 = DiTBlock(in_dim=action_dim, cond_dim=cond_dim, hidden_dim=h_dim_1)

        # Hierarchical Block 2 (Mid-Level) - Conditioned on Block 1 output
        self.block2 = DiTBlock(
            in_dim=action_dim,
            cond_dim=cond_dim,
            hidden_dim=h_dim_2,
            side_cond_dim=action_dim,
        )

        # Hierarchical Block 3 (Low-Level) - Conditioned on Block 2 output
        self.block3 = DiTBlock(
            in_dim=action_dim,
            cond_dim=cond_dim,
            hidden_dim=h_dim_3,
            side_cond_dim=action_dim,
        )

        # Output projection to continuous action space
        self.out_net = nn.Sequential(
            nn.Linear(action_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t, t, s_t, s_target):
        """
        x_t: [Batch, ActionDim] (current noisy action estimate)
        t: [Batch, 1] or [Batch, ActionDim] (current flow matching time step)
        s_t: [Batch, StateDim] (current context state embedding)
        s_target: [Batch, StateDim] (target configuration state embedding)
        """
        # ComboStoc: Handle multi-dimensional time vectors using pre-initialized projection
        if t.size(-1) > 1:
            t_input = self.combostoc_time_proj(t)
        else:
            t_input = t

        t_embed = self.time_mlp(t_input)
        # Joint conditioning vector
        cond = torch.cat([s_t, s_target, t_embed], dim=-1)

        # Run hierarchical blocks with sequential dependency
        high_level_anchor = self.block1(x_t, cond)
        mid_level_anchor = self.block2(x_t, cond, side_cond=high_level_anchor)
        low_level_trajectory = self.block3(x_t, cond, side_cond=mid_level_anchor)

        # Hierarchical action embedding (Concatenation)
        hierarchical_embeddings = torch.cat(
            [high_level_anchor, mid_level_anchor, low_level_trajectory], dim=-1
        )

        return self.out_net(hierarchical_embeddings)


class CLAPFlowMatcher(nn.Module):
    """
    CLAP-RF Rectified Flow controller updated to support dual-state target conditioning
    and Hierarchical Block Diffusion networks.
    """

    def __init__(self, action_dim=12, state_dim=512, hidden_dim=256, config=None):
        super().__init__()
        self.velocity_field = ActionVelocityField(
            action_dim, state_dim, hidden_dim=hidden_dim, config=config
        )
        self.action_dim = action_dim

    def get_cfm_loss(self, x_1, s_t, s_target):
        """
        Calculates the Conditional Flow Matching (CFM) loss.
        x_1: [Batch, ActionDim] (expert actions)
        s_t: [Batch, StateDim] (current state)
        s_target: [Batch, StateDim] (target state)
        """
        batch_size = x_1.size(0)

        # Sample random noise x_0
        x_0 = torch.randn_like(x_1)

        # Sample random time step t in [0, 1]
        t = torch.rand(batch_size, 1, device=x_1.device)

        # Flow interpolation path (Rectified Flow)
        x_t = t * x_1 + (1.0 - t) * x_0

        # Target velocity vector field (dx_t / dt = x_1 - x_0)
        target_velocity = x_1 - x_0

        # Predict velocity with dual-state conditioning
        pred_velocity = self.velocity_field(x_t, t, s_t, s_target)

        # MSE loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, s_t, s_target, num_steps=10):
        """
        Performs Euler integration of the learned velocity field from t=0 to t=1.
        s_t: [Batch, StateDim] (context representation)
        s_target: [Batch, StateDim] (target configuration)
        """
        batch_size = s_t.size(0)
        x_t = torch.randn(batch_size, self.action_dim, device=s_t.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((batch_size, 1), t_val, device=s_t.device)

            # Predict velocity at step t
            v_t = self.velocity_field(x_t, t, s_t, s_target)

            # Euler update step
            x_t = x_t + v_t * dt

        return x_t
