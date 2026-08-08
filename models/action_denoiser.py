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
        # x: [B, H, action_dim], cond: [B, cond_dim]
        B, H, _ = x.shape

        h_x = self.in_proj(x)  # [B, H, hidden_dim]

        # Explicit unsqueeze and broadcast across sequence steps without generic dimensions
        # modulation: [B, H, hidden_dim]
        modulation = self.mod_layer(cond).view(B, 1, -1).expand(-1, H, -1)

        h_modulated = self.norm1(h_x + modulation)
        attn_out, _ = self.temporal_attn(h_modulated, h_modulated, h_modulated)

        h_x = h_x + attn_out
        h_out = self.norm2(h_x + modulation)
        return x + self.out_proj(h_out)  # [B, H, action_dim]


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

        emb_feat = self.embodiment_embedding(embodiment_id)  # [B, emb_dim]

        # [B, state_dim + state_dim + action_dim * time_dim + emb_dim]
        cond = torch.cat([s_t, s_target, t_flat, emb_feat], dim=-1)

        macro_anchors = self.block1(x_t, cond)  # [B, H, action_dim]
        motion_primitives = self.block2(macro_anchors, cond)  # [B, H, action_dim]
        joint_trajectory = self.block3(motion_primitives, cond)  # [B, H, action_dim]

        return self.out_net(joint_trajectory)  # [B, H, action_dim]


class ComboStocFlowMatcher(nn.Module):
    """
    ComboStoc Action Denoiser.
    Supports independent timesteps t_i for different dimensions,
    allowing local repair, flow reversal, and joint-level timeline rollbacks.
    """

    def __init__(self, action_dim=58, state_dim=512, hidden_dim=256, config=None):
        super().__init__()
        self.velocity_field = ActionVelocityField(
            action_dim, state_dim, hidden_dim=hidden_dim, config=config
        )
        self.action_dim = action_dim

    def get_cfm_loss(self, x_1, s_t, s_target, embodiment_id=None, reduction="mean"):
        """
        Calculates CFM loss using independent timeline timesteps t_i for each joint
        with the official ComboStoc blending scheme to preserve sync coherence.
        """
        batch_size = x_1.size(0)
        horizon = x_1.size(1)
        x_0 = torch.randn_like(x_1)

        # 1. Sample unsynced independent timelines
        t_unsync = torch.rand(batch_size, horizon, self.action_dim, device=x_1.device)

        # 2. Sample synced uniform timeline
        t_sync = torch.rand(batch_size, horizon, 1, device=x_1.device).expand(
            -1, -1, self.action_dim
        )

        # 3. Blend them using the ComboStoc blend scheme
        progress = torch.rand(batch_size, 1, 1, device=x_1.device)
        t = t_sync * (1.0 - progress) + t_unsync * progress

        # Interpolate flow path along independent timesteps
        x_t = t * x_1 + (1.0 - t) * x_0
        target_velocity = x_1 - x_0

        # Pass independent time vector directly to the velocity field
        pred_velocity = self.velocity_field(x_t, t, s_t, s_target, embodiment_id)

        loss_elementwise = (pred_velocity - target_velocity) ** 2

        if reduction == "none":
            return loss_elementwise
        return torch.mean(loss_elementwise)

    def evaluate_step_transitions(
        self, x_t, t_sample, s_t, s_target, embodiment_id=None
    ):
        return self.velocity_field(x_t, t_sample, s_t, s_target, embodiment_id)

    def compute_stepwise_denoising_deltas(self, x_t, target_velocity, t_vals, dt):
        return x_t + target_velocity * dt

    @torch.no_grad()
    def sample_with_steering(
        self,
        s_t,  # [B, latent_dim]
        s_target,  # [B, latent_dim]
        embodiment_id=None,  # [B]
        horizon=8,
        num_steps=10,
        steering_timelines=None,  # [B, H * action_dim]
        stochastic_steer_scale=0.0,
        seed=42,
    ):
        generator = torch.Generator(device=s_t.device).manual_seed(seed)
        batch_size = s_t.size(0)

        # Force padding channels (joints 32..57) to zero out entirely in action space
        action_mask = torch.zeros(1, 1, self.action_dim, device=s_t.device)
        action_mask[..., :32] = 1.0

        init_sigma = 0.4
        x_t = (
            torch.randn(
                batch_size,
                horizon,
                self.action_dim,
                device=s_t.device,
                generator=generator,
            )
            * init_sigma
        ) * action_mask

        print(
            "📊 [ODE Init] x_0 Standard Normal Bounds -> "
            f"Min: {x_t.min().item():.4f} | Max: {x_t.max().item():.4f}"
        )

        if steering_timelines is None:
            steering_timelines = torch.zeros(
                batch_size, horizon, self.action_dim, device=s_t.device
            )
        else:
            steering_timelines = steering_timelines.view(
                batch_size, horizon, self.action_dim
            )

        dt = 1.0 / num_steps

        step_snrs = []
        for i in range(num_steps):
            t_vals = steering_timelines + (i * dt)
            t_vals = torch.clamp(t_vals, min=0.0, max=1.0)

            v_t = self.velocity_field(x_t, t_vals, s_t, s_target, embodiment_id)
            v_step = (v_t * dt) * action_mask

            if stochastic_steer_scale > 0.0 and i < num_steps - 1:
                raw_noise = torch.randn(
                    x_t.shape, device=x_t.device, generator=generator
                )
                steerable_mask = (t_vals < 1.0).float()
                noise_step = (
                    raw_noise * stochastic_steer_scale * steerable_mask
                ) * action_mask

                x_t = (x_t + v_step + noise_step) * action_mask
            else:
                x_t = (x_t + v_step) * action_mask

            with torch.no_grad():
                if stochastic_steer_scale > 0.0 and i < num_steps - 1:
                    raw_v_mag = v_t.abs().mean().item()
                    raw_noise_mag = (
                        (raw_noise * stochastic_steer_scale).abs().mean().item()
                    )
                    step_snr = raw_v_mag / (raw_noise_mag + 1e-8)
                else:
                    step_snr = None
                step_snrs.append(step_snr)

        return x_t, step_snrs
