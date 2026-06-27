import torch
import torch.nn as nn


class ActionVelocityField(nn.Module):
    """
    MLP-based vector field network v_θ(x_t, t, s_t, z_task) for Flow Matching.
    """

    def __init__(self, action_dim=12, state_dim=512, time_dim=64, hidden_dim=256):
        super().__init__()
        # Sinusoidal embedding for diffusion time step t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        # Main vector field layers
        self.net = nn.Sequential(
            nn.Linear(action_dim + state_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_t, t, s_t):
        """
        x_t: [Batch, ActionDim] (current action trajectory estimate at flow step t)
        t: [Batch, 1] (flow step scalar t in [0, 1])
        s_t: [Batch, StateDim] (fused context state token from MSAT, including task conditioning)
        """
        t_embed = self.time_mlp(t)
        inputs = torch.cat([x_t, t_embed, s_t], dim=-1)
        return self.net(inputs)


class CLAPFlowMatcher(nn.Module):
    """
    CLAP-RF Rectified Flow policy controller handling CFM training and ODE sampling.
    """

    def __init__(self, action_dim=12, state_dim=512, hidden_dim=256):
        super().__init__()
        self.velocity_field = ActionVelocityField(
            action_dim, state_dim, hidden_dim=hidden_dim
        )
        self.action_dim = action_dim

    def get_cfm_loss(self, x_1, s_t):
        """
        Calculates the Conditional Flow Matching (CFM) loss.
        x_1: [Batch, ActionDim] (target ground-truth actions)
        s_t: [Batch, StateDim] (fused context state)
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

        # Predict velocity
        pred_velocity = self.velocity_field(x_t, t, s_t)

        # MSE loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, s_t, num_steps=10):
        """
        Performs Euler integration of the learned velocity field from t=0 to t=1.
        s_t: [Batch, StateDim] (context representation)
        """
        batch_size = s_t.size(0)
        x_t = torch.randn(batch_size, self.action_dim, device=s_t.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            # Current time step t
            t_val = i * dt
            t = torch.full((batch_size, 1), t_val, device=s_t.device)

            # Predict velocity at step t
            v_t = self.velocity_field(x_t, t, s_t)

            # Euler update step
            x_t = x_t + v_t * dt

        return x_t
