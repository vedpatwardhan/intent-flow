import torch
import torch.nn as nn
import sys
import os

# Align paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.action_denoiser import CLAPFlowMatcher


class ComboStocFlowMatcher(CLAPFlowMatcher):
    """
    ComboStoc Action Denoiser.
    Extends CLAPFlowMatcher to support independent timesteps t_i for different dimensions,
    allowing local repair, flow reversal, and joint-level timeline rollbacks.
    """

    def __init__(self, action_dim=12, state_dim=512, hidden_dim=256, config=None):
        super().__init__(
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            config=config,
        )

    def get_cfm_loss(self, x_1, s_t, s_target, reduction="mean"):
        """
        Calculates CFM loss using independent timeline timesteps t_i for each joint
        with the official ComboStoc blending scheme to preserve sync coherence.
        """
        batch_size = x_1.size(0)
        x_0 = torch.randn_like(x_1)

        # 1. Sample unsynced independent timelines
        t_unsync = torch.rand(batch_size, self.action_dim, device=x_1.device)

        # 2. Sample synced uniform timeline
        t_sync = torch.rand(batch_size, 1, device=x_1.device).expand(
            -1, self.action_dim
        )

        # 3. Blend them using the ComboStoc blend scheme (blends uniform and unsynced)
        progress = torch.rand(batch_size, 1, device=x_1.device)
        t = t_sync * (1.0 - progress) + t_unsync * progress

        # Interpolate flow path along independent timesteps
        x_t = t * x_1 + (1.0 - t) * x_0
        target_velocity = x_1 - x_0

        # Pass independent time vector directly to the velocity field
        pred_velocity = self.velocity_field(x_t, t, s_t, s_target)

        loss_elementwise = (pred_velocity - target_velocity) ** 2

        if reduction == "none":
            return loss_elementwise
        return torch.mean(loss_elementwise)

    @torch.no_grad()
    def sample_with_steering(
        self,
        s_t,
        s_target,
        embodiment_id=None,
        horizon=8,
        num_steps=10,
        steering_timelines=None,
        step_nft_scale=0.0,
    ):
        batch_size = s_t.size(0)

        # [B, H, ActionDim]
        x_t = torch.randn(batch_size, horizon, self.action_dim, device=s_t.device)

        print(
            f"📊 [ODE Init] x_0 Standard Normal Bounds -> Min: {x_t.min().item():.4f} | Max: {x_t.max().item():.4f}"
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

        for i in range(num_steps):
            t_vals = steering_timelines + (i * dt)
            t_vals = torch.clamp(t_vals, min=0.0, max=1.0)

            # 1. Inspect raw velocity field network outputs
            v_t = self.velocity_field(x_t, t_vals, s_t, s_target, embodiment_id)

            v_min, v_max = v_t.min().item(), v_t.max().item()

            # 2. Compute the direct contribution of velocity to the step update
            v_step = v_t * dt

            # 3. Inspect Stochastic Noise Injection
            noise_min, noise_max = 0.0, 0.0
            if step_nft_scale > 0.0 and i < num_steps - 1:
                raw_noise = torch.randn_like(x_t)
                steerable_mask = (t_vals < 1.0).float()
                noise_step = raw_noise * step_nft_scale * steerable_mask
                noise_min, noise_max = noise_step.min().item(), noise_step.max().item()

                # Apply updates
                x_t = x_t + v_step + noise_step
            else:
                x_t = x_t + v_step

            print(
                f"   Step {i:02d} -> "
                f"State s_t Bounds: [{s_t.min().item():.4f}, {s_t.max().item():.4f}] | "
                f"State s_target Bounds: [{s_target.min().item():.4f}, {s_target.max().item():.4f}] | "
                f"Velocity Field Bounds: [{v_min:.4f}, {v_max:.4f}] | "
                f"Noise Step Bounds: [{noise_min:.4f}, {noise_max:.4f}] | "
                f"Resulting x_t Bounds: [{x_t.min().item():.4f}, {x_t.max().item():.4f}] | "
                f"Resulting t_vals Bounds: [{t_vals.min().item():.4f}, {t_vals.max().item():.4f}]"
            )

        return x_t

    @torch.no_grad()
    def sample_reverse(self, x_1, s_t, s_target, num_steps=10):
        """
        Flow Reversal Steering (FRS).
        Integrates backward along the learned vector field from t=1 (clean action) to t=0 (noise space).
        """
        batch_size = x_1.size(0)
        x_t = x_1.clone()
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = 1.0 - (i * dt)
            t = torch.full((batch_size, 1), t_val, device=x_1.device)
            if self.action_dim > 1:
                t = t.expand(-1, self.action_dim)

            # Predict velocity direction
            v_t = self.velocity_field(x_t, t, s_t, s_target)

            # Step backward: x_{t-dt} = x_t - v_t * dt
            x_t = x_t - v_t * dt

        return x_t
