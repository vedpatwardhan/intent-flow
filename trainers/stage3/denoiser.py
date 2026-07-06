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

    def get_cfm_loss(self, x_1, s_t, s_target):
        """
        Calculates CFM loss using independent timeline timesteps t_i for each joint.
        """
        batch_size = x_1.size(0)
        x_0 = torch.randn_like(x_1)

        # ComboStoc: Sample independent t for each dimension/joint group [Batch, ActionDim]
        t = torch.rand(batch_size, self.action_dim, device=x_1.device)

        # Interpolate flow path along independent timesteps
        x_t = t * x_1 + (1.0 - t) * x_0
        target_velocity = x_1 - x_0

        # Collapse t to single-dimension average for context time_mlp embedding inside ActionVelocityField
        t_avg = t.mean(dim=-1, keepdim=True)
        pred_velocity = self.velocity_field(x_t, t_avg, s_t, s_target)

        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        return loss

    @torch.no_grad()
    def sample_with_steering(
        self, s_t, s_target, num_steps=10, steering_timelines=None
    ):
        """
        ComboStoc Sampling.
        Performs Euler integration from independent starting timesteps.
        steering_timelines: [Batch, ActionDim] (starting noise timesteps)
        """
        batch_size = s_t.size(0)
        x_t = torch.randn(batch_size, self.action_dim, device=s_t.device)

        # If no custom timelines are passed, run standard uniform time steps
        if steering_timelines is None:
            steering_timelines = torch.zeros(
                batch_size, self.action_dim, device=s_t.device
            )

        dt = 1.0 / num_steps

        for i in range(num_steps):
            # Advance each joint along its specific timeline offset
            t_vals = steering_timelines + (i * dt)
            t_vals = torch.clamp(t_vals, 0.0, 1.0)

            t_avg = t_vals.mean(dim=-1, keepdim=True)
            v_t = self.velocity_field(x_t, t_avg, s_t, s_target)

            x_t = x_t + v_t * dt

        return x_t
