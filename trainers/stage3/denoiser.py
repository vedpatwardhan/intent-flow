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

    def get_cfm_loss(self, x_1, s_t, s_target, embodiment_id=None, reduction="mean"):
        """
        Calculates CFM loss using independent timeline timesteps t_i for each joint
        with the official ComboStoc blending scheme to preserve sync coherence.
            x_1 - [ensemble_size, horizon, action_dim]
            s_t - [ensemble_size, latent_dim]
            s_target - [ensemble_size, latent_dim]
        """
        batch_size = x_1.size(0)
        horizon = x_1.size(1)
        x_0 = torch.randn_like(x_1)

        # 1. Sample unsynced independent timelines
        # [ensemble_size, horizon, action_dim]
        t_unsync = torch.rand(batch_size, horizon, self.action_dim, device=x_1.device)

        # 2. Sample synced uniform timeline
        # [ensemble_size, horizon, action_dim]
        t_sync = torch.rand(batch_size, horizon, 1, device=x_1.device).expand(
            -1, -1, self.action_dim
        )

        # 3. Blend them using the ComboStoc blend scheme (blends uniform and unsynced)
        # [ensemble_size, 1, 1]
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
        """
        Evaluates velocity field predictions at intermediate flow timesteps t_sample.
        """
        return self.velocity_field(x_t, t_sample, s_t, s_target, embodiment_id)

    def compute_stepwise_denoising_deltas(self, x_t, target_velocity, t_vals, dt):
        """
        Calculates forward step projection target matching linear flow ODE step math: x_{t+dt} = x_t + v * dt.
        """
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
        step_nft_scale=0.0,
    ):
        batch_size = s_t.size(0)

        # Create embodiment-aware action mask (first 32 GR-1 active joints)
        action_mask = torch.zeros(1, 1, self.action_dim, device=s_t.device)
        action_mask[..., :32] = 1.0

        # [B, H, action_dim] initialized with 0.0s for padding channels
        init_sigma = 0.2
        x_t = (
            torch.randn(batch_size, horizon, self.action_dim, device=s_t.device)
            * init_sigma
        ) * action_mask

        print(
            f"📊 [ODE Init] x_0 Standard Normal Bounds -> Min: {x_t.min().item():.4f} | Max: {x_t.max().item():.4f}"
        )

        # [B, H, action_dim]
        if steering_timelines is None:
            steering_timelines = torch.zeros(
                batch_size, horizon, self.action_dim, device=s_t.device
            )
        else:
            steering_timelines = steering_timelines.view(
                batch_size, horizon, self.action_dim
            )

        dt = 1.0 / num_steps  # 0.1

        step_snrs = []
        for i in range(num_steps):
            # t_vals refers to current time of the values, target time is 1
            t_vals = steering_timelines + (i * dt)
            t_vals = torch.clamp(t_vals, min=0.0, max=1.0)

            # 1. Inspect raw velocity field network outputs
            # [B, H, action_dim]
            v_t = self.velocity_field(x_t, t_vals, s_t, s_target, embodiment_id)

            # 2. Compute the direct contribution of velocity to the step update (zero out padding)
            v_step = (v_t * dt) * action_mask

            # 3. Inspect Stochastic Noise Injection
            noise_min, noise_max = 0.0, 0.0
            if step_nft_scale > 0.0 and i < num_steps - 1:
                raw_noise = torch.randn_like(x_t)
                steerable_mask = (t_vals < 1.0).float()  # contains 1s if grid is 0s
                noise_step = (raw_noise * step_nft_scale * steerable_mask) * action_mask
                noise_min, noise_max = noise_step.min().item(), noise_step.max().item()

                # Apply updates and clamp trailing padding channels to zero
                x_t = (x_t + v_step + noise_step) * action_mask
            else:
                x_t = (x_t + v_step) * action_mask

            # Compute step-level SNR for remote endpoint telemetry logging
            with torch.no_grad():
                v_mag = v_step.abs().mean().item()
                if step_nft_scale > 0.0 and i < num_steps - 1:
                    noise_mag = noise_step.abs().mean().item()
                    step_snr = v_mag / (noise_mag + 1e-8)
                else:
                    step_snr = None  # Deterministic step (no noise injected)
                step_snrs.append(step_snr)

            if i == 0 or (i + 1) % (num_steps // 2) == 0:
                print(
                    f"   Step {i:02d} Bounds -> "
                    f"velocity: [{v_t.min().item():.3f}, {v_t.max().item():.3f}] | "
                    f"noise: [{noise_min:.3f}, {noise_max:.3f}] | "
                    f"x_t: [{x_t.min().item():.3f}, {x_t.max().item():.3f}] | "
                    f"t_vals: [{t_vals.min().item():.3f}, {t_vals.max().item():.3f}] | "
                    f"s_t: [{s_t.min().item():.3f}, {s_t.max().item():.3f}] | "
                    f"s_target: [{s_target.min().item():.3f}, "
                    f"{s_target.max().item():.3f}]"
                )

        return x_t, step_snrs

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
