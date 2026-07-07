import torch
import torch.nn as nn
import torch.nn.functional as F


class BadWorldAttacker:
    """
    BadWorld Perceptual Attacker.
    Optimizes a perturbation vector applied strictly to the context state representation (s_t)
    within the Critical Subspace Mask (M) to maximize disruption of early denoising dynamics (t -> 0).
    """

    def __init__(self, action_dim=58, state_dim=512, perturb_lr=0.02):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.perturb_lr = perturb_lr

    def generate_perturbed_context(
        self, flow_matcher, s_t, s_target, original_action, mask=None, num_steps=5
    ):
        """
        Runs the minimax optimization loop to find the worst-case state representation perturbation.
        mask: [Batch, StateDim] - restricts perturbation to target subspace (e.g., visual coordinates).
        """
        # If no mask is passed, default to target the visual subspace (first 384 dimensions of the 512 state)
        if mask is None:
            mask = torch.zeros_like(s_t)
            mask[:, :384] = 1.0  # Focus strictly on visual token coordinates

        # Initialize state perturbation vector
        perturb = torch.zeros_like(s_t, requires_grad=True)

        for _ in range(num_steps):
            # Apply masked perturbation to the context state
            perturbed_s_t = s_t + mask * perturb

            # Sample early denoising timesteps (t -> 0)
            t = torch.rand(s_t.size(0), 1, device=s_t.device) * 0.2

            # Interpolate flow path
            x_0 = torch.randn_like(original_action)
            x_t = t * original_action + (1.0 - t) * x_0
            target_vel = original_action - x_0

            # Predict velocity under perturbed context
            pred_vel = flow_matcher.velocity_field(x_t, t, perturbed_s_t, s_target)

            # Maximize CFM prediction error
            error_loss = torch.mean((pred_vel - target_vel) ** 2)

            # Get gradients w.r.t the perturbation
            grads = torch.autograd.grad(error_loss, perturb)[0]

            # Gradient ascent: maximize error
            perturb = perturb.detach() + self.perturb_lr * grads.sign()
            perturb.requires_grad = True

        # Return final perturbed context representation
        return (s_t + mask * perturb).detach()
