import torch
import torch.nn as nn
import torch.nn.functional as F


class BadWorldAttacker:
    """
    BadWorld Minimax Attacker.
    Identifies worst-case perturbations to maximize early denoising disruption,
    generating challenging training boundaries for the flow matcher.
    """

    def __init__(self, action_dim=12, perturb_lr=0.05):
        self.action_dim = action_dim
        self.perturb_lr = perturb_lr

    def generate_perturbation(
        self, flow_matcher, s_t, s_target, original_action, num_steps=5
    ):
        """
        Runs an inner minimax optimization loop to find the worst-case perturbation vector
        which maximizes the flow matcher's prediction error.
        """
        # Initialize perturbation force
        perturb = torch.zeros_like(original_action, requires_grad=True)

        for _ in range(num_steps):
            # Calculate perturbed action
            perturbed_action = original_action + perturb

            # Predict velocity under perturbation
            t = torch.rand(original_action.size(0), 1, device=original_action.device)
            x_t = t * perturbed_action + (1.0 - t) * torch.randn_like(perturbed_action)

            # Target is the clean direction
            target_vel = perturbed_action - torch.randn_like(perturbed_action)

            # Denoising velocity w.r.t context
            pred_vel = flow_matcher.velocity_field(x_t, t, s_t, s_target)

            # Loss is standard MSE prediction error (we want to MAXIMIZE this error)
            error_loss = torch.mean((pred_vel - target_vel) ** 2)

            # Backprop to get gradients of error w.r.t the perturbation
            grads = torch.autograd.grad(error_loss, perturb)[0]

            # Maximize error (Gradient Ascent)
            perturb = perturb.detach() + self.perturb_lr * grads.sign()
            perturb.requires_grad = True

        return perturb.detach()
