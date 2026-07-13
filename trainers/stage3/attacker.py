import torch
import torch.nn as nn
import torch.nn.functional as F


class BadWorldAttacker:
    def __init__(
        self, action_dim=58, state_dim=512, perturb_lr=0.02, action_search_lr=0.01
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.perturb_lr = perturb_lr
        self.action_search_lr = action_search_lr

    def generate_stochastic_ensemble_pass(
        self,
        flow_matcher,
        s_t,
        s_target,
        original_action,
        mask=None,
        ensemble_size=4,
        outer_steps=5,
        inner_steps=3,
    ):
        """
        Runs a parallelized batch minimax attack to generate an ensemble cloud of
        K distinct worst-case fuzzed state latents by perturbing s_t directly.

        Returns:
            A tensor of shape [ensemble_size, state_dim] holding the parallel adversarial cloud.
        """
        # 1. Expand the baseline latents and actions to match the parallel ensemble size [K, Dimension]
        s_t_expanded = s_t.expand(ensemble_size, -1)  # [K, 512]
        s_target_expanded = s_target.expand(ensemble_size, -1)  # [K, 512]
        action_expanded = original_action.expand(ensemble_size, -1)  # [K, 58]

        # If a mask is provided, we can restrict which latent space dimensions are fuzzed,
        # otherwise we fuzz the overall encoder latent space unconditionally (all 1s)
        if mask is None:
            mask = torch.ones_like(s_t_expanded)  # Shape: [K, 512]
        else:
            mask = mask.expand(ensemble_size, -1)

        # 2. Seed unique stochastic initialization offsets across each ensemble slot
        # This breaks symmetry and ensures each track optimizes a completely unique attack vector
        perturb = (torch.randn_like(s_t_expanded) * 0.05).requires_grad_(True)

        for _ in range(outer_steps):
            # --- INNER LOOP: Parallel Action Search (Min) ---
            search_action = action_expanded.clone().detach().requires_grad_(True)

            for _ in range(inner_steps):
                # Apply current parallel perturbation directly to the overall latent space
                perturbed_s_t = s_t_expanded + mask * perturb

                # Sample independent early-denoising trajectory slices per ensemble track
                t = torch.rand(ensemble_size, 1, device=s_t.device) * 0.2
                x_0 = torch.randn_like(search_action)
                x_t = t * search_action + (1.0 - t) * x_0
                target_vel = search_action - x_0

                # Predict velocity under fuzzed ensemble features
                pred_vel = flow_matcher.velocity_field(
                    x_t, t, perturbed_s_t, s_target_expanded
                )

                # Compute unique track errors independently (reduction='none' preserves distinct batch trajectories)
                error_loss = F.mse_loss(pred_vel, target_vel, reduction="none").mean(
                    dim=-1
                )
                total_inner_loss = error_loss.sum()

                action_grads = torch.autograd.grad(total_inner_loss, search_action)[0]
                with torch.no_grad():
                    search_action = (
                        search_action - self.action_search_lr * action_grads.sign()
                    )
                search_action.requires_grad = True

            # --- OUTER LOOP: Parallel Poison Optimization (Max) ---
            perturbed_s_t = s_t_expanded + mask * perturb
            t = torch.rand(ensemble_size, 1, device=s_t.device) * 0.2
            x_0 = torch.randn_like(search_action)
            x_t = t * search_action + (1.0 - t) * x_0
            target_vel = search_action - x_0

            pred_vel = flow_matcher.velocity_field(
                x_t, t, perturbed_s_t, s_target_expanded
            )
            final_error_loss = (
                F.mse_loss(pred_vel, target_vel, reduction="none").mean(dim=-1).sum()
            )

            outer_grads = torch.autograd.grad(final_error_loss, perturb)[0]

            # Fast gradient sign ascent across parallel tracks to maximize the planning disruption
            perturb = perturb.detach() + self.perturb_lr * outer_grads.sign()
            perturb.requires_grad = True

        # Returns the finalized parallel cloud tensor [ensemble_size, 512]
        return (s_t_expanded + mask * perturb).detach()
