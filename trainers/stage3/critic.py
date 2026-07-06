import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryDiscriminator(nn.Module):
    """
    Trajectory Discriminator (DRL style check).
    Evaluates human-like trajectory realism conditioned on the task.
    """

    def __init__(self, action_dim=12, state_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, traj, s_t):
        inputs = torch.cat([traj, s_t], dim=-1)
        return self.net(inputs)


class EBMCritic(nn.Module):
    """
    EBT-Policy Contrastive Energy-Based Critic.
    Assigns low energy to expert/successful states and high energy to negative samples.
    """

    def __init__(self, state_dim=512, action_dim=12, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Outputs energy scalar
        )

    def forward(self, s_t, action):
        inputs = torch.cat([s_t, action], dim=-1)
        return self.net(inputs)

    def compute_infonce_loss(self, s_t, pos_action, neg_actions):
        """
        InfoNCE Loss to minimize positive energy and maximize negative energies.
        neg_actions: [Batch, NumNegatives, ActionDim]
        """
        # positive energy
        pos_energy = self.forward(s_t, pos_action)  # [Batch, 1]

        # negative energies
        batch_size = s_t.size(0)
        num_neg = neg_actions.size(1)
        s_t_expanded = s_t.unsqueeze(1).expand(
            -1, num_neg, -1
        )  # [Batch, NumNeg, StateDim]
        neg_energy = self.forward(
            s_t_expanded.reshape(-1, s_t.size(-1)),
            neg_actions.reshape(-1, pos_action.size(-1)),
        ).reshape(batch_size, num_neg, 1)

        # InfoNCE: we want -pos_energy to be high, and -neg_energy to be low
        logits = torch.cat(
            [-pos_energy, -neg_energy.squeeze(-1)], dim=-1
        )  # [Batch, 1 + NumNeg]
        targets = torch.zeros(
            batch_size, dtype=torch.long, device=s_t.device
        )  # Index 0 is positive

        return F.cross_entropy(logits, targets)

    def sample_langevin_mcmc(self, s_t, num_samples=32, steps=10, step_size=0.1):
        """
        Langevin MCMC sampler to generate actions from the learned energy field.
        """
        batch_size = s_t.size(0)
        action_dim = 12

        # Initialize randomly
        actions = torch.randn(
            batch_size, action_dim, device=s_t.device, requires_grad=True
        )

        for _ in range(steps):
            energy = self.forward(s_t, actions)
            # Compute gradient of energy w.r.t action
            grads = torch.autograd.grad(energy.sum(), actions, retain_graph=True)[0]

            # Langevin update step: move in direction of lower energy + Gaussian noise
            noise = torch.randn_like(actions) * 0.05
            actions = actions.detach() - step_size * grads + noise
            actions.requires_grad = True

        return actions.detach()
