import torch
import torch.nn as nn


class TrajectoryDiscriminator(nn.Module):
    """
    Trajectory Discriminator (DRL style check).
    Evaluates human-like trajectory realism conditioned on the task state context.
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
