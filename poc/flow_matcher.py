import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# A simple vector field network to predict velocity fields
class ToyVectorField(nn.Module):
    def __init__(self, state_dim=2, cond_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + cond_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, x, t, cond):
        # x: [B, state_dim], t: [B, 1], cond: [B, cond_dim]
        inputs = torch.cat([x, t, cond], dim=-1)
        return self.net(inputs)


def train_and_eval_flow_matcher():
    print("=== Task 1.1: CLAP Flow Matcher PoC ===")
    torch.manual_seed(42)

    # 1. Setup toy data (2D target positions conditioned on a 4D context vector)
    B = 1000
    state_dim = 2
    cond_dim = 4

    # Condition: 4D one-hot target direction
    cond = torch.randn(B, cond_dim)
    # Target positions: x0 is target (expert actions)
    x0 = cond[:, :2] * 2.0 + torch.randn(B, 2) * 0.1
    # Starting noise: x1
    x1 = torch.randn(B, state_dim)

    # Init model
    model = ToyVectorField(state_dim, cond_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 2. Train using Rectified Flow objective
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Sample time t in [0, 1]
        t = torch.rand(B, 1)

        # Linear interpolation path between x0 and x1
        xt = (1 - t) * x0 + t * x1

        # Target velocity is (x1 - x0)
        target_vel = x1 - x0

        # Predict velocity field
        pred_vel = model(xt, t, cond)

        loss = nn.MSELoss()(pred_vel, target_vel)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    # 3. ODE Integration Inference Loop (Euler solver)
    print("\nRunning ODE Integration inference loop (Euler steps)...")
    steps = 20
    dt = 1.0 / steps

    # Start at x1 (noise) and integrate backward to x0 (actions)
    test_cond = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    expected_action = test_cond[:, :2] * 2.0

    # Initial noise state
    xt_test = torch.randn(1, state_dim)

    for step in range(steps):
        t_val = 1.0 - (step * dt)
        t_tensor = torch.full((1, 1), t_val)

        # Predict velocity vector
        with torch.no_grad():
            v = model(xt_test, t_tensor, test_cond)

        # Euler step backward: x_{t - dt} = x_t - v * dt
        xt_test = xt_test - v * dt

    final_error = torch.mean(torch.abs(xt_test - expected_action)).item()
    print(f"Expected Action Target: {expected_action.numpy()[0]}")
    print(f"Inference Denoised Action: {xt_test.numpy()[0]}")
    print(f"Final Convergence Error: {final_error:.4f}")

    # Validation Check
    if final_error < 0.2:
        print("PoC Result: SUCCESS (Converged with low error)")
    else:
        print("PoC Result: FAILED (High convergence error)")


if __name__ == "__main__":
    train_and_eval_flow_matcher()
