import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_2_output.png")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------
# 1. Dataset Generation (Conditional Goals)
# ----------------------------------------------------
num_samples = 4000
# Target goal coordinates c sampled uniformly
c_data = np.random.uniform(-1.8, 1.8, (num_samples, 2))
# Prior noise x0 sampled from Gaussian
x0_data = np.random.normal(0, 0.4, (num_samples, 2))

c_tensor = torch.tensor(c_data, dtype=torch.float32)
x0_tensor = torch.tensor(x0_data, dtype=torch.float32)

print(f"Generated dataset with {num_samples} goals and noise priors.")


# ----------------------------------------------------
# 2. Conditional Flow Matching Model
# ----------------------------------------------------
# Inputs: current point x_t (2D) + time t (1D) + conditioning goal c (2D) = 5D
class ConditionalFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x, t, c):
        inputs = torch.cat([x, t, c], dim=-1)
        return self.net(inputs)


flow_model = ConditionalFlowModel()
optimizer = optim.Adam(flow_model.parameters(), lr=0.003)
epochs = 1500
batch_size = 256

print("Training Conditional Flow Matching Model...")
for epoch in range(epochs):
    indices = np.random.choice(num_samples, batch_size)
    c_batch = c_tensor[indices]
    x0_batch = x0_tensor[indices]

    # In this conditional task, the target endpoint x1 is exactly the goal c
    x1_batch = c_batch

    t = torch.rand(batch_size, 1)

    # Linear path from noise x0 to goal c
    x_t = (1 - t) * x0_batch + t * x1_batch
    v_target = x1_batch - x0_batch

    optimizer.zero_grad()
    v_pred = flow_model(x_t, t, c_batch)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.6f}")

# ----------------------------------------------------
# 3. ODE Integration for Test Targets
# ----------------------------------------------------
# Define three distinct goal locations to test conditioning
test_goals = np.array(
    [
        [-1.5, 1.2],  # Goal 1: Top-Left
        [1.5, 1.2],  # Goal 2: Top-Right
        [0.0, -1.5],  # Goal 3: Bottom-Center
    ]
)

# For each goal, generate paths starting from 10 different random noise points
num_paths_per_goal = 12
eval_times = np.linspace(0.0, 1.0, 20)

plt.figure(figsize=(12, 6))

# Plot all paths
colors = ["royalblue", "forestgreen", "darkorange"]
markers = ["o", "s", "^"]

for g_idx, goal in enumerate(test_goals):
    print(f"Generating paths converging to Goal {g_idx+1}: {goal}...")

    # Sample noise starting points
    x0_test = np.random.normal(0, 0.4, (num_paths_per_goal, 2))

    # Integrate flow ODE for this target
    def flow_ode(t, x_flat):
        x_tensor = torch.tensor(x_flat.reshape(-1, 2), dtype=torch.float32)
        t_tensor = torch.full((x_tensor.shape[0], 1), t, dtype=torch.float32)
        c_tensor = torch.tensor(
            np.tile(goal, (x_tensor.shape[0], 1)), dtype=torch.float32
        )
        with torch.no_grad():
            v = flow_model(x_tensor, t_tensor, c_tensor)
        return v.numpy().flatten()

    sol = solve_ivp(
        flow_ode,
        t_span=(0.0, 1.0),
        y0=x0_test.flatten(),
        t_eval=eval_times,
        method="RK45",
    )

    # Plot each path
    for p_idx in range(num_paths_per_goal):
        path = sol.y.reshape(num_paths_per_goal, 2, len(eval_times))[p_idx]
        plt.plot(
            path[0],
            path[1],
            color=colors[g_idx],
            linestyle="--",
            alpha=0.6,
            linewidth=1.5,
        )
        # Mark start point (noise)
        if p_idx == 0:
            plt.scatter(
                path[0, 0],
                path[1, 0],
                color="black",
                marker="x",
                s=40,
                zorder=5,
                label="Noise Starts ($x_0$)",
            )
        else:
            plt.scatter(
                path[0, 0], path[1, 0], color="black", marker="x", s=25, zorder=5
            )

    # Highlight final target goal location
    plt.scatter(
        goal[0],
        goal[1],
        color=colors[g_idx],
        marker=markers[g_idx],
        s=180,
        edgecolor="black",
        zorder=10,
        label=f"Target Goal {g_idx+1} {goal}",
    )

# Format plot
plt.title(
    "Goal-Conditioned Flow Matching: Clean Convergence to Multiple Targets", fontsize=13
)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(PLOT_PATH)
print(f"Task 2 complete! Output plot saved to {PLOT_PATH}")
