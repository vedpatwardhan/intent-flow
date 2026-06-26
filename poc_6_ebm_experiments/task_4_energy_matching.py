import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_4_output.png")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------
# 1. Dataset: 8-Gaussian Mixture in a Ring
# ----------------------------------------------------
def generate_8_gaussians(num_samples=3000, std=0.2):
    centers = []
    r = 2.0
    for i in range(8):
        theta = 2 * np.pi * i / 8
        centers.append([r * np.cos(theta), r * np.sin(theta)])
    centers = np.array(centers)

    samples = []
    for _ in range(num_samples):
        center = centers[np.random.choice(8)]
        sample = center + np.random.normal(0, std, size=2)
        samples.append(sample)
    return torch.tensor(np.array(samples), dtype=torch.float32)


x1_data = generate_8_gaussians(4000)
print(f"Generated target dataset with shape: {x1_data.shape}")


# ----------------------------------------------------
# 2. Time-Independent Potential Network
# ----------------------------------------------------
class PotentialNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Scalar potential field V(x): R^2 -> R
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


model = PotentialNet()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# ----------------------------------------------------
# 3. Training Loop: Simulation-Free Gradient Matching
# ----------------------------------------------------
epochs = 2000
batch_size = 256

print("Training Energy Matching Model...")
for epoch in range(epochs + 1):
    indices = np.random.choice(len(x1_data), batch_size)
    x1 = x1_data[indices]

    # Prior noise distribution x0
    x0 = torch.randn_like(x1)

    # Uniform time steps t
    t = torch.rand(batch_size, 1)

    # Linear path interpolation
    x_t = (1 - t) * x0 + t * x1
    v_target = x1 - x0

    # Enable grad computation on x_t for gradient matching
    x_t_var = x_t.clone().detach().requires_grad_(True)

    # Compute potential V(x_t)
    pot = model(x_t_var)

    # Velocity field is defined as v(x) = -grad_x V(x)
    # Use create_graph=True to backprop through the gradient operation
    grad_V = torch.autograd.grad(pot.sum(), x_t_var, create_graph=True)[0]
    v_pred = -grad_V

    loss = nn.MSELoss()(v_pred, v_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 250 == 0:
        print(f"Epoch {epoch}/{epochs} - Gradient Matching Loss: {loss.item():.6f}")

# ----------------------------------------------------
# 4. Generative Sampling via ODE Integration
# ----------------------------------------------------
print("\nSimulating generative trajectories using learned potential gradient...")


def get_gradient_velocity(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
    pot = model(x_tensor)
    grad = torch.autograd.grad(pot.sum(), x_tensor)[0]
    return -grad.detach().numpy()


# Integrate ODE to generate samples
num_samples = 1000
x_traj = torch.randn(num_samples, 2).numpy()  # Start from prior noise

steps = 40
dt = 1.0 / steps
traj_history = [x_traj.copy()]

for step in range(steps):
    v = get_gradient_velocity(x_traj)
    x_traj = x_traj + dt * v
    traj_history.append(x_traj.copy())
    if (step + 1) % 10 == 0:
        print(f"  ODE Integration Step {step + 1}/{steps} completed.")

traj_history = np.array(traj_history)

# ----------------------------------------------------
# 5. Visualization & Plot Generation
# ----------------------------------------------------
print("\nGenerating final Energy Matching plots to:", PLOT_PATH)

grid_size = 100
x_grid = np.linspace(-3.5, 3.5, grid_size)
y_grid = np.linspace(-3.5, 3.5, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_pts = np.vstack([xx.ravel(), yy.ravel()]).T
grid_pts_tensor = torch.tensor(grid_pts, dtype=torch.float32)

with torch.no_grad():
    potentials = model(grid_pts_tensor).numpy().reshape(grid_size, grid_size)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Potential field contour landscape
im = axs[0].contourf(xx, yy, potentials, levels=30, cmap="viridis")
fig.colorbar(im, ax=axs[0], label="Potential V(x)")
axs[0].set_title("Learned Potential Landscape V(x)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

# Highlight 8 mode centers
r = 2.0
centers_x = [r * np.cos(2 * np.pi * i / 8) for i in range(8)]
centers_y = [r * np.sin(2 * np.pi * i / 8) for i in range(8)]
axs[0].scatter(
    centers_x, centers_y, color="red", marker="x", s=60, zorder=5, label="Modes"
)
axs[0].legend()

# Subplot 2: Trajectory rollouts using gradient flow
axs[1].scatter(
    x1_data[:800, 0].numpy(),
    x1_data[:800, 1].numpy(),
    color="blue",
    alpha=0.15,
    label="True Data",
)

# Plot 20 random trajectory paths
for i in range(20):
    path_x = traj_history[:, i, 0]
    path_y = traj_history[:, i, 1]
    axs[1].plot(path_x, path_y, color="red", alpha=0.6, linewidth=1.5)
    axs[1].scatter(
        path_x[0], path_y[0], color="cyan", s=25, edgecolor="black", zorder=5
    )  # Start
    axs[1].scatter(
        path_x[-1],
        path_y[-1],
        color="yellow",
        s=40,
        edgecolor="black",
        zorder=5,
        marker="*",
    )  # End

# Plot generated endpoints
axs[1].scatter(
    x_traj[:, 0], x_traj[:, 1], color="orange", alpha=0.5, label="Generated Samples"
)
axs[1].set_title("Gradient Flow Trajectories & Endpoints")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].set_xlim(-3.5, 3.5)
axs[1].set_ylim(-3.5, 3.5)
axs[1].legend()

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("Task 4 completed successfully.")
