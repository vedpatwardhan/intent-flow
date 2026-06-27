import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_1_output.png")

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
        # Choose a random cluster center
        center = centers[np.random.choice(8)]
        # Sample around the center with small variance
        sample = center + np.random.normal(0, std, size=2)
        samples.append(sample)
    return torch.tensor(np.array(samples), dtype=torch.float32)


x_data = generate_8_gaussians(4000)
print(f"Generated target dataset with shape: {x_data.shape}")


# ----------------------------------------------------
# 2. Energy-Based Model Network
# ----------------------------------------------------
class EBM(nn.Module):
    def __init__(self):
        super().__init__()
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


model = EBM()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


# ----------------------------------------------------
# 3. Persistent Replay Buffer & SGLD Sampler
# ----------------------------------------------------
class ReplayBuffer:
    def __init__(self, size=2000):
        self.size = size
        # Initialize randomly in [-3, 3]
        self.buffer = (torch.rand(size, 2) * 6.0) - 3.0

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        samples = self.buffer[indices]
        return samples, indices

    def update(self, samples, indices):
        self.buffer[indices] = samples.detach()


buffer = ReplayBuffer(2000)


def sgld_sample(model, x_start, steps=30, step_size=0.1, noise_std=0.01):
    # Enable grad computation on coordinates
    x = x_start.clone().detach().requires_grad_(True)

    # Track SGLD steps for visualization later
    history = [x.clone().detach().numpy()]

    for _ in range(steps):
        energy = model(x)
        # Gradient of energy w.r.t x
        grad = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]

        # SGLD update: step down the energy gradient + add Gaussian noise
        # Note: grad is clipped to stabilize training
        grad = torch.clamp(grad, min=-10.0, max=10.0)

        noise = noise_std * torch.randn_like(x)
        x.data = x.data - 0.5 * step_size * grad + noise

        # Keep samples bounded to avoid divergence
        x.data = torch.clamp(x.data, min=-4.0, max=4.0)
        history.append(x.clone().detach().numpy())

    return x.detach(), np.array(history)


# ----------------------------------------------------
# 4. Training Loop via Contrastive Divergence
# ----------------------------------------------------
epochs = 2000
batch_size = 128
sgld_steps = 40
sgld_lr = 0.15
sgld_noise = 0.01

print("Training Energy-Based Model using Contrastive Divergence...")
for epoch in range(epochs + 1):
    # 1. Sample real data
    indices_real = np.random.choice(len(x_data), batch_size)
    x_real = x_data[indices_real]

    # 2. Retrieve starting points from Replay Buffer (95% buffer, 5% random noise)
    x_init, buffer_indices = buffer.sample(batch_size)
    random_mask = torch.rand(batch_size) < 0.05
    x_init[random_mask] = (torch.rand(random_mask.sum(), 2) * 6.0) - 3.0

    # 3. Generate negative samples via SGLD
    x_neg, _ = sgld_sample(
        model, x_init, steps=sgld_steps, step_size=sgld_lr, noise_std=sgld_noise
    )

    # 4. Update the persistent buffer with negative samples
    buffer.update(x_neg, buffer_indices)

    # 5. Compute CD-1 loss
    energy_real = model(x_real)
    energy_neg = model(x_neg)

    # Standard CD objective: minimize real energy, maximize fake energy
    # We add L2 regularization on energy outputs to stabilize training scale
    loss = (energy_real - energy_neg).mean() + 0.1 * (
        energy_real**2 + energy_neg**2
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(
            f"Epoch {epoch}/{epochs} - CD-1 Loss: {loss.item():.4f} | E_real: {energy_real.mean().item():.3f} | E_neg: {energy_neg.mean().item():.3f}"
        )

# ----------------------------------------------------
# 5. Visualization & Plot Generation
# ----------------------------------------------------
print("\nSaving final EBM visualization to:", PLOT_PATH)

# Generate a grid to plot energy landscape contour
grid_size = 100
x_grid = np.linspace(-3.5, 3.5, grid_size)
y_grid = np.linspace(-3.5, 3.5, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_pts = np.vstack([xx.ravel(), yy.ravel()]).T
grid_pts_tensor = torch.tensor(grid_pts, dtype=torch.float32)

with torch.no_grad():
    energies = model(grid_pts_tensor).numpy().reshape(grid_size, grid_size)

# Sample a few trajectories to visualize SGLD behavior
num_trajectories = 10
x_traj_init = (torch.rand(num_trajectories, 2) * 6.0) - 3.0
_, traj_history = sgld_sample(
    model, x_traj_init, steps=60, step_size=sgld_lr, noise_std=sgld_noise
)

# Draw plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Learned Energy Landscape & Trajectories
im = axs[0].contourf(xx, yy, energies, levels=30, cmap="viridis")
fig.colorbar(im, ax=axs[0], label="Energy E(x)")
axs[0].set_title("Learned Energy Landscape & SGLD Trajectories")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

# Plot SGLD trajectory paths
for i in range(num_trajectories):
    traj_x = traj_history[:, i, 0]
    traj_y = traj_history[:, i, 1]
    axs[0].plot(traj_x, traj_y, color="red", alpha=0.6, linewidth=1.5)
    axs[0].scatter(
        traj_x[0], traj_y[0], color="cyan", edgecolor="black", zorder=5, s=40
    )  # Start
    axs[0].scatter(
        traj_x[-1],
        traj_y[-1],
        color="yellow",
        edgecolor="black",
        zorder=5,
        s=60,
        marker="*",
    )  # End

# Subplot 2: Generated vs True Data Comparison
# Draw a final batch of SGLD generated samples
x_gen_init = (torch.rand(1000, 2) * 6.0) - 3.0
x_gen, _ = sgld_sample(
    model, x_gen_init, steps=80, step_size=sgld_lr, noise_std=sgld_noise
)

axs[1].scatter(
    x_data[:1000, 0].numpy(),
    x_data[:1000, 1].numpy(),
    color="blue",
    alpha=0.3,
    label="True Data",
)
axs[1].scatter(
    x_gen[:, 0].numpy(),
    x_gen[:, 1].numpy(),
    color="orange",
    alpha=0.5,
    label="Generated Samples",
)
axs[1].set_title("Target Distribution vs. EBM Samples")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()
axs[1].set_xlim(-3.5, 3.5)
axs[1].set_ylim(-3.5, 3.5)

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("Task 1 completed successfully.")
