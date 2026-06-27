import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_2_output.png")

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


x_data = generate_8_gaussians(4000)


# EBM Network definition
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


# SGLD Sampler for inference
def sgld_sample(model, x_start, steps=80, step_size=0.15, noise_std=0.01):
    x = x_start.clone().detach().requires_grad_(True)
    for _ in range(steps):
        energy = model(x)
        grad = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]
        grad = torch.clamp(grad, min=-10.0, max=10.0)
        noise = noise_std * torch.randn_like(x)
        x.data = x.data - 0.5 * step_size * grad + noise
        x.data = torch.clamp(x.data, min=-4.0, max=4.0)
    return x.detach()


# ----------------------------------------------------
# 2. Protocol A: Contrastive Divergence (CD-1)
# ----------------------------------------------------
print("--- Training EBM A: Contrastive Divergence (CD-1) ---")
model_cd = EBM()
optimizer_cd = optim.Adam(model_cd.parameters(), lr=0.001, weight_decay=1e-4)


# Replay buffer for CD-1
class ReplayBuffer:
    def __init__(self, size=2000):
        self.size = size
        self.buffer = (torch.rand(size, 2) * 6.0) - 3.0

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return self.buffer[indices], indices

    def update(self, samples, indices):
        self.buffer[indices] = samples.detach()


buffer = ReplayBuffer(2000)
epochs = 2000
batch_size = 128

for epoch in range(epochs + 1):
    indices_real = np.random.choice(len(x_data), batch_size)
    x_real = x_data[indices_real]

    x_init, buffer_indices = buffer.sample(batch_size)
    random_mask = torch.rand(batch_size) < 0.05
    x_init[random_mask] = (torch.rand(random_mask.sum(), 2) * 6.0) - 3.0

    x_neg = sgld_sample(model_cd, x_init, steps=40, step_size=0.15, noise_std=0.01)
    buffer.update(x_neg, buffer_indices)

    energy_real = model_cd(x_real)
    energy_neg = model_cd(x_neg)

    loss_cd = (energy_real - energy_neg).mean() + 0.1 * (
        energy_real**2 + energy_neg**2
    ).mean()

    optimizer_cd.zero_grad()
    loss_cd.backward()
    optimizer_cd.step()

    if epoch % 400 == 0:
        print(f"Epoch {epoch}/{epochs} - CD Loss: {loss_cd.item():.4f}")

# ----------------------------------------------------
# 3. Protocol B: Noise Contrastive Estimation (NCE)
# ----------------------------------------------------
print("\n--- Training EBM B: Noise Contrastive Estimation (NCE) ---")
model_nce = EBM()
optimizer_nce = optim.Adam(model_nce.parameters(), lr=0.001, weight_decay=1e-4)

# Noise prior q(x): 2D Gaussian with std=2.0
noise_std = 2.0


def log_q(x):
    # Log density of 2D Gaussian: -log(2*pi*std^2) - ||x||^2 / (2*std^2)
    return -np.log(2 * np.pi * (noise_std**2)) - torch.sum(
        x**2, dim=-1, keepdim=True
    ) / (2 * (noise_std**2))


def sample_q(num_samples):
    return torch.randn(num_samples, 2) * noise_std


nu = 4  # Ratio of noise to data samples

for epoch in range(epochs + 1):
    indices_real = np.random.choice(len(x_data), batch_size)
    x_real = x_data[indices_real]

    # Sample from noise distribution q
    x_noise = sample_q(batch_size * nu)

    # Compute log p_theta (which is -E_theta(x))
    log_p_real = -model_nce(x_real)
    log_p_noise = -model_nce(x_noise)

    # Compute log q(x)
    log_q_real = log_q(x_real)
    log_q_noise = log_q(x_noise)

    # NCE Probabilities: D(x) = p / (p + nu * q) = 1 / (1 + nu * q / p)
    # We do this in log-space for numerical stability: sigmoid(log_p - log_q - log(nu))
    log_nu = np.log(nu)
    d_real = torch.sigmoid(log_p_real - log_q_real - log_nu)
    d_noise = torch.sigmoid(log_p_noise - log_q_noise - log_nu)

    # Objective: Maximize E[log D(x_real)] + E[log(1 - D(x_noise))]
    loss_nce = -(
        torch.log(d_real + 1e-7).mean() + nu * torch.log(1 - d_noise + 1e-7).mean()
    )

    optimizer_nce.zero_grad()
    loss_nce.backward()
    optimizer_nce.step()

    if epoch % 400 == 0:
        print(f"Epoch {epoch}/{epochs} - NCE Loss: {loss_nce.item():.4f}")

# ----------------------------------------------------
# 4. Generate Visualizations & Comparison Plots
# ----------------------------------------------------
print("\nGenerating final comparison plots to:", PLOT_PATH)

grid_size = 100
x_grid = np.linspace(-3.5, 3.5, grid_size)
y_grid = np.linspace(-3.5, 3.5, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_pts = np.vstack([xx.ravel(), yy.ravel()]).T
grid_pts_tensor = torch.tensor(grid_pts, dtype=torch.float32)

with torch.no_grad():
    energies_cd = model_cd(grid_pts_tensor).numpy().reshape(grid_size, grid_size)
    energies_nce = model_nce(grid_pts_tensor).numpy().reshape(grid_size, grid_size)

# Sample generated points using SGLD from both models
x_gen_init = (torch.rand(1000, 2) * 6.0) - 3.0
x_gen_cd = sgld_sample(model_cd, x_gen_init, steps=100, step_size=0.15, noise_std=0.01)
x_gen_nce = sgld_sample(
    model_nce, x_gen_init, steps=100, step_size=0.15, noise_std=0.01
)

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Row 1: Contrastive Divergence (CD-1)
im1 = axs[0, 0].contourf(xx, yy, energies_cd, levels=30, cmap="viridis")
fig.colorbar(im1, ax=axs[0, 0], label="Energy E(x)")
axs[0, 0].set_title("CD-1: Learned Energy Landscape")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")

axs[0, 1].scatter(
    x_data[:1000, 0].numpy(),
    x_data[:1000, 1].numpy(),
    color="blue",
    alpha=0.2,
    label="True Data",
)
axs[0, 1].scatter(
    x_gen_cd[:, 0].numpy(),
    x_gen_cd[:, 1].numpy(),
    color="orange",
    alpha=0.5,
    label="CD-1 Samples",
)
axs[0, 1].set_title("CD-1: Data vs. SGLD Samples")
axs[0, 1].legend()
axs[0, 1].set_xlim(-3.5, 3.5)
axs[0, 1].set_ylim(-3.5, 3.5)

# Row 2: Noise Contrastive Estimation (NCE)
im2 = axs[1, 0].contourf(xx, yy, energies_nce, levels=30, cmap="viridis")
fig.colorbar(im2, ax=axs[1, 0], label="Energy E(x)")
axs[1, 0].set_title("NCE: Learned Energy Landscape")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")

axs[1, 1].scatter(
    x_data[:1000, 0].numpy(),
    x_data[:1000, 1].numpy(),
    color="blue",
    alpha=0.2,
    label="True Data",
)
axs[1, 1].scatter(
    x_gen_nce[:, 0].numpy(),
    x_gen_nce[:, 1].numpy(),
    color="red",
    alpha=0.5,
    label="NCE Samples",
)
axs[1, 1].set_title("NCE: Data vs. SGLD Samples")
axs[1, 1].legend()
axs[1, 1].set_xlim(-3.5, 3.5)
axs[1, 1].set_ylim(-3.5, 3.5)

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("Task 2 completed successfully.")
