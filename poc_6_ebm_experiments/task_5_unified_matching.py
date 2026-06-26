import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_5_output.png")

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
# 3. Dual-Regime Training Loop (Flow Matching + NCE)
# ----------------------------------------------------
epochs = 3000
batch_size = 256
nu = 8  # NCE noise-to-data ratio
noise_std = 2.0  # Standard deviation for NCE noise distribution q(x)


def log_q(x):
    return -np.log(2 * np.pi * (noise_std**2)) - torch.sum(
        x**2, dim=-1, keepdim=True
    ) / (2 * (noise_std**2))


def sample_q(num_samples):
    return torch.randn(num_samples, 2) * noise_std


print("Training Unified Energy Matching Model (Flow + NCE)...")
for epoch in range(epochs + 1):
    indices = np.random.choice(len(x1_data), batch_size)
    x1 = x1_data[indices]
    x0 = torch.randn_like(x1)

    # Sample random times t
    t = torch.rand(batch_size, 1)

    # Split batch into Transport Regime (t < 0.8) and Density Regime (t >= 0.8)
    transport_mask = (t < 0.8).squeeze(-1)
    density_mask = ~transport_mask

    loss = 0.0

    # 1. Transport Regime: Flow Matching
    if transport_mask.sum() > 0:
        x1_trans = x1[transport_mask]
        x0_trans = x0[transport_mask]
        t_trans = t[transport_mask]

        x_t = (1 - t_trans) * x0_trans + t_trans * x1_trans
        v_target = x1_trans - x0_trans

        x_t_var = x_t.clone().detach().requires_grad_(True)
        pot = model(x_t_var)
        grad_V = torch.autograd.grad(pot.sum(), x_t_var, create_graph=True)[0]
        v_pred = -grad_V

        loss_trans = nn.MSELoss()(v_pred, v_target)
        loss += loss_trans

    # 2. Density Regime: Noise Contrastive Estimation (NCE)
    if density_mask.sum() > 0:
        x1_dense = x1[density_mask]
        n_dense = x1_dense.shape[0]

        # Sample NCE noise targets
        x_noise = sample_q(n_dense * nu)

        # EBM log-probabilities: log p = -V(x)
        log_p_real = -model(x1_dense)
        log_p_noise = -model(x_noise)

        # Log densities of Gaussian noise prior q(x)
        log_q_real = log_q(x1_dense)
        log_q_noise = log_q(x_noise)

        log_nu = np.log(nu)
        d_real = torch.sigmoid(log_p_real - log_q_real - log_nu)
        d_noise = torch.sigmoid(log_p_noise - log_q_noise - log_nu)

        loss_dense = -(
            torch.log(d_real + 1e-7).mean() + nu * torch.log(1 - d_noise + 1e-7).mean()
        )
        # Balance scale of classification loss with regression loss
        loss += 8.0 * loss_dense

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch}/{epochs} - Combined Loss: {loss.item():.4f}")

# ----------------------------------------------------
# 4. Hybrid Generative Inference
# ----------------------------------------------------
print("\nSimulating Hybrid Inference (Transport ODE + Langevin EBM refinement)...")


def get_gradient_velocity(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
    pot = model(x_tensor)
    grad = torch.autograd.grad(pot.sum(), x_tensor)[0]
    return -grad.detach().numpy()


# 1. Transport Phase (ODE integration from t=0 to t=0.8)
num_samples = 1000
x_traj = torch.randn(num_samples, 2).numpy()  # Start from prior noise

ode_steps = 24
dt = 0.8 / ode_steps
traj_history = [x_traj.copy()]

for step in range(ode_steps):
    v = get_gradient_velocity(x_traj)
    x_traj = x_traj + dt * v
    traj_history.append(x_traj.copy())

print(
    f"  Transport ODE phase completed at t=0.8. Mean coordinate magnitude: {np.mean(np.linalg.norm(x_traj, axis=1)):.3f}"
)

# 2. Density Phase (Langevin / SGLD refinement on the EBM potential)
# Run 40 steps of SGLD to push samples into the nearest sharp local mode wells
sgld_steps = 40
sgld_lr = 0.08
sgld_noise = 0.005

x_traj_tensor = torch.tensor(x_traj, dtype=torch.float32).requires_grad_(True)

for _ in range(sgld_steps):
    pot = model(x_traj_tensor)
    grad = torch.autograd.grad(pot.sum(), x_traj_tensor)[0]
    grad = torch.clamp(grad, min=-10.0, max=10.0)

    noise = sgld_noise * torch.randn_like(x_traj_tensor)
    x_traj_tensor.data = x_traj_tensor.data - 0.5 * sgld_lr * grad + noise
    x_traj_tensor.data = torch.clamp(x_traj_tensor.data, min=-4.0, max=4.0)
    traj_history.append(x_traj_tensor.clone().detach().numpy())

x_gen = x_traj_tensor.detach().numpy()
traj_history = np.array(traj_history)
print("  Langevin SGLD refinement completed.")

# ----------------------------------------------------
# 5. Visualization & Plot Generation
# ----------------------------------------------------
print("\nGenerating final comparison plots to:", PLOT_PATH)

grid_size = 100
x_grid = np.linspace(-3.5, 3.5, grid_size)
y_grid = np.linspace(-3.5, 3.5, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_pts = np.vstack([xx.ravel(), yy.ravel()]).T
grid_pts_tensor = torch.tensor(grid_pts, dtype=torch.float32)

with torch.no_grad():
    potentials = model(grid_pts_tensor).numpy().reshape(grid_size, grid_size)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Learned Potential Landscape showing separate wells
im = axs[0].contourf(xx, yy, potentials, levels=30, cmap="viridis")
fig.colorbar(im, ax=axs[0], label="Potential V(x)")
axs[0].set_title("Unified Potential Landscape V(x)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

# Highlight 8 modes
r = 2.0
centers_x = [r * np.cos(2 * np.pi * i / 8) for i in range(8)]
centers_y = [r * np.sin(2 * np.pi * i / 8) for i in range(8)]
axs[0].scatter(
    centers_x, centers_y, color="red", marker="x", s=60, zorder=5, label="Modes"
)
axs[0].legend()

# Subplot 2: Trajectories showing ODE transport + SGLD division
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

    # Plot ODE transport phase in green, SGLD refinement phase in red
    axs[1].plot(
        path_x[: ode_steps + 1],
        path_y[: ode_steps + 1],
        color="green",
        alpha=0.5,
        linewidth=1.5,
    )
    axs[1].plot(
        path_x[ode_steps:], path_y[ode_steps:], color="red", alpha=0.6, linewidth=1.5
    )

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
    x_gen[:, 0], x_gen[:, 1], color="orange", alpha=0.6, label="Unified Samples"
)
axs[1].set_title("Hybrid Trajectories (Green=ODE, Red=SGLD)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].set_xlim(-3.5, 3.5)
axs[1].set_ylim(-3.5, 3.5)
axs[1].legend()

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("Task 5 completed successfully.")
