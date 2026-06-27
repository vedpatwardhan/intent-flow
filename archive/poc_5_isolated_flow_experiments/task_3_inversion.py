import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_3_output.png")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------
# 1. Dataset: 2D Double-Ring Point Cloud
# ----------------------------------------------------
def generate_double_ring(num_samples=2000):
    samples = []
    for _ in range(num_samples):
        theta = np.random.uniform(0, 2 * np.pi)
        if np.random.rand() < 0.5:
            r = np.random.uniform(0.6, 0.9)  # Inner ring
        else:
            r = np.random.uniform(1.6, 1.9)  # Outer ring
        samples.append([r * np.cos(theta), r * np.sin(theta)])
    return torch.tensor(np.array(samples), dtype=torch.float32)


x1_data = generate_double_ring(3000)
print(f"Generated dataset of shape: {x1_data.shape}")


# ----------------------------------------------------
# 2. Flow Matching Model
# ----------------------------------------------------
class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=-1)
        return self.net(inputs)


flow_model = FlowModel()
optimizer = optim.Adam(flow_model.parameters(), lr=0.003)
epochs = 1500
batch_size = 256

print("Training Flow Matching Model for Task 3...")
for epoch in range(epochs):
    indices = np.random.choice(len(x1_data), batch_size)
    x1 = x1_data[indices]
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size, 1)

    # Linear path
    x_t = (1 - t) * x0 + t * x1
    v_target = x1 - x0

    optimizer.zero_grad()
    v_pred = flow_model(x_t, t)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.6f}")


# ----------------------------------------------------
# 3. Forward and Reverse ODE Solvers
# ----------------------------------------------------
def flow_ode(t, x_flat):
    x_tensor = torch.tensor(x_flat.reshape(-1, 2), dtype=torch.float32)
    t_tensor = torch.full((x_tensor.shape[0], 1), t, dtype=torch.float32)
    with torch.no_grad():
        v = flow_model(x_tensor, t_tensor)
    return v.numpy().flatten()


# ----------------------------------------------------
# 4. Cycle Consistency Verification
# ----------------------------------------------------
print("\nVerifying Cycle Consistency (Forward -> Backward)...")
num_test = 100
x0_test = np.random.normal(0, 1.0, (num_test, 2))

# Forward ODE (0 -> 1)
sol_forward = solve_ivp(
    flow_ode,
    t_span=(0.0, 1.0),
    y0=x0_test.flatten(),
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)
x1_gen = sol_forward.y[:, -1].reshape(num_test, 2)

# Reverse ODE (1 -> 0)
sol_backward = solve_ivp(
    flow_ode,
    t_span=(1.0, 0.0),
    y0=x1_gen.flatten(),
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)
x0_reconstructed = sol_backward.y[:, -1].reshape(num_test, 2)

# Compute errors
errors = np.linalg.norm(x0_test - x0_reconstructed, axis=1)
mean_error = np.mean(errors)
max_error = np.max(errors)
print(f"Cycle Consistency Error: Mean = {mean_error:.8f}, Max = {max_error:.8f}")

# ----------------------------------------------------
# 5. Latent Space Editing (Noise Steering)
# ----------------------------------------------------
print("\nPerforming Latent-Space Editing...")
# Select 5 spread out data points (2 from inner ring, 3 from outer ring)
inner_pts = x1_data[torch.norm(x1_data, dim=1) < 1.1].numpy()
outer_pts = x1_data[torch.norm(x1_data, dim=1) >= 1.1].numpy()

x1_real = np.array(
    [
        inner_pts[0],
        inner_pts[len(inner_pts) // 2],
        outer_pts[0],
        outer_pts[len(outer_pts) // 3],
        outer_pts[2 * len(outer_pts) // 3],
    ]
)

# Invert them to noise space (1 -> 0)
sol_invert = solve_ivp(
    flow_ode,
    t_span=(1.0, 0.0),
    y0=x1_real.flatten(),
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)
x0_inverted = sol_invert.y[:, -1].reshape(5, 2)

# Perturb the noise representation (add a constant offset/shift to slide along the manifold)
perturbation = np.array([0.5, -0.5])
x0_perturbed = x0_inverted + perturbation

# Integrate perturbed noise forward (0 -> 1)
sol_perturbed = solve_ivp(
    flow_ode,
    t_span=(0.0, 1.0),
    y0=x0_perturbed.flatten(),
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)
x1_edited = sol_perturbed.y[:, -1].reshape(5, 2)

# ----------------------------------------------------
# 6. Visualization
# ----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Plot 1: Cycle Consistency Reconstruction Errors
axes[0].hist(errors, bins=15, color="teal", edgecolor="black", alpha=0.7)
axes[0].axvline(
    mean_error,
    color="red",
    linestyle="dashed",
    linewidth=1.5,
    label=f"Mean Error: {mean_error:.2e}",
)
axes[0].set_title("Cycle Consistency Error Distribution\n(x0 -> x1 -> x0')")
axes[0].set_xlabel("L2 Reconstruction Error")
axes[0].set_ylabel("Count")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Forward & Backward Paths
axes[1].scatter(
    x1_data[:, 0], x1_data[:, 1], color="gray", alpha=0.1, s=5, label="Data Manifold"
)
# Draw paths for a few samples
t_steps = sol_forward.t
path_y = sol_forward.y.reshape(num_test, 2, -1)
back_y = sol_backward.y.reshape(num_test, 2, -1)

for idx in range(3):
    # Forward path (noise to data)
    axes[1].plot(
        path_y[idx, 0, :],
        path_y[idx, 1, :],
        color="blue",
        alpha=0.7,
        linestyle="-",
        label="Forward Path" if idx == 0 else "",
    )
    # Backward path (data to noise)
    axes[1].plot(
        back_y[idx, 0, :],
        back_y[idx, 1, :],
        color="red",
        alpha=0.7,
        linestyle="--",
        label="Backward Path" if idx == 0 else "",
    )

    # Endpoints
    axes[1].scatter(
        x0_test[idx, 0],
        x0_test[idx, 1],
        color="green",
        marker="o",
        s=40,
        zorder=5,
        label="Original x0" if idx == 0 else "",
    )
    axes[1].scatter(
        x1_gen[idx, 0],
        x1_gen[idx, 1],
        color="purple",
        marker="x",
        s=55,
        zorder=5,
        label="Generated x1" if idx == 0 else "",
    )
    axes[1].scatter(
        x0_reconstructed[idx, 0],
        x0_reconstructed[idx, 1],
        color="orange",
        marker="s",
        s=30,
        zorder=5,
        label="Reconstructed x0'" if idx == 0 else "",
    )

axes[1].set_title("Forward/Reverse ODE Trajectories")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
axes[1].set_xlim(-2.5, 2.5)
axes[1].set_ylim(-2.5, 2.5)
axes[1].grid(True, alpha=0.3)

# Plot 3: Latent Space Editing (Noise-Space Steering)
axes[2].scatter(
    x1_data[:, 0], x1_data[:, 1], color="gray", alpha=0.15, s=5, label="Data Manifold"
)
# Plot original points
axes[2].scatter(
    x1_real[:, 0],
    x1_real[:, 1],
    color="magenta",
    marker="o",
    s=60,
    edgecolors="black",
    zorder=5,
    label="Original Real x1",
)
# Plot latent noise points
axes[2].scatter(
    x0_inverted[:, 0],
    x0_inverted[:, 1],
    color="cyan",
    marker="d",
    s=60,
    edgecolors="black",
    zorder=5,
    label="Latent Noise x0",
)
# Plot perturbed noise points
axes[2].scatter(
    x0_perturbed[:, 0],
    x0_perturbed[:, 1],
    color="orange",
    marker="v",
    s=60,
    edgecolors="black",
    zorder=5,
    label="Perturbed Noise x0 + delta",
)
# Plot edited points
axes[2].scatter(
    x1_edited[:, 0],
    x1_edited[:, 1],
    color="lime",
    marker="p",
    s=80,
    edgecolors="black",
    zorder=5,
    label="Edited Real x1'",
)

# Connect original points to their latent noise representation, then to edited points
for idx in range(5):
    # Arrow showing inversion x1 -> x0
    axes[2].annotate(
        "",
        xy=(x0_inverted[idx, 0], x0_inverted[idx, 1]),
        xytext=(x1_real[idx, 0], x1_real[idx, 1]),
        arrowprops=dict(arrowstyle="->", color="purple", lw=1.2, ls=":"),
    )
    # Arrow showing editing x0 -> x0_perturbed
    axes[2].annotate(
        "",
        xy=(x0_perturbed[idx, 0], x0_perturbed[idx, 1]),
        xytext=(x0_inverted[idx, 0], x0_inverted[idx, 1]),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.2),
    )
    # Arrow showing forward path of edited point x0_perturbed -> x1_edited
    axes[2].annotate(
        "",
        xy=(x1_edited[idx, 0], x1_edited[idx, 1]),
        xytext=(x0_perturbed[idx, 0], x0_perturbed[idx, 1]),
        arrowprops=dict(arrowstyle="->", color="green", lw=1.2, ls="-."),
    )

    # Find the nearest 3 points in the initial noise distribution (t=0) for justification
    # We compare the perturbed noise point to a reference set of training noise points
    x0_ref = np.random.normal(0, 1.0, (3000, 2))
    dists = np.linalg.norm(x0_ref - x0_perturbed[idx], axis=1)
    nearest_idxs = np.argsort(dists)[:3]
    nearest_pts = x0_ref[nearest_idxs]

    # Plot nearest training noise points
    axes[2].scatter(
        nearest_pts[:, 0],
        nearest_pts[:, 1],
        color="red",
        marker="*",
        s=45,
        zorder=4,
        label="Nearest Train Noise x0" if idx == 0 else "",
    )
    # Draw line from perturbed noise point to its 3 nearest training noise points
    for pt in nearest_pts:
        axes[2].plot(
            [x0_perturbed[idx, 0], pt[0]],
            [x0_perturbed[idx, 1], pt[1]],
            color="red",
            linestyle=":",
            alpha=0.6,
            lw=1,
        )

    print(
        f"Perturbed Noise {idx+1} at [{x0_perturbed[idx, 0]:.4f}, {x0_perturbed[idx, 1]:.4f}]: Distances to 3 nearest train noise pts: {dists[nearest_idxs]}"
    )

axes[2].set_title(
    "Latent Space Editing / FRS\n(Real x1 -> Noise x0 -> +shift -> Edited x1')"
)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].legend()
axes[2].set_xlim(-2.5, 2.5)
axes[2].set_ylim(-2.5, 2.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Task 3 complete! Output plot saved to {PLOT_PATH}")
