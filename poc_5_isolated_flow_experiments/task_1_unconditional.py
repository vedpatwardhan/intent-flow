import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_1_output.png")

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
print(f"Generated target dataset with shape: {x1_data.shape}")

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
            nn.Linear(64, 2)
        )
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=-1)
        return self.net(inputs)

flow_model = FlowModel()
optimizer = optim.Adam(flow_model.parameters(), lr=0.003)
epochs = 1500
batch_size = 256

print("Training Unconditional Flow Matching Model...")
for epoch in range(epochs):
    # Sample a batch
    indices = np.random.choice(len(x1_data), batch_size)
    x1 = x1_data[indices]
    
    # Sample Gaussian noise
    x0 = torch.randn_like(x1)
    
    # Sample random times t
    t = torch.rand(batch_size, 1)
    
    # Linear path interpolation
    x_t = (1 - t) * x0 + t * x1
    v_target = x1 - x0
    
    # Train
    optimizer.zero_grad()
    v_pred = flow_model(x_t, t)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    optimizer.step()
    
    if epoch % 300 == 0:
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.6f}")

# ----------------------------------------------------
# 3. ODE Integration (t=0 to t=1)
# ----------------------------------------------------
print("\nGenerating new samples via ODE integration...")
def flow_ode(t, x_flat):
    x_tensor = torch.tensor(x_flat.reshape(-1, 2), dtype=torch.float32)
    t_tensor = torch.full((x_tensor.shape[0], 1), t, dtype=torch.float32)
    with torch.no_grad():
        v = flow_model(x_tensor, t_tensor)
    return v.numpy().flatten()

# Sample 1000 noise points
num_gen = 1000
x0_test = torch.randn(num_gen, 2).numpy().flatten()
num_eval_steps = 20
eval_times = np.linspace(0.0, 1.0, num_eval_steps)

sol = solve_ivp(
    flow_ode,
    t_span=(0.0, 1.0),
    y0=x0_test,
    t_eval=eval_times,
    method='RK45'
)

# Generated coordinates at t=1.0
x1_gen = sol.y[:, -1].reshape(-1, 2)

# Extract trajectory history for 5 sample paths
trajectories = []
for i in range(5):
    # Reshape sol.y to (num_gen, 2, num_eval_steps)
    path = sol.y.reshape(num_gen, 2, num_eval_steps)[i]  # Shape: (2, num_eval_steps)
    trajectories.append(path.T)

# ----------------------------------------------------
# 4. Streamplot Vector Field Grid
# ----------------------------------------------------
print("Computing vector field flow grid...")
grid_range = np.linspace(-2.5, 2.5, 30)
X, Y = np.meshgrid(grid_range, grid_range)
grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)

# Evaluate velocity field at t = 0.5
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
t_mid = torch.full((grid_tensor.shape[0], 1), 0.5, dtype=torch.float32)
with torch.no_grad():
    v_mid = flow_model(grid_tensor, t_mid).numpy()

U = v_mid[:, 0].reshape(X.shape)
V = v_mid[:, 1].reshape(Y.shape)

# ----------------------------------------------------
# 5. Visualization
# ----------------------------------------------------
plt.figure(figsize=(18, 5))

# Subplot 1: Target vs Generated Distributions
plt.subplot(1, 3, 1)
plt.scatter(x1_data[:, 0], x1_data[:, 1], color='gray', alpha=0.2, s=8, label='Target Data')
plt.scatter(x1_gen[:, 0], x1_gen[:, 1], color='crimson', alpha=0.5, s=8, label='Generated')
plt.title("Target vs. Generated Distributions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid(True)

# Subplot 2: Trajectory Paths (Noise -> Ring)
plt.subplot(1, 3, 2)
plt.scatter(x1_data[:, 0], x1_data[:, 1], color='gray', alpha=0.1, s=8)
for i, path in enumerate(trajectories):
    plt.plot(path[:, 0], path[:, 1], '--o', markersize=3, alpha=0.8, label=f'Path {i+1}')
    plt.scatter(path[0, 0], path[0, 1], color='black', marker='s', s=35, zorder=5)  # Start (Noise)
    plt.scatter(path[-1, 0], path[-1, 1], color='red', marker='*', s=60, zorder=5)   # End (Data)
plt.title("Sample Integration Paths (t=0 -> t=1)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid(True)

# Subplot 3: Vector Field Streamplot (at t=0.5)
plt.subplot(1, 3, 3)
plt.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap='autumn', density=1.2)
plt.scatter(x1_gen[:, 0], x1_gen[:, 1], color='black', alpha=0.15, s=4)
plt.title("Learned Velocity Field Streamlines (t=0.5)")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Task 1 complete! Output plot saved to {PLOT_PATH}")
