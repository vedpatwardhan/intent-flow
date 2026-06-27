import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "trajectory_comparison.png")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------
# 1. Data Generation (Bimodal Obstacle Avoidance)
# ----------------------------------------------------
def generate_data(num_samples=200):
    obstacle = np.array([0.5, 0.5])
    target = np.array([1.0, 1.0])

    x_list = []
    while len(x_list) < num_samples:
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.8, 1.2)
        pt = np.array([r * np.cos(theta), r * np.sin(theta)])
        if np.linalg.norm(pt - obstacle) >= 0.45:
            x_list.append(pt)

    x = np.stack(x_list, axis=0)
    y = np.zeros_like(x)

    for i in range(num_samples):
        direction = target - x[i]
        to_obstacle = obstacle - x[i]
        projection = np.dot(to_obstacle, direction) / np.dot(direction, direction)
        closest_point = x[i] + np.clip(projection, 0, 1) * direction
        dist = np.linalg.norm(closest_point - obstacle)

        if dist < 0.4:
            perp = np.array([-direction[1], direction[0]])
            perp = perp / np.linalg.norm(perp)
            if np.dot(perp, to_obstacle) > 0:
                perp = -perp
            y[i] = target + 0.35 * perp
        else:
            y[i] = target

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


x_data, y_target = generate_data()


# ----------------------------------------------------
# 2. Define the Target Model Architecture
# ----------------------------------------------------
class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)


# Get the initial (untrained) output distribution p_0
init_model = TargetModel()
with torch.no_grad():
    y_init = init_model(x_data).clone()


# ----------------------------------------------------
# 3. Train the Flow Matching Model (Geodesic Prior)
# ----------------------------------------------------
# Learn the straight vector field (flow) transporting p_0 (init) to p_1 (target)
class FlowMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, y, t, x):
        inputs = torch.cat([y, t, x], dim=-1)
        return self.net(inputs)


flow_model = FlowMatchingModel()
flow_optimizer = optim.Adam(flow_model.parameters(), lr=0.005)


# Function to define a mathematically consistent curved path avoiding the obstacle
def get_curved_path(y0, y1, t):
    yt = (1 - t) * y0 + t * y1
    obstacle = torch.tensor([0.5, 0.5])
    to_obstacle = yt - obstacle
    dist = torch.norm(to_obstacle, dim=-1, keepdim=True)

    # Boundary slightly larger than the obstacle radius (0.45 vs 0.4)
    repulsion_radius = 0.45
    repulsion_mask = (dist < repulsion_radius).float()
    repulsive_dir = to_obstacle / (dist + 1e-6)

    # Scale repulsion by t*(1-t) so it smoothly fades to 0 at the endpoints (t=0 and t=1)
    fade = 4.0 * t * (1.0 - t)

    # Deflect the path coordinates
    yt_curved = (
        yt + 0.35 * (repulsion_radius - dist) * repulsion_mask * repulsive_dir * fade
    )
    return yt_curved


print("Training Flow Matching Model (Curved Geodesic Transport)...")
flow_epochs = 1200
for epoch in range(flow_epochs):
    flow_optimizer.zero_grad()

    # Sample random virtual time t in [0, 1]
    t = torch.rand(y_init.shape[0], 1)

    # Compute the curved path coordinates yt
    y_t = get_curved_path(y_init, y_target, t)

    # Compute velocity target numerically to satisfy the continuity equation
    dt = 1e-3
    y_t_plus_dt = get_curved_path(y_init, y_target, t + dt)
    v_target = (y_t_plus_dt - y_t) / dt

    # Predict velocity
    v_pred = flow_model(y_t, t, x_data)

    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    flow_optimizer.step()

    if epoch % 100 == 0:
        print(f"Flow Matching Epoch {epoch}/{flow_epochs} - Loss: {loss.item():.6f}")

# ----------------------------------------------------
# 4. Integrate Flow to get Intermediate Targets
# ----------------------------------------------------
# Compute the flow-matching guided path targets by integrating the velocity field
print("\nGenerating guided path targets via ODE integration...")


def flow_ode(t, y_flat, x_np):
    y_tensor = torch.tensor(y_flat.reshape(-1, 2), dtype=torch.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    t_tensor = torch.full((y_tensor.shape[0], 1), t, dtype=torch.float32)
    with torch.no_grad():
        v = flow_model(y_tensor, t_tensor, x_tensor)
    return v.numpy().flatten()


x_np = x_data.numpy()
y_init_np = y_init.numpy().flatten()
num_eval_steps = 11
eval_times = np.linspace(0.0, 1.0, num_eval_steps)

sol = solve_ivp(
    flow_ode,
    t_span=(0.0, 1.0),
    y0=y_init_np,
    args=(x_np,),
    t_eval=eval_times,
    method="RK45",
)

# Extract guided targets for each training epoch step
guided_targets = []
for i in range(num_eval_steps):
    guided_targets.append(torch.tensor(sol.y[:, i].reshape(-1, 2), dtype=torch.float32))

# ----------------------------------------------------
# 5. A/B Comparative Training Runs
# ----------------------------------------------------
epochs = 100
eval_interval = epochs // (num_eval_steps - 1)

# --- RUN A: Baseline Training (Unguided) ---
print("\nRunning Baseline Training (Unguided)...")
baseline_model = TargetModel()
baseline_model.load_state_dict(init_model.state_dict())  # start from same init
baseline_opt = optim.Adam(baseline_model.parameters(), lr=0.01)

baseline_losses = []
baseline_trajectories = []

for epoch in range(epochs + 1):
    baseline_opt.zero_grad()
    outputs = baseline_model(x_data)
    # Unguided training always pushes directly to final targets
    loss = nn.MSELoss()(outputs, y_target)
    loss.backward()
    baseline_opt.step()

    if epoch % eval_interval == 0:
        baseline_losses.append(loss.item())
        baseline_trajectories.append(outputs.detach().clone())
        print(f"Baseline Epoch {epoch} - Loss: {loss.item():.4f}")

# --- RUN B: Flow-Guided Training (Geodesic Flow) ---
print("\nRunning Flow-Guided Training...")
guided_model = TargetModel()
guided_model.load_state_dict(init_model.state_dict())  # start from same init
guided_opt = optim.Adam(guided_model.parameters(), lr=0.01)

guided_losses = []
guided_trajectories = []

for epoch in range(epochs + 1):
    guided_opt.zero_grad()
    outputs = guided_model(x_data)

    # Determine which flow step targets to guide with
    step_idx = min(epoch // eval_interval, num_eval_steps - 1)
    target_step = guided_targets[step_idx]

    # Guide output towards the intermediate flow-predicted geodesic state
    loss = nn.MSELoss()(outputs, target_step)
    loss.backward()
    guided_opt.step()

    # Track loss relative to the true final targets for fair comparison
    with torch.no_grad():
        true_loss = nn.MSELoss()(outputs, y_target)

    if epoch % eval_interval == 0:
        guided_losses.append(true_loss.item())
        guided_trajectories.append(outputs.detach().clone())
        print(f"Guided Epoch {epoch} - True Target Loss: {true_loss.item():.4f}")

# ----------------------------------------------------
# 6. Visualization & Trajectory Straightness Comparison
# ----------------------------------------------------
obstacle = np.array([0.5, 0.5])
plt.figure(figsize=(20, 5))

# Subplot 1: Convergence Speed Comparison
plt.subplot(1, 4, 1)
plt.plot(eval_times * epochs, baseline_losses, "b-o", label="Baseline (Unguided)")
plt.plot(eval_times * epochs, guided_losses, "r-o", label="Flow-Guided")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss to Final Targets")
plt.title("Convergence Rate Comparison")
plt.legend()
plt.grid(True)

# Subplot 2: Baseline Output Trajectories (First 5 samples)
plt.subplot(1, 4, 2)
plt.scatter(obstacle[0], obstacle[1], color="red", s=150, marker="x", label="Obstacle")
for sample_idx in range(5):
    traj_x = [t[sample_idx, 0].item() for t in baseline_trajectories]
    traj_y = [t[sample_idx, 1].item() for t in baseline_trajectories]
    plt.plot(traj_x, traj_y, "b--o", alpha=0.6)
plt.title("Baseline Paths (Curved)")
plt.grid(True)

# Subplot 3: Ideal Flow Trajectories (First 5 samples from ODE integration)
plt.subplot(1, 4, 3)
plt.scatter(obstacle[0], obstacle[1], color="red", s=150, marker="x", label="Obstacle")
for sample_idx in range(5):
    traj_x = [t[sample_idx, 0].item() for t in guided_targets]
    traj_y = [t[sample_idx, 1].item() for t in guided_targets]
    plt.plot(traj_x, traj_y, "g--o", alpha=0.6)
plt.title("Ideal Flow Paths (Geodesic)")
plt.grid(True)

# Subplot 4: Flow-Guided Output Trajectories (First 5 samples)
plt.subplot(1, 4, 4)
plt.scatter(obstacle[0], obstacle[1], color="red", s=150, marker="x", label="Obstacle")
for sample_idx in range(5):
    traj_x = [t[sample_idx, 0].item() for t in guided_trajectories]
    traj_y = [t[sample_idx, 1].item() for t in guided_trajectories]
    plt.plot(traj_x, traj_y, "r--o", alpha=0.6)
plt.title("Flow-Guided Paths (Straightened)")
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nA/B test complete! Trajectory comparison saved to {PLOT_PATH}")

# Generate and save a separate plot for all 200 ideal flow trajectories to visualize the vector field
plt.figure(figsize=(8, 8))
plt.scatter(obstacle[0], obstacle[1], color="red", s=200, marker="x", label="Obstacle")
plt.scatter(1.0, 1.0, color="green", s=250, marker="*", zorder=5, label="Target")
for sample_idx in range(len(x_data)):
    traj_x = [t[sample_idx, 0].item() for t in guided_targets]
    traj_y = [t[sample_idx, 1].item() for t in guided_targets]
    plt.plot(traj_x, traj_y, "g-", alpha=0.15)
plt.title("Ideal Geodesic Flow Paths (All 200 Samples)")
plt.legend()
plt.grid(True)
ideal_flow_plot_path = os.path.join(SCRIPT_DIR, "ideal_flow_paths.png")
plt.savefig(ideal_flow_plot_path)
print(f"Separate ideal flow paths plot saved to {ideal_flow_plot_path}")
