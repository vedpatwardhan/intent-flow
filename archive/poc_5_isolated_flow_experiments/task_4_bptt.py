import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_4_output.png")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------
# 1. Base Dataset and Reward Setup
# ----------------------------------------------------
def generate_double_ring(num_samples=2000):
    samples = []
    for _ in range(num_samples):
        theta = np.random.uniform(0, 2 * np.pi)
        if np.random.rand() < 0.5:
            r = np.random.uniform(0.6, 0.9)
        else:
            r = np.random.uniform(1.6, 1.9)
        samples.append([r * np.cos(theta), r * np.sin(theta)])
    return torch.tensor(np.array(samples), dtype=torch.float32)


x1_data = generate_double_ring(2000)

target_pt = np.array([1.75, 0.0])
obstacle_pt = np.array([1.0, 0.0])


def reward_fn(x1):
    dist_to_target = np.linalg.norm(x1 - target_pt, axis=1)
    dist_to_obstacle = np.linalg.norm(x1 - obstacle_pt, axis=1)
    r_target = np.exp(-(dist_to_target**2) / 0.4)
    r_obstacle = -1.5 * np.exp(-(dist_to_obstacle**2) / 0.15)
    return r_target + r_obstacle


# ----------------------------------------------------
# 2. Base Flow Model Definition and Pre-training
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

print("Pre-training base Flow Matching model on double ring...")
pretrain_epochs = 1000
batch_size = 256

for epoch in range(pretrain_epochs):
    indices = np.random.choice(len(x1_data), batch_size)
    x1 = x1_data[indices]
    x0 = torch.randn_like(x1) * 0.5
    t = torch.rand(batch_size, 1)

    x_t = (1 - t) * x0 + t * x1
    v_target = x1 - x0

    optimizer.zero_grad()
    v_pred = flow_model(x_t, t)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    optimizer.step()

# Save pre-trained state
pretrained_state = {k: v.clone() for k, v in flow_model.state_dict().items()}


# Helper to run ODE integration
def get_ode_solver(model):
    def flow_ode(t, x_flat):
        x_tensor = torch.tensor(x_flat.reshape(-1, 2), dtype=torch.float32)
        t_tensor = torch.full((x_tensor.shape[0], 1), t, dtype=torch.float32)
        with torch.no_grad():
            v = model(x_tensor, t_tensor)
        return v.numpy().flatten()

    return flow_ode


# ----------------------------------------------------
# 3. Differentiable Path RL alignment
# ----------------------------------------------------
base_model = FlowModel()
base_model.load_state_dict(pretrained_state)
base_model.eval()

print("\nRunning Differentiable Path RL alignment...")
rl_optimizer = optim.Adam(flow_model.parameters(), lr=0.005)
rl_epochs = 1000
rl_batch_size = 128
num_steps = 15

# Track progression snapshots
eval_samples = 40
x0_eval = np.random.normal(0, 0.2, (eval_samples, 2))
eval_times = np.linspace(0.0, 1.0, 25)
snapshots = {}
snapshot_epochs = [0, 250, 500, 750]

for epoch in range(rl_epochs):
    # Log path snapshot at current stage
    if epoch in snapshot_epochs:
        sol_snapshot = solve_ivp(
            get_ode_solver(flow_model),
            t_span=(0.0, 1.0),
            y0=x0_eval.flatten(),
            t_eval=eval_times,
            method="RK45",
        )
        snapshots[epoch] = sol_snapshot.y.reshape(eval_samples, 2, len(eval_times))

    x0 = torch.randn(rl_batch_size, 2) * 0.2

    x = x0
    dt = 1.0 / num_steps
    max_obstacle_penalty = torch.zeros(rl_batch_size, device=x.device)

    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((rl_batch_size, 1), t_val, dtype=torch.float32)
        v = flow_model(x, t_tensor)
        x = x + v * dt

        dist_to_obstacle = torch.norm(
            x - torch.tensor(obstacle_pt, dtype=torch.float32), dim=-1
        )
        penalty_step = 3.5 * torch.exp(-(dist_to_obstacle**2) / 0.12)
        max_obstacle_penalty = torch.max(max_obstacle_penalty, penalty_step)

    dist_to_target = torch.norm(
        x - torch.tensor(target_pt, dtype=torch.float32), dim=-1
    )
    r_target = torch.exp(-(dist_to_target**2) / 0.15)

    rewards = r_target - max_obstacle_penalty
    # Minimize negative reward (maximize reward)
    loss_rl = -torch.mean(rewards)

    # Supervised regularization to preserve the double-ring prior for non-target trajectories
    indices = np.random.choice(len(x1_data), rl_batch_size)
    x1_sup = x1_data[indices]
    x0_sup = torch.randn_like(x1_sup) * 0.2
    t_sup = torch.rand(rl_batch_size, 1)
    x_t_sup = (1 - t_sup) * x0_sup + t_sup * x1_sup
    v_target_sup = x1_sup - x0_sup
    loss_reg = nn.MSELoss()(flow_model(x_t_sup, t_sup), v_target_sup)

    loss_total = loss_rl + 1.5 * loss_reg

    rl_optimizer.zero_grad()
    loss_total.backward()
    rl_optimizer.step()

    if epoch % 200 == 0:
        print(
            f"RL Epoch {epoch}/{rl_epochs} | Loss: {loss_total.item():.4f} | Mean Reward: {torch.mean(rewards).item():.4f}"
        )

# ----------------------------------------------------
# 4. Verification and Visualization
# ----------------------------------------------------
print("\nGenerating evaluation plots...")

# Integrate Pre-trained paths
sol_base = solve_ivp(
    get_ode_solver(base_model),
    t_span=(0.0, 1.0),
    y0=x0_eval.flatten(),
    t_eval=eval_times,
    method="RK45",
)
paths_base = sol_base.y.reshape(eval_samples, 2, len(eval_times))

# Integrate RAM aligned paths (Final Epoch 1000)
sol_ram = solve_ivp(
    get_ode_solver(flow_model),
    t_span=(0.0, 1.0),
    y0=x0_eval.flatten(),
    t_eval=eval_times,
    method="RK45",
)
paths_ram = sol_ram.y.reshape(eval_samples, 2, len(eval_times))
snapshots[1000] = paths_ram

# Generate reward landscape grid
grid_range = np.linspace(-2.2, 2.2, 100)
X, Y = np.meshgrid(grid_range, grid_range)
grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
Z = reward_fn(grid_points).reshape(X.shape)

# Create 5-panel Progression Plot
plt.figure(figsize=(24, 4.8))
plot_epochs = [0, 250, 500, 750, 1000]
for idx, ep in enumerate(plot_epochs):
    plt.subplot(1, 5, idx + 1)
    plt.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.2)
    paths_ep = snapshots[ep]
    for i in range(eval_samples):
        plt.plot(
            paths_ep[i, 0], paths_ep[i, 1], color="green", alpha=0.5, linewidth=1.0
        )
    plt.scatter(
        target_pt[0],
        target_pt[1],
        color="red",
        marker="*",
        s=120,
        edgecolors="black",
        zorder=10,
    )
    plt.scatter(
        obstacle_pt[0],
        obstacle_pt[1],
        color="black",
        marker="X",
        s=120,
        edgecolors="red",
        zorder=10,
    )
    plt.title(f"Epoch {ep}")
    plt.xlim(-2.2, 2.2)
    plt.ylim(-2.2, 2.2)
    plt.grid(True, alpha=0.2)
plt.tight_layout()
PROGRESSION_PLOT_PATH = os.path.join(SCRIPT_DIR, "task_4_progression.png")
plt.savefig(PROGRESSION_PLOT_PATH)
print(f"Progression plot saved to {PROGRESSION_PLOT_PATH}")


# Create figures
plt.figure(figsize=(18, 5.5))

# Plot 1: Pre-trained Model Trajectories on Reward Landscape
plt.subplot(1, 3, 1)
plt.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.3)
plt.colorbar(label="Reward Value")
for i in range(eval_samples):
    plt.plot(paths_base[i, 0], paths_base[i, 1], color="blue", alpha=0.4, linewidth=1)
plt.scatter(
    target_pt[0],
    target_pt[1],
    color="red",
    marker="*",
    s=150,
    edgecolors="black",
    zorder=10,
    label="Target",
)
plt.scatter(
    obstacle_pt[0],
    obstacle_pt[1],
    color="black",
    marker="X",
    s=150,
    edgecolors="red",
    zorder=10,
    label="Obstacle",
)
plt.title("Pre-trained Model Paths (No RL)\n(Collide with Obstacle / Straight Paths)")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: RAM-Aligned Model Trajectories on Reward Landscape
plt.subplot(1, 3, 2)
plt.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.3)
plt.colorbar(label="Reward Value")
for i in range(eval_samples):
    plt.plot(paths_ram[i, 0], paths_ram[i, 1], color="green", alpha=0.5, linewidth=1.2)
plt.scatter(
    target_pt[0],
    target_pt[1],
    color="red",
    marker="*",
    s=150,
    edgecolors="black",
    zorder=10,
)
plt.scatter(
    obstacle_pt[0],
    obstacle_pt[1],
    color="black",
    marker="X",
    s=150,
    edgecolors="red",
    zorder=10,
)
plt.title("RAM-Aligned Paths (After RL)\n(Bends around Obstacle to reach Target)")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)
plt.grid(True, alpha=0.3)

# Plot 3: RAM Velocity Field Streamplot at t=0.5
plt.subplot(1, 3, 3)
grid_stream = np.linspace(-2.2, 2.2, 30)
XS, YS = np.meshgrid(grid_stream, grid_stream)
grid_stream_pts = np.stack([XS.flatten(), YS.flatten()], axis=1)
grid_tensor = torch.tensor(grid_stream_pts, dtype=torch.float32)
t_mid = torch.full((grid_tensor.shape[0], 1), 0.5, dtype=torch.float32)
with torch.no_grad():
    v_mid = flow_model(grid_tensor, t_mid).numpy()
US = v_mid[:, 0].reshape(XS.shape)
VS = v_mid[:, 1].reshape(YS.shape)

plt.contourf(XS, YS, Z[:30, :30], levels=20, cmap="viridis", alpha=0.1)
plt.streamplot(XS, YS, US, VS, color=np.sqrt(US**2 + VS**2), cmap="autumn", density=1.2)
plt.scatter(
    target_pt[0],
    target_pt[1],
    color="red",
    marker="*",
    s=150,
    edgecolors="black",
    zorder=10,
)
plt.scatter(
    obstacle_pt[0],
    obstacle_pt[1],
    color="black",
    marker="X",
    s=150,
    edgecolors="red",
    zorder=10,
)
plt.title("RAM-Aligned Velocity Field (t=0.5)")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Task 4 complete! Output plot saved to {PLOT_PATH}")
