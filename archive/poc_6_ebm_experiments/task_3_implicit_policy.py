import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "task_3_output.png")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------
# 1. Dataset Generation: Multi-Modal Avoidance Paths
# ----------------------------------------------------
def generate_expert_trajectories(num_episodes=150, step_size=0.15):
    states = []
    actions = []

    for ep in range(num_episodes):
        go_right = ep % 2 == 0
        curr_state = np.array([0.0, -2.0])

        while curr_state[1] < 2.0:
            y = curr_state[1]
            next_y = y + step_size

            sign = 1.0 if go_right else -1.0
            if np.abs(next_y) > 1.0:
                next_x = 0.0
            else:
                next_x = sign * 0.85 * np.cos(next_y * np.pi / 2)

            next_state = np.array([next_x, next_y])
            action = next_state - curr_state
            # Add small noise
            action += np.random.normal(0, 0.015, size=2)

            states.append(curr_state.copy())
            actions.append(action)
            curr_state = curr_state + action

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.float32),
    )


x_states, y_actions = generate_expert_trajectories(200)
print(
    f"Generated behavior cloning dataset. States: {x_states.shape}, Actions: {y_actions.shape}"
)


# ----------------------------------------------------
# 2. Action-Conditioned EBM Policy Network
# ----------------------------------------------------
class ActionEBM(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: state (2) + action (2) = 4
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)
        return self.net(inputs)


model = ActionEBM()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# ----------------------------------------------------
# 3. NCE Policy Training
# ----------------------------------------------------
epochs = 2500
batch_size = 256
nu = 4  # Noise-to-data ratio
action_noise_std = 0.25  # Standard deviation of action noise distribution q(a)


# Noise prior log q(a)
def log_q_action(a):
    return -np.log(2 * np.pi * (action_noise_std**2)) - torch.sum(
        a**2, dim=-1, keepdim=True
    ) / (2 * (action_noise_std**2))


print("Training Action-Conditioned EBM via NCE...")
for epoch in range(epochs + 1):
    indices = np.random.choice(len(x_states), batch_size)
    s_batch = x_states[indices]
    a_batch = y_actions[indices]

    # 1. Sample negative actions from q(a)
    a_noise = torch.randn(batch_size * nu, 2) * action_noise_std
    s_noise = s_batch.repeat_interleave(
        nu, dim=0
    )  # Repeat states to match noise actions

    # 2. Compute unnormalized log probability log p = -E(a | s)
    log_p_real = -model(s_batch, a_batch)
    log_p_noise = -model(s_noise, a_noise)

    # 3. Compute log q(a)
    log_q_real = log_q_action(a_batch)
    log_q_noise = log_q_action(a_noise)

    # NCE Loss computation
    log_nu = np.log(nu)
    d_real = torch.sigmoid(log_p_real - log_q_real - log_nu)
    d_noise = torch.sigmoid(log_p_noise - log_q_noise - log_nu)

    loss = -(
        torch.log(d_real + 1e-7).mean() + nu * torch.log(1 - d_noise + 1e-7).mean()
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 250 == 0:
        print(f"Epoch {epoch}/{epochs} - NCE Policy Loss: {loss.item():.4f}")


# ----------------------------------------------------
# 4. SGLD Sampler for Action Selection at Runtime
# ----------------------------------------------------
def select_action_sgld(model, state, steps=40, step_size=0.03, noise_std=0.005):
    # Initialize actions randomly
    a = torch.randn(1, 2) * 0.1
    a.requires_grad_(True)

    # Replicate state to match action batch size (1)
    s = state.unsqueeze(0)

    for _ in range(steps):
        energy = model(s, a)
        grad = torch.autograd.grad(energy.sum(), a)[0]
        grad = torch.clamp(grad, min=-2.0, max=2.0)

        noise = noise_std * torch.randn_like(a)
        a.data = a.data - 0.5 * step_size * grad + noise
        # Bound actions to physical velocity limit
        a.data = torch.clamp(a.data, min=-0.3, max=0.3)

    return a.detach().squeeze(0)


# ----------------------------------------------------
# 5. Policy Evaluation & Rollout Simulation
# ----------------------------------------------------
print("\nRunning EBM policy rollouts...")
rollout_paths = []
num_rollouts = 15

for i in range(num_rollouts):
    state = torch.tensor([0.0, -2.0], dtype=torch.float32)
    path = [state.numpy().copy()]

    # Step simulation
    step = 0
    max_steps = 35
    while state[1] < 1.9 and step < max_steps:
        # Action selection via SGLD
        action = select_action_sgld(model, state, steps=50, step_size=0.03)
        state = state + action
        path.append(state.numpy().copy())
        step += 1

    print(f"  Rollout {i+1}/{num_rollouts} completed in {step} steps.")
    rollout_paths.append(np.array(path))

# ----------------------------------------------------
# 6. Visualization & Plot Generation
# ----------------------------------------------------
print("Generating final EBM policy visualizations to:", PLOT_PATH)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Action Energy Landscape E(a | s) at s = (0.0, -0.5)
# This is when the agent is approaching the obstacle and must decide left or right
state_test = torch.tensor([0.0, -0.5], dtype=torch.float32)

grid_size = 100
ax_grid = np.linspace(-0.35, 0.35, grid_size)
ay_grid = np.linspace(-0.1, 0.3, grid_size)
axx, ayy = np.meshgrid(ax_grid, ay_grid)
action_pts = np.vstack([axx.ravel(), ayy.ravel()]).T
action_pts_tensor = torch.tensor(action_pts, dtype=torch.float32)
state_pts_tensor = state_test.unsqueeze(0).repeat(len(action_pts_tensor), 1)

with torch.no_grad():
    action_energies = (
        model(state_pts_tensor, action_pts_tensor).numpy().reshape(grid_size, grid_size)
    )

im = axs[0].contourf(axx, ayy, action_energies, levels=30, cmap="viridis")
fig.colorbar(im, ax=axs[0], label="Energy E(a | s)")
axs[0].set_title("Action Space Energy Landscape at s = (0, -0.5)")
axs[0].set_xlabel("Action dx (Horizontal Velocity)")
axs[0].set_ylabel("Action dy (Vertical Velocity)")

# Highlight the two expected expert action modes at this state
# Left mode: dx < 0, dy > 0. Right mode: dx > 0, dy > 0
axs[0].scatter(
    [-0.18, 0.18],
    [0.12, 0.12],
    color="red",
    marker="x",
    s=80,
    label="Expert Modes",
    zorder=5,
)
axs[0].legend()

# Subplot 2: Trajectory rollouts around the obstacle
# Draw obstacle
obstacle_circle = plt.Circle((0, 0), 0.6, color="grey", alpha=0.3, label="Obstacle")
axs[1].add_patch(obstacle_circle)

# Plot a subset of expert trajectories
for i in range(15):
    expert_idx = i * 20
    # Reconstruct path from dataset
    states_subset = x_states[expert_idx : expert_idx + 25].numpy()
    if len(states_subset) > 0:
        axs[1].plot(
            states_subset[:, 0],
            states_subset[:, 1],
            color="blue",
            alpha=0.15,
            linestyle="--",
        )

# Plot rollout trajectories
first_rollout = True
for path in rollout_paths:
    label = "EBM Rollout" if first_rollout else None
    axs[1].plot(
        path[:, 0], path[:, 1], color="red", alpha=0.7, linewidth=2.0, label=label
    )
    first_rollout = False

axs[1].set_title("Expert Demonstrations vs. EBM Rollouts")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].set_xlim(-1.5, 1.5)
axs[1].set_ylim(-2.2, 2.2)
axs[1].legend()
axs[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("Task 3 completed successfully.")
