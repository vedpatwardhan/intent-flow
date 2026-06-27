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

# Define static coordinates
KEY = np.array([0.2, 0.8])
DOOR = np.array([0.8, 0.5])
TREASURE = np.array([0.9, 0.9])


# ----------------------------------------------------
# 1. Dataset Generation (Heuristic Oracle Solutions)
# ----------------------------------------------------
# State representation (10D):
# [pos_x, pos_y, key_x, key_y, door_x, door_y, treasure_x, treasure_y, carrying_key, door_unlocked]
# Action representation (2D): [v_x, v_y]
def generate_dataset(num_trajectories=150, max_steps=50, step_size=0.05):
    states_list = []
    actions_list = []

    for _ in range(num_trajectories):
        pos = np.random.uniform(0.05, 0.45, 2)
        carrying_key = 0.0
        door_unlocked = 0.0

        for _ in range(max_steps):
            # Form state vector
            state = np.concatenate(
                [pos, KEY, DOOR, TREASURE, [carrying_key, door_unlocked]]
            )

            # Determine target element
            if carrying_key == 0.0:
                target = KEY
                dist = np.linalg.norm(target - pos)
                if dist < 0.06:
                    carrying_key = 1.0
            elif door_unlocked == 0.0:
                target = DOOR
                dist = np.linalg.norm(target - pos)
                if dist < 0.06:
                    door_unlocked = 1.0
            else:
                target = TREASURE
                dist = np.linalg.norm(target - pos)
                if dist < 0.06:
                    # Final step reached
                    action = np.zeros(2)
                    states_list.append(state)
                    actions_list.append(action)
                    break

            # Compute oracle action
            direction = target - pos
            action = direction / (np.linalg.norm(direction) + 1e-6)

            states_list.append(state)
            actions_list.append(action)

            # Move agent
            pos += action * step_size

    return (
        torch.tensor(np.array(states_list), dtype=torch.float32),
        torch.tensor(np.array(actions_list), dtype=torch.float32),
    )


states_data, actions_target = generate_dataset()
print(f"Generated dataset with {states_data.shape[0]} state-action pairs.")


# ----------------------------------------------------
# 2. Target Model Definition
# ----------------------------------------------------
class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2)
        )

    def forward(self, s):
        return self.net(s)


# Get the initial (untrained) output distribution p_0
init_model = TargetModel()
with torch.no_grad():
    actions_init = init_model(states_data).clone()


# ----------------------------------------------------
# 3. Flow Matching Model: Learning Geodesic Action Flow
# ----------------------------------------------------
class FlowMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: action_t (2D) + time t (1D) + conditioning state s (10D)
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 10, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )

    def forward(self, a, t, s):
        inputs = torch.cat([a, t, s], dim=-1)
        return self.net(inputs)


flow_model = FlowMatchingModel()
flow_optimizer = optim.Adam(flow_model.parameters(), lr=0.005)

print("\nTraining Flow Matching Model on Action Space...")
flow_epochs = 1200
for epoch in range(flow_epochs):
    flow_optimizer.zero_grad()

    t = torch.rand(actions_init.shape[0], 1)

    # Linear interpolation (geodesic path) in Action Space
    a_t = (1 - t) * actions_init + t * actions_target
    v_target = actions_target - actions_init

    v_pred = flow_model(a_t, t, states_data)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    flow_optimizer.step()

    if epoch % 200 == 0:
        print(f"Flow Matching Epoch {epoch}/{flow_epochs} - Loss: {loss.item():.6f}")

# ----------------------------------------------------
# 4. Integrate Flow to get Intermediate Action Targets
# ----------------------------------------------------
print("\nGenerating guided action targets via ODE integration...")


def flow_ode(t, a_flat, s_np):
    a_tensor = torch.tensor(a_flat.reshape(-1, 2), dtype=torch.float32)
    s_tensor = torch.tensor(s_np, dtype=torch.float32)
    t_tensor = torch.full((a_tensor.shape[0], 1), t, dtype=torch.float32)
    with torch.no_grad():
        v = flow_model(a_tensor, t_tensor, s_tensor)
    return v.numpy().flatten()


s_np = states_data.numpy()
a_init_np = actions_init.numpy().flatten()
num_eval_steps = 11
eval_times = np.linspace(0.0, 1.0, num_eval_steps)

sol = solve_ivp(
    flow_ode,
    t_span=(0.0, 1.0),
    y0=a_init_np,
    args=(s_np,),
    t_eval=eval_times,
    method="RK45",
)

guided_targets = []
for i in range(num_eval_steps):
    guided_targets.append(torch.tensor(sol.y[:, i].reshape(-1, 2), dtype=torch.float32))

import copy

# ----------------------------------------------------
# 5. A/B Comparative Training Runs
# ----------------------------------------------------
epochs = 80
eval_interval = epochs // (num_eval_steps - 1)

# Keep track of checkpoints (state dicts) for rollouts
baseline_checkpoints = []
guided_checkpoints = []

# --- RUN A: Baseline Training (Unguided) ---
print("\nRunning Baseline Training (Unguided)...")
baseline_model = TargetModel()
baseline_model.load_state_dict(init_model.state_dict())
baseline_opt = optim.Adam(baseline_model.parameters(), lr=0.01)

baseline_losses = []

for epoch in range(epochs + 1):
    baseline_opt.zero_grad()
    outputs = baseline_model(states_data)
    loss = nn.MSELoss()(outputs, actions_target)
    loss.backward()
    baseline_opt.step()

    if epoch % eval_interval == 0:
        baseline_losses.append(loss.item())
        baseline_checkpoints.append(copy.deepcopy(baseline_model.state_dict()))
        print(f"Baseline Epoch {epoch} - Loss: {loss.item():.4f}")

# --- RUN B: Flow-Guided Training ---
print("\nRunning Flow-Guided Training...")
guided_model = TargetModel()
guided_model.load_state_dict(init_model.state_dict())
guided_opt = optim.Adam(guided_model.parameters(), lr=0.01)

guided_losses = []

for epoch in range(epochs + 1):
    guided_opt.zero_grad()
    outputs = guided_model(states_data)

    step_idx = min(epoch // eval_interval, num_eval_steps - 1)
    target_step = guided_targets[step_idx]

    loss = nn.MSELoss()(outputs, target_step)
    loss.backward()
    guided_opt.step()

    with torch.no_grad():
        true_loss = nn.MSELoss()(outputs, actions_target)

    if epoch % eval_interval == 0:
        guided_losses.append(true_loss.item())
        guided_checkpoints.append(copy.deepcopy(guided_model.state_dict()))
        print(f"Guided Epoch {epoch} - True Target Loss: {true_loss.item():.4f}")


# ----------------------------------------------------
# 6. Closed-Loop Rollout Simulator
# ----------------------------------------------------
# Evaluates how the model actually navigates the environment
def run_rollouts(state_dict_list, starting_positions, max_steps=60, step_size=0.05):
    # Returns a list of paths (shape: num_samples, steps, 2) for the FINAL trained checkpoint
    final_state_dict = state_dict_list[-1]
    model = TargetModel()
    model.load_state_dict(final_state_dict)

    paths = []
    for start_pos in starting_positions:
        pos = np.copy(start_pos)
        path = [np.copy(pos)]
        carrying_key = 0.0
        door_unlocked = 0.0

        for _ in range(max_steps):
            s = torch.tensor(
                np.concatenate(
                    [pos, KEY, DOOR, TREASURE, [carrying_key, door_unlocked]]
                ),
                dtype=torch.float32,
            )
            with torch.no_grad():
                a = model(s).numpy()

            # Normalize step velocity
            a_norm = np.linalg.norm(a)
            if a_norm > 1.0:
                a = a / a_norm

            pos += a * step_size
            path.append(np.copy(pos))

            # Update sequence states
            if carrying_key == 0.0 and np.linalg.norm(pos - KEY) < 0.06:
                carrying_key = 1.0
            elif (
                carrying_key == 1.0
                and door_unlocked == 0.0
                and np.linalg.norm(pos - DOOR) < 0.06
            ):
                door_unlocked = 1.0
            elif (
                carrying_key == 1.0
                and door_unlocked == 1.0
                and np.linalg.norm(pos - TREASURE) < 0.06
            ):
                break
        paths.append(np.array(path))
    return paths


# Closed-loop simulation for the Ideal Flow model (integrating the ODE at t=1.0)
def run_ideal_flow_rollouts(starting_positions, max_steps=60, step_size=0.05):
    paths = []
    for start_pos in starting_positions:
        pos = np.copy(start_pos)
        path = [np.copy(pos)]
        carrying_key = 0.0
        door_unlocked = 0.0

        for _ in range(max_steps):
            s = np.concatenate(
                [pos, KEY, DOOR, TREASURE, [carrying_key, door_unlocked]]
            )
            s_tensor = torch.tensor(s, dtype=torch.float32)

            with torch.no_grad():
                a_init = init_model(s_tensor).numpy()

            def ode_func(t_val, a_val):
                a_tensor = torch.tensor(a_val, dtype=torch.float32).unsqueeze(0)
                t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
                cond_tensor = s_tensor.unsqueeze(0)
                with torch.no_grad():
                    v = flow_model(a_tensor, t_tensor, cond_tensor)
                return v.numpy().flatten()

            # Integrate to tau=1.0 (final target distribution)
            sol = solve_ivp(ode_func, (0.0, 1.0), a_init, method="RK45")
            a_tau = sol.y[:, -1]

            a_norm = np.linalg.norm(a_tau)
            if a_norm > 1.0:
                a_tau = a_tau / a_norm

            pos += a_tau * step_size
            path.append(np.copy(pos))

            if carrying_key == 0.0 and np.linalg.norm(pos - KEY) < 0.06:
                carrying_key = 1.0
            elif (
                carrying_key == 1.0
                and door_unlocked == 0.0
                and np.linalg.norm(pos - DOOR) < 0.06
            ):
                door_unlocked = 1.0
            elif (
                carrying_key == 1.0
                and door_unlocked == 1.0
                and np.linalg.norm(pos - TREASURE) < 0.06
            ):
                break
        paths.append(np.array(path))
    return paths


# Closed-loop simulation for the heuristic Oracle (ground truth)
def run_oracle_rollouts(starting_positions, max_steps=60, step_size=0.05):
    paths = []
    for start_pos in starting_positions:
        pos = np.copy(start_pos)
        path = [np.copy(pos)]
        carrying_key = 0.0
        door_unlocked = 0.0

        for _ in range(max_steps):
            # Determine target element
            if carrying_key == 0.0:
                target = KEY
                dist = np.linalg.norm(target - pos)
                if dist < 0.06:
                    carrying_key = 1.0
            elif door_unlocked == 0.0:
                target = DOOR
                dist = np.linalg.norm(target - pos)
                if dist < 0.06:
                    door_unlocked = 1.0
            else:
                target = TREASURE
                dist = np.linalg.norm(target - pos)
                if dist < 0.06:
                    break

            # Compute oracle action
            direction = target - pos
            action = direction / (np.linalg.norm(direction) + 1e-6)

            # Normalize step velocity
            a_norm = np.linalg.norm(action)
            if a_norm > 1.0:
                action = action / a_norm

            pos += action * step_size
            path.append(np.copy(pos))

            # Double check landmarks
            if carrying_key == 0.0 and np.linalg.norm(pos - KEY) < 0.06:
                carrying_key = 1.0
            elif (
                carrying_key == 1.0
                and door_unlocked == 0.0
                and np.linalg.norm(pos - DOOR) < 0.06
            ):
                door_unlocked = 1.0
            elif (
                carrying_key == 1.0
                and door_unlocked == 1.0
                and np.linalg.norm(pos - TREASURE) < 0.06
            ):
                break
        paths.append(np.array(path))
    return paths


# Evaluate rollouts from random starting points
eval_starts = [np.random.uniform(0.05, 0.45, 2) for _ in range(10)]
oracle_paths = run_oracle_rollouts(eval_starts)
baseline_paths = run_rollouts(baseline_checkpoints, eval_starts)
ideal_paths = run_ideal_flow_rollouts(eval_starts)
guided_paths = run_rollouts(guided_checkpoints, eval_starts)

# ----------------------------------------------------
# 7. Visualization
# ----------------------------------------------------
plt.figure(figsize=(24, 4.8))

# Subplot 1: Convergence Comparison
plt.subplot(1, 5, 1)
plt.plot(eval_times * epochs, baseline_losses, "b-o", label="Baseline (Unguided)")
plt.plot(eval_times * epochs, guided_losses, "r-o", label="Flow-Guided")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss to Final Targets")
plt.title("Convergence Rate Comparison")
plt.legend()
plt.grid(True)

# Subplot 2: Oracle Ground Truth Paths
plt.subplot(1, 5, 2)
plt.scatter(KEY[0], KEY[1], color="gold", s=100, marker="P", zorder=5)
plt.scatter(DOOR[0], DOOR[1], color="brown", s=120, marker="s", zorder=5)
plt.scatter(TREASURE[0], TREASURE[1], color="green", s=120, marker="*", zorder=5)
for path in oracle_paths:
    plt.plot(path[:, 0], path[:, 1], "m--o", alpha=0.7, markersize=3)
    plt.scatter(path[0, 0], path[0, 1], color="magenta", s=40)
plt.title("Oracle Ground Truth Paths")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)

# Subplot 3: Baseline Rollout Paths
plt.subplot(1, 5, 3)
plt.scatter(KEY[0], KEY[1], color="gold", s=100, marker="P", zorder=5)
plt.scatter(DOOR[0], DOOR[1], color="brown", s=120, marker="s", zorder=5)
plt.scatter(TREASURE[0], TREASURE[1], color="green", s=120, marker="*", zorder=5)
for path in baseline_paths:
    plt.plot(path[:, 0], path[:, 1], "b--o", alpha=0.7, markersize=3)
    plt.scatter(path[0, 0], path[0, 1], color="blue", s=40)
plt.title("Baseline Paths (Unguided)")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)

# Subplot 4: Ideal Flow Rollout Paths
plt.subplot(1, 5, 4)
plt.scatter(KEY[0], KEY[1], color="gold", s=100, marker="P", zorder=5)
plt.scatter(DOOR[0], DOOR[1], color="brown", s=120, marker="s", zorder=5)
plt.scatter(TREASURE[0], TREASURE[1], color="green", s=120, marker="*", zorder=5)
for path in ideal_paths:
    plt.plot(path[:, 0], path[:, 1], "g--o", alpha=0.7, markersize=3)
    plt.scatter(path[0, 0], path[0, 1], color="green", s=40)
plt.title("Ideal Geodesic Flow Paths")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)

# Subplot 5: Flow-Guided Rollout Paths
plt.subplot(1, 5, 5)
plt.scatter(KEY[0], KEY[1], color="gold", s=100, marker="P", zorder=5)
plt.scatter(DOOR[0], DOOR[1], color="brown", s=120, marker="s", zorder=5)
plt.scatter(TREASURE[0], TREASURE[1], color="green", s=120, marker="*", zorder=5)
for path in guided_paths:
    plt.plot(path[:, 0], path[:, 1], "r--o", alpha=0.7, markersize=3)
    plt.scatter(path[0, 0], path[0, 1], color="red", s=40)
plt.title("Flow-Guided Paths")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nA/B test complete! Trajectory comparison saved to {PLOT_PATH}")
