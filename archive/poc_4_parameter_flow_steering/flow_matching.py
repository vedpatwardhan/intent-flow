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
# 1. Target MLP Definition (25 parameters)
# ----------------------------------------------------
class TargetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, x):
        return self.net(x)


# Flat weight helper functions
def get_flat_params(model):
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1))
    return torch.cat(params).clone().numpy()


def set_flat_params(model, flat_params):
    flat_params_tensor = torch.tensor(flat_params, dtype=torch.float32)
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params_tensor[offset : offset + numel].view(p.shape))
        offset += numel


# Pre-train a "base model" to fit a standard sine wave y = sin(x)
# This serves as our structured starting weights theta_0
print("Pre-training base model weights...")
base_model = TargetMLP()
optimizer = optim.SGD(base_model.parameters(), lr=0.1)
x_pre = torch.linspace(-3.0, 3.0, 100).unsqueeze(1)
y_pre = torch.sin(x_pre)

for _ in range(500):
    optimizer.zero_grad()
    loss = nn.MSELoss()(base_model(x_pre), y_pre)
    loss.backward()
    optimizer.step()

THETA_0 = get_flat_params(base_model)
print("Base model pre-trained.")

# ----------------------------------------------------
# 2. Meta-Dataset Generation (Weight Trajectories)
# ----------------------------------------------------
# Generate training trajectories for various sine wave tasks: y = A * sin(x + phi)
num_tasks = 250
num_steps = 50
step_size = 0.05

weight_trajectories = []  # Shape: (num_tasks, num_steps, 25)
task_contexts = []  # Shape: (num_tasks, 10)

print(f"Generating weight trajectories for {num_tasks} tasks...")
for task_idx in range(num_tasks):
    # Sample task parameters
    A = np.random.uniform(0.5, 2.0)
    phi = np.random.uniform(0, np.pi)

    # Generate task data
    x_train = torch.linspace(-3.0, 3.0, 20).unsqueeze(1)
    y_train = A * torch.sin(x_train + phi)

    # Context vector: 5 random context points (flattened to 10D)
    context_indices = np.random.choice(20, 5, replace=False)
    x_ctx = x_train[context_indices].flatten().numpy()
    y_ctx = y_train[context_indices].flatten().numpy()
    context_vector = np.concatenate([x_ctx, y_ctx])

    # Train the MLP from the base weights THETA_0
    model = TargetMLP()
    set_flat_params(model, THETA_0)
    opt = optim.SGD(model.parameters(), lr=0.08)

    trajectory = []
    for step in range(num_steps):
        trajectory.append(get_flat_params(model))

        opt.zero_grad()
        loss = nn.MSELoss()(model(x_train), y_train)
        loss.backward()
        opt.step()

    weight_trajectories.append(np.stack(trajectory))
    task_contexts.append(context_vector)

weight_trajectories = np.array(weight_trajectories)  # (250, 50, 25)
task_contexts = np.array(task_contexts)  # (250, 10)
print("Trajectories generated successfully.")


# ----------------------------------------------------
# 3. Flow Matching Model in Parameter Space (4,953 parameters)
# ----------------------------------------------------
# Inputs: weight vector theta_t (25D) + time t (1D) + task context c (10D) = 36D
class ParameterFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25 + 1 + 10, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 25),
        )

    def forward(self, theta, t, cond):
        inputs = torch.cat([theta, t, cond], dim=-1)
        return self.net(inputs)


flow_model = ParameterFlowModel()
flow_optimizer = optim.Adam(flow_model.parameters(), lr=0.002)
flow_epochs = 1200

print("\nTraining Flow Matching Model in Weight Space...")
for epoch in range(flow_epochs):
    flow_optimizer.zero_grad()

    # Sample random tasks and time steps
    batch_tasks = np.random.choice(num_tasks, 128)
    batch_t_idx = np.random.choice(num_steps, 128)

    t_vals = batch_t_idx / (num_steps - 1)
    t_tensor = torch.tensor(t_vals, dtype=torch.float32).unsqueeze(1)

    theta_0 = torch.tensor(weight_trajectories[batch_tasks, 0], dtype=torch.float32)
    theta_1 = torch.tensor(weight_trajectories[batch_tasks, -1], dtype=torch.float32)

    # Linear interpolation in parameter space
    theta_t = (1 - t_tensor) * theta_0 + t_tensor * theta_1
    v_target = theta_1 - theta_0

    contexts = torch.tensor(task_contexts[batch_tasks], dtype=torch.float32)

    v_pred = flow_model(theta_t, t_tensor, contexts)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    flow_optimizer.step()

    if epoch % 200 == 0:
        print(f"Flow Matching Epoch {epoch}/{flow_epochs} - Loss: {loss.item():.6f}")


# ----------------------------------------------------
# 4. Integrate Flow for Zero-Backprop Adaptation
# ----------------------------------------------------
def steer_weights(context):
    def ode_func(t, theta_flat):
        theta_tensor = torch.tensor(theta_flat, dtype=torch.float32).unsqueeze(0)
        t_tensor = torch.tensor([[t]], dtype=torch.float32)
        cond_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            v = flow_model(theta_tensor, t_tensor, cond_tensor)
        return v.numpy().flatten()

    sol = solve_ivp(ode_func, (0.0, 1.0), THETA_0, method="RK45")
    return sol.y[:, -1]


# ----------------------------------------------------
# 5. A/B Evaluation Benchmark
# ----------------------------------------------------
# Generate a new, unseen sine wave task
test_A = 1.4
test_phi = 1.2
x_test = torch.linspace(-3.0, 3.0, 100).unsqueeze(1)
y_test_gt = test_A * torch.sin(x_test + test_phi)

# Create task context (5 points)
ctx_idx = [15, 35, 55, 75, 95]
x_test_ctx = x_test[ctx_idx].flatten().numpy()
y_test_ctx = y_test_gt[ctx_idx].flatten().numpy()
test_context = np.concatenate([x_test_ctx, y_test_ctx])

# --- Run A: Base Model (No adaptation) ---
base_test_model = TargetMLP()
set_flat_params(base_test_model, THETA_0)
with torch.no_grad():
    y_pred_base = base_test_model(x_test).numpy()

# --- Run B: Zero-Backprop Flow Steered Model ---
print("\nRunning Zero-Backprop Parameter-Flow Steering...")
steered_flat_weights = steer_weights(test_context)
steered_model = TargetMLP()
set_flat_params(steered_model, steered_flat_weights)
with torch.no_grad():
    y_pred_steered = steered_model(x_test).numpy()

# --- Run C: Baseline SGD Adaptation (5, 15, and 50 steps) ---
print("Running Baseline SGD runs...")


def run_sgd_adaptation(steps):
    m = TargetMLP()
    set_flat_params(m, THETA_0)
    opt = optim.SGD(m.parameters(), lr=0.08)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.MSELoss()(m(x_test), y_test_gt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return m(x_test).numpy()


y_pred_sgd_5 = run_sgd_adaptation(5)
y_pred_sgd_15 = run_sgd_adaptation(15)
y_pred_sgd_50 = run_sgd_adaptation(num_steps)

# Track loss progression to compare convergence
sgd_loss_progression = []
m_sgd = TargetMLP()
set_flat_params(m_sgd, THETA_0)
opt_sgd = optim.SGD(m_sgd.parameters(), lr=0.08)
for _ in range(num_steps + 1):
    with torch.no_grad():
        loss = nn.MSELoss()(m_sgd(x_test), y_test_gt).item()
        sgd_loss_progression.append(loss)
    opt_sgd.zero_grad()
    loss_val = nn.MSELoss()(m_sgd(x_test), y_test_gt)
    loss_val.backward()
    opt_sgd.step()

# Compute losses for comparison
base_loss = nn.MSELoss()(torch.tensor(y_pred_base), y_test_gt).item()
steered_loss = nn.MSELoss()(torch.tensor(y_pred_steered), y_test_gt).item()
sgd_50_loss = nn.MSELoss()(torch.tensor(y_pred_sgd_50), y_test_gt).item()

print(f"\nFinal Test Losses:")
print(f"Base Model (Untrained on Task): {base_loss:.5f}")
print(f"Flow-Steered Model (Zero-Backprop): {steered_loss:.5f}")
print(f"Baseline SGD (50 steps): {sgd_50_loss:.5f}")

# ----------------------------------------------------
# 6. Visualization
# ----------------------------------------------------
plt.figure(figsize=(15, 6))

# Subplot 1: Curve Fitting Comparison
plt.subplot(1, 2, 1)
plt.plot(
    x_test.numpy(),
    y_test_gt.numpy(),
    "k-",
    label="Ground Truth (Target Wave)",
    linewidth=2.5,
)
plt.scatter(
    x_test_ctx,
    y_test_ctx,
    color="gold",
    s=120,
    marker="P",
    zorder=5,
    label="Task Context Points",
)
plt.plot(
    x_test.numpy(), y_pred_base, "gray", linestyle=":", label="Base Model (THETA_0)"
)
plt.plot(x_test.numpy(), y_pred_sgd_5, "b--", alpha=0.5, label="SGD (5 steps)")
plt.plot(x_test.numpy(), y_pred_sgd_15, "b-.", alpha=0.7, label="SGD (15 steps)")
plt.plot(
    x_test.numpy(),
    y_pred_steered,
    "r-",
    linewidth=2,
    label="Flow-Steered (Zero-Backprop)",
)
plt.title(f"Curve Fitting Comparison\n(Flow-Steered Loss: {steered_loss:.4f})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Subplot 2: Optimization Convergence Speed
plt.subplot(1, 2, 2)
plt.plot(range(num_steps + 1), sgd_loss_progression, "b-o", label="Baseline SGD Steps")
# Flow-steered is a single zero-shot inference (equivalent to 1 ODE integration pass)
# We plot it as a constant line or point to compare relative efficiency
plt.axhline(
    y=steered_loss,
    color="r",
    linestyle="-",
    linewidth=2,
    label="Flow-Steered Final Loss",
)
plt.scatter(0, base_loss, color="gray", s=80, zorder=5, label="Base Model Loss")
plt.xlabel("Optimization Step / Iteration")
plt.ylabel("MSE Loss to Ground Truth")
plt.title("Convergence Speed / Adaptation Efficiency")
plt.legend()
plt.yscale("log")
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nParameter flow steering complete! Trajectory comparison saved to {PLOT_PATH}")
