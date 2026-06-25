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
# 1. Task Definition: Reaching around an obstacle
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
# 2. Target Model Definition & Training Trajectory Collection
# ----------------------------------------------------
class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)


target_model = TargetModel()
optimizer = optim.Adam(target_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 100
trajectory_data = {}

print("Training target model and collecting output trajectory...")
for epoch in range(epochs + 1):
    optimizer.zero_grad()
    outputs = target_model(x_data)
    loss = criterion(outputs, y_target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        trajectory_data[epoch / epochs] = outputs.detach().clone()
        print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")


# ----------------------------------------------------
# 3. Flow Matching Model: Learning the Output Distribution Flow
# ----------------------------------------------------
class FlowOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, y, tau, x):
        inputs = torch.cat([y, tau, x], dim=-1)
        return self.net(inputs)


flow_oracle = FlowOracle()
flow_optimizer = optim.Adam(flow_oracle.parameters(), lr=0.005)

print("\nTraining Flow Matching Oracle on output distributions...")
flow_epochs = 500
tau_keys = sorted(list(trajectory_data.keys()))

for epoch in range(flow_epochs):
    flow_optimizer.zero_grad()

    idx = np.random.choice(len(tau_keys) - 1)
    t0, t1 = tau_keys[idx], tau_keys[idx + 1]

    y0 = trajectory_data[t0]
    y1 = trajectory_data[t1]

    t = torch.rand(y0.shape[0], 1)
    yt = (1 - t) * y0 + t * y1
    v_target = (y1 - y0) / (t1 - t0)
    tau = torch.tensor([[t0 + (t1 - t0) * float(t_val)] for t_val in t])

    v_pred = flow_oracle(yt, tau, x_data)

    flow_loss = nn.MSELoss()(v_pred, v_target)
    flow_loss.backward()
    flow_optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Flow Matching Epoch {epoch}/{flow_epochs} - Loss: {flow_loss.item():.6f}"
        )

# ----------------------------------------------------
# 4. Integrate Flow ODE and Compare Trajectories
# ----------------------------------------------------
print("\nIntegrating Flow Matching ODE to reconstruct training trajectory...")


def flow_ode(t, y_flat, x_np):
    y_tensor = torch.tensor(y_flat.reshape(-1, 2), dtype=torch.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    tau_tensor = torch.full((y_tensor.shape[0], 1), t, dtype=torch.float32)

    with torch.no_grad():
        v = flow_oracle(y_tensor, tau_tensor, x_tensor)
    return v.numpy().flatten()


y_init = trajectory_data[0.0].numpy().flatten()
x_np = x_data.numpy()

sol = solve_ivp(
    flow_ode,
    t_span=(0.0, 1.0),
    y0=y_init,
    args=(x_np,),
    t_eval=np.linspace(0.0, 1.0, 11),
    method="RK45",
)

# ----------------------------------------------------
# 5. Plotting and Verification
# ----------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_np[:, 0], x_np[:, 1], color="gray", alpha=0.5, label="Inputs (x)")
plt.scatter(0.5, 0.5, color="red", s=200, marker="x", label="Obstacle")
plt.scatter(1.0, 1.0, color="green", s=200, marker="*", label="Target")
plt.title("Workspace Definition")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
y_final_pred = sol.y[:, -1].reshape(-1, 2)
y_final_actual = trajectory_data[1.0].numpy()

plt.scatter(
    y_final_actual[:, 0],
    y_final_actual[:, 1],
    color="blue",
    alpha=0.6,
    label="Actual Trained Outputs",
)
plt.scatter(
    y_final_pred[:, 0],
    y_final_pred[:, 1],
    color="orange",
    alpha=0.6,
    label="Flow Reconstructed Outputs",
)
plt.title("Reconstructed vs Actual Output Distribution")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nVerification complete! Plot saved to {PLOT_PATH}")
