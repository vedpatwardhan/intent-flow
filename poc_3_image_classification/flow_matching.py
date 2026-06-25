import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "trajectory_comparison.png")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------
# 1. Dataset Setup (Lightweight Subsampled MNIST)
# ----------------------------------------------------
print("Loading and preparing subsampled MNIST dataset...")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Download dataset locally
mnist_dir = os.path.join(SCRIPT_DIR, "mnist_data")
train_dataset = datasets.MNIST(
    mnist_dir, train=True, download=True, transform=transform
)

# Subsample to 2000 images
subsample_indices = np.random.choice(len(train_dataset), 2000, replace=False)
images = []
labels = []
for idx in subsample_indices:
    img, lbl = train_dataset[idx]
    images.append(img.view(-1))
    labels.append(lbl)

x_data = torch.stack(images)  # Shape: (2000, 784)
labels_raw = torch.tensor(labels, dtype=torch.long)
y_target = nn.functional.one_hot(
    labels_raw, num_classes=10
).float()  # One-hot target probabilities

print(f"Dataset ready. Inputs: {x_data.shape}, Target Probabilities: {y_target.shape}")

# Create fixed random projection matrix (784 -> 16)
PROJ_DIM = 128
random_matrix = torch.randn(784, PROJ_DIM) / np.sqrt(PROJ_DIM)
x_cond = torch.matmul(x_data, random_matrix)  # Shape: (2000, 128)


# ----------------------------------------------------
# 2. Target Model (7,850 parameters)
# ----------------------------------------------------
class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 10))

    def forward(self, x):
        return self.net(x)


init_model = TargetModel()
with torch.no_grad():
    logits_init = init_model(x_data)
    # Convert initial logits to probability space via Softmax
    probs_init = nn.functional.softmax(logits_init, dim=-1).clone()

target_params = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
print(f"Target Model loaded. Total Parameters: {target_params:,}")


# ----------------------------------------------------
# 3. Flow Matching Model in Probability Space (618 parameters)
# ----------------------------------------------------
# Inputs: current probability p_t (10D) + time t (1D) + conditioning context x_cond (16D) = 27D
class FlowMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10 + 1 + PROJ_DIM, 10),
            # nn.Linear(10 + 1 + PROJ_DIM, 16),
            # nn.Tanh(),
            # nn.Linear(16, 10)
        )

    def forward(self, p, t, cond):
        inputs = torch.cat([p, t, cond], dim=-1)
        return self.net(inputs)


flow_model = FlowMatchingModel()
flow_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
print(f"Flow Matching Model loaded. Total Parameters: {flow_params:,}")

# Train the Flow Matching Model on output probability space
flow_optimizer = optim.Adam(flow_model.parameters(), lr=0.005)
flow_epochs = 800

print("\nTraining Flow Matching Model on Probability space...")
for epoch in range(flow_epochs):
    flow_optimizer.zero_grad()
    t = torch.rand(probs_init.shape[0], 1)

    # Linear interpolation path in Probability space (constrained to the simplex)
    p_t = (1 - t) * probs_init + t * y_target
    v_target = y_target - probs_init

    v_pred = flow_model(p_t, t, x_cond)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    flow_optimizer.step()

# ----------------------------------------------------
# 4. Integrate Flow to get Guided Targets (Probability vectors)
# ----------------------------------------------------
print("\nIntegrating Flow to get intermediate guided probability targets...")


def flow_ode(t, p_flat, cond_np):
    p_tensor = torch.tensor(p_flat.reshape(-1, 10), dtype=torch.float32)
    cond_tensor = torch.tensor(cond_np, dtype=torch.float32)
    t_tensor = torch.full((p_tensor.shape[0], 1), t, dtype=torch.float32)
    with torch.no_grad():
        v = flow_model(p_tensor, t_tensor, cond_tensor)
    return v.numpy().flatten()


cond_np = x_cond.numpy()
p_init_np = probs_init.numpy().flatten()
num_eval_steps = 11
eval_times = np.linspace(0.0, 1.0, num_eval_steps)

sol = solve_ivp(
    flow_ode,
    t_span=(0.0, 1.0),
    y0=p_init_np,
    args=(cond_np,),
    t_eval=eval_times,
    method="RK45",
)

guided_targets = []
for i in range(num_eval_steps):
    raw_p = torch.tensor(sol.y[:, i].reshape(-1, 10), dtype=torch.float32)
    # Clip and normalize to ensure they are valid probabilities (simplex constraint)
    raw_p = torch.clamp(raw_p, min=1e-8)
    normalized_p = raw_p / raw_p.sum(dim=-1, keepdim=True)
    guided_targets.append(normalized_p)


# Helper to capture flat weight vector of target model's key layer
def copy_model_weights(model):
    return model.net[0].weight.data.clone().numpy().flatten()


# ----------------------------------------------------
# 5. Comparative A/B Training Runs (Cross-Entropy & KL-Div)
# ----------------------------------------------------
epochs = 80
eval_interval = epochs // (num_eval_steps - 1)

# --- RUN A: Baseline Training (Standard Cross-Entropy) ---
print("\nRunning Baseline Training (Cross-Entropy)...")
baseline_model = TargetModel()
baseline_model.load_state_dict(init_model.state_dict())
baseline_opt = optim.Adam(baseline_model.parameters(), lr=0.01)
loss_fn_baseline = nn.CrossEntropyLoss()

baseline_losses = []
baseline_accs = []
baseline_weights = []
baseline_probs_history = []  # Store softmax probabilities

for epoch in range(epochs + 1):
    baseline_opt.zero_grad()
    outputs = baseline_model(x_data)
    loss = loss_fn_baseline(outputs, labels_raw)
    loss.backward()
    baseline_opt.step()

    if epoch % eval_interval == 0:
        baseline_losses.append(loss.item())
        with torch.no_grad():
            acc = (outputs.argmax(dim=-1) == labels_raw).float().mean().item()
            baseline_accs.append(acc)
            baseline_weights.append(copy_model_weights(baseline_model))
            # Convert to probability for PCA plotting
            probs = nn.functional.softmax(outputs, dim=-1).detach().numpy()
            baseline_probs_history.append(probs)
        print(
            f"Baseline Epoch {epoch} - CE Loss: {loss.item():.4f} - Acc: {acc*100:.2f}%"
        )

# --- RUN B: Flow-Guided Training (KL-Divergence on Soft Targets) ---
print("\nRunning Flow-Guided Training (KL-Divergence)...")
guided_model = TargetModel()
guided_model.load_state_dict(init_model.state_dict())
guided_opt = optim.Adam(guided_model.parameters(), lr=0.01)
# KL Div takes log-probabilities as input and targets as soft probability distributions
loss_fn_guided = nn.KLDivLoss(reduction="batchmean")

guided_losses = []
guided_accs = []
guided_weights = []
guided_logits_history = []

# Linear scheduling of step_idx
for epoch in range(epochs + 1):
    guided_opt.zero_grad()
    outputs = guided_model(x_data)

    step_idx = min(epoch // eval_interval, num_eval_steps - 1)
    target_p = guided_targets[step_idx]

    # Input to KL Div must be log-softmax
    log_probs = nn.functional.log_softmax(outputs, dim=-1)
    loss = loss_fn_guided(log_probs, target_p)
    loss.backward()
    guided_opt.step()

    if epoch % eval_interval == 0:
        with torch.no_grad():
            # True Cross Entropy loss to raw labels
            true_loss = loss_fn_baseline(outputs, labels_raw).item()
            acc = (outputs.argmax(dim=-1) == labels_raw).float().mean().item()
            guided_losses.append(true_loss)
            guided_accs.append(acc)
            guided_weights.append(copy_model_weights(guided_model))
            # Convert to probability for PCA plotting
            probs = nn.functional.softmax(outputs, dim=-1).detach().numpy()
            guided_logits_history.append(probs)
        print(
            f"Guided Epoch {epoch} - True CE Loss: {true_loss:.4f} - Acc: {acc*100:.2f}%"
        )

# Convert histories to numpy arrays
baseline_probs_history = np.stack(baseline_probs_history)
guided_probs_history = np.stack(guided_logits_history)
ideal_probs_history = np.stack([gt.numpy() for gt in guided_targets])

# ----------------------------------------------------
# 6. Quantitative Straightness Analysis (Geodesic Ratio)
# ----------------------------------------------------
print("\nPerforming path straightness analysis in Probability space...")


def compute_straightness(probs_history):
    num_steps, num_samples, _ = probs_history.shape
    direct_dist = np.linalg.norm(probs_history[-1] - probs_history[0], axis=1)

    path_len = np.zeros(num_samples)
    for t in range(num_steps - 1):
        path_len += np.linalg.norm(probs_history[t + 1] - probs_history[t], axis=1)

    straightness = np.zeros(num_samples)
    mask = path_len > 1e-8
    straightness[mask] = direct_dist[mask] / path_len[mask]
    return straightness


baseline_straightness = compute_straightness(baseline_probs_history)
guided_straightness = compute_straightness(guided_probs_history)
ideal_straightness = compute_straightness(ideal_probs_history)

# ----------------------------------------------------
# 7. Rank Analysis of Parameter Updates (SVD)
# ----------------------------------------------------
print("\nPerforming SVD update-rank analysis...")


def compute_singular_values(weight_history):
    orig_shape = init_model.net[0].weight.shape
    w0 = weight_history[0].reshape(orig_shape)
    w_final = weight_history[-1].reshape(orig_shape)
    delta_w = w_final - w0
    U, S, Vt = np.linalg.svd(delta_w)
    return S


baseline_sing_vals = compute_singular_values(baseline_weights)
guided_sing_vals = compute_singular_values(guided_weights)

# ----------------------------------------------------
# 8. Visualization in Probability Space
# ----------------------------------------------------
# Fit stable 2D PCA representation of target probabilities
pca = PCA(n_components=2)
pca.fit(y_target.numpy())

# Select 5 specific samples
traj_sample_indices = []
for digit in range(5):
    idx = np.where(labels_raw.numpy() == digit)[0][0]
    traj_sample_indices.append(idx)

plt.figure(figsize=(18, 11))

# Subplot 1: Convergence Comparison (CE Loss)
plt.subplot(2, 3, 1)
plt.plot(eval_times * epochs, baseline_losses, "b-o", label="Baseline (Unguided)")
plt.plot(eval_times * epochs, guided_losses, "r-o", label="Flow-Guided")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss to Labels")
plt.title("Convergence Comparison (CE Loss)")
plt.legend()
plt.grid(True)

# Subplot 2: SVD Update-Rank Analysis
plt.subplot(2, 3, 2)
plt.semilogy(baseline_sing_vals, "b-", label="Baseline ΔW")
plt.semilogy(guided_sing_vals, "r-", label="Flow-Guided ΔW")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value Magnitude (Log)")
plt.title("Singular Value Decay of ΔW")
plt.legend()
plt.grid(True)

# Subplot 3: Path Straightness Distribution
plt.subplot(2, 3, 3)
plt.hist(baseline_straightness, bins=25, alpha=0.5, label="Baseline", color="blue")
plt.hist(guided_straightness, bins=25, alpha=0.5, label="Flow-Guided", color="red")
plt.hist(ideal_straightness, bins=25, alpha=0.3, label="Ideal Flow", color="green")
plt.xlabel("Straightness Metric (Arc-Length Ratio)")
plt.ylabel("Sample Count")
plt.title("Probability Space Straightness")
plt.legend()
plt.grid(True)


# Helper to plot PCA trajectories
def plot_pca_trajectories(history_array, title, color_line):
    final_outputs = history_array[-1]
    coords_2d = pca.transform(final_outputs)
    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels_raw.numpy(),
        cmap="tab10",
        alpha=0.3,
        s=8,
    )

    for sample_idx in traj_sample_indices:
        sample_history = history_array[:, sample_idx, :]
        proj_history = pca.transform(sample_history)

        plt.plot(
            proj_history[:, 0],
            proj_history[:, 1],
            color=color_line,
            linestyle="--",
            marker="o",
            markersize=3,
            alpha=0.9,
        )
        plt.scatter(
            proj_history[0, 0],
            proj_history[0, 1],
            color="black",
            marker="s",
            s=35,
            zorder=5,
        )
        plt.scatter(
            proj_history[-1, 0],
            proj_history[-1, 1],
            color="red",
            marker="*",
            s=60,
            zorder=5,
        )

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    return scatter


# Subplot 4: Baseline Output Trajectories
plt.subplot(2, 3, 4)
plot_pca_trajectories(baseline_probs_history, "Baseline Paths (Unguided)", "blue")

# Subplot 5: Ideal Flow Trajectories
plt.subplot(2, 3, 5)
plot_pca_trajectories(ideal_probs_history, "Ideal Geodesic Flow Paths", "green")

# Subplot 6: Flow-Guided Output Trajectories
plt.subplot(2, 3, 6)
scatter_handle = plot_pca_trajectories(guided_probs_history, "Flow-Guided Paths", "red")
plt.colorbar(scatter_handle, label="Digit Class")

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"\nA/B test complete! Trajectory comparison saved to {PLOT_PATH}")
