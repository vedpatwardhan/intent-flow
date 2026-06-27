import os
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "initial_data_visualization.png")

# Set random seed
np.random.seed(42)

# Define target elements
KEY = np.array([0.2, 0.8])
DOOR = np.array([0.8, 0.5])
TREASURE = np.array([0.9, 0.9])


# Heuristic oracle to generate a successful path
def generate_heuristic_trajectory(start_pos, max_steps=60, step_size=0.05):
    pos = np.copy(start_pos)
    trajectory = [np.copy(pos)]

    carrying_key = 0.0
    door_unlocked = 0.0

    for _ in range(max_steps):
        # 1. If key not collected, move to Key
        if carrying_key == 0.0:
            target = KEY
            dist = np.linalg.norm(target - pos)
            if dist < 0.06:
                carrying_key = 1.0

        # 2. If key collected but door locked, move to Door
        elif door_unlocked == 0.0:
            target = DOOR
            dist = np.linalg.norm(target - pos)
            if dist < 0.06:
                door_unlocked = 1.0

        # 3. If door unlocked, move to Treasure
        else:
            target = TREASURE
            dist = np.linalg.norm(target - pos)
            if dist < 0.06:
                trajectory.append(np.copy(target))
                break

        # Take step towards current target
        direction = target - pos
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        pos += direction * step_size
        trajectory.append(np.copy(pos))

    return np.array(trajectory)


# Generate and plot 5 sample trajectories
plt.figure(figsize=(8, 8))

# Plot environment layout
plt.scatter(
    KEY[0], KEY[1], color="gold", s=200, marker="P", zorder=5, label="Key (0.2, 0.8)"
)
plt.scatter(
    DOOR[0],
    DOOR[1],
    color="brown",
    s=250,
    marker="s",
    zorder=5,
    label="Door (0.8, 0.5)",
)
plt.scatter(
    TREASURE[0],
    TREASURE[1],
    color="green",
    s=250,
    marker="*",
    zorder=5,
    label="Treasure (0.9, 0.9)",
)

# Generate starting positions and plot paths
num_samples = 5
for i in range(num_samples):
    # Generate random starting point in bottom-left region
    start_pos = np.random.uniform(0.05, 0.45, 2)
    traj = generate_heuristic_trajectory(start_pos)

    # Plot starting point
    plt.scatter(start_pos[0], start_pos[1], color="blue", s=60, alpha=0.7, zorder=4)
    if i == 0:
        plt.scatter(
            start_pos[0],
            start_pos[1],
            color="blue",
            s=60,
            alpha=0.7,
            zorder=4,
            label="Start Position",
        )

    # Plot trajectory path
    plt.plot(
        traj[:, 0],
        traj[:, 1],
        linestyle="-",
        marker="o",
        markersize=4,
        alpha=0.6,
        label=f"Sample Path {i+1}",
    )

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("PoC 2: Key-Door-Treasure Navigation Trajectories")
plt.legend(loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"PoC 2 Data visualization saved to {PLOT_PATH}")
