import os
import numpy as np
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "initial_data_visualization.png")

# Set random seed for consistency
np.random.seed(42)


def generate_data(num_samples=200):
    obstacle = np.array([0.5, 0.5])
    target = np.array([1.0, 1.0])

    x_list = []
    # Loop to generate points outside the obstacle boundary
    while len(x_list) < num_samples:
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.8, 1.2)
        pt = np.array([r * np.cos(theta), r * np.sin(theta)])
        # Check if starting point is outside obstacle (radius 0.4)
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
            # Generate perpendicular deflection direction
            perp = np.array([-direction[1], direction[0]])
            perp = perp / np.linalg.norm(perp)
            # Deflect AWAY from the obstacle center
            if np.dot(perp, to_obstacle) > 0:
                perp = -perp
            y[i] = target + 0.35 * perp
        else:
            y[i] = target

    return x, y


x_data, y_data = generate_data()

# Plotting the raw data configuration
plt.figure(figsize=(8, 8))

# Plot starting positions (Inputs x)
plt.scatter(
    x_data[:, 0],
    x_data[:, 1],
    color="blue",
    alpha=0.5,
    s=40,
    label="Starting Positions (Inputs $x$)",
)

# Plot the target goal (Green Star)
plt.scatter(
    1.0, 1.0, color="green", s=250, marker="*", zorder=5, label="Goal Target (1.0, 1.0)"
)

# Plot the obstacle (Red circle constraint boundary)
obstacle_circle = plt.Circle(
    (0.5, 0.5),
    0.4,
    color="red",
    fill=True,
    alpha=0.2,
    label="Obstacle Constraint (Radius 0.4)",
)
plt.gca().add_patch(obstacle_circle)
plt.scatter(0.5, 0.5, color="red", s=100, marker="X", zorder=5, label="Obstacle Center")

# Plot target outputs (y) and draw trajectories for a few samples
plt.scatter(
    y_data[:, 0],
    y_data[:, 1],
    color="purple",
    alpha=0.6,
    s=40,
    label="Target Final Outputs (Labels $y$)",
)

# Draw lines representing the path mapping for 15 random samples
indices = np.random.choice(len(x_data), 15, replace=False)
for idx in indices:
    plt.plot(
        [x_data[idx, 0], y_data[idx, 0]],
        [x_data[idx, 1], y_data[idx, 1]],
        color="gray",
        linestyle="--",
        alpha=0.6,
    )

plt.xlim(-1.5, 1.8)
plt.ylim(-1.5, 1.8)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title(
    "Workspace Visualization: Inputs ($x$) to Targets ($y$) with Obstacle Avoidance"
)
plt.legend(loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Data visualization plot successfully saved to {PLOT_PATH}")
