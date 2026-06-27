# PoC 1: Kinematic Obstacle Avoidance

This proof of concept validates the core hypothesis of **training trajectory acceleration and constraint regularization** using Flow Matching. It tests a continuous control reaching task inside a constrained 2D workspace.

## Files

*   `visualize_data.py`: Generates the dataset showing starting positions, the goal, the obstacle, and the bimodal target deflected trajectories. Saves plot to `initial_data_visualization.png`.
*   `flow_matching.py`: The main simulation code that trains the baseline model, trains the flow matching oracle, integrates the flow ODE, runs flow-guided training, and plots comparison results. Saves plot to `trajectory_comparison.png`.
*   `initial_data_visualization.png`: Visualizes the workspace input-to-output mapping.
*   `trajectory_comparison.png`: Visualizes the comparative results of the A/B test.

## How to Run

Generate the initial data visualization:
```bash
.venv/bin/python lewm-flow/poc_1_obstacle_avoidance/visualize_data.py
```

Run the comparative training simulation:
```bash
.venv/bin/python lewm-flow/poc_1_obstacle_avoidance/flow_matching.py
```

## Key Findings

1.  **Obstacle Violation in Baseline**: Unguided training pushes the model directly toward the final target, causing it to take high-velocity, erratic shortcuts that pass directly through the obstacle center (violating the safety boundary) before contracting.
2.  **Smooth, Obstacle-Avoiding Flow**: The Flow Matching model learns a mathematically consistent curved vector field by deflecting the interpolation path itself (using a potential field repulsion scaled by $4t(1-t)$) and calculating target velocities numerically, satisfying the continuity equation.
3.  **Controlled Geodesic Guidance**: The Flow-Guided model follows this vector field precisely. It converges to the optimal target smoothly and linearly, routing its outputs safely around the obstacle boundary at all epochs.
4.  **Decoupled Model Sizing**: The flow matching model's size depends on the output/data space dimension, not the target model's parameter footprint, proving that the flow guide can scale to be a small fraction (often $< 1\%$) of the target model's parameter footprint in large-scale systems.
