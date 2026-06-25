# PoC 2: Continuous Navigation Game (Key-Door-Treasure)

This proof of concept verifies the flow-guided training paradigm on a sequential navigation task. It introduces temporal task sequencing and conditional action vector fields.

## Files

*   `visualize_data.py`: Generates the continuous grid-world coordinate layouts, key, door, and treasure locations, and traces the bimodal sequential path trajectories. Saves plot to `initial_data_visualization.png`.
*   `flow_matching.py`: Implements dataset generation from heuristic trajectories, trains the action-space flow matching model, integrates the velocity ODE, performs A/B baseline vs guided training runs, and plots comparative vector fields. Saves plot to `trajectory_comparison.png`.
*   `initial_data_visualization.png`: Visualizes the sequential trajectories in coordinate space.
*   `trajectory_comparison.png`: Visualizes the A/B test convergence rates and action fields.

## How to Run

Generate the initial trajectories plot:
```bash
.venv/bin/python lewm-flow/poc_2_continuous_navigation/visualize_data.py
```

Run the comparative training benchmark:
```bash
.venv/bin/python lewm-flow/poc_2_continuous_navigation/flow_matching.py
```

## Task Paradigm

*   **Inputs ($s \in \mathbb{R}^{10}$)**: The agent's 2D position, fixed targets (Key, Door, Treasure), and binary task flags (`carrying_key`, `door_unlocked`).
*   **Outputs ($a \in \mathbb{R}^2$)**: Velocity action vectors $(v_x, v_y)$.
*   **The Flow**: The flow matching model $v_\phi(a, t; s)$ learns the straight geodesic flow transporting the target model's untrained action outputs ($a_{\text{init}}$) to the dataset's target actions ($a_{\text{target}}$).
*   **Guidance**: The target model is trained on intermediate actions evaluated along this geodesic flow, preventing parameter updates from taking high-variance, curved steps in the loss landscape.
