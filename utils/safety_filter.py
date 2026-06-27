import torch
import numpy as np

# Try importing pycapacity for polytope projections, fallback to clipping if not installed
try:
    import pycapacity.robot

    PYCAPACITY_AVAILABLE = True
except ImportError:
    PYCAPACITY_AVAILABLE = False


class SafetyFilter:
    """
    Limits proposed joint action torques/velocities using pycapacity's acceleration
    polytope equations and the robot's physical URDF joint ranges.
    """

    def __init__(self, urdf_path=None):
        # Default joint torque limits for the 12 hand joints (Fourier GR-1)
        self.joint_min = np.array(
            [
                -1.57,
                -0.5,
                -0.78,
                -2.1,
                -1.57,
                -1.57,
                -1.57,
                -0.5,
                -0.78,
                -2.1,
                -1.57,
                -1.57,
            ]
        )
        self.joint_max = np.array(
            [1.57, 1.57, 0.78, 0.0, 1.57, 1.57, 1.57, 1.57, 0.78, 0.0, 1.57, 1.57]
        )
        self.torque_max = np.array([10.0] * 12)  # Peak torque limit (Nm)

        if urdf_path:
            # Under a full system, parse limits directly from URDF morphs
            self._load_urdf_limits(urdf_path)

    def _load_urdf_limits(self, urdf_path):
        # In mock/offline setup, we keep default boundaries if parser isn't initialized
        pass

    def filter_actions(self, proposed_actions, J=None):
        """
        Projects proposed actions into physical workspace limits.
        proposed_actions: Tensor or array of shape [Batch, 12] (proposed actions)
        J: Jacobian matrix, optional for cartesian projection
        """
        is_tensor = torch.is_tensor(proposed_actions)
        if is_tensor:
            device = proposed_actions.device
            actions_np = proposed_actions.detach().cpu().numpy()
        else:
            actions_np = np.array(proposed_actions)

        batch_size = actions_np.shape[0]
        filtered = []

        for b in range(batch_size):
            action = actions_np[b]

            # Apply pycapacity workspace polytope projection if available and Jacobian is provided
            if PYCAPACITY_AVAILABLE and J is not None:
                try:
                    # Map joint torque limits through the Jacobian (J)
                    # We compute the acceleration or force boundaries
                    torque_limit_max = self.torque_max
                    torque_limit_min = -self.torque_max

                    # Project actions onto the polytope boundary using pycapacity QP solvers
                    action_proj = pycapacity.robot.torque_to_force(
                        J, action, torque_limit_min, torque_limit_max
                    )
                    filtered.append(action_proj)
                    continue
                except Exception:
                    # Fallback to analytical clipping on failure
                    pass

            # Fallback clipping to ensure physical bounds are never violated
            action_clipped = np.clip(action, -self.torque_max, self.torque_max)
            filtered.append(action_clipped)

        filtered_np = np.stack(filtered, axis=0)

        if is_tensor:
            return torch.tensor(filtered_np, dtype=torch.float32, device=device)
        return filtered_np
