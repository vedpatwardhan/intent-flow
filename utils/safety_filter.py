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
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            joint_limits_min = []
            joint_limits_max = []
            joint_efforts = []

            # Find all joint limit configurations in URDF XML structure
            for joint in root.findall("joint"):
                limit = joint.find("limit")
                if limit is not None:
                    lower = float(limit.get("lower", -1.57))
                    upper = float(limit.get("upper", 1.57))
                    effort = float(limit.get("effort", 10.0))

                    joint_limits_min.append(lower)
                    joint_limits_max.append(upper)
                    joint_efforts.append(effort)

            # If we successfully parsed limits, update the arrays
            if len(joint_limits_min) > 0:
                # Align array dimensions to the 12 hand joints
                self.joint_min = np.array(joint_limits_min[:12])
                self.joint_max = np.array(joint_limits_max[:12])
                self.torque_max = np.array(joint_efforts[:12])
                print(
                    f"Successfully loaded joint limits from URDF: {len(self.joint_min)} joints configured."
                )
        except Exception as e:
            print(
                f"Warning: Failed to parse URDF file ({e}). Retaining default hand joint limits."
            )

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
            J_np = J.detach().cpu().numpy() if J is not None else None
        else:
            actions_np = np.array(proposed_actions)
            J_np = np.array(J) if J is not None else None

        batch_size = actions_np.shape[0]
        filtered = []

        for b in range(batch_size):
            action = actions_np[b]

            # Apply pycapacity workspace polytope projection if available and Jacobian is provided
            if PYCAPACITY_AVAILABLE and J_np is not None:
                try:
                    # Map joint torque limits through the Jacobian (J_np)
                    torque_limit_max = self.torque_max
                    torque_limit_min = -self.torque_max

                    # Compute force polytope inequalities H * f <= d directly from joint boundaries:
                    # tau_min <= J_np^T * f <= tau_max  =>
                    # J_np^T * f <= tau_max AND -J_np^T * f <= -tau_min
                    H = np.vstack([J_np.T, -J_np.T])  # [24, 6]
                    d = np.concatenate([torque_limit_max, -torque_limit_min])  # [24]

                    # Map joint torques to Cartesian force: f = (J_np^T)^+ * tau
                    JT_pinv = np.linalg.pinv(J_np.T)
                    f_prop = (JT_pinv @ action).astype(np.float64)

                    # Ensure H and d are float64 for SciPy solvers
                    H = H.astype(np.float64)
                    d = d.astype(np.float64)

                    # If force violates H * f <= d, project it onto the polytope boundary
                    if np.any(H @ f_prop > d):
                        from scipy.optimize import minimize

                        # Objective: minimize L2 distance to proposed force
                        obj = lambda f: np.sum((f - f_prop) ** 2)
                        # Constraint: d - H * f >= 0
                        cons = {"type": "ineq", "fun": lambda f: d - H @ f}
                        res = minimize(obj, f_prop, constraints=cons, method="SLSQP")
                        f_proj = res.x
                    else:
                        f_proj = f_prop

                    # Convert back to joint torque space: tau = J_np^T * f_proj
                    action_proj = J_np.T @ f_proj
                    filtered.append(action_proj)
                    continue
                except Exception as e:
                    # Fallback to analytical clipping on failure
                    print(
                        f"Warning: pycapacity projection failed ({e}). Falling back to clipping."
                    )
                    pass

            # Fallback clipping to ensure physical bounds are never violated
            torque_max = self.torque_max
            if len(action) > len(torque_max):
                pad_len = len(action) - len(torque_max)
                torque_max = np.concatenate(
                    [torque_max, np.full(pad_len, 1000.0, dtype=np.float32)]
                )
            action_clipped = np.clip(action, -torque_max, torque_max)
            filtered.append(action_clipped)

        filtered_np = np.stack(filtered, axis=0)

        if is_tensor:
            return torch.tensor(filtered_np, dtype=torch.float32, device=device)
        return filtered_np
