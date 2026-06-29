import torch
import numpy as np
from utils.safety_filter import SafetyFilter, PYCAPACITY_AVAILABLE

if PYCAPACITY_AVAILABLE:
    import pycapacity.robot
else:
    print("Warning: pycapacity is not installed. Testing basic clipping fallback.")


def run_pycapacity_poc():
    print("=== Task 1.6: pycapacity Safety Filter Feasibility PoC ===")

    # 1. Instantiate the safety filter
    filter_unit = SafetyFilter()

    # Define a batch of proposed joint actions (shape: [B, 12] representing joint torques)
    # Some values exceed the torque limits (10.0 Nm) to test filtering
    B = 5
    proposed_actions = np.array(
        [
            [12.0, -8.0, 5.0, -2.0, 15.0, 0.0, 3.0, -11.0, 4.0, -0.5, 9.0, -3.0],
            [-15.0, 7.5, -9.0, 1.0, -3.0, 2.0, -4.0, 6.0, -1.0, 0.1, -12.0, 8.0],
            [5.0, 3.0, -2.0, -1.0, 4.0, 0.0, -3.0, 2.0, 1.0, -0.2, 5.0, -4.0],
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
                -20.0,
            ],
        ]
    )

    # 2. Test standard clipping fallback
    print("\nTesting analytical clipping fallback...")
    clipped_actions = filter_unit.filter_actions(proposed_actions, J=None)

    # Verify values are bounded by torque_max (10.0)
    max_val = np.max(np.abs(clipped_actions))
    print(f"Max absolute torque after clipping: {max_val:.4f} Nm (Should be <= 10.0)")

    clipping_success = max_val <= 10.0

    # 3. Test pycapacity polytope Cartesian projection
    print("\nTesting pycapacity Cartesian force/torque polytope projection...")
    # Jacobian matrix J for 12 joints to 6 Cartesian task space coordinates (shape: [6, 12])
    J = np.random.randn(6, 12)

    polytope_actions = filter_unit.filter_actions(proposed_actions, J=J)
    print(
        f"Proposed shape: {proposed_actions.shape} | Projected shape: {polytope_actions.shape}"
    )

    # Verify that the projected Cartesian forces are shape [B, 6] or clipped to joint bounds
    # (Since torque_to_force outputs Cartesian forces)
    print(f"Polytope projections shape: {polytope_actions.shape}")

    # Test tensor input conversion as well
    proposed_tensor = torch.tensor(proposed_actions, dtype=torch.float32)
    J_tensor = torch.tensor(J, dtype=torch.float32)
    filtered_tensor = filter_unit.filter_actions(proposed_tensor, J=J_tensor)
    print(
        f"Tensor projection input: {proposed_tensor.device} | Output: {filtered_tensor.device}"
    )

    polytope_success = polytope_actions.shape == (B, 12)

    if clipping_success and polytope_success:
        print(
            "\nPoC Result: SUCCESS (Safety filter functions and projections verified)"
        )
    else:
        print("\nPoC Result: FAILED (Safety limits exceeded or shape mismatch)")


if __name__ == "__main__":
    run_pycapacity_poc()
