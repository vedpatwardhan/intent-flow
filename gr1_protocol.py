import numpy as np
from gr1_config import JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, COMPACT_WIRE_JOINTS


class StandardScaler:
    """
    Grounds model outputs in physical limits (Canonical Min-Max Normalization).
    Provides the universal [-1, 1] <-> Radians handshake for the GR-1 protocol.
    """

    def __init__(self):
        # STAGE 0: RE-INITIALIZATION (Dynamic Handshake)
        self.lmin = JOINT_LIMITS_MIN
        self.lmax = JOINT_LIMITS_MAX

        # Calculate range safely
        self.range = self.lmax - self.lmin
        self.range = np.where(self.range < 1e-6, 1.0, self.range)

    def unscale_action(self, norm_action):
        """Maps model output [-1, 1] to Radians."""
        return (
            np.array(norm_action, dtype=np.float32) + 1.0
        ) * self.range / 2.0 + self.lmin

    def scale_state(self, raw_state):
        """Maps Raw Radians to [-1, 1] with Global Integrity Guard."""
        norm_state = (
            2.0 * (np.array(raw_state, dtype=np.float32) - self.lmin) / self.range
        ) - 1.0

        # GLOBAL INTEGRITY GUARD: Silent monitor for all 32 joints
        # Only prints if a protocol violation is detected
        if np.any(np.abs(norm_state) > 1.01):
            bad_indices = np.where(np.abs(norm_state) > 1.01)[0]

            print(f"\n🚨 [PROTOCOL VIOLATION] Action exceeds [-1, 1] boundary!")
            for b_idx in bad_indices:
                print(
                    f"   - Joint {b_idx:02d} ({COMPACT_WIRE_JOINTS[b_idx]}): {norm_state[b_idx]:.4f} (Raw: {raw_state[b_idx]:.4f})"
                )

        return norm_state
