import os
import numpy as np
import mujoco

# -----------------------------------------------------------------------------
# GR1 Robot Configuration
# -----------------------------------------------------------------------------


def load_joint_order():
    order_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gr1_joint_order.txt"
    )
    if not os.path.exists(order_path):
        raise FileNotFoundError(f"❌ Canonical protocol missing: {order_path}")

    with open(order_path, "r") as f:
        joints = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    return joints


COMPACT_WIRE_JOINTS = load_joint_order()

FROZEN_JOINTS = {
    "head_pitch_joint": 0.0,
    "head_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
}

# Joints that are temporarily locked to a specific pose ONLY during IK demonstations
# to stabilize biomechanics (e.g., preventing palm flips).
# These remain ACTIVE in the VLA action space with full range.
IK_POSTURE_LOCKS = {
    "R_thumb_proximal_yaw_joint": -1.676,  # Normalized -1.0
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(BASE_DIR, "sim_assets/scene_gr1_pickup.xml")


def get_protocol_limits():
    """
    CANONICAL HANDSHAKE:
    Extracts joint limits directly from the compiled MuJoCo model to ensure
    perfect parity between simulation physics and dataset normalization.
    """
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)

    mins, maxs = [], []
    for name in COMPACT_WIRE_JOINTS:
        if name in FROZEN_JOINTS:
            val = FROZEN_JOINTS[name]
            mins.append(val)
            maxs.append(val)
            continue

        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id == -1:
            # Fallback for non-existent joints (should not happen in valid protocol)
            mins.append(-1.0)
            maxs.append(1.0)
            continue

        # Ground Truth from MuJoCo Engine
        r = model.jnt_range[j_id]
        mins.append(r[0])
        maxs.append(r[1])

    return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)


# Global constants for legacy support (Calculated at import time)
# Note: Re-importing this module will refresh from the XML.
JOINT_LIMITS_MIN, JOINT_LIMITS_MAX = get_protocol_limits()
