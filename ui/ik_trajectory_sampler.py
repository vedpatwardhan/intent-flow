import base64
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import os


def frame_to_base64(rgb_array: np.ndarray) -> str:
    img = Image.fromarray(rgb_array).resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def solve_ik_target(
    sim, target_3d: np.ndarray, site_name: str = "right_hand_roll_link"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves inverse kinematics for a target 3D world position using sim.solve_ik.
    Matches le-probe flow: sim.solve_ik -> qpos_to_action_32 -> unscaler.scale_state.

    Returns:
        tuple[np.ndarray, np.ndarray]: (q_target_norm_32, q_target_full_qpos)
    """
    # Downward grasping orientation quaternion
    quat_down = np.array([0, 1, 0, 0])
    q_target_full = sim.solve_ik(target_3d, quat_down)
    raw_32 = sim.qpos_to_action_32(q_target_full)
    norm_32 = sim.unscaler.scale_state(raw_32)
    return norm_32, q_target_full


CAM_NAMES = [
    "world_center",
    "world_top",
    "world_left",
    "world_right",
    "world_wrist",
]


def generate_ik_trajectories(
    sim,
    initial_state: dict,
    target_3d: np.ndarray,
    target_3d_bounds: dict,
    site_name: str = "R_wrist_roll_link",
    n: int = 4,
    scale_multiplier: float = 0.3,
    is_positive: bool = True,
) -> list[dict]:
    """
    Generates n IK trajectories surrounding the target 3D object for a dynamically selected site_name.
    Uses target_3d_bounds and scale_multiplier (0.3 for positive, 0.9 for negative) to perturb target positions.
    Renders 5 camera views per step.
    """
    trajectories = []

    # Calculate perturbation scale based on object 3D spatial extents
    extents = np.array(target_3d_bounds["extents_3d"]).max()
    delta = max(0.01, float(extents) * scale_multiplier)
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([delta, 0.0, 0.0]),
        np.array([-delta, 0.0, 0.0]),
        np.array([0.0, delta, 0.0]),
        np.array([0.0, -delta, 0.0]),
    ]
    if not is_positive:
        offsets = offsets[1:] + [
            np.array([delta, delta, 0.0]),
            np.array([-delta, -delta, 0.0]),
            np.array([delta, -delta, 0.0]),
            np.array([-delta, delta, 0.0]),
        ]

    for k in range(n):
        # Reset sim to identical initial state
        sim.data.qpos[:] = initial_state["qpos"].copy()
        sim.data.qvel[:] = initial_state["qvel"].copy()
        sim.data.ctrl[:] = initial_state["ctrl"].copy()
        mujoco.mj_forward(sim.model, sim.data)

        target_k = target_3d + offsets[k % len(offsets)]
        print(f"target_k ({'positive' if is_positive else 'negative'}): {target_k}")
        q_target_norm, q_target_full = solve_ik_target(
            sim, target_k, site_name=site_name
        )

        q_start_norm = sim.get_state_32()
        actions = []
        observations = dict()

        # Render initial start frame (t=0) across all 5 cameras before stepping
        for cam in CAM_NAMES:
            sim.renderer.update_scene(sim.data, camera=cam)
            rgb = sim.renderer.render().copy()
            observations[cam] = [frame_to_base64(rgb)]

        for h in range(7):
            alpha = (h + 1) / 7.0
            q_interp = (1.0 - alpha) * q_start_norm + alpha * q_target_norm
            actions.append(q_interp.tolist())

            # Dispatch action to step physical sim
            sim.dispatch_action(
                action_32_norm=q_interp,
                target_q=q_target_full,
                n_steps=20,
                render_freq=0,
                reset_start=False,
            )

            # Render frames across all 5 cameras
            if h in [1, 3, 6]:
                for cam in CAM_NAMES:
                    sim.renderer.update_scene(sim.data, camera=cam)
                    rgb = sim.renderer.render().copy()
                    observations[cam].append(frame_to_base64(rgb))

        # Constructed the final observations in right format
        final_observations = []
        for cam_idx, cam in enumerate(CAM_NAMES):
            for frame_idx, frame in enumerate(observations[cam]):
                if cam_idx == 0:
                    final_observations.append({cam: frame})
                else:
                    final_observations[frame_idx][cam] = frame

        # Pad actions [7, 32] to [7, 58]
        actions = np.array(actions)
        final_actions = np.concatenate(
            [actions, np.zeros((actions.shape[0], 58 - actions.shape[1]))],
            axis=1,
        )

        # ToDo: Include tactile information
        trajectories.append(
            {
                "track_idx": k,
                "target_pos": target_k.tolist(),
                "frames": final_observations[-1],
                "history_frames": final_observations,
                "proprioception": sim.unscaler.scale_state(sim.get_state_32()).tolist(),
                "actions": final_actions.tolist(),
                "is_positive": is_positive,
            }
        )

    return trajectories


def save_ik_trajectory_diagnostic_plots(
    sim, pos_trajectories: list, neg_trajectories: list
):
    """
    Renders diagnostic 3D scatter plots comparing 4 Positive IK trajectories vs 10 Negative IK trajectories.
    """
    try:
        fig = plt.figure(figsize=(12, 6))

        # --- Panel 1: Positive Trajectories (D+) ---
        ax1 = fig.add_subplot(121, projection="3d")
        for tr in pos_trajectories:
            t3d = tr["target_pos"]
            ax1.scatter(t3d[0], t3d[1], t3d[2], color="green", s=60, label="D+ Target")
        ax1.set_title("1. Positive IK Trajectories (D+)", fontsize=10)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")

        # --- Panel 2: Negative Distractor Trajectories (D-) ---
        ax2 = fig.add_subplot(122, projection="3d")
        for tr in neg_trajectories:
            t3d = tr["target_pos"]
            ax2.scatter(
                t3d[0], t3d[1], t3d[2], color="red", s=40, label="D- Distractor"
            )
        ax2.set_title("2. Negative Distractor Trajectories (D-)", fontsize=10)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")

        plt.tight_layout()
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "debug_ik_positive_and_negative_trajectories.png",
        )
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"📸 Saved IK trajectory diagnostic plot to: {out_path}")
    except Exception as e:
        print(f"Error saving IK diagnostic plot: {e}")


def save_ik_trajectory_video(
    pos_trajectories: list,
    neg_trajectories: list,
    output_dir: str = "latent-flow/ui/logs/training/goals",
    fps: int = 4,
):
    """
    Encodes rendered RGB frames of positive (D+) and negative (D-) IK trajectories
    into separate MP4 videos per trajectory track (1 per camera view per track) inside output_dir.
    """
    try:
        abs_output_dir = os.path.abspath(output_dir)
        pos_output_dir = os.path.join(abs_output_dir, "positive")
        neg_output_dir = os.path.join(abs_output_dir, "negative")

        for cam in CAM_NAMES:
            pos_cam_dir = os.path.join(pos_output_dir, cam)
            neg_cam_dir = os.path.join(neg_output_dir, cam)
            os.makedirs(pos_cam_dir, exist_ok=True)
            os.makedirs(neg_cam_dir, exist_ok=True)

            # 1. Save individual Positive Trajectory Videos
            for tr in pos_trajectories:
                track_idx = tr.get("track_idx", 0)
                mv_frames = tr.get("multi_view_frames", {})
                if cam in mv_frames and mv_frames[cam]:
                    frames = mv_frames[cam]
                    h, w, _ = frames[0].shape
                    pos_video_path = os.path.join(
                        pos_cam_dir,
                        f"track_{track_idx}.mp4",
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    writer = cv2.VideoWriter(pos_video_path, fourcc, fps, (w, h))
                    for frame in frames:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(bgr)
                    writer.release()
                    print(
                        f"🎬 Saved Positive IK Video ({cam} track {track_idx}): {pos_video_path}"
                    )

            # 2. Save individual Negative Trajectory Videos
            for tr in neg_trajectories:
                track_idx = tr.get("track_idx", 0)
                mv_frames = tr.get("multi_view_frames", {})
                if cam in mv_frames and mv_frames[cam]:
                    frames = mv_frames[cam]
                    h, w, _ = frames[0].shape
                    neg_video_path = os.path.join(
                        neg_cam_dir,
                        f"track_{track_idx}.mp4",
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    writer = cv2.VideoWriter(neg_video_path, fourcc, fps, (w, h))
                    for frame in frames:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(bgr)
                    writer.release()
                    print(
                        f"🎬 Saved Negative IK Video ({cam} track {track_idx}): {neg_video_path}"
                    )

    except Exception as e:
        print(f"Error saving IK trajectory videos: {e}")
