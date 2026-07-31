import copy
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import cv2
import os


def solve_ik_target(
    sim, target_3d: np.ndarray, site_name: str = "R_wrist_roll_link"
) -> np.ndarray:
    """
    Solves inverse kinematics for a target 3D world position using Jacobian pseudo-inverse Damped Least Squares (DLS).

    Parameters:
        sim: Simulation object.
        target_3d: Target 3D coordinates [X_w, Y_w, Z_w].
        site_name: End-effector link or body site name dynamically passed from 3D unprojection.

    Returns:
        np.ndarray: Solved 32-dim joint target position array.
    """
    try:
        body_id = sim.model.body(site_name).id
    except Exception:
        body_id = sim.model.body("R_index_tip_link").id

    qpos_ik = sim.data.qpos.copy()
    step_size = 0.5
    damping = 1e-2

    for _ in range(100):
        # Forward kinematics
        mujoco.mj_forward(sim.model, sim.data)
        curr_pos = sim.data.xpos[body_id]
        error = target_3d - curr_pos

        if np.linalg.norm(error) < 0.005:  # 5mm convergence threshold
            break

        # Compute translation Jacobian
        jacp = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp, None, body_id)

        # Damped Least Squares Inverse: J^T * (J * J^T + lambda^2 * I)^-1
        jjt = jacp @ jacp.T + damping * np.eye(3)
        delta_qvel = jacp.T @ np.linalg.solve(jjt, error)

        # Integrate joint positions
        sim.data.qpos[:] += step_size * delta_qvel
        sim.data.qpos[:] = np.clip(
            sim.data.qpos[:], sim.model.jnt_range[:, 0], sim.model.jnt_range[:, 1]
        )

    solved_qpos = sim.data.qpos.copy()
    sim.data.qpos[:] = qpos_ik  # Restore
    mujoco.mj_forward(sim.model, sim.data)

    # Scale 32-dim state via unscaler if available
    if hasattr(sim, "unscaler"):
        return sim.unscaler.scale_state(solved_qpos[:32])
    return solved_qpos[:32]


def generate_ik_positive_trajectories(
    sim,
    initial_state: dict,
    target_3d: np.ndarray,
    target_3d_bounds: dict | None = None,
    site_name: str = "R_wrist_roll_link",
    n: int = 4,
) -> list[dict]:
    """
    Generates n positive IK trajectories surrounding the target 3D object for a dynamically selected site_name.
    Uses target_3d_bounds to perturb target positions along the physical 3D extents of the object.
    """
    trajectories = []

    # Calculate perturbation scale based on object 3D spatial extents
    if target_3d_bounds and "extents_3d" in target_3d_bounds:
        extents = np.array(target_3d_bounds["extents_3d"])
        dx = max(0.01, float(extents[0]) * 0.5)
        dy = max(0.01, float(extents[1]) * 0.5)
        offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([dx, 0.0, 0.0]),
            np.array([-dx, 0.0, 0.0]),
            np.array([0.0, dy, 0.0]),
        ]
    else:
        offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.015, 0.0, 0.0]),
            np.array([-0.015, 0.0, 0.0]),
            np.array([0.0, 0.015, 0.0]),
        ]

    for k in range(n):
        # Reset sim to identical initial state
        sim.data.qpos[:] = initial_state["qpos"].copy()
        sim.data.qvel[:] = initial_state["qvel"].copy()
        sim.data.ctrl[:] = initial_state["ctrl"].copy()
        mujoco.mj_forward(sim.model, sim.data)

        target_k = target_3d + offsets[k % len(offsets)]
        q_target = solve_ik_target(sim, target_k, site_name=site_name)

        # Interpolate 7-step trajectory and render RGB frames
        q_start = sim.unscaler.scale_state(sim.get_state_32())
        step_trajectories = []
        rendered_frames = []

        for h in range(7):
            alpha = (h + 1) / 7.0
            q_interp = (1.0 - alpha) * q_start + alpha * q_target
            step_trajectories.append(q_interp.tolist())

            # Apply step to sim and render RGB frame
            sim.process_target_32(q_interp)
            sim.dispatch_action(
                action_32_norm=q_interp,
                target_q=sim.last_target_q,
                n_steps=2,
                render_freq=0,
                reset_start=False,
            )
            sim.renderer.update_scene(sim.data, camera="world_center")
            rgb = sim.renderer.render().copy()
            rendered_frames.append(rgb)

        trajectories.append(
            {
                "track_idx": k,
                "target_3d": target_k.tolist(),
                "trajectory_steps": step_trajectories,
                "rendered_frames": rendered_frames,
                "is_positive": True,
            }
        )

    return trajectories


def generate_ik_negative_trajectories(
    sim,
    initial_state: dict,
    target_3d: np.ndarray,
    target_3d_bounds: dict | None = None,
    site_name: str = "R_wrist_roll_link",
    n: int = 10,
) -> list[dict]:
    """
    Generates n negative distractor IK trajectories away from the target 3D object for a dynamically selected site_name.
    Resets sim to initial_state for each trajectory.
    """
    trajectories = []
    np.random.seed(42)

    for k in range(n):
        # Reset sim to identical initial state
        sim.data.qpos[:] = initial_state["qpos"].copy()
        sim.data.qvel[:] = initial_state["qvel"].copy()
        sim.data.ctrl[:] = initial_state["ctrl"].copy()
        mujoco.mj_forward(sim.model, sim.data)

        random_dir = np.random.uniform(-1.0, 1.0, size=3)
        random_dir /= np.linalg.norm(random_dir) + 1e-8
        distractor_dist = np.random.uniform(0.15, 0.30)
        target_k = target_3d + random_dir * distractor_dist

        q_target = solve_ik_target(sim, target_k, site_name=site_name)
        # Unroll steps and capture rendered RGB frames
        step_trajectories = []
        rendered_frames = []

        q_start = sim.unscaler.scale_state(sim.get_state_32())

        for h in range(7):
            alpha = (h + 1) / 7.0
            q_interp = (1.0 - alpha) * q_start + alpha * q_target
            step_trajectories.append(q_interp.tolist())

            # Apply step to sim and render RGB frame
            sim.process_target_32(q_interp)
            sim.dispatch_action(
                action_32_norm=q_interp,
                target_q=sim.last_target_q,
                n_steps=2,
                render_freq=0,
                reset_start=False,
            )
            sim.renderer.update_scene(sim.data, camera="world_center")
            rgb = sim.renderer.render().copy()
            rendered_frames.append(rgb)

        trajectories.append(
            {
                "track_idx": k,
                "target_3d": target_k.tolist(),
                "trajectory_steps": step_trajectories,
                "rendered_frames": rendered_frames,
                "is_positive": False,
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
            t3d = tr["target_3d"]
            ax1.scatter(t3d[0], t3d[1], t3d[2], color="green", s=60, label="D+ Target")
        ax1.set_title("1. Positive IK Trajectories (D+)", fontsize=10)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")

        # --- Panel 2: Negative Distractor Trajectories (D-) ---
        ax2 = fig.add_subplot(122, projection="3d")
        for tr in neg_trajectories:
            t3d = tr["target_3d"]
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
    pos_trajectories: list, neg_trajectories: list, fps: int = 4
):
    """
    Encodes rendered RGB frames of positive (D+) and negative (D-) IK trajectories into MP4 videos.
    """
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 1. Save Positive (D+) Trajectories Video
        pos_video_path = os.path.join(base_dir, "debug_ik_positive_trajectories.mp4")
        all_pos_frames = []
        for tr in pos_trajectories:
            all_pos_frames.extend(tr.get("rendered_frames", []))

        if all_pos_frames:
            h, w, _ = all_pos_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(pos_video_path, fourcc, fps, (w, h))
            for frame in all_pos_frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            print(f"🎬 Saved Positive IK Trajectories Video: {pos_video_path}")

        # 2. Save Negative (D-) Trajectories Video
        neg_video_path = os.path.join(base_dir, "debug_ik_negative_trajectories.mp4")
        all_neg_frames = []
        for tr in neg_trajectories:
            all_neg_frames.extend(tr.get("rendered_frames", []))

        if all_neg_frames:
            h, w, _ = all_neg_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(neg_video_path, fourcc, fps, (w, h))
            for frame in all_neg_frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            print(f"🎬 Saved Negative IK Trajectories Video: {neg_video_path}")

    except Exception as e:
        print(f"Error saving IK trajectory videos: {e}")
