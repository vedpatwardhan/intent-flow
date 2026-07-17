import asyncio
import base64
import io
import json
import os
import sys
import traceback
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
import numpy as np
import cv2
import httpx
import copy

# Add parent directory (latent-flow root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr1_config import COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler
from simulation_base import GR1MuJoCoBase
import mujoco


class GR1SimulationServer(GR1MuJoCoBase):
    """
    Stand-alone MuJoCo Simulation Server.
    Completely isolated from the le-probe project.
    """

    def __init__(self):
        super().__init__(restrict_ik=True)

    def _render_needs_depth(self) -> bool:
        return True

    def get_point_cloud_numpy(self, cam_name: str, rgb_224: np.ndarray) -> list:
        # 1. Render and capture depth buffer
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=cam_name)
        depth_cam = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        # 2. Metric depth (meters) is returned directly by MuJoCo's Python renderer
        near = self.model.vis.map.znear
        far = self.model.vis.map.zfar
        metric_depth = depth_cam

        # 3. Resize to 224x224 matching image shape
        metric_depth_224 = cv2.resize(
            metric_depth, (224, 224), interpolation=cv2.INTER_NEAREST
        )

        # 4. Grid generation (70x70) on 224x224 resolution
        w, h = 224, 224
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, w - 1, 70).astype(int),
            np.linspace(0, h - 1, 70).astype(int),
        )
        xs = grid_x.flatten()
        ys = grid_y.flatten()
        zs = metric_depth_224[ys, xs]

        # 5. Filter out background points on raw metric depth first (in meters)
        foreground_mask = zs < 3.0
        if foreground_mask.sum() > 0:
            xs = xs[foreground_mask]
            ys = ys[foreground_mask]
            zs = zs[foreground_mask]

        # 6. Projection: Emulate disparity convention where closer is larger and farther is smaller
        focal_length = max(w, h)
        cx = w / 2.0
        cy = h / 2.0
        zs_proj = far - zs
        xs_proj = (xs - cx) * zs_proj / focal_length
        ys_proj = (cy - ys) * zs_proj / focal_length

        # 7. Colors
        colors = rgb_224[ys, xs]
        rs = colors[:, 0] / 255.0
        gs = colors[:, 1] / 255.0
        bs = colors[:, 2] / 255.0

        # 7. Range check & Jitter (Anti-Degeneracy)
        x_range = xs_proj.max() - xs_proj.min() if len(xs_proj) > 0 else 0
        y_range = ys_proj.max() - ys_proj.min() if len(ys_proj) > 0 else 0
        z_range = zs_proj.max() - zs_proj.min() if len(zs_proj) > 0 else 0

        if max(x_range, y_range, z_range) < 1e-3:
            xs_proj = xs_proj + np.random.normal(0, 1e-5, xs_proj.shape)
            ys_proj = ys_proj + np.random.normal(0, 1e-5, ys_proj.shape)
            zs_proj = zs_proj + np.random.normal(0, 1e-5, zs_proj.shape)
            x_range = xs_proj.max() - xs_proj.min()
            y_range = ys_proj.max() - ys_proj.min()
            z_range = zs_proj.max() - zs_proj.min()

        max_range = max(x_range, y_range, z_range, 1e-4)

        # 8. Normalization & Viewport Offset
        xs_norm = (xs_proj - xs_proj.mean()) / max_range * 1.5
        ys_norm = (ys_proj - ys_proj.mean()) / max_range * 1.5 - 0.25
        zs_norm = (zs_proj - zs_proj.mean()) / max_range * 1.5

        point_cloud = np.stack([xs_norm, ys_norm, zs_norm, rs, gs, bs], axis=1)
        return point_cloud.tolist()

    def _handle_ik_pickup_logic(self, phase=0, offset_cm=5):
        """Standard multi-phase IK solver for the red cube."""
        self.current_phase = phase + 1
        print(
            f"🎯 Executing IK Pickup Phase {phase} (Global ID: {self.current_phase})..."
        )

        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_pos = self.data.qpos[
            self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id] + 3
        ].copy()
        quat_down = [0, 1, 0, 0]

        if phase == 0:
            # Phase 1: Lift (Approach)
            pos_i_h, pos_t_h, pos_w_h = (
                cube_pos + [0.02, 0.02, 0.02 + offset_cm / 100.0],
                cube_pos + [-0.02, 0, 0.02 + offset_cm / 100.0],
                cube_pos + [0, 0, 0.08 + offset_cm / 100.0],
            )
            q_reach_h = self.solve_ik(
                pos_w_h, quat_down, pos_i_h, pos_t_h, posture_cost=1e-6
            )
            self.dispatch_action(
                self.qpos_to_action_32(q_reach_h),
                q_reach_h,
                n_steps=240,
                render_freq=30,
            )

        elif phase == 1:
            # Phase 2: Descent
            pos_i_l, pos_t_l, pos_w_l = (
                cube_pos + [-0.02, 0.02, 0],
                cube_pos + [-0.06, 0, 0],
                cube_pos + [0, 0, 0.06],
            )
            q_reach_l = self.solve_ik(
                pos_w_l, quat_down, pos_i_l, pos_t_l, posture_cost=1e-6
            )
            for f_idx in [50, 51, 52, 53, 54, 55, 56]:
                if f_idx < len(q_reach_l):
                    q_reach_l[f_idx] = 0.0

            self.dispatch_action(
                self.qpos_to_action_32(q_reach_l),
                q_reach_l,
                n_steps=240,
                render_freq=30,
            )

        elif phase == 2:
            # Phase 3: Grasp
            pos_i_l, pos_t_l, pos_w_l = (
                cube_pos + [0, 0.02, 0],
                cube_pos + [0, 0, 0],
                cube_pos + [0, 0, 0],
            )
            q_reach_l = self.solve_ik(
                pos_w_l, quat_down, pos_i_l, pos_t_l, posture_cost=1e-6
            )
            q_grasp = q_reach_l.copy()
            q_grasp[48] = 1.1
            for g_id in [50, 52, 54, 56]:
                q_grasp[g_id] = -1.1
            self.dispatch_action(
                self.qpos_to_action_32(q_grasp), q_grasp, n_steps=240, render_freq=30
            )

        elif phase == 3:
            # Phase 4: Lift (Retract)
            pos_i_up, pos_t_up, pos_w_up = (
                cube_pos + [0, 0.02, 0.25],
                cube_pos + [0, 0, 0.25],
                cube_pos + [0, 0, 0.25],
            )
            q_lift = self.solve_ik(
                pos_w_up, quat_down, pos_i_up, pos_t_up, posture_cost=1e-6
            )
            q_lift[48] = 1.1
            for g_id in [50, 52, 54, 56]:
                q_lift[g_id] = -1.1
            self.dispatch_action(
                self.qpos_to_action_32(q_lift), q_lift, n_steps=240, render_freq=30
            )


# Instantiate local server
sim = GR1SimulationServer()
sim.reset_env(lock_posture=True)

app = FastAPI()

active_camera = "world_center"
encoder_processing_enabled = True
combostoc_noise = {"torso": 0.0, "arm": 0.0, "hand": 0.0, "vision": 0.0}
attack_active = False

colab_url = None
for i, arg in enumerate(sys.argv):
    if arg == "--colab-url" and i + 1 < len(sys.argv):
        colab_url = sys.argv[i + 1]
colab_url = colab_url or os.environ.get("COLAB_URL")

click_x = None
click_y = None
click_type = None
text_prompt = "cube block"
text_modifier = None
frame_history = deque(maxlen=5)
ui_annotations = {}

colab_is_processing = False
is_training_active = False
needs_colab_processing = False
last_colab_query_time = 0.0

cached_dino_attn = None
cached_clip_sim = None
cached_sam_mask = None
cached_point_cloud = []
cached_vggt_tracks = []
cached_task_isolated_features = None


async def run_stage3_training_loop(
    websocket, sim, colab_url, text_prompt, ui_annotations
):
    # Called from the websocket endpoint
    if not colab_url:
        print("[Training Error] Colab URL is not set. Cannot run training.")
        await websocket.send_text(
            json.dumps(
                {
                    "type": "training_progress",
                    "status": "Error: Colab URL is not set.",
                    "progress": 0.0,
                    "episode": 0,
                    "total_episodes": 0,
                }
            )
        )
        return

    index_id = sim.model.body("R_index_tip_link").id
    thumb_id = sim.model.body("R_thumb_tip_link").id
    cube_id = sim.model.body("cube").id
    initial_state = {
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "ctrl": sim.data.ctrl.copy(),
    }

    num_episodes = 5
    max_steps = 20

    print(
        f"[Training] Starting Stage 3 training sandbox: {num_episodes} episodes, {max_steps} steps."
    )

    for ep_idx in range(num_episodes):
        if ep_idx > 0:
            sim.reset_env(lock_posture=True)
        frame_history = []

        # Reflected in the progress bar on the UI
        await websocket.send_text(
            json.dumps(
                {
                    "type": "training_progress",
                    "status": f"Episode {ep_idx + 1}/{num_episodes} in progress...",
                    "progress": float(ep_idx) / num_episodes,
                    "episode": ep_idx + 1,
                    "total_episodes": num_episodes,
                }
            )
        )

        episode_reward = 0.0

        for env_step in range(max_steps):
            # Capture observation
            sim.renderer.update_scene(sim.data, camera="world_center")

            # Computes distance between fingers and cube
            # ToDo: Needs to be generalized for other tasks, where the user configures
            # which fingers matter, relative positions, etc. in the UI.
            index_pos = sim.data.xpos[index_id]
            thumb_pos = sim.data.xpos[thumb_id]
            cube_pos = sim.data.xpos[cube_id]

            d_index = float(np.linalg.norm(index_pos - cube_pos))
            d_thumb = float(np.linalg.norm(thumb_pos - cube_pos))

            touch_index = max(0.0, 1.0 - (d_index / 0.04)) if d_index < 0.04 else 0.0
            touch_thumb = max(0.0, 1.0 - (d_thumb / 0.04)) if d_thumb < 0.04 else 0.0

            tactile_grid = [[0.0] * 4 for _ in range(4)]
            tactile_grid[0][0] = touch_index
            tactile_grid[1][1] = touch_thumb

            # Capture all 5 camera views for multi-view processing
            frames_all_views = {}
            point_clouds_all_views = {}
            for cam_name in sim.cam_names:
                sim.renderer.update_scene(sim.data, camera=cam_name)
                rgb_cam = sim.renderer.render()
                img_cam = Image.fromarray(rgb_cam)
                img_cam_224 = img_cam.resize((224, 224))
                buf_cam = io.BytesIO()
                img_cam_224.save(buf_cam, format="JPEG", quality=75)
                frames_all_views[cam_name] = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(buf_cam.getvalue()).decode("utf-8")
                )
                point_clouds_all_views[cam_name] = sim.get_point_cloud_numpy(
                    cam_name, np.array(img_cam_224)
                )

            if len(frame_history) < 2:
                frame_history.append(frames_all_views)
            else:
                frame_history.pop(0)
                frame_history.append(frames_all_views)

            proprio_list = sim.get_state_32()[:24].tolist()
            if any(np.isnan(val) for val in proprio_list):
                print(
                    f"⚠️ [NaN Warning] MuJoCo joint proprioception contains NaN at step {env_step}! Simulator exploded."
                )

            current_obs = {
                "frames": frames_all_views,
                "history_frames": list(frame_history),
                "proprioception": proprio_list,
                "tactile": tactile_grid,
                "text_prompt": text_prompt or "grasp cube",
                "ui_annotations": ui_annotations
                or {"crops": [], "vectors": [], "segments": []},
                "is_easy_task": False,
                "point_clouds": point_clouds_all_views,
            }

            action_taken_ensemble = None
            energy_ensemble = None
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        f"{colab_url}/stage3/step", json=current_obs, timeout=100.0
                    )
                    if r.status_code == 200:
                        res = r.json()
                        action_taken_ensemble = res.get("action")
                        energy_ensemble = res.get("energy")
                        perturbed_payloads = res.get("perturbed_payloads", [])
            except Exception as e:
                print(
                    f"[Training Error] Episode {ep_idx + 1} Step "
                    f"{env_step} Colab step query failed: {e}"
                )
                break

            if action_taken_ensemble is None or energy_ensemble is None:
                print("[Training Error] No action or energy returned from Colab.")
                break

            # Convert to numpy to slice precisely
            action_np = np.array(action_taken_ensemble, dtype=np.float32)  # [16, 7, 58]

            # Snapshot initial state
            initial_qpos = sim.data.qpos.copy()
            initial_qvel = sim.data.qvel.copy()
            initial_ctrl = sim.data.ctrl.copy()

            transitions = []
            committed_qpos = None

            # Replay all 16 trajectories inside the simulator
            for track_idx in range(action_np.shape[0]):
                # Rewind physics cleanly to the starting coordinates of the rollout window
                sim.data.qpos[:] = initial_qpos
                sim.data.qvel[:] = initial_qvel
                sim.data.ctrl[:] = initial_ctrl
                mujoco.mj_forward(sim.model, sim.data)

                # Initialize frame recording buffer
                track_frames = []

                # Capture starting frame (t=0)
                start_frames = {}
                for cam_name in sim.cam_names:
                    sim.renderer.update_scene(sim.data, camera=cam_name)
                    start_frames[cam_name] = sim.renderer.render().copy()
                track_frames.append(start_frames)

                track_actions_flat = []

                # Execute the full 8-step trajectory sequence open-loop
                for h in range(action_np.shape[1]):
                    track_action = action_np[track_idx, h, :]  # Shape: (58,)
                    track_actions_flat.extend(track_action.tolist())
                    action_32 = track_action[:32]
                    action_32_clamped = np.clip(action_32, -1.0, 1.0)
                    action_rad = sim.unscaler.unscale_action(action_32_clamped)

                    # Execute motor posture loop
                    for _ in range(2):
                        for i, j_id in enumerate(sim.protocol_joint_ids):
                            if j_id != -1:
                                q_idx = sim.model.jnt_qposadr[j_id]
                                sim.last_target_q[q_idx] = action_rad[i]
                                if i in sim.coupling_map:
                                    for distal_idx in sim.coupling_map[i]:
                                        sim.last_target_q[distal_idx] = action_rad[i]
                        sim.sync_ctrl_to_qpos(sim.last_target_q)
                        sim.data.qpos[sim.root_q_idx : sim.root_q_idx + 3] = [
                            0.0,
                            0.0,
                            0.95,
                        ]
                        sim.data.qpos[sim.root_q_idx + 3 : sim.root_q_idx + 7] = [
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                        sim.data.qvel[:6] = 0.0
                        mujoco.mj_step(sim.model, sim.data)

                    # Get next observation for this specific candidate step
                    frames_all_views_next = {}
                    step_frames = {}
                    for cam_name in sim.cam_names:
                        sim.renderer.update_scene(sim.data, camera=cam_name)
                        rgb_cam_next = sim.renderer.render()
                        step_frames[cam_name] = rgb_cam_next.copy()
                        img_cam_next = Image.fromarray(rgb_cam_next)
                        img_cam_next_224 = img_cam_next.resize((224, 224))
                        buf_cam_next = io.BytesIO()
                        img_cam_next_224.save(buf_cam_next, format="JPEG", quality=75)
                        frames_all_views_next[cam_name] = (
                            "data:image/jpeg;base64,"
                            + base64.b64encode(buf_cam_next.getvalue()).decode("utf-8")
                        )
                    track_frames.append(step_frames)

                    index_pos = sim.data.xpos[index_id]
                    thumb_pos = sim.data.xpos[thumb_id]
                    cube_pos = sim.data.xpos[cube_id]
                    d_index = float(np.linalg.norm(index_pos - cube_pos))
                    d_thumb = float(np.linalg.norm(thumb_pos - cube_pos))
                    touch_index_next = (
                        max(0.0, 1.0 - (d_index / 0.04)) if d_index < 0.04 else 0.0
                    )
                    touch_thumb_next = (
                        max(0.0, 1.0 - (d_thumb / 0.04)) if d_thumb < 0.04 else 0.0
                    )
                    tactile_grid_next = [[0.0] * 4 for _ in range(4)]
                    tactile_grid_next[0][0] = touch_index_next
                    tactile_grid_next[1][1] = touch_thumb_next

                    track_next_obs = {
                        "frames": frames_all_views_next,
                        "history_frames": list(frame_history),
                        "proprioception": sim.get_state_32()[:24].tolist(),
                        "tactile": tactile_grid_next,
                        "text_prompt": text_prompt or "grasp cube",
                        "ui_annotations": {},
                        "is_easy_task": False,
                    }

                    if h == action_np.shape[1] - 1:
                        point_clouds_next = {}
                        for cam_name in sim.cam_names:
                            img_cam_next = Image.fromarray(step_frames[cam_name])
                            img_cam_next_224 = img_cam_next.resize((224, 224))
                            point_clouds_next[cam_name] = sim.get_point_cloud_numpy(
                                cam_name, np.array(img_cam_next_224)
                            )
                        track_next_obs["point_clouds"] = point_clouds_next

                    # Keep track 0's final step outcomes as the committed path for the environment
                    if track_idx == 0 and h == action_np.shape[1] - 1:
                        committed_qpos = sim.data.qpos.copy()
                        committed_qvel = sim.data.qvel.copy()
                        committed_ctrl = sim.data.ctrl.copy()
                        committed_next_obs = track_next_obs
                        committed_touch_index_next = touch_index_next
                        committed_touch_thumb_next = touch_thumb_next

                    # Append the full 16-step transition once lookahead completes
                    if h == action_np.shape[1] - 1:
                        grasp_success = (
                            touch_index_next > 0.5 and touch_thumb_next > 0.5
                        )

                        # Track 0-3 belong to payload 0 (clean), 4-7 to payload 1 (blur), etc.
                        visual_variant_idx = track_idx // 4

                        # Enforce adversarial data contract: visual variants must exist
                        assert (
                            perturbed_payloads
                        ), "🔥 FATAL: Colab endpoint failed to return perturbed_payloads!"
                        assigned_obs = copy.deepcopy(
                            perturbed_payloads[visual_variant_idx]
                        )
                        assigned_obs["point_clouds"] = current_obs["point_clouds"]

                        transitions.append(
                            {
                                "current_obs": assigned_obs,
                                "action_taken": track_actions_flat,
                                "next_obs": track_next_obs,
                                "energy": energy_ensemble[track_idx],
                                "tactile": float(grasp_success),
                            }
                        )

                # Save rollout video compilation for this track using cv2
                video_dir = (
                    f"logs/training/latent-flow/rollouts/ep_{ep_idx}_step_{env_step}"
                )
                os.makedirs(video_dir, exist_ok=True)
                video_path = os.path.join(video_dir, f"track_{track_idx:02d}.mp4")

                # Setup cv2.VideoWriter: 2x3 grid of 240x240 frames -> 720 width, 480 height
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                video_writer = cv2.VideoWriter(video_path, fourcc, 4.0, (720, 480))

                for frame_idx, frame_dict in enumerate(track_frames):
                    grid = np.zeros((480, 720, 3), dtype=np.uint8)

                    # Resize views to 240x240
                    c_center = cv2.resize(frame_dict["world_center"], (240, 240))
                    c_top = cv2.resize(frame_dict["world_top"], (240, 240))
                    c_left = cv2.resize(frame_dict["world_left"], (240, 240))
                    c_right = cv2.resize(frame_dict["world_right"], (240, 240))
                    c_wrist = cv2.resize(frame_dict["world_wrist"], (240, 240))

                    # Tile grid: center, top, left in row 1; right, wrist in row 2
                    grid[0:240, 0:240] = c_center
                    grid[0:240, 240:480] = c_top
                    grid[0:240, 480:720] = c_left
                    grid[240:480, 0:240] = c_right
                    grid[240:480, 240:480] = c_wrist

                    # Draw text in the remaining black telemetry quadrant (bottom right)
                    energy_val = energy_ensemble[track_idx]
                    cv2.putText(
                        grid,
                        f"Track {track_idx:02d}",
                        (495, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        grid,
                        f"Energy: {energy_val:.6f}",
                        (495, 340),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        grid,
                        f"Step: {frame_idx}",
                        (495, 380),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        1,
                    )

                    # Convert to BGR format for OpenCV
                    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                    video_writer.write(grid_bgr)

                video_writer.release()
                print(f"[Video Utility] Saved rollout video: {video_path}")

            # Rewind physics back to the committed path's final outcome to capture its state
            sim.data.qpos[:] = committed_qpos
            sim.data.qvel[:] = committed_qvel
            sim.data.ctrl[:] = committed_ctrl
            mujoco.mj_forward(sim.model, sim.data)

            # Map outer variables to the committed outcome
            next_obs = committed_next_obs
            touch_index_next = committed_touch_index_next
            touch_thumb_next = committed_touch_thumb_next

            physics_state = sim.get_physics_state()
            step_reward = -physics_state["target_dist"]
            episode_reward += step_reward

            # Report calibration
            calibrate_payload = {"transitions": transitions}

            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{colab_url}/stage3/calibrate",
                        json=calibrate_payload,
                        timeout=100.0,
                    )
            except Exception as e:
                print(
                    f"[Training Error] Episode {ep_idx + 1} Step {env_step} Colab calibrate failed: {e}"
                )

            # Reset physics back to the pre-training layout for the next step start
            sim.data.qpos[:] = initial_state["qpos"]
            sim.data.qvel[:] = initial_state["qvel"]
            sim.data.ctrl[:] = initial_state["ctrl"]
            mujoco.mj_forward(sim.model, sim.data)

        # Distillation
        touch_count = 0
        if touch_index_next > 0.5:
            touch_count += 2
        if touch_thumb_next > 0.5:
            touch_count += 2

        final_reward = float(touch_count >= 4)

        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{colab_url}/stage3/distill",
                    json={"reward": final_reward},
                    timeout=20.0,
                )
                if r.status_code == 200:
                    res = r.json()
                    print(
                        f"[Training] Distill completed for Episode {ep_idx + 1}."
                        f" Loss: {res.get('opsd_loss')}"
                    )
        except Exception as e:
            print(f"[Training Error] Episode {ep_idx + 1} distill failed: {e}")

    # Restore simulation physics cleanly back to the pre-training layout
    sim.data.qpos[:] = initial_state["qpos"]
    sim.data.qvel[:] = initial_state["qvel"]
    sim.data.ctrl[:] = initial_state["ctrl"]
    mujoco.mj_forward(sim.model, sim.data)

    await websocket.send_text(
        json.dumps(
            {
                "type": "training_progress",
                "status": "Training Completed Successfully!",
                "progress": 1.0,
                "episode": num_episodes,
                "total_episodes": num_episodes,
            }
        )
    )
    print("[Training] Stage 3 training sandbox finished.")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_camera, encoder_processing_enabled, attack_active, combostoc_noise, click_x, click_y, click_type, text_prompt, text_modifier
    global colab_is_processing, needs_colab_processing, last_colab_query_time
    global cached_dino_attn, cached_clip_sim, cached_sam_mask, cached_point_cloud, cached_vggt_tracks, cached_task_isolated_features
    global ui_annotations
    global is_training_active
    await websocket.accept()
    print("UI Connected via WebSocket")
    needs_colab_processing = (
        True  # Auto-trigger default Colab processing request to populate panels on load
    )

    index_id = sim.model.body("R_index_tip_link").id
    thumb_id = sim.model.body("R_thumb_tip_link").id
    cube_id = sim.model.body("cube").id

    step_count = 0
    is_moving = False
    moving_check_steps = 0
    cached_data_updated = True  # Send once on initial connection
    last_sent_payload = None

    try:
        while True:
            should_send = False
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                payload = json.loads(data)
                should_send = True

                if payload.get("type") == "select_camera":
                    active_camera = payload["camera"]
                    needs_colab_processing = True
                    print(f"Selected Camera Focus: {active_camera}")

                elif payload.get("type") == "ik_command":
                    phase = payload["phase"]
                    sim._handle_ik_pickup_logic(phase=phase)
                    needs_colab_processing = True

                elif payload.get("type") == "reset":
                    sim.reset_env(lock_posture=True)
                    needs_colab_processing = True
                    is_moving = False

                elif payload.get("type") == "wild_randomize":
                    sim.wild_reset()
                    needs_colab_processing = True
                    is_moving = False

                elif payload.get("type") == "combostoc_noise":
                    group = payload["group"]
                    val = payload["value"]
                    combostoc_noise[group] = val

                elif payload.get("type") == "toggle_encoders":
                    encoder_processing_enabled = bool(payload["enabled"])
                    print(
                        f"Set Encoder Processing Enabled: {encoder_processing_enabled}"
                    )

                elif payload.get("type") == "trigger_attack":
                    attack_active = payload["active"]

                elif payload.get("type") == "clear_selections":
                    click_x = None
                    click_y = None
                    click_type = None
                    needs_colab_processing = False
                    cached_dino_attn = None
                    cached_clip_sim = None
                    cached_sam_mask = None
                    cached_point_cloud = []
                    cached_vggt_tracks = []
                    cached_task_isolated_features = {}
                    print("Cleared active camera click selections.")

                elif payload.get("type") == "add_crop":
                    cam = payload.get("camera", "world_center")
                    if cam not in ui_annotations:
                        ui_annotations[cam] = {
                            "crops": [],
                            "vectors": [],
                            "segments": [],
                        }
                    ui_annotations[cam]["crops"].append(payload["coordinates"])
                    needs_colab_processing = True
                    print(f"Added crop annotation on {cam}: {payload['coordinates']}")

                elif payload.get("type") == "add_vector":
                    cam = payload.get("camera", "world_center")
                    if cam not in ui_annotations:
                        ui_annotations[cam] = {
                            "crops": [],
                            "vectors": [],
                            "segments": [],
                        }
                    ui_annotations[cam]["vectors"].append(payload["coordinates"])
                    needs_colab_processing = True
                    print(f"Added vector annotation on {cam}: {payload['coordinates']}")

                elif payload.get("type") == "clear_annotations":
                    ui_annotations = {}
                    needs_colab_processing = True
                    print("Cleared all UI annotations")

                elif payload.get("type") == "sync_annotations":
                    ui_annotations = payload["annotations"]
                    needs_colab_processing = True
                    print(f"Synchronized UI annotations: {ui_annotations}")

                elif payload.get("type") in [
                    "original_click",
                    "track_click",
                    "goal_click",
                ]:
                    click_x = int(payload["x"])
                    click_y = int(payload["y"])
                    click_type = payload.get("type")
                    needs_colab_processing = True
                    print(
                        f"Set click coordinates via {payload.get('type')} to: ({click_x}, {click_y})"
                    )

                elif payload.get("type") == "text_prompt":
                    text_prompt = payload["text"]
                    needs_colab_processing = True
                    print(f"Set text prompt to: {text_prompt}")

                elif payload.get("type") == "set_joint":
                    idx = int(payload["index"])
                    val = float(payload["value"])
                    act_norm = np.full(32, np.nan, dtype=np.float32)
                    act_norm[idx] = val
                    sim.process_target_32(act_norm)
                    is_moving = True
                    moving_check_steps = 0

                # Entrypoint --> called from the UI
                elif payload.get("type") == "start_training":
                    print("Starting Stage 3 training loop...")
                    is_training_active = True
                    try:
                        await run_stage3_training_loop(
                            websocket, sim, colab_url, text_prompt, ui_annotations
                        )
                    finally:
                        is_training_active = False

            except asyncio.TimeoutError:
                pass

            # Step physics 16 times per tick
            if not is_training_active:
                for _ in range(16):
                    sim.sync_ctrl_to_qpos(sim.last_target_q)
                    sim.data.qpos[sim.root_q_idx : sim.root_q_idx + 3] = [
                        0.0,
                        0.0,
                        0.95,
                    ]
                    sim.data.qpos[sim.root_q_idx + 3 : sim.root_q_idx + 7] = [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                    sim.data.qvel[:6] = 0.0
                    mujoco.mj_step(sim.model, sim.data)

                # Check if slider movement completed
                if is_moving:
                    moving_check_steps += 1
                    max_error = 0.0
                    for i in sim.active_joints_this_command:
                        j_id = sim.protocol_joint_ids[i]
                        if j_id != -1:
                            q_idx = sim.model.jnt_qposadr[j_id]
                            error = abs(sim.data.qpos[q_idx] - sim.last_target_q[q_idx])
                            if error > max_error:
                                max_error = error
                    if max_error < 0.02 or moving_check_steps > 30:
                        is_moving = False
                        needs_colab_processing = True

            if is_training_active and last_sent_payload is not None:
                ws_payload = copy.deepcopy(last_sent_payload)
            else:
                # Render and encode all 5 cameras for the UI grid
                frames = {}
                for name in sim.cam_names:
                    sim.renderer.update_scene(sim.data, camera=name)
                    rgb = sim.renderer.render()

                    # Apply visual noise if BadWorld Attack is active on active camera
                    if attack_active and name == active_camera:
                        rgb = rgb.copy()
                        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + 50, 0, 255)

                    img = Image.fromarray(rgb)
                    # Resizing to 480x480 for higher resolution UI display
                    img_resized = img.resize((480, 480))
                    buf = io.BytesIO()
                    img_resized.save(buf, format="JPEG", quality=75)
                    frames[name] = "data:image/jpeg;base64," + base64.b64encode(
                        buf.getvalue()
                    ).decode("utf-8")

                # Calculate actual physics telemetry
                physics = sim.get_physics_state()
                target_dist = physics["target_dist"]

                energy = min(target_dist * 2.0, 1.2)
                if attack_active:
                    energy = min(energy + 0.5, 1.2)

                index_pos = sim.data.xpos[index_id]
                thumb_pos = sim.data.xpos[thumb_id]
                cube_pos = sim.data.xpos[cube_id]

                d_index = float(np.linalg.norm(index_pos - cube_pos))
                d_thumb = float(np.linalg.norm(thumb_pos - cube_pos))

                touch_index = (
                    max(0.0, 1.0 - (d_index / 0.04)) if d_index < 0.04 else 0.0
                )
                touch_thumb = (
                    max(0.0, 1.0 - (d_thumb / 0.04)) if d_thumb < 0.04 else 0.0
                )

                tactile_grid = [[touch_index, 0.0], [0.0, touch_thumb]]

                joints_data = {
                    "positions": sim.get_state_32()[:4].tolist(),
                    "torques": [float(sim.data.ctrl[i]) for i in range(4)],
                }

                skills_data = [
                    {
                        "id": "1",
                        "name": "reach_cube",
                        "type": "internalized",
                        "x": 60,
                        "y": 75,
                        "active": target_dist > 0.15,
                    },
                    {
                        "id": "2",
                        "name": "pinch_cube",
                        "type": "internalized",
                        "x": 150,
                        "y": 45,
                        "active": (0.04 < target_dist <= 0.15),
                    },
                    {
                        "id": "3",
                        "name": "lift_cube",
                        "type": "externalized",
                        "x": 240,
                        "y": 75,
                        "active": (target_dist <= 0.04 and cube_pos[2] > 0.85),
                    },
                ]

                ws_payload = {
                    "frames": frames,
                    "energy": energy,
                    "tactile_grid": tactile_grid,
                    "joints": joints_data,
                    "skills": skills_data,
                }
                last_sent_payload = ws_payload

            current_time = asyncio.get_event_loop().time()
            # Query Colab server for advanced frame processing (DINO, CLIP, SAM, VGGT)
            if colab_url and needs_colab_processing and not colab_is_processing:
                # Retrieve the active camera's rendered image and resize to 224x224 for Colab processing
                sim.renderer.update_scene(sim.data, camera=active_camera)
                rgb_active = sim.renderer.render()
                if attack_active:
                    rgb_active = rgb_active.copy()
                    rgb_active[:, :, 0] = np.clip(rgb_active[:, :, 0] + 50, 0, 255)
                img_active = Image.fromarray(rgb_active)
                img_224 = img_active.resize((224, 224))
                buf_224 = io.BytesIO()
                img_224.save(buf_224, format="JPEG", quality=75)
                base64_frame_224 = "data:image/jpeg;base64," + base64.b64encode(
                    buf_224.getvalue()
                ).decode("utf-8")

                if base64_frame_224:
                    # Generate point cloud for active_camera natively in sim server
                    pc_active = sim.get_point_cloud_numpy(
                        active_camera, np.array(img_224)
                    )

                    frame_history.append(base64_frame_224)
                    colab_is_processing = True
                    needs_colab_processing = False
                    last_colab_query_time = current_time

                    async def run_colab_query(payload_data):
                        global colab_is_processing
                        global cached_dino_attn, cached_clip_sim, cached_sam_mask, cached_point_cloud, cached_vggt_tracks, cached_task_isolated_features
                        nonlocal cached_data_updated
                        try:
                            async with httpx.AsyncClient() as client:
                                r = await client.post(
                                    f"{colab_url}/process",
                                    json=payload_data,
                                    timeout=10.0,
                                )
                                if r.status_code == 200:
                                    res_data = r.json()
                                    cached_dino_attn = res_data.get("dino_attn")
                                    cached_clip_sim = res_data.get("clip_sim")
                                    cached_sam_mask = res_data.get("sam_mask")
                                    cached_point_cloud = res_data.get("point_cloud")
                                    cached_vggt_tracks = res_data.get("vggt_tracks")
                                    cached_task_isolated_features = res_data.get(
                                        "task_isolated_features"
                                    )
                                    if cached_task_isolated_features:
                                        print(
                                            f"[server.py] Received task_isolated_features from Colab: {list(cached_task_isolated_features.keys())}"
                                        )
                                    else:
                                        print(
                                            "[server.py] No task_isolated_features in Colab response"
                                        )
                                    cached_data_updated = True
                        except Exception as e:
                            print("Colab communication error:")
                            traceback.print_exc()
                        finally:
                            colab_is_processing = False

                    post_payload = {
                        "frame": base64_frame_224,
                        "click_x": click_x,
                        "click_y": click_y,
                        "click_type": click_type,
                        "text_prompt": text_prompt,
                        "text_modifier": text_modifier,
                        "ui_annotations": ui_annotations,
                        "history_frames": list(frame_history),
                        "point_clouds": {active_camera: pc_active},
                    }
                    asyncio.create_task(run_colab_query(post_payload))

            if is_moving or cached_data_updated or (step_count % 20 == 0):
                should_send = True

            if should_send:
                if cached_data_updated:
                    ws_payload["dino_attn"] = cached_dino_attn
                    ws_payload["clip_sim"] = cached_clip_sim
                    ws_payload["sam_mask"] = cached_sam_mask
                    ws_payload["point_cloud"] = cached_point_cloud
                    ws_payload["vggt_tracks"] = cached_vggt_tracks
                    ws_payload["task_isolated_features"] = cached_task_isolated_features
                    cached_data_updated = False

                await websocket.send_text(json.dumps(ws_payload))
            step_count += 1
            await asyncio.sleep(0.05)  # ~20fps

    except WebSocketDisconnect:
        print("UI Disconnected")
