import asyncio
import base64
import io
import json
import os
import sys
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
import numpy as np
import httpx

# Add parent directory (latent-flow root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr1_config import COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler
from simulation_base import GR1MuJoCoBase


class GR1SimulationServer(GR1MuJoCoBase):
    """
    Stand-alone MuJoCo Simulation Server.
    Completely isolated from the le-probe project.
    """

    def __init__(self):
        super().__init__(restrict_ik=True)

    def _handle_ik_pickup_logic(self, phase=0, offset_cm=5):
        """Standard multi-phase IK solver for the red cube."""
        self.current_phase = phase + 1
        print(
            f"🎯 Executing IK Pickup Phase {phase} (Global ID: {self.current_phase})..."
        )

        import mujoco

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
ui_annotations = {"crops": [], "vectors": [], "segments": []}

colab_is_processing = False
needs_colab_processing = False
last_colab_query_time = 0.0

cached_dino_attn = None
cached_clip_sim = None
cached_sam_mask = None
cached_point_cloud = []
cached_vggt_tracks = []
cached_task_isolated_features = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_camera, encoder_processing_enabled, attack_active, combostoc_noise, click_x, click_y, click_type, text_prompt, text_modifier
    global colab_is_processing, needs_colab_processing, last_colab_query_time
    global cached_dino_attn, cached_clip_sim, cached_sam_mask, cached_point_cloud, cached_vggt_tracks, cached_task_isolated_features
    global ui_annotations
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

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                payload = json.loads(data)

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
                    ui_annotations["crops"].append(payload["coordinates"])
                    needs_colab_processing = True
                    print(f"Added crop annotation: {payload['coordinates']}")

                elif payload.get("type") == "add_vector":
                    ui_annotations["vectors"].append(payload["coordinates"])
                    needs_colab_processing = True
                    print(f"Added vector annotation: {payload['coordinates']}")

                elif payload.get("type") == "clear_annotations":
                    ui_annotations = {"crops": [], "vectors": [], "segments": []}
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

            except asyncio.TimeoutError:
                pass

            # Step physics 16 times per tick
            import mujoco

            for _ in range(16):
                sim.sync_ctrl_to_qpos(sim.last_target_q)
                sim.data.qpos[sim.root_q_idx : sim.root_q_idx + 3] = [0.0, 0.0, 0.95]
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

            touch_index = max(0.0, 1.0 - (d_index / 0.04)) if d_index < 0.04 else 0.0
            touch_thumb = max(0.0, 1.0 - (d_thumb / 0.04)) if d_thumb < 0.04 else 0.0

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
                "dino_attn": cached_dino_attn,
                "clip_sim": cached_clip_sim,
                "sam_mask": cached_sam_mask,
                "point_cloud": cached_point_cloud,
                "vggt_tracks": cached_vggt_tracks,
                "task_isolated_features": cached_task_isolated_features,
            }

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
                    frame_history.append(base64_frame_224)
                    colab_is_processing = True
                    needs_colab_processing = False
                    last_colab_query_time = current_time

                    async def run_colab_query(payload_data):
                        global colab_is_processing
                        global cached_dino_attn, cached_clip_sim, cached_sam_mask, cached_point_cloud, cached_vggt_tracks, cached_task_isolated_features
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
                        except Exception as e:
                            import traceback

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
                    }
                    asyncio.create_task(run_colab_query(post_payload))

            await websocket.send_text(json.dumps(ws_payload))
            step_count += 1
            await asyncio.sleep(0.05)  # ~20fps

    except WebSocketDisconnect:
        print("UI Disconnected")
