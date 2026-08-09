from tqdm import tqdm
import asyncio
import base64
import copy
import io
import json
import os
import pickle
import numpy as np
import httpx
import mujoco
from PIL import Image
from fastapi import WebSocketDisconnect

from . import config
from .helpers import capture_sim_frames, render_camera_views
from .telemetry import get_body_ids


def build_stage3_obs_payload(sim, text_prompt, ui_annotations, ep_idx, env_step):
    current_frames = capture_sim_frames(sim)
    frame_history = [current_frames, current_frames, current_frames, current_frames]
    tactile_grid = [[0.0] * 4 for _ in range(4)]
    return {
        "frames": current_frames,
        "history_frames": frame_history,
        "proprioception": sim.unscaler.scale_state(sim.get_state_32()).tolist(),
        "tactile": tactile_grid,
        "text_prompt": text_prompt or "grasp cube",
        "ui_annotations": (
            ui_annotations or {"crops": [], "vectors": [], "segments": []}
        ),
        "pos_trajectories": [],
        "is_easy_task": False,
        "episode_idx": ep_idx,
        "step_idx": env_step,
    }


async def websocket_endpoint_handler(websocket, sim, eval_sim):
    await websocket.accept()
    print("UI Connected via WebSocket")

    body_ids = get_body_ids(sim)
    r_index_id = body_ids["r_index_id"]
    r_thumb_id = body_ids["r_thumb_id"]
    l_index_id = body_ids["l_index_id"]
    l_thumb_id = body_ids["l_thumb_id"]
    cube_id = body_ids["cube_id"]

    step_count = 0
    is_moving = False
    moving_check_steps = 0
    cached_data_updated = True
    last_sent_payload = None
    baseline_exemplar_frames = None

    try:
        while True:
            should_send = False
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                payload = json.loads(data)
                should_send = True

                if payload.get("type") == "select_camera":
                    config.active_camera = payload["camera"]
                    config.needs_colab_processing = True
                    print(f"Selected Camera Focus: {config.active_camera}")

                elif payload.get("type") == "ik_pickup":
                    phase = payload["phase"]
                    sim._handle_ik_pickup_logic(phase=phase)
                    config.needs_colab_processing = True

                elif payload.get("type") == "reset":
                    print("🔄 [Server Received] Reset / Home All request")
                    sim.reset_env(lock_posture=True)
                    sim.sync_ctrl_to_qpos(sim.data.qpos)
                    baseline_exemplar_frames = None
                    config.needs_colab_processing = True
                    is_moving = False

                elif payload.get("type") == "wild_randomize":
                    print("🎲 [Server Received] Wild Randomize request")
                    sim.wild_reset()
                    baseline_exemplar_frames = None
                    config.needs_colab_processing = True
                    is_moving = False

                elif payload.get("type") == "combostoc_noise":
                    group = payload["group"]
                    val = payload["value"]
                    config.combostoc_noise[group] = val

                elif payload.get("type") == "toggle_encoders":
                    config.encoder_processing_enabled = bool(payload["enabled"])

                elif payload.get("type") == "trigger_attack":
                    config.attack_active = payload["active"]

                elif payload.get("type") == "sync_annotations":
                    config.ui_annotations = payload.get("annotations", {})
                    config.needs_colab_processing = True
                    print(
                        f"✏️ [Server Received] Annotations updated: {list(config.ui_annotations.keys())}"
                    )

                elif payload.get("type") == "clear_annotations":
                    config.ui_annotations = {}
                    config.click_x = None
                    config.click_y = None
                    config.click_type = None
                    config.needs_colab_processing = True
                    config.cached_dino_attn = None
                    config.cached_sobel_edge = None
                    config.cached_sam_mask = None
                    config.cached_motion_field = None
                    config.cached_task_isolated_features = {}

                elif payload.get("type") == "clear_selections":
                    config.click_x = None
                    config.click_y = None
                    config.click_type = None
                    config.needs_colab_processing = True
                    config.cached_dino_attn = None
                    config.cached_sobel_edge = None
                    config.cached_sam_mask = None
                    config.cached_motion_field = None
                    config.cached_task_isolated_features = {}

                elif payload.get("type") == "set_joint":
                    idx = int(payload["index"])
                    val = float(payload["value"])
                    act_norm = np.full(32, np.nan, dtype=np.float32)
                    act_norm[idx] = val
                    sim.process_target_32(act_norm)
                    is_moving = True
                    moving_check_steps = 0

                elif payload.get("type") == "record_exemplar":
                    ex_name = payload.get("name", "phase_0")
                    print(f"📸 [Server Received] Record Exemplar request: {ex_name}")
                    if config.colab_url:

                        async def send_exemplar():
                            try:
                                obs_payload = build_stage3_obs_payload(
                                    sim, config.text_prompt, config.ui_annotations, 0, 0
                                )
                                async with httpx.AsyncClient() as client:
                                    r = await client.post(
                                        f"{config.colab_url}/record_exemplar?name={ex_name}",
                                        json=obs_payload,
                                        timeout=10.0,
                                    )
                                    if r.status_code == 200:
                                        print(
                                            f"✅ Exemplar '{ex_name}' recorded on Colab server."
                                        )
                            except Exception as e:
                                print(f"⚠️ Exemplar recording error: {e}")

                        asyncio.create_task(send_exemplar())

                elif payload.get("type") == "start_training":
                    print(
                        "🚀 [Server Received] Starting Stage 3 training sandbox loop..."
                    )
                    if config.colab_url:

                        async def run_training_wrapper():
                            try:
                                config.is_training_active = True
                                from .training_loop import run_stage3_training_loop

                                await run_stage3_training_loop(
                                    websocket,
                                    sim,
                                    eval_sim,
                                    config.colab_url,
                                    config.text_prompt,
                                    config.ui_annotations,
                                    config.cached_task_isolated_features,
                                )
                            except Exception as e:
                                print(f"⚠️ Stage 3 training loop error: {e}")
                            finally:
                                config.is_training_active = False

                        asyncio.create_task(run_training_wrapper())
                    else:
                        print("⚠️ Cannot trigger training: colab_url is not set.")

                elif payload.get("type") == "execute_checkpoint":
                    ckpt_name = payload.get("checkpoint_name", "stage3_rl_final.pt")
                    noise_scale = float(
                        payload.get(
                            "stochastic_steer_scale",
                            payload.get("step_nft_scale", 0.08),
                        )
                    )
                    print(
                        f"⚡ [Execute Checkpoint] Executing '{ckpt_name}' with noise scale {noise_scale}..."
                    )

                    obs_payload = build_stage3_obs_payload(
                        sim, config.text_prompt, config.ui_annotations, 0, 0
                    )

                    execute_data = {
                        "obs": obs_payload,
                        "checkpoint_name": ckpt_name,
                        "stochastic_steer_scale": noise_scale,
                        "seed": 42,
                    }

                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{config.colab_url}/stage3/execute",
                                json=execute_data,
                                timeout=120.0,
                            )
                            if r.status_code == 200:
                                res = r.json()
                                action_candidates = res.get("action_candidates", [])
                                print(
                                    f"✅ [Execute Checkpoint] Received {len(action_candidates)} candidate trajectories from Colab."
                                )
                                action_np = np.array(
                                    action_candidates, dtype=np.float32
                                )

                                initial_qpos = sim.data.qpos.copy()
                                initial_qvel = sim.data.qvel.copy()
                                initial_ctrl = sim.data.ctrl.copy()

                                # --- PASS 1: Pure Physics Rollout & Distance Scoring (Zero Camera Rendering) ---
                                screened_candidates = []
                                for candidate_idx in tqdm(
                                    range(16), desc="Screening Candidates"
                                ):
                                    eval_sim.data.qpos[:] = initial_qpos
                                    eval_sim.data.qvel[:] = initial_qvel
                                    eval_sim.data.ctrl[:] = initial_ctrl
                                    mujoco.mj_forward(eval_sim.model, eval_sim.data)

                                    track_frames_per_cam = {
                                        cam: []
                                        for cam in [
                                            "world_center",
                                            "world_top",
                                            "world_left",
                                            "world_right",
                                            "world_wrist",
                                        ]
                                    }

                                    step_phys_distances = []
                                    for h in range(7):
                                        act_k = action_np[candidate_idx, h, :]
                                        act_32_clamped = np.clip(act_k[:32], -1.0, 1.0)
                                        eval_sim.process_target_32(act_32_clamped)
                                        eval_sim.dispatch_action(
                                            action_32_norm=act_32_clamped,
                                            target_q=eval_sim.last_target_q,
                                            n_steps=2,
                                            render_freq=0,
                                            reset_start=False,
                                        )

                                        if h in [0, 3, 6]:
                                            cur_frames, _ = render_camera_views(
                                                eval_sim
                                            )
                                            for cam_key in track_frames_per_cam:
                                                track_frames_per_cam[cam_key].append(
                                                    cur_frames[cam_key]
                                                )

                                        r_index_pos = eval_sim.data.xpos[r_index_id]
                                        r_thumb_pos = eval_sim.data.xpos[r_thumb_id]
                                        l_index_pos = eval_sim.data.xpos[l_index_id]
                                        l_thumb_pos = eval_sim.data.xpos[l_thumb_id]
                                        cube_pos = eval_sim.data.xpos[cube_id]

                                        r_hand_center = (
                                            r_index_pos + r_thumb_pos
                                        ) / 2.0
                                        l_hand_center = (
                                            l_index_pos + l_thumb_pos
                                        ) / 2.0

                                        r_dist = float(
                                            np.linalg.norm(r_hand_center - cube_pos)
                                        )
                                        l_dist = float(
                                            np.linalg.norm(l_hand_center - cube_pos)
                                        )

                                        h_phys_dist = min(r_dist, l_dist)
                                        step_phys_distances.append(h_phys_dist)

                                    min_phys_dist = float(np.min(step_phys_distances))
                                    final_phys_dist = float(step_phys_distances[-1])
                                    mean_phys_dist = float(np.mean(step_phys_distances))

                                    screened_candidates.append(
                                        {
                                            "candidate_idx": candidate_idx,
                                            "mean_phys_dist": mean_phys_dist,
                                            "min_phys_dist": min_phys_dist,
                                            "final_phys_dist": final_phys_dist,
                                            "final_frames": {
                                                cam: track_frames_per_cam[cam][-1]
                                                for cam in track_frames_per_cam
                                                if track_frames_per_cam[cam]
                                            },
                                            "frame_sequences": {
                                                cam: track_frames_per_cam[cam]
                                                for cam in track_frames_per_cam
                                                if track_frames_per_cam[cam]
                                            },
                                            "actions": action_np[
                                                candidate_idx
                                            ].tolist(),
                                        }
                                    )

                                # Rank all 16 candidates by ascending physical distance to goal
                                screened_candidates.sort(
                                    key=lambda c: (
                                        c["min_phys_dist"],
                                        c["final_phys_dist"],
                                    )
                                )
                                for i, cand in enumerate(screened_candidates):
                                    cand["rank"] = i
                                top_8_screened = screened_candidates[:8]
                                top_candidate_dist = top_8_screened[0]["min_phys_dist"]
                                print(
                                    "📊 [Execute Checkpoint] Top Candidate #1 Min "
                                    f"Physical Distance: {top_candidate_dist:.4f}m"
                                )

                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "checkpoint_execution_results",
                                            "checkpoint": ckpt_name,
                                            "top_candidates": [
                                                {
                                                    "rank": c["rank"],
                                                    "candidate_idx": c["candidate_idx"],
                                                    "mean_phys_dist": round(
                                                        c["min_phys_dist"], 3
                                                    ),
                                                    "frames": c["final_frames"],
                                                    "frame_sequences": c[
                                                        "frame_sequences"
                                                    ],
                                                }
                                                for c in top_8_screened
                                            ],
                                        }
                                    )
                                )

                    except Exception as e:
                        print(f"❌ [Execute Checkpoint Error] {e}")
                        import traceback

                        traceback.print_exc()

            except asyncio.TimeoutError:
                pass

            if is_moving and not config.is_training_active:
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
                    config.needs_colab_processing = True

            if config.is_training_active and last_sent_payload is not None:
                ws_payload = copy.deepcopy(last_sent_payload)
            else:
                frames = {}
                for name in sim.cam_names:
                    sim.renderer.update_scene(sim.data, camera=name)
                    rgb = sim.renderer.render()

                    if config.attack_active and name == config.active_camera:
                        rgb = rgb.copy()
                        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + 50, 0, 255)

                    img = Image.fromarray(rgb)
                    img_resized = img.resize((480, 480))
                    buf = io.BytesIO()
                    img_resized.save(buf, format="JPEG", quality=75)
                    frames[name] = "data:image/jpeg;base64," + base64.b64encode(
                        buf.getvalue()
                    ).decode("utf-8")

                physics = sim.get_physics_state()
                target_dist = physics["target_dist"]

                energy = min(target_dist * 2.0, 1.2)
                if config.attack_active:
                    energy = min(energy + 0.5, 1.2)

                index_pos = sim.data.xpos[r_index_id]
                thumb_pos = sim.data.xpos[r_thumb_id]
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
            if (
                config.colab_url
                and config.needs_colab_processing
                and not config.colab_is_processing
            ):
                # Retrieve the active camera's rendered image and resize to 224x224 for Colab processing
                config.needs_colab_processing = False
                sim.renderer.update_scene(sim.data, camera=config.active_camera)

                # Render frames from all cameras
                frame_all_views = {}
                for cam_name in sim.cam_names:
                    sim.renderer.update_scene(sim.data, camera=cam_name)
                    rgb_cam = sim.renderer.render()
                    img_cam = Image.fromarray(rgb_cam)
                    img_cam_224 = img_cam.resize((224, 224))
                    buf_cam = io.BytesIO()
                    img_cam_224.save(buf_cam, format="JPEG", quality=75)
                    frame_all_views[cam_name] = (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(buf_cam.getvalue()).decode("utf-8")
                    )

                config.colab_is_processing = True
                config.last_colab_query_time = current_time

                # Append the first frame or any frame with actual movement
                if is_moving or step_count == 0:
                    print("Appending Frame")
                    config.frame_history.append(frame_all_views.copy())
                    is_moving = False

                async def run_colab_query(payload_data):
                    nonlocal cached_data_updated
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{config.colab_url}/process",
                                json=payload_data,
                                timeout=100.0,
                            )
                            if r.status_code == 200:
                                res_data = r.json()
                                config.cached_dino_attn = res_data.get("dino_attn")
                                config.cached_sobel_edge = res_data.get("sobel_edge")
                                config.cached_sam_mask = res_data.get("sam_mask")
                                config.cached_motion_field = res_data.get(
                                    "motion_field"
                                )
                                print(
                                    f"Motion Field: {np.array(config.cached_motion_field).shape if config.cached_motion_field else None}"
                                )
                                config.cached_task_isolated_features = res_data.get(
                                    "task_isolated_features"
                                )
                                if config.cached_task_isolated_features:
                                    print(
                                        f"[server.py] Received task_isolated_features from Colab: {list(config.cached_task_isolated_features.keys())}"
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
                        config.colab_is_processing = False

                post_payload = {
                    "frame": frame_all_views[config.active_camera],
                    "click_x": config.click_x,
                    "click_y": config.click_y,
                    "click_type": config.click_type,
                    "text_prompt": config.text_prompt,
                    "text_modifier": config.text_modifier,
                    "ui_annotations": config.ui_annotations,
                    "history_frames": [
                        frames[config.active_camera] for frames in config.frame_history
                    ],
                    "view_name": config.active_camera,
                }
                asyncio.create_task(run_colab_query(post_payload))

            if is_moving or cached_data_updated or (step_count % 20 == 0):
                should_send = True

            if should_send:
                if cached_data_updated:
                    ws_payload["dino_attn"] = config.cached_dino_attn
                    ws_payload["sobel_edge"] = config.cached_sobel_edge
                    ws_payload["sam_mask"] = config.cached_sam_mask
                    ws_payload["motion_field"] = config.cached_motion_field
                    ws_payload["task_isolated_features"] = (
                        config.cached_task_isolated_features
                    )
                    cached_data_updated = False

                await websocket.send_text(json.dumps(ws_payload))
            step_count += 1
            await asyncio.sleep(0.05)  # ~20fps

            if step_count == 1:
                # Process the secondary initialization frame manually for the startup frame history
                act_init = np.full(32, np.nan, dtype=np.float32)
                act_init[17] = sim.last_target_q[17] + 0.3
                sim.process_target_32(act_init)
                is_moving = True
                moving_check_steps = 0

    except WebSocketDisconnect:
        print("UI Disconnected")
