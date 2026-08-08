import asyncio
import json
import os
import numpy as np
import httpx
import mujoco
from PIL import Image
from fastapi import WebSocket

from .helpers import capture_sim_frames, render_camera_views, write_tiled_mp4
from .visualize import plot_energy_landscape
from .depth_unprojector import unproject_ui_annotations_to_3d
from .ik_trajectory_sampler import (
    generate_ik_trajectories,
    save_ik_trajectory_diagnostic_plots,
    save_ik_trajectory_video,
)
from .telemetry import get_body_ids


async def async_track_unroll_worker(
    all_track_frames,
    energy_ensemble,
    ep_idx,
    env_step,
):
    """
    Asynchronously handles writing telemetry videos to disk from pre-rendered frame buffers
    without blocking main environment training loop execution.
    """
    try:
        video_dir = f"logs/training/latent-flow/rollouts/ep_{ep_idx}_step_{env_step}"
        os.makedirs(video_dir, exist_ok=True)

        for track_idx, track_frames in enumerate(all_track_frames):
            video_path = os.path.join(video_dir, f"track_{track_idx:02d}.mp4")
            write_tiled_mp4(
                video_path, track_frames, track_idx, float(energy_ensemble[track_idx])
            )
    except Exception as background_err:
        print(
            f"⚠️ [Background Worker Warning] Telemetry video unroll skipped: {background_err}"
        )


async def run_stage3_training_loop(
    websocket: WebSocket,
    sim,
    eval_sim,
    colab_url: str,
    text_prompt: str,
    ui_annotations: dict,
    task_isolated_features: dict,
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

    body_ids = get_body_ids(sim)
    r_index_id = body_ids["r_index_id"]
    r_thumb_id = body_ids["r_thumb_id"]
    l_index_id = body_ids["l_index_id"]
    l_thumb_id = body_ids["l_thumb_id"]
    cube_id = body_ids["cube_id"]

    # Use eval_sim for offline IK trajectory sampling so main UI sim state remains completely unaffected
    eval_sim.data.qpos[:] = sim.data.qpos.copy()
    eval_sim.data.qvel[:] = sim.data.qvel.copy()
    eval_sim.data.ctrl[:] = sim.data.ctrl.copy()
    mujoco.mj_forward(eval_sim.model, eval_sim.data)

    initial_state = {
        "qpos": eval_sim.data.qpos.copy(),
        "qvel": eval_sim.data.qvel.copy(),
        "ctrl": eval_sim.data.ctrl.copy(),
    }

    num_epochs = 10
    max_steps = 16
    calibrate_steps = 4

    # --- STAGE 3 IK TRAJECTORY GENERATION EVALUATION HOOK (INSERTED ABOVE LINE 414) ---
    print(f"UI ANNOTATIONS: {ui_annotations}")
    for view_name in ui_annotations:
        print(
            "\n🚀 [IK Evaluation Mode] Triggering 2D-to-3D Unprojection & Trajectory Generation (via eval_sim)..."
        )
        try:
            # 1. 2D-to-3D Depth Unprojection & 3D Coordinates Printing (using isolated eval_sim)
            (
                start_3d,
                target_3d,
                target_3d_bounds,
                selected_body_name,
            ) = unproject_ui_annotations_to_3d(
                eval_sim,
                ui_annotations[view_name],
                camera_name=view_name,
                task_isolated_features=task_isolated_features,
            )
            print(
                f"📍 [3D Unprojection Result] Selected Effector Body/Link: '{selected_body_name}'"
            )
            print(f"📍 [3D Unprojection Result] Effector Link 3D Start: {start_3d}")
            print(f"📍 [3D Unprojection Result] Target Object 3D Goal: {target_3d}")
            print(
                f"📍 [3D Unprojection Result] Target Object 3D Bounds: {target_3d_bounds}"
            )

            if selected_body_name is None:
                raise ValueError("Could not find end effector")

            # 2. IK Trajectory Generation (Unrolling on isolated eval_sim)
            pos_trajectories = generate_ik_trajectories(
                eval_sim,
                initial_state,
                target_3d,
                target_3d_bounds=target_3d_bounds,
                site_name=selected_body_name,
                n=5,
                scale_multiplier=0.5,
                is_positive=True,
            )
            neg_trajectories = generate_ik_trajectories(
                eval_sim,
                initial_state,
                target_3d,
                target_3d_bounds=target_3d_bounds,
                site_name=selected_body_name,
                n=8,
                scale_multiplier=1,
                is_positive=False,
            )
            print(
                f"✅ [IK Sampler] Generated {len(pos_trajectories)} Positive (D+) & "
                f"{len(neg_trajectories)} Negative (D-) IK Trajectories!"
            )

            # 3. Save Trajectory Diagnostic Plot & Video Files
            save_ik_trajectory_diagnostic_plots(
                eval_sim, pos_trajectories, neg_trajectories
            )
            save_ik_trajectory_video(
                pos_trajectories,
                neg_trajectories,
                output_dir="logs/training/latent-flow/goals",
                fps=4,
            )

            # 4. Trajectory Generation Complete - Proceed to Epoch Loop
            print(
                f"🚀 [IK Trajectory Sampler] Positives ({len(pos_trajectories)}) "
                f"& Negatives ({len(neg_trajectories)}) ready. "
                "Starting training epochs...\n"
            )
        except Exception as e:
            print(f"❌ [IK Evaluation Mode Error] Failed to generate trajectories: {e}")
            import traceback

            traceback.print_exc()

    for ep_idx in range(num_epochs):
        if ep_idx > 0:
            sim.reset_env(lock_posture=True)

        # Reflected in the progress bar on the UI
        await websocket.send_text(
            json.dumps(
                {
                    "type": "training_progress",
                    "status": f"Epoch {ep_idx + 1}/{num_epochs} in progress...",
                    "progress": float(ep_idx) / num_epochs,
                    "epoch": ep_idx + 1,
                    "total_epochs": num_epochs,
                }
            )
        )

        buffered_transitions = []

        for env_step in range(max_steps):
            # 1. Randomize robot posture and cube position on all steps EXCEPT Episode 0 Step 0
            # to preserve exact posture & cube position where user drew UI annotations.
            if not (ep_idx == 0 and env_step == 0):
                sim.reset_env(lock_posture=True, randomize_cube=True)

            # 2. Seed a fresh, clean zero-velocity frame history for this step
            # Prevents VGGT motion extractor from comparing across teleported worlds
            step_init_frames = capture_sim_frames(sim)
            frame_history = [
                step_init_frames,
                step_init_frames,
                step_init_frames,
                step_init_frames,
            ]
            frame_all_views = step_init_frames.copy()

            # 3. Capture observation natively
            sim.renderer.update_scene(sim.data, camera="world_center")

            # Computes distance between fingers and cube
            # ToDo: Needs to be generalized for other tasks, where the user configures
            # which fingers matter, relative positions, etc. in the UI.
            index_pos = sim.data.xpos[r_index_id]
            thumb_pos = sim.data.xpos[r_thumb_id]
            cube_pos = sim.data.xpos[cube_id]

            d_index = float(np.linalg.norm(index_pos - cube_pos))
            d_thumb = float(np.linalg.norm(thumb_pos - cube_pos))

            touch_index = max(0.0, 1.0 - (d_index / 0.04)) if d_index < 0.04 else 0.0
            touch_thumb = max(0.0, 1.0 - (d_thumb / 0.04)) if d_thumb < 0.04 else 0.0

            tactile_grid = [[0.0] * 4 for _ in range(4)]
            tactile_grid[0][0] = touch_index
            tactile_grid[1][1] = touch_thumb

            current_obs = {
                "frames": frame_all_views,
                "history_frames": list(frame_history),  # length 4
                "proprioception": sim.unscaler.scale_state(sim.get_state_32()).tolist(),
                "tactile": tactile_grid,
                "text_prompt": text_prompt or "grasp cube",
                "ui_annotations": ui_annotations
                or {"crops": [], "vectors": [], "segments": []},
            }
            print(
                f"Frame History: {len(frame_history)}, "
                f"Views: {list(frame_all_views.keys())}"
            )

            step_payload = {
                "obs": current_obs,
                "pos_trajectories": pos_trajectories,
                "is_easy_task": False,
                "episode_idx": ep_idx,
                "step_idx": env_step,
                "eval_mean_physical_distance": getattr(
                    sim, "_last_step_mean_phys_dist", 0.0
                ),
                "eval_median_physical_distance": getattr(
                    sim, "_last_step_median_phys_dist", 0.0
                ),
                "eval_min_physical_distance": getattr(
                    sim, "_last_step_min_phys_dist", 0.0
                ),
                "eval_energy_distance_correlation": getattr(
                    sim, "_last_step_energy_dist_corr", 0.0
                ),
            }

            # 4. Step Colab Stage3 API
            action_taken_ensemble = None
            energy_ensemble = None
            s_target = None
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        f"{colab_url}/stage3/step",
                        json=step_payload,
                        timeout=2000.0,
                    )
                    if r.status_code == 200:
                        res = r.json()
                        action_taken_ensemble = res.get("action")  # [16, 7, 58]
                        energy_ensemble = res.get("energy")  # [16]
                        s_target = res.get("s_target")  # [1, 512]
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

            # UNIFIED 16-TRACK OBSERVATION & TELEMETRY EVALUATION LOOP (ALL CANDIDATES 0..15)
            all_track_frames = []
            for track_k in range(action_np.shape[0]):
                eval_sim.data.qpos[:] = initial_qpos
                eval_sim.data.qvel[:] = initial_qvel
                eval_sim.data.ctrl[:] = initial_ctrl
                mujoco.mj_forward(eval_sim.model, eval_sim.data)

                track_frames_k = []
                # Initial posture camera rendering (h = 0 start keyframe 1)
                start_frames_k, frames_all_views_start_k = render_camera_views(eval_sim)
                track_frames_k.append(start_frames_k)
                recording_history_frames_k = [frames_all_views_start_k]

                track_actions_flat_k = []
                frames_all_views_next_k = {}

                for h in range(action_np.shape[1]):
                    action_k = action_np[track_k, h, :]
                    track_actions_flat_k.extend(action_k.tolist())
                    action_32_clamped = np.clip(action_k[:32], -1.0, 1.0)

                    eval_sim.process_target_32(action_32_clamped)
                    eval_sim.dispatch_action(
                        action_32_norm=action_32_clamped,
                        target_q=eval_sim.last_target_q,
                        n_steps=2,
                        render_freq=0,
                        reset_start=False,
                    )

                    # Action step camera rendering (t = 1..7)
                    step_frames_k, frames_all_views_next_k = render_camera_views(
                        eval_sim
                    )
                    track_frames_k.append(step_frames_k)

                    # Sample 4 keyframes across candidate action execution: h=1,3,6
                    if h in [1, 3, 6]:
                        recording_history_frames_k.append(frames_all_views_next_k)

                all_track_frames.append(track_frames_k)

                r_index_pos = eval_sim.data.xpos[r_index_id]
                r_thumb_pos = eval_sim.data.xpos[r_thumb_id]
                l_index_pos = eval_sim.data.xpos[l_index_id]
                l_thumb_pos = eval_sim.data.xpos[l_thumb_id]
                cube_pos = eval_sim.data.xpos[cube_id]

                d_index = float(np.linalg.norm(r_index_pos - cube_pos))
                d_thumb = float(np.linalg.norm(r_thumb_pos - cube_pos))
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
                    "frames": frames_all_views_next_k,
                    "history_frames": recording_history_frames_k,
                    "proprioception": eval_sim.unscaler.scale_state(
                        eval_sim.get_state_32()
                    ).tolist(),
                    "tactile": tactile_grid_next,
                    "text_prompt": text_prompt or "grasp cube",
                    "ui_annotations": {},
                }

                grasp_success = touch_index_next > 0.5 and touch_thumb_next > 0.5
                r_hand_center = (r_index_pos + r_thumb_pos) / 2.0
                l_hand_center = (l_index_pos + l_thumb_pos) / 2.0
                r_phys_dist = float(np.linalg.norm(r_hand_center - cube_pos))
                l_phys_dist = float(np.linalg.norm(l_hand_center - cube_pos))
                phys_dist = min(r_phys_dist, l_phys_dist)

                if not hasattr(sim, "_step_physical_distances"):
                    sim._step_physical_distances = []
                sim._step_physical_distances.append(phys_dist)

                transitions.append(
                    {
                        "current_obs": current_obs,
                        "action_taken": track_actions_flat_k,
                        "next_obs": track_next_obs,
                        "energy": energy_ensemble[track_k],
                        "tactile": float(grasp_success),
                        "s_target": s_target,
                        "candidate_idx": track_k,
                    }
                )

                if track_k == 0:
                    committed_qpos = eval_sim.data.qpos.copy()
                    committed_qvel = eval_sim.data.qvel.copy()
                    committed_ctrl = eval_sim.data.ctrl.copy()
                    committed_next_obs = track_next_obs
                    committed_touch_index_next = touch_index_next
                    committed_touch_thumb_next = touch_thumb_next

            # 2. DISPATCH BACKGROUND UNROLL WORKER FOR VIDEO DISK ENCODING (NON-BLOCKING)
            asyncio.create_task(
                async_track_unroll_worker(
                    all_track_frames, energy_ensemble, ep_idx, env_step
                )
            )

            # Compute correlation between latent energies and physical distances across ALL 16 candidate tracks
            phys_dists_np = np.array(sim._step_physical_distances, dtype=np.float32)
            energies_np = np.array(energy_ensemble, dtype=np.float32)
            step_mean_phys_dist = float(np.mean(phys_dists_np))
            step_median_phys_dist = float(np.median(phys_dists_np))
            step_min_phys_dist = float(np.min(phys_dists_np))
            if len(phys_dists_np) <= 1 or len(phys_dists_np) != len(energies_np):
                raise RuntimeError(
                    f"❌ [Telemetry Error] Shape mismatch: phys_dists ({len(phys_dists_np)}) vs candidate energies ({len(energies_np)})"
                )

            std_dist = np.std(phys_dists_np)
            std_energy = np.std(energies_np)
            if std_dist == 0.0 or std_energy == 0.0:
                raise RuntimeError(
                    f"❌ [Telemetry Error] Zero variance in candidate rollouts: std_dist={std_dist:.6f}, std_energy={std_energy:.6f}"
                )

            corr_mat = np.corrcoef(phys_dists_np, energies_np)
            step_energy_dist_corr = corr_mat[0, 1]
            if np.isnan(step_energy_dist_corr):
                raise RuntimeError(
                    "❌ [Telemetry Error] Energy-distance correlation computed as NaN!"
                )

            sim._last_step_mean_phys_dist = step_mean_phys_dist
            sim._last_step_median_phys_dist = step_median_phys_dist
            sim._last_step_min_phys_dist = step_min_phys_dist
            sim._last_step_energy_dist_corr = step_energy_dist_corr
            sim._step_physical_distances = []

            print(
                f"📈 [Telemetry Summary] Step {env_step} -> Dist (16 tracks) "
                f"Mean: {step_mean_phys_dist:.3f}m | "
                f"Median: {step_median_phys_dist:.3f}m | "
                f"Min: {step_min_phys_dist:.3f}m ({phys_dists_np.argmin()}) | "
                f"Energy-Dist Corr: {step_energy_dist_corr:.3f}"
            )

            # Rewind physics back to the committed path's final outcome to capture its state
            sim.data.qpos[:] = committed_qpos
            sim.data.qvel[:] = committed_qvel
            sim.data.ctrl[:] = committed_ctrl
            mujoco.mj_forward(sim.model, sim.data)

            # Map outer variables to the committed outcome
            touch_index_next = committed_touch_index_next
            touch_thumb_next = committed_touch_thumb_next

            # Update transitions
            buffered_transitions.extend(transitions)

            # Report calibration payload to Colab once every 5 steps (or on final step)
            if (env_step + 1) % calibrate_steps == 0 or env_step == max_steps - 1:
                calibrate_payload = {
                    "transitions": buffered_transitions,
                    "episode_idx": ep_idx,
                    "step_idx": env_step,
                }

                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.post(
                            f"{colab_url}/stage3/calibrate",
                            json=calibrate_payload,
                            timeout=120.0,
                        )
                        if r.status_code == 200:
                            job_data = r.json()
                            job_id = job_data.get("job_id")
                            if job_id:
                                print(
                                    f"[Calibrate] Dispatched job {job_id}. Polling Colab for completion..."
                                )
                                while True:
                                    await asyncio.sleep(6.0)
                                    status_resp = await client.get(
                                        f"{colab_url}/stage3/calibrate/status/{job_id}",
                                        timeout=30.0,
                                    )
                                    if status_resp.status_code == 200:
                                        s_data = status_resp.json()
                                        st = s_data.get("status")
                                        if st == "completed":
                                            print(
                                                f"✅ [Calibrate] Job {job_id} completed. Dynamics Loss: {s_data.get('loss'):.6f}"
                                            )
                                            break
                                        elif st == "failed":
                                            print(
                                                f"❌ [Calibrate Error] Job {job_id} failed: {s_data.get('error')}"
                                            )
                                            break
                except Exception as e:
                    print(
                        f"[Training Error] Episode {ep_idx + 1} Step {env_step} Colab calibrate failed: {e}"
                    )
                    import traceback

                    traceback.print_exc()

                buffered_transitions = []

            # Step loop ends cleanly; next step starts with sim.reset_env(lock_posture=True, randomize_cube=True)

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
                    json={
                        "reward": final_reward,
                        "pos_trajectories": pos_trajectories,
                        "neg_trajectories": neg_trajectories,
                    },
                    timeout=30.0,
                )
                if r.status_code == 200:
                    res = r.json()
                    job_id = res.get("job_id")
                    if job_id:
                        print(
                            f"[Distill] Dispatched job {job_id}. Polling Colab for completion..."
                        )
                        while True:
                            await asyncio.sleep(3.0)
                            status_resp = await client.get(
                                f"{colab_url}/stage3/distill/status/{job_id}",
                                timeout=30.0,
                            )
                            if status_resp.status_code == 200:
                                s_data = status_resp.json()
                                st = s_data.get("status")
                                if st == "completed":
                                    opsd_loss = s_data.get("opsd_loss")
                                    landscape_data = s_data.get("energy_landscape")

                                    print(
                                        f"✅ [Distill] Job {job_id} completed for Episode {ep_idx + 1}. OPSD Loss: {opsd_loss}"
                                    )

                                    if landscape_data:
                                        landscape_dir = (
                                            "logs/training/latent-flow/analytics"
                                        )
                                        os.makedirs(landscape_dir, exist_ok=True)
                                        json_path = os.path.join(
                                            landscape_dir,
                                            f"epoch_{ep_idx + 1:02d}_energy_landscape.json",
                                        )
                                        with open(json_path, "w") as f:
                                            json.dump(landscape_data, f, indent=2)

                                        plot_energy_landscape(
                                            landscape_data, ep_idx + 1, landscape_dir
                                        )

                                    break
                                elif st == "failed":
                                    print(
                                        f"❌ [Distill Error] Job {job_id} failed: {s_data.get('error')}"
                                    )
                                    break
        except Exception as e:
            print(f"[Training Error] Episode {ep_idx + 1} distill failed: {e}")

    # Restore simulation physics cleanly back to the pre-training layout
    sim.data.qpos[:] = initial_state["qpos"]
