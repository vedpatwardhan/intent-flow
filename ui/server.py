import asyncio
import base64
import io
import json
import os
import pickle
import sys
import traceback
from collections import deque
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image
import numpy as np
import cv2
import httpx
import copy

# Add parent directory (latent-flow root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation_base import GR1MuJoCoBase
import mujoco
from depth_unprojector import unproject_ui_annotations_to_3d
from ik_trajectory_sampler import (
    generate_ik_trajectories,
    save_ik_trajectory_diagnostic_plots,
    save_ik_trajectory_video,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_energy_landscape(landscape_data: dict, epoch: int, output_dir: str):
    """
    Renders an interactive HTML Energy Landscape plot (Plotly).
    Supports PCA 2D Latent Space Map (with positive/negative anchor markers)
    alongside Trajectory Pull Ratio distributions.
    """
    if (
        "mean_pos_dist_per_track" not in landscape_data
        or "mean_neg_dist_per_track" not in landscape_data
    ):
        return

    mean_pos = np.array(landscape_data["mean_pos_dist_per_track"])
    mean_neg = np.array(landscape_data["mean_neg_dist_per_track"])
    ratios = np.array(landscape_data["pos_neg_ratio"])
    metadata = landscape_data.get("trajectory_metadata", [])

    has_pca = "pca_rollout_coords" in landscape_data

    # Extract metadata labels for interactive hover tooltips
    hover_texts = []
    for idx in range(len(mean_pos)):
        meta = metadata[idx] if idx < len(metadata) else {}
        ep = meta.get("episode_idx", epoch)
        st = meta.get("step_idx", 0)
        cand = meta.get("candidate_idx", idx)
        text = (
            f"<b>Trajectory #{idx + 1}</b><br>"
            f"Episode: {ep} | Step: {st} | Candidate: {cand}<br>"
            f"D+ (Pos Dist): {mean_pos[idx]:.4f}<br>"
            f"D- (Neg Dist): {mean_neg[idx]:.4f}<br>"
            f"Pull Ratio (D+/D-): {ratios[idx]:.4f}"
        )
        hover_texts.append(text)

    # Render interactive Plotly HTML plot
    left_title = (
        f"Epoch {epoch}: PCA 2D Latent Space Map"
        if has_pca
        else f"Epoch {epoch}: Latent Energy Landscape (D+ vs D-)"
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            left_title,
            f"Epoch {epoch}: Trajectory Pull Distribution",
        ),
    )

    if has_pca:
        pca_rollouts = np.array(landscape_data["pca_rollout_coords"])
        pca_pos = np.array(landscape_data.get("pca_pos_coords", []))
        pca_neg = np.array(landscape_data.get("pca_neg_coords", []))

        # Subplot 1: PCA Rollouts
        fig.add_trace(
            go.Scatter(
                x=pca_rollouts[:, 0],
                y=pca_rollouts[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=ratios,
                    colorscale="rdylbu",
                    showscale=True,
                    colorbar=dict(title="Pull Ratio (D+/D-)", x=0.45),
                    line=dict(width=1, color="black"),
                ),
                text=hover_texts,
                hoverinfo="text",
                name="Rollouts",
            ),
            row=1,
            col=1,
        )

        # Positive Anchors (Green Stars)
        if len(pca_pos) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pca_pos[:, 0],
                    y=pca_pos[:, 1],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="green",
                        symbol="star",
                        line=dict(width=1, color="black"),
                    ),
                    name="Positive Anchors (Expert)",
                ),
                row=1,
                col=1,
            )

        # Negative Anchors (Red Crosses)
        if len(pca_neg) > 0:
            fig.add_trace(
                go.Scatter(
                    x=pca_neg[:, 0],
                    y=pca_neg[:, 1],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="red",
                        symbol="x",
                        line=dict(width=2, color="black"),
                    ),
                    name="Negative Anchors (Failure)",
                ),
                row=1,
                col=1,
            )

        fig.update_xaxes(title_text="PCA Component 1", row=1, col=1)
        fig.update_yaxes(title_text="PCA Component 2", row=1, col=1)
    else:
        # Subplot 1: D+ vs D- Scatter
        fig.add_trace(
            go.Scatter(
                x=mean_pos,
                y=mean_neg,
                mode="markers",
                marker=dict(
                    size=10,
                    color=ratios,
                    colorscale="rdylbu",
                    showscale=True,
                    colorbar=dict(title="Pull Ratio (D+/D-)", x=0.45),
                    line=dict(width=1, color="black"),
                ),
                text=hover_texts,
                hoverinfo="text",
                name="Rollouts",
            ),
            row=1,
            col=1,
        )

        # Parity Line (D+ = D-)
        max_val = max(float(mean_pos.max()), float(mean_neg.max()), 1.0)
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Parity Line (D+ = D-)",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Mean Normalized Cosine Distance to Positive Anchors (D+)",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Mean Normalized Cosine Distance to Negative Anchors (D-)",
            row=1,
            col=1,
        )

    # Subplot 2: Histogram
    fig.add_trace(
        go.Histogram(
            x=ratios,
            nbinsx=20,
            marker=dict(color="teal", line=dict(color="black", width=1)),
            name="Pull Ratios",
        ),
        row=1,
        col=2,
    )

    # Parity Threshold line
    fig.add_vline(x=1.0, line_width=2, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_xaxes(title_text="Pull Ratio (D+ / D-)", row=1, col=2)
    fig.update_yaxes(title_text="Trajectory Count", row=1, col=2)

    fig.update_layout(
        title_text=f"Stage 3 Epoch {epoch} Latent Energy Landscape Diagnostics",
        width=1400,
        height=600,
        template="plotly_white",
    )

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"epoch_{epoch:02d}_energy_landscape.html")
    fig.write_html(html_path)
    print(f"📊 [Landscape Analytics] Saved interactive HTML plot to: {html_path}")


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


# import rerun as rr

# Instantiate local server for live UI
sim = GR1SimulationServer()
sim.reset_env(lock_posture=True)

# Print initial 3D positions of cube and right wrist link upon server startup
try:
    cube_id = sim.model.body("cube").id
    cube_pos_3d = sim.data.xpos[cube_id].copy()
    print(f"\n📍 [Server Startup Ground Truth] Cube 3D World Position: {cube_pos_3d}")
except Exception as e:
    print(f"⚠️ [Server Startup] Could not query cube body: {e}")

try:
    wrist_id = sim.model.body("R_pinky_proximal_link").id
    wrist_pos_3d = sim.data.xpos[wrist_id].copy()
    print(
        f"📍 [Server Startup Ground Truth] Right Wrist Link 3D World Position: {wrist_pos_3d}\n"
    )
except Exception as e:
    print(f"⚠️ [Server Startup] Could not query right wrist link: {e}")

# Dedicated evaluation simulator for offline candidate track unrolling
eval_sim = GR1SimulationServer()
eval_sim.reset_env(lock_posture=True)


# Initialize Rerun logger for evaluation stream
# rr.init("latent_flow_offline_eval", spawn=False)
# try:
#     rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
#     print("✅ Rerun connected to rerun+http://127.0.0.1:9876/proxy")
# except Exception as e:
#     print(f"⚠️ Rerun connection fallback: {e}")

import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "logs", "training", "latent-flow")
)
if os.path.exists(LOGS_DIR):
    app.mount("/rollout_videos", StaticFiles(directory=LOGS_DIR), name="rollout_videos")


@app.get("/api/rollouts")
def list_rollouts():
    if not os.path.exists(LOGS_DIR):
        return []
    runs = []
    for name in os.listdir(LOGS_DIR):
        match = re.match(r"^rollouts_(\d+)$", name)
        if match and os.path.isdir(os.path.join(LOGS_DIR, name)):
            idx = int(match.group(1))
            runs.append({"id": name, "index": idx, "label": f"Run {idx}"})
    runs.sort(key=lambda r: r["index"], reverse=True)
    print(f"runs {runs}")
    return runs


@app.get("/api/rollouts/{rollout_id}/steps")
def list_rollout_steps(rollout_id: str):
    target_dir = os.path.join(LOGS_DIR, rollout_id)
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        return []
    steps = []
    for name in os.listdir(target_dir):
        match = re.match(r"^ep_(\d+)_step_(\d+)$", name)
        if match and os.path.isdir(os.path.join(target_dir, name)):
            ep = int(match.group(1))
            st = int(match.group(2))
            step_dir = os.path.join(target_dir, name)
            tracks = []
            for fname in os.listdir(step_dir):
                t_match = re.match(r"^track_(\d+)\.mp4$", fname)
                if t_match:
                    t_idx = int(t_match.group(1))
                    tracks.append(
                        {
                            "track_id": f"track_{t_idx:02d}",
                            "index": t_idx,
                            "filename": fname,
                            "url": f"/rollout_videos/{rollout_id}/{name}/{fname}",
                        }
                    )
            tracks.sort(key=lambda t: t["index"])
            steps.append(
                {
                    "step_id": name,
                    "epoch": ep,
                    "step": st,
                    "label": f"Epoch {ep} • Step {st}",
                    "tracks": tracks,
                }
            )
    steps.sort(key=lambda s: (s["epoch"], s["step"]))
    return steps


active_camera = "world_center"
encoder_processing_enabled = True
combostoc_noise = {"torso": 0.0, "arm": 0.0, "hand": 0.0, "vision": 0.0}
attack_active = False

colab_url = None
for i, arg in enumerate(sys.argv):
    if arg == "--colab-url" and i + 1 < len(sys.argv):
        colab_url = sys.argv[i + 1]
colab_url = colab_url or os.environ.get("COLAB_URL")

cached_checkpoints = []


@app.on_event("startup")
async def fetch_checkpoints_on_startup():
    global cached_checkpoints
    if not colab_url:
        print("[Startup Warning] COLAB_URL not set; skipping startup checkpoint query.")
        return
    try:
        async with httpx.AsyncClient() as client:
            print(
                f"🔄 [Server Startup] Querying available checkpoints from Colab: {colab_url}"
            )
            r = await client.get(f"{colab_url}/stage3/checkpoints", timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                cached_checkpoints = data.get("checkpoints", [])
                print(
                    f"✅ [Server Startup] Retrieved {len(cached_checkpoints)} checkpoints: {cached_checkpoints}"
                )
    except Exception as e:
        print(f"⚠️ [Server Startup Checkpoints Query Failed] {e}")


@app.get("/api/checkpoints")
async def get_checkpoints():
    global cached_checkpoints
    if cached_checkpoints:
        return cached_checkpoints
    if not colab_url:
        return []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{colab_url}/stage3/checkpoints", timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                cached_checkpoints = data.get("checkpoints", [])
                return cached_checkpoints
    except Exception as e:
        print(f"⚠️ [Server Checkpoints Query Failed] {e}")
    return []


click_x = None
click_y = None
click_type = None
text_prompt = "right hand to the red cube"
text_modifier = None
frame_history = deque(maxlen=5)
frame_all_views = {}
ui_annotations = {}

colab_is_processing = False
is_training_active = False
needs_colab_processing = False
last_colab_query_time = 0.0

cached_dino_attn = None
cached_clip_sim = None
cached_sam_mask = None
cached_motion_field = None
cached_task_isolated_features = None


def capture_sim_frames(sim):
    frame_all_views = {}
    for cam_name in sim.cam_names:
        sim.renderer.update_scene(sim.data, camera=cam_name)
        rgb = sim.renderer.render().copy()
        img = Image.fromarray(rgb)
        img_224 = img.resize((224, 224))
        buf = io.BytesIO()
        img_224.save(buf, format="JPEG", quality=75)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(
            "utf-8"
        )
        frame_all_views[cam_name] = b64
    return frame_all_views


def build_stage3_obs_payload(
    sim,
    text_prompt="grasp cube",
    ui_annotations=None,
    ep_idx=0,
    env_step=0,
    base_history_frames=None,
):
    current_frames = capture_sim_frames(sim)
    base_frames = (
        base_history_frames
        if base_history_frames is not None
        else [current_frames, current_frames, current_frames]
    )
    frame_history = [*base_frames, current_frames]
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


def render_camera_views(eval_sim):
    raw_frames = {}
    base64_frames = {}
    for cam_name in eval_sim.cam_names:
        eval_sim.renderer.update_scene(eval_sim.data, camera=cam_name)
        rgb_cam = eval_sim.renderer.render().copy()
        raw_frames[cam_name] = rgb_cam.copy()
        img_cam = Image.fromarray(rgb_cam).resize((224, 224))
        buf = io.BytesIO()
        img_cam.save(buf, format="JPEG", quality=75)
        base64_frames[cam_name] = "data:image/jpeg;base64," + base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")
    return raw_frames, base64_frames


def write_tiled_mp4(
    video_path: str, track_frames: list, track_idx: int, energy_val: float
):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(video_path, fourcc, 4.0, (720, 480))
    for frame_idx, frame_dict in enumerate(track_frames):
        grid = np.zeros((480, 720, 3), dtype=np.uint8)
        grid[0:240, 0:240] = cv2.resize(frame_dict["world_center"], (240, 240))
        grid[0:240, 240:480] = cv2.resize(frame_dict["world_top"], (240, 240))
        grid[0:240, 480:720] = cv2.resize(frame_dict["world_left"], (240, 240))
        grid[240:480, 0:240] = cv2.resize(frame_dict["world_right"], (240, 240))
        grid[240:480, 240:480] = cv2.resize(frame_dict["world_wrist"], (240, 240))
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
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        video_writer.write(grid_bgr)
    video_writer.release()


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
    websocket: WebSocket, sim, colab_url: str, text_prompt: str, ui_annotations: dict
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

    r_index_id = sim.model.body("R_index_tip_link").id
    r_thumb_id = sim.model.body("R_thumb_tip_link").id
    l_index_id = sim.model.body("L_index_tip_link").id
    l_thumb_id = sim.model.body("L_thumb_tip_link").id
    cube_id = sim.model.body("cube").id
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

    num_epochs = 20
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
                task_isolated_features=cached_task_isolated_features,
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
    sim.data.qvel[:] = initial_state["qvel"]
    sim.data.ctrl[:] = initial_state["ctrl"]
    mujoco.mj_forward(sim.model, sim.data)

    await websocket.send_text(
        json.dumps(
            {
                "type": "training_progress",
                "status": "Training Completed Successfully!",
                "progress": 1.0,
                "epoch": num_epochs,
                "total_epochs": num_epochs,
            }
        )
    )
    print("[Training] Stage 3 training sandbox finished.")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_camera, encoder_processing_enabled, attack_active, combostoc_noise, click_x, click_y, click_type, text_prompt, text_modifier
    global colab_is_processing, needs_colab_processing, last_colab_query_time
    global cached_dino_attn, cached_clip_sim, cached_sam_mask, cached_motion_field, cached_task_isolated_features
    global ui_annotations, is_training_active
    await websocket.accept()
    print("UI Connected via WebSocket")

    # Auto-trigger default Colab processing request to populate panels on load
    needs_colab_processing = True

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
                    baseline_exemplar_frames = None
                    needs_colab_processing = True
                    is_moving = False

                elif payload.get("type") == "wild_randomize":
                    sim.wild_reset()
                    baseline_exemplar_frames = None
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

                elif payload.get("type") == "record_exemplar":
                    name = payload.get("name", "phase_1")
                    print(f"📸 Recording Stage 3 Exemplar Snapshot locally: {name}...")
                    try:
                        exemplar_dir = os.path.abspath(
                            os.path.join(
                                os.path.dirname(__file__),
                                "..",
                                "checkpoints",
                                "exemplars",
                            )
                        )
                        os.makedirs(exemplar_dir, exist_ok=True)
                        target_path = os.path.join(exemplar_dir, f"{name}.pkl")

                        if name == "phase_0":
                            baseline_exemplar_frames = capture_sim_frames(sim)
                            print(
                                "📸 Baseline Anchor Snapshot (Phase 0) set successfully!"
                            )

                        obs_payload = build_stage3_obs_payload(
                            sim, text_prompt, ui_annotations, 0, 0
                        )
                        with open(target_path, "wb") as f:
                            pickle.dump(obs_payload, f)

                        print(
                            f"💾 [Local Exemplar Factory] Captured and saved state checkpoint: {target_path}"
                        )
                    except Exception as e:
                        print(f"❌ Error recording exemplar '{name}': {e}")

                elif payload.get("type") == "clear_selections":
                    click_x = None
                    click_y = None
                    click_type = None
                    needs_colab_processing = False
                    cached_dino_attn = None
                    cached_clip_sim = None
                    cached_sam_mask = None
                    cached_motion_field = None
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

                elif payload.get("type") == "execute_checkpoint":
                    ckpt_name = payload.get("checkpoint_name", "stage3_rl_final.pt")
                    noise_scale = float(payload.get("step_nft_scale", 0.08))
                    print(
                        f"⚡ [Execute Checkpoint] Executing '{ckpt_name}' with noise scale {noise_scale}..."
                    )

                    # Capture 5-camera observation payload
                    obs_payload = build_stage3_obs_payload(
                        sim, text_prompt, ui_annotations, 0, 0
                    )

                    execute_data = {
                        "obs": obs_payload,
                        "checkpoint_name": ckpt_name,
                        "step_nft_scale": noise_scale,
                        "seed": 42,
                    }

                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{colab_url}/stage3/execute",
                                json=execute_data,
                                timeout=120.0,
                            )
                            if r.status_code == 200:
                                res = r.json()
                                action_candidates = res.get("action_candidates", [])
                                print(
                                    f"✅ [Execute Checkpoint] Received {len(action_candidates)} candidate trajectories from Colab."
                                )
                                if action_candidates and len(action_candidates) == 16:
                                    action_np = np.array(
                                        action_candidates, dtype=np.float32
                                    )  # [16, 7, 58]

                                    # Snapshot initial state
                                    initial_qpos = sim.data.qpos.copy()
                                    initial_qvel = sim.data.qvel.copy()
                                    initial_ctrl = sim.data.ctrl.copy()

                                    r_index_id = mujoco.mj_name2id(
                                        sim.model,
                                        mujoco.mjtObj.mjOBJ_SITE,
                                        "r_index_tip",
                                    )
                                    r_thumb_id = mujoco.mj_name2id(
                                        sim.model,
                                        mujoco.mjtObj.mjOBJ_SITE,
                                        "r_thumb_tip",
                                    )
                                    l_index_id = mujoco.mj_name2id(
                                        sim.model,
                                        mujoco.mjtObj.mjOBJ_SITE,
                                        "l_index_tip",
                                    )
                                    l_thumb_id = mujoco.mj_name2id(
                                        sim.model,
                                        mujoco.mjtObj.mjOBJ_SITE,
                                        "l_thumb_tip",
                                    )
                                    cube_id = mujoco.mj_name2id(
                                        sim.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube"
                                    )

                                    evaluated_candidates = []

                                    # Roll out each of the 16 candidate trajectories in isolated evaluation environment
                                    for candidate_idx in range(16):
                                        eval_sim = copy.deepcopy(sim)
                                        eval_sim.data.qpos[:] = initial_qpos
                                        eval_sim.data.qvel[:] = initial_qvel
                                        eval_sim.data.ctrl[:] = initial_ctrl
                                        mujoco.mj_forward(eval_sim.model, eval_sim.data)

                                        step_phys_distances = []
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

                                        for h in range(7):
                                            act_k = action_np[candidate_idx, h, :]
                                            act_32_clamped = np.clip(
                                                act_k[:32], -1.0, 1.0
                                            )
                                            eval_sim.process_target_32(act_32_clamped)
                                            eval_sim.dispatch_action(
                                                action_32_norm=act_32_clamped,
                                                target_q=eval_sim.last_target_q,
                                                n_steps=2,
                                                reset_start=False,
                                            )

                                            # Calculate physical distance to cube target at step h
                                            r_index_pos = (
                                                eval_sim.data.xpos[r_index_id]
                                                if r_index_id != -1
                                                else eval_sim.data.qpos[:3]
                                            )
                                            r_thumb_pos = (
                                                eval_sim.data.xpos[r_thumb_id]
                                                if r_thumb_id != -1
                                                else eval_sim.data.qpos[:3]
                                            )
                                            l_index_pos = (
                                                eval_sim.data.xpos[l_index_id]
                                                if l_index_id != -1
                                                else eval_sim.data.qpos[:3]
                                            )
                                            l_thumb_pos = (
                                                eval_sim.data.xpos[l_thumb_id]
                                                if l_thumb_id != -1
                                                else eval_sim.data.qpos[:3]
                                            )
                                            cube_pos = (
                                                eval_sim.data.xpos[cube_id]
                                                if cube_id != -1
                                                else np.array([0.45, 0.0, 0.85])
                                            )

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

                                            # Render camera frames
                                            cur_frames, _ = render_camera_views(
                                                eval_sim
                                            )
                                            for cam_key in track_frames_per_cam:
                                                if cam_key in cur_frames:
                                                    track_frames_per_cam[
                                                        cam_key
                                                    ].append(cur_frames[cam_key])

                                        mean_phys_dist = float(
                                            np.mean(step_phys_distances)
                                        )
                                        evaluated_candidates.append(
                                            {
                                                "candidate_idx": candidate_idx,
                                                "mean_phys_dist": mean_phys_dist,
                                                "final_frames": {
                                                    cam: track_frames_per_cam[cam][-1]
                                                    for cam in track_frames_per_cam
                                                    if track_frames_per_cam[cam]
                                                },
                                                "actions": action_np[
                                                    candidate_idx
                                                ].tolist(),
                                            }
                                        )

                                    # Rank candidates by ascending mean physical distance
                                    evaluated_candidates.sort(
                                        key=lambda c: c["mean_phys_dist"]
                                    )
                                    top_8_candidates = evaluated_candidates[:8]

                                    for rank, cand in enumerate(
                                        top_8_candidates, start=1
                                    ):
                                        cand["rank"] = rank

                                    print(
                                        f"📊 [Execute Checkpoint] Top Candidate #1 Mean Physical Distance: {top_8_candidates[0]['mean_phys_dist']:.4f}m"
                                    )

                                    # Send top 8 candidate evaluation results to UI
                                    await websocket.send_text(
                                        json.dumps(
                                            {
                                                "type": "checkpoint_execution_results",
                                                "checkpoint": ckpt_name,
                                                "top_candidates": [
                                                    {
                                                        "rank": c["rank"],
                                                        "candidate_idx": c[
                                                            "candidate_idx"
                                                        ],
                                                        "mean_phys_dist": round(
                                                            c["mean_phys_dist"], 4
                                                        ),
                                                        "frames": c["final_frames"],
                                                    }
                                                    for c in top_8_candidates
                                                ],
                                            }
                                        )
                                    )

                                    # Commit Best Candidate (#1) trajectory to live simulation
                                    best_actions = np.array(
                                        top_8_candidates[0]["actions"], dtype=np.float32
                                    )
                                    for h_step in range(7):
                                        step_act = best_actions[h_step, :32]
                                        sim.process_target_32(step_act)
                                        for _ in range(16):
                                            sim.sync_ctrl_to_qpos(sim.last_target_q)
                                            sim.data.qpos[
                                                sim.root_q_idx : sim.root_q_idx + 3
                                            ] = [0.0, 0.0, 0.95]
                                            sim.data.qpos[
                                                sim.root_q_idx + 3 : sim.root_q_idx + 7
                                            ] = [1.0, 0.0, 0.0, 0.0]
                                            sim.data.qvel[:6] = 0.0
                                            mujoco.mj_step(sim.model, sim.data)
                                        await asyncio.sleep(0.02)
                    except Exception as e:
                        print(f"❌ [Execute Checkpoint Error] {e}")

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
            if is_moving and not is_training_active:
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
                needs_colab_processing = False
                sim.renderer.update_scene(sim.data, camera=active_camera)

                # Render frames from all cameras
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

                colab_is_processing = True
                last_colab_query_time = current_time

                # Append the first frame or any frame with actual movement
                if is_moving or step_count == 0:
                    print("Appending Frame")
                    frame_history.append(frame_all_views.copy())
                    is_moving = False

                async def run_colab_query(payload_data):
                    global colab_is_processing
                    global cached_dino_attn, cached_clip_sim, cached_sam_mask, cached_motion_field, cached_task_isolated_features
                    nonlocal cached_data_updated
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{colab_url}/process",
                                json=payload_data,
                                timeout=100.0,
                            )
                            if r.status_code == 200:
                                res_data = r.json()
                                cached_dino_attn = res_data.get("dino_attn")
                                cached_clip_sim = res_data.get("clip_sim")
                                cached_sam_mask = res_data.get("sam_mask")
                                cached_motion_field = res_data.get("motion_field")
                                print(
                                    f"Motion Field: {np.array(cached_motion_field).shape}"
                                )
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
                    "frame": frame_all_views[active_camera],
                    "click_x": click_x,
                    "click_y": click_y,
                    "click_type": click_type,
                    "text_prompt": text_prompt,
                    "text_modifier": text_modifier,
                    "ui_annotations": ui_annotations,
                    "history_frames": [
                        frames[active_camera] for frames in frame_history
                    ],
                    "view_name": active_camera,
                }
                asyncio.create_task(run_colab_query(post_payload))

            if is_moving or cached_data_updated or (step_count % 20 == 0):
                should_send = True

            if should_send:
                if cached_data_updated:
                    ws_payload["dino_attn"] = cached_dino_attn
                    ws_payload["clip_sim"] = cached_clip_sim
                    ws_payload["sam_mask"] = cached_sam_mask
                    ws_payload["motion_field"] = cached_motion_field
                    ws_payload["task_isolated_features"] = cached_task_isolated_features
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
