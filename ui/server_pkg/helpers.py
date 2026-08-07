import base64
import io
import os
import cv2
import numpy as np
from PIL import Image


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


def render_camera_views(eval_sim):
    raw_frames = {}
    base64_frames = {}
    for cam_name in eval_sim.cam_names:
        eval_sim.renderer.update_scene(eval_sim.data, camera=cam_name)
        rgb_cam = eval_sim.renderer.render().copy()
        raw_frames[cam_name] = rgb_cam.copy()
        img_cam = Image.fromarray(rgb_cam).resize((480, 480))
        buf = io.BytesIO()
        img_cam.save(buf, format="JPEG", quality=75)
        base64_frames[cam_name] = "data:image/jpeg;base64," + base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")
    return base64_frames, raw_frames


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
    from .config import LOGS_DIR

    step_dir = os.path.join(
        LOGS_DIR,
        f"rollouts_{ep_idx + 1:02d}",
        f"ep_{ep_idx + 1:02d}_step_{env_step:02d}",
    )
    os.makedirs(step_dir, exist_ok=True)
    for track_idx, track_frames in enumerate(all_track_frames):
        mp4_filename = f"track_{track_idx:02d}.mp4"
        mp4_path = os.path.join(step_dir, mp4_filename)
        e_val = float(energy_ensemble[track_idx])
        write_tiled_mp4(mp4_path, track_frames, track_idx, e_val)
