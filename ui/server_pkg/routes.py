import os
import re
import httpx
from fastapi import APIRouter
from . import config

router = APIRouter()


@router.get("/api/rollouts")
def list_rollouts():
    if not os.path.exists(config.LOGS_DIR):
        return []
    runs = []
    for name in os.listdir(config.LOGS_DIR):
        match = re.match(r"^rollouts_(\d+)$", name)
        if match and os.path.isdir(os.path.join(config.LOGS_DIR, name)):
            idx = int(match.group(1))
            runs.append({"id": name, "index": idx, "label": f"Run {idx}"})
    runs.sort(key=lambda r: r["index"], reverse=True)
    return runs


@router.get("/api/rollouts/{rollout_id}/steps")
def list_rollout_steps(rollout_id: str):
    target_dir = os.path.join(config.LOGS_DIR, rollout_id)
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


@router.get("/api/checkpoints")
async def get_checkpoints():
    if config.cached_checkpoints:
        return config.cached_checkpoints
    if not config.colab_url:
        return []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{config.colab_url}/stage3/checkpoints", timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                config.cached_checkpoints = data.get("checkpoints", [])
                return config.cached_checkpoints
    except Exception as e:
        print(f"⚠️ [Server Checkpoints Query Failed] {e}")
    return []
