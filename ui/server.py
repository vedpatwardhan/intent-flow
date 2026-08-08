import os
import sys
import httpx
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add parent directory (intent-flow root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation_base import GR1MuJoCoBase
from server_pkg import config
from server_pkg.routes import router as api_router
from server_pkg.websocket_handler import websocket_endpoint_handler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(config.LOGS_DIR):
    app.mount(
        "/rollout_videos", StaticFiles(directory=config.LOGS_DIR), name="rollout_videos"
    )

app.include_router(api_router)


class GR1SimulationServer(GR1MuJoCoBase):
    def __init__(self):
        super().__init__(restrict_ik=True)

    def _handle_ik_pickup_logic(self, phase=0, offset_cm=5):
        self.current_phase = phase + 1
        cube_id = self.model.body("cube").id
        cube_pos = self.data.qpos[
            self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id] + 3
        ].copy()
        quat_down = [0, 1, 0, 0]

        if phase == 0:
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


sim = GR1SimulationServer()
sim.reset_env(lock_posture=True)

eval_sim = GR1SimulationServer()
eval_sim.reset_env(lock_posture=True)


@app.on_event("startup")
async def fetch_checkpoints_on_startup():
    if not config.colab_url:
        print("[Startup Warning] COLAB_URL not set; skipping startup checkpoint query.")
        return
    try:
        async with httpx.AsyncClient() as client:
            print(
                f"🔄 [Server Startup] Querying available checkpoints from Colab: {config.colab_url}"
            )
            r = await client.get(f"{config.colab_url}/stage3/checkpoints", timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                config.cached_checkpoints = data.get("checkpoints", [])
                print(
                    f"✅ [Server Startup] Retrieved {len(config.cached_checkpoints)} checkpoints: {config.cached_checkpoints}"
                )
    except Exception as e:
        print(f"⚠️ [Server Startup Checkpoints Query Failed] {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_endpoint_handler(websocket, sim, eval_sim)
