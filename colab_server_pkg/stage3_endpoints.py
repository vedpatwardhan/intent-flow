import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException

from colab_server_pkg.config import device
from colab_server_pkg.models_state import stage3_models, stage3_trajectory_history
from colab_server_pkg.feature_extractor import extract_stage3_obs_features

from models.adapters import (
    VisualAdapter,
    TextAdapter,
    PointNeXtAdapter,
    TactileAdapter,
    ActionAdapter,
    VGGTAdapter,
)
from models.msat import MultiStreamActionTransformer
from models.jepa_predictor import JepaPredictor
from trainers.stage3.denoiser import ComboStocFlowMatcher
from trainers.stage3.discriminator import TrajectoryDiscriminator
from trainers.stage3.attacker import BadWorldAttacker
from trainers.stage3.trainer import GNNSkillLibrary, HybridMemoryTriad


class Stage3StepPayload(BaseModel):
    frame: str
    history_frames: List[str]
    proprioception: List[float]
    tactile: List[List[float]]
    text_prompt: str
    ui_annotations: dict
    is_easy_task: bool = False


class Stage3CalibratePayload(BaseModel):
    current_obs: Stage3StepPayload
    action_taken: List[float]
    next_obs: Stage3StepPayload


class Stage3DistillPayload(BaseModel):
    reward: float


def ensure_stage3_models():
    import colab_server_pkg.models_state as state

    if "msat" in state.stage3_models:
        return

    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    action_dim = config["model"]["action_dim"]
    latent_dim = config["model"]["latent_dim"]

    state.stage3_models["vis_adapter"] = VisualAdapter(d_in=384).to(device)
    state.stage3_models["txt_adapter"] = TextAdapter(d_in=512).to(device)
    state.stage3_models["pt_adapter"] = PointNeXtAdapter(d_in=384).to(device)
    state.stage3_models["vggt_adapter"] = VGGTAdapter(
        d_in=config["model"]["vggt_dim"]
    ).to(device)
    state.stage3_models["tactile_adapter"] = TactileAdapter().to(device)
    state.stage3_models["action_adapter"] = ActionAdapter(
        d_in=action_dim, d_out=512
    ).to(device)
    state.stage3_models["action_down_proj"] = torch.nn.Linear(512, 16).to(device)

    state.stage3_models["msat"] = MultiStreamActionTransformer(
        latent_dim=latent_dim,
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    state.stage3_models["predictor"] = JepaPredictor(
        state_dim=latent_dim,
        action_dim=16,
        hidden_dim=latent_dim,
    ).to(device)

    state.stage3_models["flow_matcher"] = ComboStocFlowMatcher(
        action_dim=action_dim, config=config
    ).to(device)

    state.stage3_models["gnn_library"] = GNNSkillLibrary(
        state.stage3_models["flow_matcher"], state_dim=latent_dim
    ).to(device)

    state.stage3_models["discriminator"] = TrajectoryDiscriminator(
        action_dim=action_dim
    ).to(device)

    state.stage3_models["attacker"] = BadWorldAttacker(action_dim=action_dim)

    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", config["paths"]["checkpoint_dir"])
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    s3_ckpt_path = os.path.join(checkpoint_dir, "stage3_rl_final.pt")
    s2_ckpt_path = os.path.join(checkpoint_dir, "stage2_sft.pt")

    if os.path.exists(s3_ckpt_path):
        print(f"[Colab] Loading Stage 3 checkpoint from: {s3_ckpt_path}")
        checkpoint = torch.load(s3_ckpt_path, map_location=device)
        state.stage3_models["vis_adapter"].load_state_dict(checkpoint["vis_adapter"])
        state.stage3_models["txt_adapter"].load_state_dict(checkpoint["txt_adapter"])
        state.stage3_models["pt_adapter"].load_state_dict(checkpoint["pt_adapter"])
        state.stage3_models["vggt_adapter"].load_state_dict(checkpoint["vggt_adapter"])
        state.stage3_models["tactile_adapter"].load_state_dict(
            checkpoint["tactile_adapter"]
        )
        state.stage3_models["action_adapter"].load_state_dict(
            checkpoint["action_adapter"]
        )
        state.stage3_models["action_down_proj"].load_state_dict(
            checkpoint["action_down_proj"]
        )
        state.stage3_models["msat"].load_state_dict(checkpoint["msat"])
        state.stage3_models["predictor"].load_state_dict(checkpoint["predictor"])
        state.stage3_models["gnn_library"].nodes.load_state_dict(
            checkpoint["gnn_nodes"]
        )
        state.stage3_models["gnn_library"].specialists.load_state_dict(
            checkpoint["gnn_specialists"]
        )
    elif os.path.exists(s2_ckpt_path):
        print(f"[Colab] Loading Stage 2 checkpoint from: {s2_ckpt_path}")
        checkpoint = torch.load(s2_ckpt_path, map_location=device)
        state.stage3_models["vis_adapter"].load_state_dict(checkpoint["vis_adapter"])
        state.stage3_models["txt_adapter"].load_state_dict(checkpoint["txt_adapter"])
        state.stage3_models["pt_adapter"].load_state_dict(checkpoint["pt_adapter"])
        state.stage3_models["vggt_adapter"].load_state_dict(checkpoint["vggt_adapter"])
        state.stage3_models["tactile_adapter"].load_state_dict(
            checkpoint["tactile_adapter"]
        )
        state.stage3_models["action_adapter"].load_state_dict(
            checkpoint["action_adapter"]
        )
        state.stage3_models["action_down_proj"].load_state_dict(
            checkpoint["action_down_proj"]
        )
        state.stage3_models["msat"].load_state_dict(checkpoint["msat"])
        state.stage3_models["predictor"].load_state_dict(checkpoint["predictor"])
        state.stage3_models["flow_matcher"].load_state_dict(checkpoint["flow_matcher"])

    state.stage3_optimizer = torch.optim.AdamW(
        list(state.stage3_models["flow_matcher"].parameters())
        + list(state.stage3_models["gnn_library"].parameters())
        + list(state.stage3_models["predictor"].parameters())
        + list(state.stage3_models["discriminator"].parameters())
        + list(state.stage3_models["action_adapter"].parameters())
        + list(state.stage3_models["action_down_proj"].parameters()),
        lr=config["stage3"]["lr"],
    )

    state.stage3_memory = HybridMemoryTriad()


async def handle_stage3_step(payload: Stage3StepPayload):
    # Called on every step of every epoch
    try:
        # instantiates all models and loads parameters from checkpoints
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        # get all global and filtered features
        obs_dict = extract_stage3_obs_features(payload)

        with torch.no_grad():
            vis_tok = state.stage3_models["vis_adapter"](obs_dict["vision"])
            txt_tok = state.stage3_models["txt_adapter"](obs_dict["text"].squeeze(1))
            pt_tok = state.stage3_models["pt_adapter"](obs_dict["pointnext"])
            vggt_tok = state.stage3_models["vggt_adapter"](obs_dict["vggt"])
            tactile_emb = state.stage3_models["tactile_adapter"](obs_dict["tactile"])

            modality_dict = {
                "vision": vis_tok,
                "text": txt_tok,
                "pointnext": pt_tok,
                "vggt": vggt_tok,
                "tactile": tactile_emb,
                "proprioception": obs_dict["proprioception"],
            }
            s_t = state.stage3_models["msat"](modality_dict)

            if len(state.stage3_memory.short_term) >= 2:
                s_t = s_t + state.stage3_memory.gist.to(device)

            if len(state.stage3_memory.anchors) > 0:
                s_target = state.stage3_memory.anchors[-1].to(device)
            else:
                s_target = s_t.clone()

            active_policy, active_node_key = state.stage3_models[
                "gnn_library"
            ].route_or_spawn(s_t, state.stage3_models["flow_matcher"])

        a_candidate = active_policy.sample_with_steering(s_t, s_target, num_steps=10)
        a_candidate = a_candidate.clone().detach().requires_grad_(True)
        eta = 0.1

        for k in range(5):
            z_action = state.stage3_models["action_adapter"](a_candidate)
            z_action_16 = state.stage3_models["action_down_proj"](z_action)
            s_next_pred = state.stage3_models["predictor"](s_t, z_action_16)
            energy = torch.mean((s_next_pred - s_target) ** 2)
            grad_a = torch.autograd.grad(energy, a_candidate, retain_graph=True)[0]
            with torch.no_grad():
                a_candidate = a_candidate - eta * grad_a
            a_candidate = a_candidate.clone().detach().requires_grad_(True)

        final_action = a_candidate.detach()

        if payload.is_easy_task:
            with torch.no_grad():
                final_action = state.stage3_models[
                    "attacker"
                ].generate_perturbed_context(active_policy, s_t, s_target, final_action)

        return {
            "action": final_action.squeeze(0).cpu().numpy().tolist(),
            "active_node_key": active_node_key,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Stage 3 step error: {str(e)}")


async def handle_stage3_calibrate(payload: Stage3CalibratePayload):
    try:
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        obs_t = extract_stage3_obs_features(payload.current_obs)
        obs_next = extract_stage3_obs_features(payload.next_obs)
        action = torch.tensor([payload.action_taken], dtype=torch.float32).to(device)

        with torch.no_grad():
            s_t = state.stage3_models["msat"](
                {
                    "vision": state.stage3_models["vis_adapter"](obs_t["vision"]),
                    "text": state.stage3_models["txt_adapter"](
                        obs_t["text"].squeeze(1)
                    ),
                    "pointnext": state.stage3_models["pt_adapter"](obs_t["pointnext"]),
                    "vggt": state.stage3_models["vggt_adapter"](obs_t["vggt"]),
                    "tactile": state.stage3_models["tactile_adapter"](obs_t["tactile"]),
                    "proprioception": obs_t["proprioception"],
                }
            ).detach()

            s_next = state.stage3_models["msat"](
                {
                    "vision": state.stage3_models["vis_adapter"](obs_next["vision"]),
                    "text": state.stage3_models["txt_adapter"](
                        obs_next["text"].squeeze(1)
                    ),
                    "pointnext": state.stage3_models["pt_adapter"](
                        obs_next["pointnext"]
                    ),
                    "vggt": state.stage3_models["vggt_adapter"](obs_next["vggt"]),
                    "tactile": state.stage3_models["tactile_adapter"](
                        obs_next["tactile"]
                    ),
                    "proprioception": obs_next["proprioception"],
                }
            ).detach()

        state.stage3_trajectory_history.append((s_t, action, s_next))
        if len(state.stage3_trajectory_history) > 100:
            state.stage3_trajectory_history.pop(0)

        batch_size = min(len(state.stage3_trajectory_history), 8)
        indices = np.random.choice(
            len(state.stage3_trajectory_history), batch_size, replace=False
        )

        batch_s_t = torch.cat(
            [state.stage3_trajectory_history[idx][0] for idx in indices], dim=0
        )
        batch_action = torch.cat(
            [state.stage3_trajectory_history[idx][1] for idx in indices], dim=0
        )
        batch_s_next = torch.cat(
            [state.stage3_trajectory_history[idx][2] for idx in indices], dim=0
        )

        state.stage3_optimizer.zero_grad()
        z_action = state.stage3_models["action_adapter"](batch_action)
        z_action_16 = state.stage3_models["action_down_proj"](z_action)
        s_next_pred = state.stage3_models["predictor"](batch_s_t, z_action_16)

        loss_dynamics = F.mse_loss(s_next_pred, batch_s_next)
        loss_dynamics.backward()
        state.stage3_optimizer.step()

        tactile_spike = float(payload.next_obs.tactile[0][0] > 0.5)
        state.stage3_memory.update(s_t[0], tactile_spike)

        return {"status": "calibrated", "loss": loss_dynamics.item()}
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Stage 3 calibration error: {str(e)}"
        )


async def handle_stage3_distill(payload: Stage3DistillPayload):
    try:
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        if len(state.stage3_trajectory_history) == 0:
            return {"status": "no data to distill"}

        config_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "config", "default_config.yaml"
            )
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        state.stage3_optimizer.zero_grad()

        sample_idx = np.random.randint(0, len(state.stage3_trajectory_history))
        s_t_sample, a_sample, s_next_sample = state.stage3_trajectory_history[
            sample_idx
        ]

        x_0 = torch.randn_like(a_sample)
        t_rand = torch.rand(a_sample.size(0), 1, device=device)
        t_vector = t_rand.expand(-1, state.stage3_models["flow_matcher"].action_dim)

        x_t = t_vector * a_sample + (1.0 - t_vector) * x_0
        target_vel = a_sample - x_0

        if len(state.stage3_memory.anchors) > 0:
            s_target_sample = state.stage3_memory.anchors[-1].to(device).unsqueeze(0)
        else:
            s_target_sample = s_t_sample.clone()

        pred_vel = state.stage3_models["flow_matcher"].velocity_field(
            x_t, t_vector, s_t_sample, s_target_sample
        )
        cfm_loss = F.mse_loss(pred_vel, target_vel)

        combined_reward = max(0.05, 1.0 + payload.reward)
        ram_loss = cfm_loss * combined_reward

        ram_loss.backward()
        state.stage3_optimizer.step()

        checkpoint_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", config["paths"]["checkpoint_dir"]
            )
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "stage3_rl_final.pt")

        torch.save(
            {
                "vis_adapter": state.stage3_models["vis_adapter"].state_dict(),
                "txt_adapter": state.stage3_models["txt_adapter"].state_dict(),
                "pt_adapter": state.stage3_models["pt_adapter"].state_dict(),
                "vggt_adapter": state.stage3_models["vggt_adapter"].state_dict(),
                "tactile_adapter": state.stage3_models["tactile_adapter"].state_dict(),
                "msat": state.stage3_models["msat"].state_dict(),
                "action_adapter": state.stage3_models["action_adapter"].state_dict(),
                "action_down_proj": state.stage3_models[
                    "action_down_proj"
                ].state_dict(),
                "predictor": state.stage3_models["predictor"].state_dict(),
                "gnn_nodes": state.stage3_models["gnn_library"].nodes.state_dict(),
                "gnn_specialists": state.stage3_models[
                    "gnn_library"
                ].specialists.state_dict(),
            },
            final_path,
        )

        return {
            "status": "distilled",
            "ram_loss": ram_loss.item(),
            "checkpoint": final_path,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Stage 3 distillation error: {str(e)}"
        )
