import pickle
import io
import base64
import json
import os
import yaml
import numpy as np
from pydantic import BaseModel

EXEMPLAR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "exemplars")
)
from typing import List
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFilter
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def ensure_wandb_init(project_name="latent-flow-stage3"):
    if HAS_WANDB and wandb.run is None:
        try:
            wandb.init(project=project_name, reinit=False)
            print(f"✅ Weights & Biases initialized: project='{project_name}'")
        except Exception as e:
            print(f"⚠️ W&B initialization warning: {e}")


from colab_server_pkg.config import device
from colab_server_pkg.models_state import (
    stage3_models,
    stage3_trajectory_history,
    models,
)
from colab_server_pkg.feature_extractor import (
    extract_stage3_obs_features,
    extract_single_view_stage3_obs_features,
    construct_stage3_latent_goal_features,
)
from colab_server_pkg.image_utils import (
    decode_base64_image,
    save_stage3_debug_plots,
    save_stage3_goal_features_plots,
)

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


class LatentAlignmentAdapter(torch.nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.LayerNorm(feature_dim),
            torch.nn.GELU(),
            torch.nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, g):
        return g + self.adapter(g)


from trainers.stage3.trainer import GNNSkillLibrary


class Stage3StepPayload(BaseModel):
    frames: dict[str, str]  # Multi-view frames: {camera_name: base64_image}
    history_frames: List[dict[str, str]]
    proprioception: List[float]
    tactile: List[List[float]]
    text_prompt: str
    ui_annotations: dict
    is_easy_task: bool = False
    point_clouds: dict | None = None
    episode_idx: int = 0
    step_idx: int = 0


class Stage3CalibrateTransition(BaseModel):
    current_obs: Stage3StepPayload
    action_taken: List[float]
    next_obs: Stage3StepPayload
    energy: float
    tactile: float
    s_target: List[List[float]] | None = None


class Stage3CalibratePayload(BaseModel):
    transitions: List[Stage3CalibrateTransition]
    eval_mean_physical_distance: float | None = None
    eval_energy_distance_correlation: float | None = None


class Stage3DistillPayload(BaseModel):
    reward: float


def encode_obs_to_latent(obs_dict, state):
    """
    Passes observation features through the respective adapter blocks and the
    Multi-Stream Action Transformer (MSAT) to yield the multi-modal latent state.
    """
    with torch.amp.autocast("cuda"):
        # Adapters
        vis_tok = state.stage3_models["vis_adapter"](obs_dict["vision"])
        # txt_tok = state.stage3_models["txt_adapter"](obs_dict["text"])
        # pt_tok = state.stage3_models["pt_adapter"](obs_dict["pointnext"])
        vggt_tok = state.stage3_models["vggt_adapter"](obs_dict["vggt"])
        # tactile_tok = state.stage3_models["tactile_adapter"](obs_dict["tactile"])
        proprio_tok = state.stage3_models["state_adapter"](obs_dict["proprioception"])

        # MSAT
        modality_dict = {
            "vision": vis_tok,
            # "text": txt_tok,
            # "pointnext": pt_tok,
            "vggt": vggt_tok,
            # "tactile": tactile_tok,
            "proprioception": proprio_tok,
        }
        out = state.stage3_models["msat"](modality_dict)
        return out


def run_exemplar_diagnostic_check(s_target, state, eval_payloads):
    """
    Computes direct latent distance from a set of fixed environment states to the current target anchor.
    eval_payloads: Dict containing your pre-captured Near-Goal, Mid-Phase, and OOD observation structures.
    """
    import colab_server_pkg.models_state as model_state

    diagnostic_distances = {}
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            for name, payload in eval_payloads.items():
                # 1. Extract observation dictionaries natively
                obs_dict, combined_obs = extract_stage3_obs_features(payload)
                # 2. Encode to the shared latent state space
                s_encoded = encode_obs_to_latent(combined_obs, model_state)
                # 3. Calculate mean squared distance directly to your goal anchor
                dist = torch.mean((s_encoded - s_target) ** 2).item()
                diagnostic_distances[f"exemplar_distance/{name}"] = dist

    print(f"📊 Exemplar Goal Distances: {json.dumps(diagnostic_distances, indent=2)}")
    if HAS_WANDB and wandb.run is not None:
        wandb.log(diagnostic_distances)


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
    horizon = config["model"]["horizon"]
    latent_dim = config["model"]["latent_dim"]

    state.stage3_models["vis_adapter"] = VisualAdapter(d_in=384).to(device)
    state.stage3_models["txt_adapter"] = TextAdapter(d_in=384).to(device)
    state.stage3_models["pt_adapter"] = PointNeXtAdapter(d_in=384).to(device)
    state.stage3_models["vggt_adapter"] = VGGTAdapter(
        d_in=config["model"]["vggt_dim"]
    ).to(device)
    state.stage3_models["tactile_adapter"] = TactileAdapter().to(device)
    state.stage3_models["action_adapter"] = ActionAdapter(
        d_in=(horizon - 1) * action_dim, d_out=512
    ).to(device)
    state.stage3_models["state_adapter"] = ActionAdapter(
        d_in=config["model"]["state_dim"], d_out=512
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
    if "flow_matcher_ref" not in state.stage3_models:
        state.stage3_models["flow_matcher_ref"] = ComboStocFlowMatcher(
            action_dim=action_dim, config=config
        ).to(device)
        state.stage3_models["flow_matcher_ref"].load_state_dict(
            state.stage3_models["flow_matcher"].state_dict()
        )
        state.stage3_models["flow_matcher_ref"].eval()

    state.stage3_models["gnn_library"] = GNNSkillLibrary(
        state.stage3_models["flow_matcher"], state_dim=latent_dim
    ).to(device)

    state.stage3_models["discriminator"] = TrajectoryDiscriminator(
        action_dim=action_dim
    ).to(device)

    state.stage3_models["attacker"] = BadWorldAttacker(action_dim=action_dim)

    # Add goal attention head for converting predicted latent to goal latent
    state.stage3_models["goal_attention"] = torch.nn.MultiheadAttention(
        embed_dim=latent_dim, num_heads=8, batch_first=True
    ).to(device)

    # Add latent adapter for mapping foveated/blurred goal representations to SFT space
    state.stage3_models["latent_adapter"] = LatentAlignmentAdapter(
        feature_dim=latent_dim
    ).to(device)

    ckpt_dir_config = config["paths"]["checkpoint_dir"]
    if ckpt_dir_config.startswith("latent-flow/"):
        ckpt_dir_config = ckpt_dir_config[len("latent-flow/") :]
    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ckpt_dir_config)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    s3_ckpt_path = os.path.join(checkpoint_dir, "stage3_rl_final.pt")
    s2_ckpt_path = os.path.join(checkpoint_dir, "stage2_sft.pt")
    print(f"[Colab] Resolved checkpoint directory path: {checkpoint_dir}")

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
        if "state_adapter" in checkpoint:
            state.stage3_models["state_adapter"].load_state_dict(
                checkpoint["state_adapter"]
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
        if "latent_adapter" in checkpoint:
            state.stage3_models["latent_adapter"].load_state_dict(
                checkpoint["latent_adapter"]
            )
    elif os.path.exists(s2_ckpt_path):
        print(f"[Colab] Loading Stage 2 checkpoint from: {s2_ckpt_path}")
        checkpoint = torch.load(s2_ckpt_path, map_location=device)
        # SFT checkpoints are always PyTorch Lightning format; extract from flat state_dict
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        print("[Colab] Loading PyTorch Lightning SFT weights layout...")
        for module_name in [
            "vis_adapter",
            # "txt_adapter",
            "pt_adapter",
            # "vggt_adapter",
            "tactile_adapter",
            "action_adapter",
            "state_adapter",
            "action_down_proj",
            "msat",
            "predictor",
            # "flow_matcher",
        ]:
            m_state_dict = {}
            prefix = f"{module_name}."
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    m_state_dict[k[len(prefix) :]] = v
            if len(m_state_dict) > 0 and module_name in state.stage3_models:
                state.stage3_models[module_name].load_state_dict(m_state_dict)
    else:
        print(
            "[Colab] WARNING: Neither Stage 3 nor Stage 2 checkpoints were found. Models initialized with default random weights!"
        )

    stage3_lr = max(config["stage3"]["lr"], 2e-4)
    state.stage3_optimizer = torch.optim.AdamW(
        list(state.stage3_models["flow_matcher"].parameters())
        + list(state.stage3_models["gnn_library"].parameters())
        + list(state.stage3_models["predictor"].parameters())
        + list(state.stage3_models["discriminator"].parameters())
        + list(state.stage3_models["action_adapter"].parameters())
        + list(state.stage3_models["state_adapter"].parameters())
        + list(state.stage3_models["action_down_proj"].parameters())
        + list(state.stage3_models["goal_attention"].parameters())
        + list(state.stage3_models["latent_adapter"].parameters()),
        lr=stage3_lr,
    )


async def handle_stage3_step(payload: Stage3StepPayload):
    # Called on every step of every epoch
    try:
        torch.cuda.empty_cache()
        # instantiates all models and loads parameters from checkpoints
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        # get all global and filtered features
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                obs_dict, combined_obs = extract_stage3_obs_features(payload)

                # Construct and cache target goal representation strictly at Episode 0 Step 0
                is_start_of_training = (
                    payload.episode_idx == 0 and payload.step_idx == 0
                ) or not hasattr(state, "active_goal_combined_obs")

                if is_start_of_training:
                    annotations = payload.ui_annotations
                    goal_obs_dict, goal_combined_obs = (
                        construct_stage3_latent_goal_features(obs_dict, annotations)
                    )
                    state.active_goal_combined_obs = goal_combined_obs

                    # Save 4-panel diagnostic comparison plots for the initial frame
                    pil_frame = obs_dict["world_center"]["features"]["pil_frame"]
                    save_stage3_goal_features_plots(
                        pil_frame,
                        annotations,
                        goal_obs_dict,
                        view_name="world_center",
                    )
                else:
                    goal_combined_obs = state.active_goal_combined_obs

        with torch.no_grad():
            # Get clean live state latent [1, latent_dim]
            s_t = encode_obs_to_latent(combined_obs, state)

            # Get clean transformed target goal state latent [1, latent_dim]
            s_target = encode_obs_to_latent(goal_combined_obs, state)

        # Initialize the 2D Space-Time Grid (Horizon=7, Joints=58)
        horizon = 7
        joint_dim = 58
        total_gen_dim = horizon * joint_dim

        # Create a master grid
        # ToDo: temporarily setting it all to 0 for baseline exploration
        # grid = torch.ones(1, horizon, joint_dim, device=device)
        # grid[0, 0:3, :] = 0.0  # immediate steps get the full denoising path
        # grid[0, 3:6, :] = 0.5  # intermediate steps get coarse denoising
        # grid[0, 6:, :] = 0.8  # far steps initialized near convergence
        grid = torch.zeros(1, horizon, joint_dim, device=device)
        steering_timelines = grid.view(1, total_gen_dim)  # Shape: [1, Horizon * Joints]

        # Detach s_t and s_target to prevent graph leaks and runtime crashes
        s_t = s_t.detach()
        s_target = s_target.detach()
        embodiment_id = torch.tensor([2], dtype=torch.long, device=device)

        # --- 16-CANDIDATE STEPNFT STOCHASTIC NOISE GENERATION BLOCK ---
        # Comment out observation-space visual perturbations (blurring/exposure/noise)
        # and generate all 16 candidate trajectories directly using StepNFT stochastic noise.
        ensemble_size = 16
        s_t_ensemble = s_t.expand(ensemble_size, -1)
        s_target_expanded = s_target.expand(ensemble_size, -1)
        embodiment_id_expanded = embodiment_id.expand(ensemble_size)

        steering_timelines_expanded = (
            steering_timelines.expand(ensemble_size, -1)
            .view(ensemble_size, horizon, joint_dim)
            .clone()
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                a_candidates, step_snrs = state.stage3_models[
                    "flow_matcher"
                ].sample_with_steering(
                    s_t_ensemble,
                    s_target_expanded,
                    embodiment_id=embodiment_id_expanded,
                    horizon=horizon,
                    num_steps=10,
                    steering_timelines=steering_timelines_expanded,
                    step_nft_scale=0.2,
                )

        # Log ODE step SNR telemetry to W&B on remote Colab server
        if HAS_WANDB and step_snrs:
            ensure_wandb_init()
            if wandb.run is not None:
                try:
                    snr_dict = {
                        f"snr_trajectory/step_{idx}": val
                        for idx, val in enumerate(step_snrs)
                        if val is not None
                    }
                    if snr_dict:
                        wandb.log(snr_dict)
                except Exception:
                    pass

        # Learning rate for action adjustment and timeline rollback scale
        eta = 0.12
        timeline_advance_rate = 0.05
        error_threshold = 0.0008

        # Create embodiment-aware action mask (first 32 GR-1 active joints)
        action_mask = torch.zeros(1, 1, joint_dim, device=device)
        action_mask[..., :32] = 1.0

        # Mask initial candidate actions to ensure padding channels start at zero
        a_candidates = a_candidates * action_mask

        steering_history = {
            "episode_idx": payload.episode_idx,
            "step_idx": payload.step_idx,
            "iterations": [],
        }

        for k in range(8):
            a_candidates = a_candidates.clone().detach().requires_grad_(True)

            # Record candidate trajectory state before this iteration update
            steering_history["iterations"].append(
                {
                    "iteration": k,
                    "a_candidates": a_candidates.detach().cpu().tolist(),
                }
            )

            # Flatten step layouts to match ActionAdapter's footprint contract
            a_flat = a_candidates.view(ensemble_size, -1)  # Shape: [16, 406]

            # Get the action representation and next latent state
            z_action = state.stage3_models["action_adapter"](a_flat)
            z_action_16 = state.stage3_models["action_down_proj"](z_action)

            s_next_pred = state.stage3_models["predictor"](s_t_ensemble, z_action_16)

            # Direct energy computation between predicted next state and target goal anchor
            energy = torch.mean((s_next_pred - s_target_expanded) ** 2)
            grad_a = torch.autograd.grad(energy, a_candidates)[0]

            # Normalize gradients per ensemble instance and mask padding channels
            # [16, 7, 58]
            grad_norm = grad_a.norm(dim=(1, 2), keepdim=True) + 1e-8
            grad_a_normalized = (grad_a / grad_norm) * action_mask

            # Masked Energy Guidance matching 3D coordinates
            # Dynamically calculate the guidance mask (shape: [4, 7, 58])
            t_j = 1.0 - steering_timelines_expanded

            # Sculpt action parameters along the tracking vector field
            diff = eta * grad_a_normalized * t_j
            a_candidates = (a_candidates - diff) * action_mask

            # Raw gradient forces per joint: Mean over ensemble (dim 0) and horizon (dim 1) -> Shape: [58]
            raw_joint_errors = grad_a.abs().mean(dim=(0, 1))

            # Store initial unsteered raw gradient forces at iteration k == 0
            if k == 0:
                g_0_joint_errors = raw_joint_errors.clone().detach()

            # Isolate active structural dimensions (first 32 joints)
            active_selector = action_mask.squeeze(0).squeeze(0) > 0

            # Relative Force Convergence Ratio:
            # Joint j is STABLE if its current gradient force has dropped to <= 40% of its initial force at iteration 0
            # (i.e. >= 60% force reduction / convergence across lookahead steering)
            alpha = 0.40
            relative_ratio = raw_joint_errors / (g_0_joint_errors + 1e-8)

            # Stable and drifting masks scoped strictly to active joints
            stable_joints_mask = (relative_ratio <= alpha) & active_selector
            drifting_joints_mask = (~stable_joints_mask) & active_selector

            # Force padding channels to zero out entirely in the timeline grid
            padding_mask = ~active_selector
            steering_timelines_expanded[:, :, padding_mask] = 0.0

            # Advance timelines for stable active joints, reset drifting ones to noise
            steering_timelines_expanded[
                :, :, stable_joints_mask
            ] += timeline_advance_rate
            steering_timelines_expanded[:, :, drifting_joints_mask] = 0.0

            # Keep boundaries locked within standard flow matching bounds [0.0, 1.0]
            steering_timelines_expanded = torch.clamp(
                steering_timelines_expanded, 0.0, 1.0
            )

        # Detach the candidates after the loop
        final_actions = a_candidates.clone().detach()

        # Record final steered candidate trajectories (iteration 5)
        steering_history["iterations"].append(
            {
                "iteration": 5,
                "a_candidates": final_actions.cpu().tolist(),
            }
        )

        # Save steering trajectory JSON file
        steering_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "logs",
            "training",
            "latent-flow",
            "steering",
        )
        os.makedirs(steering_dir, exist_ok=True)
        steering_json_path = os.path.join(
            steering_dir,
            f"ep_{payload.episode_idx}_step_{payload.step_idx}.json",
        )
        with open(steering_json_path, "w") as f:
            json.dump(steering_history, f, indent=2)
        print(f"📊 [Telemetry] Saved steering trajectory JSON: {steering_json_path}")

        # Compute per-candidate final energy scores directly against target anchor
        with torch.no_grad():
            final_energies = torch.mean((s_next_pred - s_target_expanded) ** 2, dim=-1)

        # Log real steering telemetry metrics to Weights & Biases
        if HAS_WANDB:
            ensure_wandb_init()
            if wandb.run is not None:
                with torch.no_grad():
                    candidate_var = final_actions.var(dim=0).mean().item()
                    mean_timeline = steering_timelines_expanded[..., :32].mean().item()
                    stable_joints_cnt = stable_joints_mask.sum().item()
                    drifting_joints_cnt = drifting_joints_mask.sum().item()
                    mean_energy = final_energies.mean().item()
                    min_energy = final_energies.min().item()

                    # --- DINO Spatial Distribution Diagnostics ---
                    dino_entropy = 0.0
                    dino_par = 0.0
                    if (
                        isinstance(obs_dict, dict)
                        and "world_center" in obs_dict
                        and isinstance(obs_dict["world_center"], dict)
                        and "features" in obs_dict["world_center"]
                        and isinstance(obs_dict["world_center"]["features"], dict)
                        and "dino_attn" in obs_dict["world_center"]["features"]
                    ):
                        raw_dino_attn = np.array(
                            obs_dict["world_center"]["features"]["dino_attn"],
                            dtype=np.float32,
                        )
                        dino_prob = raw_dino_attn.flatten() / (
                            np.sum(raw_dino_attn) + 1e-8
                        )
                        dino_entropy = float(
                            -np.sum(dino_prob * np.log(dino_prob + 1e-8))
                        )
                        dino_max = float(np.max(raw_dino_attn))
                        dino_mean = float(np.mean(raw_dino_attn))
                        dino_par = float(dino_max / (dino_mean + 1e-8))

                try:
                    wandb.log(
                        {
                            "steering/candidate_variance": candidate_var,
                            "steering/mean_timeline": mean_timeline,
                            "steering/stable_joints": stable_joints_cnt,
                            "steering/drifting_joints": drifting_joints_cnt,
                            "steering/mean_energy": mean_energy,
                            "steering/min_energy": min_energy,
                            "dino_diagnostics/spatial_entropy": dino_entropy,
                            "dino_diagnostics/peak_to_average_ratio": dino_par,
                        }
                    )
                except Exception:
                    pass

        # Print step trajectories telemetry
        print("--- Stage 3 Step Trajectories ---")
        for i in range(ensemble_size):
            energy_val = final_energies[i].item()
            action_norm = final_actions[i].norm().item()
            print(
                f"  Track {i:02d}: Energy = {energy_val:.10f} | Action Norm = {action_norm:.4f}"
            )
        print("---------------------------------")

        return {
            "action": final_actions.cpu().numpy().tolist(),
            "energy": final_energies.cpu().numpy().tolist(),
            "s_target": s_target.cpu().numpy().tolist(),
            "active_node_key": "skill_0",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Stage 3 step error: {str(e)}")


async def handle_stage3_calibrate(payload: Stage3CalibratePayload):
    try:
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        s_t_first = None
        num_transitions = len(payload.transitions)
        print(f"\n[CALIBRATE] Starting calibration on {num_transitions} transitions...")

        for i, trans in enumerate(payload.transitions):
            print(
                f"[CALIBRATE] Processing transition {i + 1}/{num_transitions} ({(i + 1) / num_transitions * 100:.1f}%)..."
            )
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    _, combined_obs_t = extract_stage3_obs_features(trans.current_obs)
                    _, combined_obs_next = extract_stage3_obs_features(trans.next_obs)

                    # Check for NaNs/Infs in combined features before encoding
                    for name, d_dict in [
                        ("current", combined_obs_t),
                        ("next", combined_obs_next),
                    ]:
                        for k, v in d_dict.items():
                            if torch.is_tensor(v) and (
                                torch.isnan(v).any() or torch.isinf(v).any()
                            ):
                                print(
                                    f"⚠️ [NaN/Inf Warning] {name} observation contains "
                                    f"NaN/Inf in feature: '{k}'!"
                                )

                    # s_t, s_next: Shape [1, 512] (shared state latent dimension)
                    s_t = encode_obs_to_latent(combined_obs_t, state).detach()
                    s_next = encode_obs_to_latent(combined_obs_next, state).detach()

            if s_t_first is None:
                s_t_first = s_t

            # action: Shape [1, 406] (406 = 7 steps * 58 action_dim)
            action = torch.tensor([trans.action_taken], dtype=torch.float32).to(device)
            print(
                f"[CALIBRATE] s_t bounds: [{s_t.min().item():.6f}, "
                f"{s_t.max().item():.6f}] "
                f"s_next bounds: [{s_next.min().item():.6f}, "
                f"{s_next.max().item():.6f}] "
                f"action bounds: [{action.min().item():.6f}, {action.max().item():.6f}]"
            )

            # Retrieve real target goal anchor vector if present, or fall back to s_t
            s_target = torch.tensor(trans.s_target, dtype=torch.float32, device=device)

            # Track the current state, action block, true next state, energy,
            # tactile success, and the target goal anchor
            state.stage3_trajectory_history.append(
                (s_t, action, s_next, trans.energy, trans.tactile, s_target)
            )

        while len(state.stage3_trajectory_history) > 100:
            state.stage3_trajectory_history.pop(0)

        batch_size = min(len(state.stage3_trajectory_history), 8)
        indices = np.random.choice(
            len(state.stage3_trajectory_history), batch_size, replace=False
        )

        # batch_s_t: Shape [batch_size, 512]
        batch_s_t = torch.cat(
            [state.stage3_trajectory_history[idx][0] for idx in indices], dim=0
        )
        # batch_action: Shape [batch_size, 406]
        batch_action = torch.cat(
            [state.stage3_trajectory_history[idx][1] for idx in indices], dim=0
        )
        # batch_s_next: Shape [batch_size, 512]
        batch_s_next = torch.cat(
            [state.stage3_trajectory_history[idx][2] for idx in indices], dim=0
        )

        state.stage3_optimizer.zero_grad()

        # z_action: Shape [batch_size, 512] (projected action trajectory latents)
        z_action = state.stage3_models["action_adapter"](batch_action)

        # z_action_16: Shape [batch_size, 16] (bottleneck dynamics conditioning latent)
        z_action_16 = state.stage3_models["action_down_proj"](z_action)

        # s_next_pred: Shape [batch_size, 512] (predicted macro-step outcome latent state)
        s_next_pred = state.stage3_models["predictor"](batch_s_t, z_action_16)

        # Dynamics loss (predictor update without goal attention) - Scalar loss
        print(
            f"s_next_pred bounds: [{s_next_pred.min().item():.6f}, {s_next_pred.max().item():.6f}]"
        )
        print(
            f"batch_s_next bounds: [{batch_s_next.min().item():.6f}, {batch_s_next.max().item():.6f}]"
        )
        loss_dynamics = F.mse_loss(s_next_pred, batch_s_next)

        # Anti-Collapse Regularization: Enforce diversity on predicted future states
        if s_next_pred.size(0) > 1:
            # Project predicted states onto random directions to check for isotropic distribution
            random_dirs = torch.randn(s_next_pred.size(-1), 10, device=device)
            random_dirs = random_dirs / (random_dirs.norm(dim=0, keepdim=True) + 1e-8)

            # Project the predicted outputs, not the identical inputs
            projected = torch.matmul(s_next_pred, random_dirs)
            mean_proj = projected.mean(dim=0, keepdim=True)

            # Calculate variance safely on the predictor's diverse outputs
            var_proj = torch.clamp(projected.var(dim=0, keepdim=True), min=0.0)
            std_proj = torch.sqrt(var_proj) + 1e-8

            loss_sigreg = F.mse_loss(std_proj, torch.ones_like(std_proj)) + F.mse_loss(
                mean_proj, torch.zeros_like(mean_proj)
            )
            loss_total = loss_dynamics + 0.01 * loss_sigreg
        else:
            loss_total = loss_dynamics

        # Backpropagate the gradients cleanly
        loss_total.backward()

        # Extract parameters of the predictor
        model_obj = state.stage3_models.get("predictor")
        if not model_obj or not hasattr(model_obj, "parameters"):
            raise RuntimeError(
                "🔥 FATAL: The 'predictor' module was not found in state.stage3_models! "
                "Calibration cannot proceed without active dynamics updates."
            )
        trainable_params = list(model_obj.parameters())

        # Apply safety rails to clip exploding gradient updates
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # Step the optimizer forward safely
        state.stage3_optimizer.step()

        # Release fragmented allocation pools
        torch.cuda.empty_cache()

        print(
            f"[Calibrate] Optimization complete. Dynamics loss: {loss_dynamics.item():.6f}"
        )
        if batch_s_t.size(0) > 1:
            print(
                f"            SIGReg loss: {loss_sigreg.item():.6f} | Total loss: {loss_total.item():.6f}\n"
            )

        # Log evaluation telemetry passed from simulation step payload
        if HAS_WANDB and payload.eval_mean_physical_distance is not None:
            ensure_wandb_init()
            if wandb.run is not None:
                try:
                    wandb.log(
                        {
                            "eval/mean_physical_distance": payload.eval_mean_physical_distance,
                            "eval/energy_distance_correlation": payload.eval_energy_distance_correlation
                            or 0.0,
                        }
                    )
                except Exception:
                    pass

        return {"status": "batch_calibrated", "loss": loss_dynamics.item()}
    except Exception as e:
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

        num_opsd_steps = 45
        batch_size = min(len(state.stage3_trajectory_history), 16)
        total_loss = 0.0
        accumulated_diagnostics = []

        for opsd_step in range(num_opsd_steps):
            state.stage3_optimizer.zero_grad()

            # Sample batch from trajectory history
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
            batch_energy = torch.tensor(
                [state.stage3_trajectory_history[idx][3] for idx in indices],
                dtype=torch.float32,
                device=device,
            )
            batch_tactile = torch.tensor(
                [state.stage3_trajectory_history[idx][4] for idx in indices],
                dtype=torch.float32,
                device=device,
            )

            # Retrieve the true baseline step targets saved during the calibration trace
            s_target_batch = torch.cat(
                [state.stage3_trajectory_history[idx][5] for idx in indices], dim=0
            )

            # Generative Flow Matcher Loss (CFM) via Native Blended ComboStoc Method
            B_size = batch_action.size(0)

            # Unflatten back to the true 3D trajectory grid layout [B, 7, 58]
            batch_action_3d = batch_action.view(B_size, 7, 58)

            # RUN 102 FIX: Direct Manifold Target Routing (Bypassing goal_attention cross-attention head)
            s_target_adapted = s_target_batch

            # 2. CFM Flow Matching with True \pi-StepNFT Contrastive Push-Pull Loss
            batch_size = batch_action_3d.size(0)
            embodiment_id = torch.tensor([2], dtype=torch.long, device=device).expand(
                batch_size
            )

            num_solver_steps = 10
            dt = 1.0 / num_solver_steps
            x_0 = torch.randn_like(batch_action_3d)
            target_velocity = batch_action_3d - x_0

            all_denoise_step_losses = []

            # Explicitly evaluate and supervise across ALL 10 discrete denoising solver timesteps
            for k in range(num_solver_steps):
                t_val = k * dt
                t_sample_k = torch.full_like(
                    batch_action_3d, fill_value=t_val, device=device
                )

                # Intermediate noise point x_t and target next solver checkpoint x_next_target at step k
                x_t_k = t_sample_k * batch_action_3d + (1.0 - t_sample_k) * x_0
                x_next_target_k = state.stage3_models[
                    "flow_matcher"
                ].compute_stepwise_denoising_deltas(
                    x_t_k, target_velocity, t_sample_k, dt
                )

                # Query trainable and reference velocity fields at denoise step k
                v_theta_k = state.stage3_models[
                    "flow_matcher"
                ].evaluate_step_transitions(
                    batch_action_3d,
                    t_sample_k,
                    batch_s_t,
                    s_target_adapted,
                    embodiment_id,
                )
                with torch.no_grad():
                    v_old_k = state.stage3_models[
                        "flow_matcher_ref"
                    ].evaluate_step_transitions(
                        batch_action_3d,
                        t_sample_k,
                        batch_s_t,
                        s_target_adapted,
                        embodiment_id,
                    )

                # Velocity drift delta (Δv) and clipped perturbations at denoise step k
                delta_v_k = v_theta_k - v_old_k
                delta_v_norm_k = delta_v_k.norm(dim=(1, 2), keepdim=True) + 1e-8
                delta_v_clipped_k = delta_v_k * torch.clamp(
                    1.0 / delta_v_norm_k, max=1.0
                )

                beta = 0.1
                v_pos_k = v_old_k + beta * delta_v_clipped_k
                v_neg_k = v_old_k - beta * delta_v_clipped_k

                # Step projections at denoise step k
                x_next_pos_k = x_t_k + v_pos_k * dt
                x_next_neg_k = x_t_k + v_neg_k * dt

                # Step reconstruction errors at denoise step k
                err_pos_k = torch.mean(
                    (x_next_target_k - x_next_pos_k) ** 2, dim=(1, 2)
                )
                err_neg_k = torch.mean(
                    (x_next_target_k - x_next_neg_k) ** 2, dim=(1, 2)
                )

                # Softplus preference penalty at denoise step k
                dpo_beta = 1.0
                step_logit_k = (dpo_beta / 2.0) * (err_pos_k - err_neg_k)
                all_denoise_step_losses.append(F.softplus(step_logit_k))

            # Average softplus preference loss across ALL 10 DENOISING STEPS
            mean_step_softplus = torch.stack(all_denoise_step_losses, dim=1).mean(dim=1)

            # Scale across batch elements by continuous JEPA Predictor energy reward
            energy_reward = torch.exp(-batch_energy)
            cfm_loss = torch.mean(mean_step_softplus * energy_reward)

            # 3. Predictor Loss (JEPA dynamics)
            z_action = state.stage3_models["action_adapter"](batch_action)
            z_action_16 = state.stage3_models["action_down_proj"](z_action)
            s_next_pred = state.stage3_models["predictor"](batch_s_t, z_action_16)
            predictor_loss = F.mse_loss(s_next_pred, batch_s_next)

            # 4. CASA (Contrastive Action-State Alignment) loss insulated to Positive (D+) trajectories
            z_s = batch_s_t / (batch_s_t.norm(dim=-1, keepdim=True) + 1e-8)
            z_a = z_action / (z_action.norm(dim=-1, keepdim=True) + 1e-8)
            sim_matrix = torch.matmul(z_s, z_a.T) / 0.07
            labels = torch.arange(sim_matrix.size(0), device=device)
            casa_loss = F.cross_entropy(sim_matrix, labels)

            # 5. Anti-collapse regularization (SIGReg) with Step Decay Schedule
            random_dirs = torch.randn(s_next_pred.size(-1), 10, device=device)
            random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)
            projected = torch.matmul(s_next_pred, random_dirs)
            mean_proj = projected.mean(dim=0, keepdim=True)
            std_proj = projected.std(dim=0, keepdim=True)
            sigreg_loss = F.mse_loss(std_proj, torch.ones_like(std_proj)) + F.mse_loss(
                mean_proj, torch.zeros_like(mean_proj)
            )

            # Persistent Global SIGReg Decay (smooth decay across global run steps, no epoch resets)
            if not hasattr(state, "global_distill_step_count"):
                state.global_distill_step_count = 0
            state.global_distill_step_count += 1

            beta_sig = max(0.001, 0.01 * (0.995**state.global_distill_step_count))

            # Penalty on action magnitude and jerk for overall smoothness
            reg_action_norm = torch.mean(batch_action**2)

            # Multi-scale sliding window jerk regularization (1st and 2nd order deltas over H=7 horizon)
            deltas_1 = batch_action_3d[:, 1:, :] - batch_action_3d[:, :-1, :]
            deltas_2 = batch_action_3d[:, 2:, :] - batch_action_3d[:, :-2, :]
            loss_smoothness = torch.mean(deltas_1**2) + 0.5 * torch.mean(deltas_2**2)

            # Combined total optimization payload for Run 102 (Direct Target Routing, No Goal Attn Losses)
            loss_opsd = (
                cfm_loss
                + casa_loss * 0.2
                + predictor_loss * 0.5
                + sigreg_loss * beta_sig
                + reg_action_norm * 0.0025
                + loss_smoothness * 0.65
            )

            # --- EXTENDED STAGE 1 & 2 PARITY DIAGNOSTIC TELEMETRY ---
            with torch.no_grad():
                state_magnitude = batch_s_t.norm(dim=-1).mean().item()
                state_variance = batch_s_t.var(dim=-1).mean().item()

                action_magnitude = batch_action.norm(dim=-1).mean().item()
                action_steps = batch_action.view(batch_action.size(0), 7, 58)
                action_deltas = action_steps[:, 1:, :] - action_steps[:, :-1, :]
                action_smoothness = action_deltas.abs().mean().item()

                # Core Parity Metrics from Stage 1/2 Checkpoints
                identity_error = F.mse_loss(batch_s_t, batch_s_next).item()
                noop_ratio = predictor_loss.item() / max(identity_error, 1e-6)

                z_random = torch.randn_like(z_action_16)
                s_next_pred_rand = state.stage3_models["predictor"](batch_s_t, z_random)
                action_drift = F.mse_loss(s_next_pred, s_next_pred_rand).item()

            # Read the cached activation profiles from the MSAT instance layer
            msat_profile = getattr(
                state.stage3_models["msat"], "last_modality_profile", {}
            )

            diagnostics = {
                "epoch_step": opsd_step,
                "loss/cfm_loss": cfm_loss.item(),
                "loss/casa_loss": casa_loss.item(),
                "loss/sigreg_loss": sigreg_loss.item(),
                "loss/predictor_loss": predictor_loss.item(),
                "drift/state_magnitude": state_magnitude,
                "drift/state_variance": state_variance,
                "policy/action_magnitude": action_magnitude,
                "policy/action_smoothness": action_smoothness,
                "metrics/noop_loss_ratio": noop_ratio,
                "metrics/action_drift": action_drift,
                **msat_profile,
            }
            accumulated_diagnostics.append(diagnostics)

            if HAS_WANDB:
                ensure_wandb_init()
                if wandb.run is not None:
                    try:
                        wandb.log(diagnostics)
                    except Exception:
                        pass

            loss_opsd.backward()
            state.stage3_optimizer.step()
            total_loss += loss_opsd.item()

        # Release fragmented allocation pools
        torch.cuda.empty_cache()

        avg_loss = total_loss / num_opsd_steps

        # Compute and print mean diagnostics across all completed distillation steps
        mean_diagnostics = {}
        if accumulated_diagnostics:
            for k in accumulated_diagnostics[0].keys():
                if k == "epoch_step":
                    continue
                vals = [
                    d[k]
                    for d in accumulated_diagnostics
                    if k in d and isinstance(d[k], (int, float))
                ]
                if vals:
                    mean_diagnostics[k] = float(np.mean(vals))

        print(
            f"📊 [DISTILL SUMMARY] Completed {num_opsd_steps} steps | Mean Distill Loss: {avg_loss:.6f}"
        )
        print(
            f"📊 Mean OPSD Diagnostics Summary:\n{json.dumps(mean_diagnostics, indent=2)}"
        )

        # Check and run exemplar diagnostic checks if any saved exemplars exist in EXEMPLAR_DIR
        if os.path.exists(EXEMPLAR_DIR):
            phase_0_frames = None
            phase_0_fp = os.path.join(EXEMPLAR_DIR, "phase_0.pkl")
            if os.path.exists(phase_0_fp):
                with open(phase_0_fp, "rb") as f:
                    raw_0 = pickle.load(f)
                    if isinstance(raw_0, dict):
                        phase_0_frames = raw_0.get("frames")
                    else:
                        phase_0_frames = getattr(raw_0, "frames", None)

            eval_payloads = {}
            for fn in sorted(os.listdir(EXEMPLAR_DIR)):
                if fn.endswith(".pkl") and fn != "phase_0.pkl":
                    name = fn[:-4]
                    fp = os.path.join(EXEMPLAR_DIR, fn)
                    with open(fp, "rb") as f:
                        raw_payload = pickle.load(f)
                        if isinstance(raw_payload, dict):
                            curr_frames = raw_payload.get("frames", {})
                            base_frames = (
                                phase_0_frames
                                if phase_0_frames is not None
                                else curr_frames
                            )
                            raw_payload["history_frames"] = [
                                base_frames,
                                curr_frames,
                            ]
                            raw_payload = Stage3StepPayload(**raw_payload)
                        else:
                            curr_frames = getattr(raw_payload, "frames", {})
                            base_frames = (
                                phase_0_frames
                                if phase_0_frames is not None
                                else curr_frames
                            )
                            raw_payload.history_frames = [base_frames, curr_frames]
                        eval_payloads[name] = raw_payload
            if eval_payloads:
                run_exemplar_diagnostic_check(s_target_batch[0], state, eval_payloads)

        # Synchronize frozen reference snapshot model with newly learned weights
        state.stage3_models["flow_matcher_ref"].load_state_dict(
            state.stage3_models["flow_matcher"].state_dict()
        )

        # ... Rest of checkpoint saving code remains identical ...
        checkpoint_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", config["paths"]["checkpoint_dir"]
            )
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, "stage3_rl_final.pt")

        checkpoint_payload = {
            "flow_matcher": state.stage3_models["flow_matcher"].state_dict(),
            "vis_adapter": state.stage3_models["vis_adapter"].state_dict(),
            "txt_adapter": state.stage3_models["txt_adapter"].state_dict(),
            "pt_adapter": state.stage3_models["pt_adapter"].state_dict(),
            "vggt_adapter": state.stage3_models["vggt_adapter"].state_dict(),
            "tactile_adapter": state.stage3_models["tactile_adapter"].state_dict(),
            "msat": state.stage3_models["msat"].state_dict(),
            "action_adapter": state.stage3_models["action_adapter"].state_dict(),
            "state_adapter": state.stage3_models["state_adapter"].state_dict(),
            "action_down_proj": state.stage3_models["action_down_proj"].state_dict(),
            "predictor": state.stage3_models["predictor"].state_dict(),
            "goal_attention": state.stage3_models["goal_attention"].state_dict(),
            "latent_adapter": state.stage3_models["latent_adapter"].state_dict(),
            "gnn_nodes": state.stage3_models["gnn_library"].nodes.state_dict(),
            "gnn_specialists": state.stage3_models[
                "gnn_library"
            ].specialists.state_dict(),
        }

        # 1. Save final rolling checkpoint
        torch.save(checkpoint_payload, final_path)

        # 2. Multi-Epoch Checkpointing: Save intermediate checkpoint every 2 epochs
        if not hasattr(state, "epoch_counter"):
            state.epoch_counter = 0
        state.epoch_counter += 1
        if state.epoch_counter % 2 == 0:
            epoch_ckpt_path = os.path.join(
                checkpoint_dir, f"stage3_epoch_{state.epoch_counter:02d}.pt"
            )
            torch.save(checkpoint_payload, epoch_ckpt_path)
            print(
                f"💾 [Checkpoint] Persisted dynamic epoch checkpoint: {epoch_ckpt_path}"
            )

        return {"status": "distilled", "opsd_loss": avg_loss, "checkpoint": final_path}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Stage 3 distillation error: {str(e)}"
        )
