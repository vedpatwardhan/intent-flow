from time import perf_counter
import threading
import uuid
import pickle
import json
import os
import yaml
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel

EXEMPLAR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "exemplars")
)
from fastapi import HTTPException
import matplotlib

matplotlib.use("Agg")
import traceback
import torch
import torch.nn.functional as F

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
from colab_server_pkg.feature_extractor import (
    extract_stage3_obs_features,
    extract_batch_stage3_obs_features,
)
from colab_server_pkg.image_utils import save_stage3_obs_feature_plots

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


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D) -> Sequence length, Batch size, Latent dimension
        """
        # sample random projections dynamically
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time steps


from trainers.stage3.trainer import GNNSkillLibrary


class Stage3StepPayload(BaseModel):
    frames: dict[str, str]  # Multi-view frames: {camera_name: base64_image}
    history_frames: list[dict[str, str]]
    proprioception: list[float]
    tactile: list[list[float]]
    text_prompt: str
    ui_annotations: dict
    pos_trajectories: list[dict]
    episode_idx: int
    step_idx: int
    is_easy_task: bool
    eval_mean_physical_distance: float
    eval_median_physical_distance: float
    eval_min_physical_distance: float
    eval_energy_distance_correlation: float
    point_clouds: dict | None = None


class Stage3CalibrateTransition(BaseModel):
    current_obs: Stage3StepPayload
    action_taken: list[float]
    next_obs: Stage3StepPayload
    energy: float
    tactile: float
    s_target: list[list[float]]


class Stage3CalibratePayload(BaseModel):
    transitions: list[Stage3CalibrateTransition]
    episode_idx: int
    step_idx: int


class Stage3DistillPayload(BaseModel):
    reward: float  # placeholder for the tactile stuff
    pos_trajectories: list[dict]
    neg_trajectories: list[dict]


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
    Computes direct latent distance from a set of fixed environment states to the current target anchor s_target [1, 512].
    eval_payloads: Dict containing pre-captured Near-Goal, Mid-Phase, and OOD observation structures.
    """
    diagnostic_distances = {}
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            payloads = list(eval_payloads.values())
            obs_dict, combined_obs = extract_batch_stage3_obs_features(payloads)
            s_encoded = encode_obs_to_latent(combined_obs, state)  # Shape [K, 512]

            # Calculate mean squared distance directly to single target anchor [1, 512]
            dist = torch.mean((s_encoded - s_target) ** 2, dim=1)  # Shape [K]
            for idx, name in enumerate(eval_payloads):
                diagnostic_distances[f"exemplar_distance/{name}"] = dist[idx].item()

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
    # state.stage3_models["txt_adapter"] = TextAdapter(d_in=384).to(device)
    # state.stage3_models["pt_adapter"] = PointNeXtAdapter(d_in=384).to(device)
    state.stage3_models["vggt_adapter"] = VGGTAdapter(
        d_in=config["model"]["vggt_dim"]
    ).to(device)
    # state.stage3_models["tactile_adapter"] = TactileAdapter().to(device)
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

    # Instantiate the new Epps-Pulley statistical sketch regularizer
    state.stage3_models["sigreg_module"] = SIGReg(knots=17, num_proj=1024).to(device)

    state.stage3_models["flow_matcher"] = ComboStocFlowMatcher(
        action_dim=action_dim, config=config
    ).to(device)
    # if "flow_matcher_ref" not in state.stage3_models:
    #     state.stage3_models["flow_matcher_ref"] = ComboStocFlowMatcher(
    #         action_dim=action_dim, config=config
    #     ).to(device)
    #     state.stage3_models["flow_matcher_ref"].load_state_dict(
    #         state.stage3_models["flow_matcher"].state_dict()
    #     )
    #     state.stage3_models["flow_matcher_ref"].eval()

    # state.stage3_models["gnn_library"] = GNNSkillLibrary(
    #     state.stage3_models["flow_matcher"], state_dim=latent_dim
    # ).to(device)

    # state.stage3_models["discriminator"] = TrajectoryDiscriminator(
    #     action_dim=action_dim
    # ).to(device)

    # state.stage3_models["attacker"] = BadWorldAttacker(action_dim=action_dim)

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
        # state.stage3_models["txt_adapter"].load_state_dict(checkpoint["txt_adapter"])
        # state.stage3_models["pt_adapter"].load_state_dict(checkpoint["pt_adapter"])
        state.stage3_models["vggt_adapter"].load_state_dict(checkpoint["vggt_adapter"])
        # state.stage3_models["tactile_adapter"].load_state_dict(
        #     checkpoint["tactile_adapter"]
        # )
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
            # "tactile_adapter",
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
        # + list(state.stage3_models["gnn_library"].parameters())
        + list(state.stage3_models["predictor"].parameters())
        # + list(state.stage3_models["discriminator"].parameters())
        + list(state.stage3_models["action_adapter"].parameters())
        + list(state.stage3_models["state_adapter"].parameters())
        + list(state.stage3_models["action_down_proj"].parameters()),
        lr=stage3_lr,
    )


async def handle_stage3_step(payload: Stage3StepPayload):
    # Called on every step of every epoch
    try:
        torch.cuda.empty_cache()
        # instantiates all models and loads parameters from checkpoints
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        # get all global and filtered features for current live observation
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                obs_dict, combined_obs = extract_batch_stage3_obs_features([payload])
                # Get clean live state latent: Shape [1, 512]
                s_t = encode_obs_to_latent(combined_obs, state)
                print(f"s_t shape: {s_t.shape}")

                # Encode positive IK trajectories into a target bank in 1 batched GPU call: Shape [M, 512]
                tr_obs_payloads = [
                    Stage3StepPayload(
                        frames=tr["frames"],
                        history_frames=tr["history_frames"],
                        proprioception=tr["proprioception"],
                        tactile=payload.tactile,
                        text_prompt=payload.text_prompt,
                        ui_annotations=payload.ui_annotations,
                        pos_trajectories=[],
                        episode_idx=0,
                        step_idx=0,
                        is_easy_task=True,
                        eval_mean_physical_distance=0,
                        eval_median_physical_distance=0,
                        eval_min_physical_distance=0,
                        eval_energy_distance_correlation=0,
                    )
                    for tr in payload.pos_trajectories
                ]
                tr_obs_dicts_batch, tr_combined_obs_batch = (
                    extract_batch_stage3_obs_features(tr_obs_payloads)
                )

                # Save 4-panel diagnostic plot for first goal trajectory observation
                if payload.episode_idx == 0 and payload.step_idx == 0:
                    save_stage3_obs_feature_plots(
                        history_frames=tr_obs_payloads[0].history_frames,
                        obs_features=tr_obs_dicts_batch[0],
                        title_prefix="Goal Trajectory 0",
                        output_filename="debug_goal_tr_0_world_center.png",
                        view_name="world_center",
                    )

                # Shape [M, 512]
                s_target_bank = encode_obs_to_latent(tr_combined_obs_batch, state)
                print(f"s_target_bank shape: {s_target_bank.shape}")

        # Initialize the 2D Space-Time Grid (Horizon=7, Joints=58)
        horizon = 7
        joint_dim = 58
        total_gen_dim = horizon * joint_dim

        grid = torch.zeros(1, horizon, joint_dim, device=device)  # Shape [1, 7, 58]
        steering_timelines = grid.view(1, total_gen_dim)  # Shape [1, 406]

        s_t = s_t.detach()  # Shape [1, 512]
        embodiment_id = torch.tensor([2], dtype=torch.long, device=device)  # Shape [1]

        ensemble_size = 16
        s_t_ensemble = s_t.expand(ensemble_size, -1)  # Shape [16, 512]
        s_target_conditioned = torch.max(s_target_bank, dim=0)[0].unsqueeze(
            0
        )  # Shape [1, 512]

        embodiment_id_expanded = embodiment_id.expand(ensemble_size)  # Shape [16]

        steering_timelines_expanded = (
            steering_timelines.expand(ensemble_size, -1)
            .view(ensemble_size, horizon, joint_dim)
            .clone()
        )  # Shape [16, 7, 58]

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                a_candidates, step_snrs = state.stage3_models[
                    "flow_matcher"
                ].sample_with_steering(
                    s_t_ensemble,
                    s_target_conditioned.expand(ensemble_size, -1),  # [16, 512]
                    embodiment_id=embodiment_id_expanded,
                    horizon=horizon,
                    num_steps=10,
                    steering_timelines=steering_timelines_expanded,
                    step_nft_scale=0.3,
                )  # a_candidates Shape [16, 7, 58]

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

        # Create embodiment-aware action mask (first 32 GR-1 active joints)
        action_mask = torch.zeros(1, 1, joint_dim, device=device)  # Shape [1, 1, 58]
        action_mask[..., :32] = 1.0

        # Mask initial candidate actions to ensure padding channels start at zero
        a_candidates = a_candidates * action_mask  # Shape [16, 7, 58]

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
            a_flat = a_candidates.view(ensemble_size, -1)  # Shape [16, 406]

            # Get the action representation and next latent state
            z_action = state.stage3_models["action_adapter"](a_flat)  # Shape [16, 512]
            z_action_16 = state.stage3_models["action_down_proj"](
                z_action
            )  # Shape [16, 16]

            s_next_pred = state.stage3_models["predictor"](
                s_t_ensemble, z_action_16
            )  # Shape [16, 512]

            # Multi-target energy computation: distance to closest goal in s_target_bank [M, 512]
            # s_next_pred: Shape [16, 512] -> [16, 1, 512]
            # s_target_bank: Shape [M, 512] -> [1, M, 512]
            dists_to_bank = torch.mean(
                (s_next_pred.unsqueeze(1) - s_target_bank.unsqueeze(0)) ** 2, dim=-1
            )  # Shape [16, M]
            min_bank_dists, _ = torch.min(dists_to_bank, dim=1)  # Shape [16]
            energy = torch.mean(min_bank_dists)
            grad_a = torch.autograd.grad(energy, a_candidates)[0]  # Shape [16, 7, 58]

            # Normalize gradients per ensemble instance and mask padding channels
            # Shape [16, 1, 1]
            grad_norm = grad_a.norm(dim=(1, 2), keepdim=True) + 1e-8
            grad_a_normalized = (grad_a / grad_norm) * action_mask  # Shape [16, 7, 58]

            # Masked Energy Guidance matching 3D coordinates
            # Dynamically calculate the guidance mask (shape: [16, 7, 58])
            t_j = 1.0 - steering_timelines_expanded

            # Sculpt action parameters along the tracking vector field
            diff = eta * grad_a_normalized * t_j  # Shape [16, 7, 58]
            a_candidates = (a_candidates - diff) * action_mask  # Shape [16, 7, 58]

            # Raw gradient forces per joint: Mean over ensemble (dim 0) and horizon (dim 1) -> Shape: [58]
            raw_joint_errors = grad_a.abs().mean(dim=(0, 1))

            # Store initial unsteered raw gradient forces at iteration k == 0
            if k == 0:
                g_0_joint_errors = raw_joint_errors.clone().detach()

            # Isolate active structural dimensions (first 32 joints)
            active_selector = action_mask.squeeze(0).squeeze(0) > 0  # Shape [58]

            # Relative Force Convergence Ratio:
            # Joint j is STABLE if its current gradient force has dropped to <= 40% of its initial force at iteration 0
            # (i.e. >= 60% force reduction / convergence across lookahead steering)
            alpha = 0.40
            relative_ratio = raw_joint_errors / (g_0_joint_errors + 1e-8)  # Shape [58]

            # Stable and drifting masks scoped strictly to active joints
            stable_joints_mask = (
                relative_ratio <= alpha
            ) & active_selector  # Shape [58]
            drifting_joints_mask = (~stable_joints_mask) & active_selector  # Shape [58]

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
            )  # Shape [16, 7, 58]

        # Detach the candidates after the loop
        final_actions = a_candidates.clone().detach()  # Shape [16, 7, 58]

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
            final_energies = torch.mean(
                (s_next_pred.unsqueeze(1) - s_target_bank.unsqueeze(0)) ** 2, dim=-1
            )  # Shape [16, 5]
            final_energies = torch.min(final_energies, dim=1)[0]  # [16]

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
                    log_payload = {
                        "telemetry/candidate_variance": candidate_var,
                        "telemetry/mean_timeline_progress": mean_timeline,
                        "telemetry/stable_joints_count": stable_joints_cnt,
                        "telemetry/drifting_joints_count": drifting_joints_cnt,
                        "telemetry/mean_candidate_energy": mean_energy,
                        "telemetry/min_candidate_energy": min_energy,
                        "telemetry/dino_attn_entropy": dino_entropy,
                        "telemetry/dino_attn_par": dino_par,
                    }
                    if payload.eval_mean_physical_distance is not None:
                        log_payload.update(
                            {
                                "eval/mean_physical_distance": payload.eval_mean_physical_distance,
                                "eval/median_physical_distance": payload.eval_median_physical_distance,
                                "eval/min_physical_distance": payload.eval_min_physical_distance,
                                "eval/energy_distance_correlation": payload.eval_energy_distance_correlation,
                            }
                        )
                    wandb.log(log_payload)
                except Exception:
                    pass

        # Return action array [16, 7, 58] and energy vector [16] to client
        return {
            "action": final_actions.cpu().tolist(),  # [16, 7, 58]
            "energy": final_energies.cpu().tolist(),  # [16]
            "s_target": s_target_conditioned.cpu().tolist(),  # [1, 512]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Stage 3 step error: {str(e)}")


import uuid

calibration_jobs = {}


def run_calibration_worker(job_id: str, payload: Stage3CalibratePayload):
    try:
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        num_trans = len(payload.transitions)
        print(
            f"⚡ [CALIBRATE JOB {job_id}] Batched processing across {num_trans} transitions..."
        )

        current_obs_list = [trans.current_obs for trans in payload.transitions]
        next_obs_list = [trans.next_obs for trans in payload.transitions]

        # 1. Batched Feature Extraction & Latent Encoding (No Gradient Tracking + Mixed Precision)
        start_time = perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                _, combined_obs_t_batch = extract_batch_stage3_obs_features(
                    current_obs_list
                )
                obs_dict_next_batch, combined_obs_next_batch = (
                    extract_batch_stage3_obs_features(next_obs_list)
                )

                # Save 4-panel diagnostic plot for first transition outcome features
                if payload.episode_idx == 0 and payload.step_idx == 3:
                    save_stage3_obs_feature_plots(
                        history_frames=payload.transitions[0].next_obs.history_frames,
                        obs_features=obs_dict_next_batch[0],
                        title_prefix="Calibrate Track 0",
                        output_filename="debug_calibrate_track_0_world_center.png",
                        view_name="world_center",
                    )

                # 2. Batched MSAT GPU Latent Encoding across all transitions
                # Shape [B, 512]
                s_t_batch = encode_obs_to_latent(combined_obs_t_batch, state).detach()
                # Shape [B, 512]
                s_next_batch = encode_obs_to_latent(
                    combined_obs_next_batch, state
                ).detach()

        print(
            "time taken for feature extraction & latent encoding: "
            f"{perf_counter() - start_time:.4f}s"
        )

        for idx, trans in enumerate(payload.transitions):
            s_t = s_t_batch[idx : idx + 1]  # Shape [1, 512]
            s_next = s_next_batch[idx : idx + 1]  # Shape [1, 512]

            # action: Shape [1, 406] (406 = 7 steps * 58 action_dim)
            action = torch.tensor([trans.action_taken], dtype=torch.float32).to(device)

            # Retrieve real target goal anchor vector if present, or fall back to s_t: Shape [1, 512]
            s_target = torch.tensor(trans.s_target, dtype=torch.float32, device=device)

            # Track current state, action, next state, energy, tactile, s_target, and is_positive_trajectory label
            state.stage3_trajectory_history.append(
                (
                    s_t,
                    action,
                    s_next,
                    trans.energy,
                    trans.tactile,
                    s_target,
                )
            )

        while len(state.stage3_trajectory_history) > 2000:
            print(
                f"[CALIBRATE] WARNING: Trajectory history is too long! Current size: "
                f"{len(state.stage3_trajectory_history)}. "
                "Popping oldest trajectories..."
            )
            state.stage3_trajectory_history.pop(0)

        num_transitions = len(payload.transitions)
        loss_dynamics_accum = 0.0
        loss_sigreg_accum = 0.0
        loss_total_accum = 0.0

        for grad_step in range(num_transitions):
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

            loss_dynamics = F.mse_loss(s_next_pred, batch_s_next)

            # Anti-Collapse Regularization: Enforce diversity on predicted future states
            loss_sigreg = state.stage3_models["sigreg_module"](s_next_pred.unsqueeze(0))
            loss_total = loss_dynamics + 0.03 * loss_sigreg

            loss_total.backward()

            model_obj = state.stage3_models.get("predictor")
            if not model_obj or not hasattr(model_obj, "parameters"):
                raise RuntimeError(
                    "🔥 FATAL: The 'predictor' module was not found in state.stage3_models! "
                    "Calibration cannot proceed without active dynamics updates."
                )
            trainable_params = list(model_obj.parameters())

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            state.stage3_optimizer.step()

            loss_dynamics_accum += loss_dynamics.item()
            loss_sigreg_accum += loss_sigreg.item()
            loss_total_accum += loss_total.item()

        torch.cuda.empty_cache()

        avg_loss_dynamics = loss_dynamics_accum / num_transitions
        avg_loss_sigreg = loss_sigreg_accum / num_transitions
        avg_loss_total = loss_total_accum / num_transitions

        print(
            f"[Calibrate] Ingested {num_transitions} transitions across {num_transitions} optimization steps. Avg Dynamics Loss: {avg_loss_dynamics:.6f} | Avg SIGReg: {avg_loss_sigreg:.6f} | Avg Total: {avg_loss_total:.6f}\n"
        )

        calibration_jobs[job_id] = {
            "status": "completed",
            "loss": float(avg_loss_dynamics),
            "error": None,
        }
    except Exception as e:
        traceback.print_exc()
        calibration_jobs[job_id] = {
            "status": "failed",
            "loss": None,
            "error": str(e),
        }


async def handle_stage3_calibrate(payload: Stage3CalibratePayload):
    job_id = f"calib_{uuid.uuid4().hex[:10]}"
    calibration_jobs[job_id] = {"status": "processing", "loss": None, "error": None}
    threading.Thread(
        target=run_calibration_worker, args=(job_id, payload), daemon=True
    ).start()
    return {"job_id": job_id, "status": "processing"}


async def get_calibration_job_status(job_id: str):
    if job_id not in calibration_jobs:
        raise HTTPException(status_code=404, detail="Calibration job ID not found")
    return calibration_jobs[job_id]


distill_jobs = {}


def run_distill_worker(job_id: str, payload: Stage3DistillPayload):
    try:
        ensure_stage3_models()
        import colab_server_pkg.models_state as state

        if len(state.stage3_trajectory_history) == 0:
            distill_jobs[job_id] = {
                "status": "completed",
                "opsd_loss": 0.0,
                "error": None,
                "message": "no data to distill",
            }
            return

        config_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "config", "default_config.yaml"
            )
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Deterministic Epoch Partitioning: Isolate fresh rollout transitions (up to 400)
        total_history_len = len(state.stage3_trajectory_history)
        print(f"[DISTILL JOB {job_id}] Total History Length: {total_history_len}")
        fresh_data_size = min(total_history_len, 400)
        fresh_indices = np.arange(
            total_history_len - fresh_data_size, total_history_len
        )
        np.random.shuffle(fresh_indices)

        target_batch_size = 16
        batches_pool = [
            fresh_indices[i : i + target_batch_size]
            for i in range(0, len(fresh_indices), target_batch_size)
        ]

        num_opsd_steps = len(batches_pool)
        print(
            f"📊 [Epoch Partition] Processing exactly {num_opsd_steps} non-overlapping "
            f"batches across {fresh_data_size} fresh transitions. Parity guaranteed."
        )

        total_loss = 0.0
        accumulated_diagnostics = []

        opsd_pbar = tqdm(
            range(num_opsd_steps), desc=f"⚡ [OPSD Distill {job_id}]", unit="batch"
        )
        for opsd_step in opsd_pbar:
            state.stage3_optimizer.zero_grad()

            indices = batches_pool[opsd_step]
            # batch_s_t: Shape [B, 512]
            batch_s_t = torch.cat(
                [state.stage3_trajectory_history[idx][0] for idx in indices], dim=0
            )
            # batch_action: Shape [B, 406]
            batch_action = torch.cat(
                [state.stage3_trajectory_history[idx][1] for idx in indices], dim=0
            )
            # batch_s_next: Shape [B, 512]
            batch_s_next = torch.cat(
                [state.stage3_trajectory_history[idx][2] for idx in indices], dim=0
            )
            # batch_energy: Shape [B]
            batch_energy = torch.tensor(
                [state.stage3_trajectory_history[idx][3] for idx in indices],
                dtype=torch.float32,
                device=device,
            )
            # batch_tactile: Shape [B]
            batch_tactile = torch.tensor(
                [state.stage3_trajectory_history[idx][4] for idx in indices],
                dtype=torch.float32,
                device=device,
            )

            # Retrieve the true baseline step targets saved during the calibration trace: Shape [B, 512]
            batch_s_target = torch.cat(
                [state.stage3_trajectory_history[idx][5] for idx in indices], dim=0
            ).to(device, dtype=torch.float32)

            # Initialize data for flow matcher
            B_size = batch_action.size(0)
            batch_action_3d = batch_action.view(B_size, 7, 58)  # Shape [B, 7, 58]
            embodiment_id = torch.tensor([2], dtype=torch.long, device=device).expand(
                B_size
            )
            x_0 = torch.randn_like(batch_action_3d)  # Shape [B, 7, 58]
            target_velocity = batch_action_3d - x_0  # Shape [B, 7, 58]
            t_sample = torch.full_like(batch_action_3d, fill_value=0.5, device=device)

            # Evaluate model vector field output
            v_theta = state.stage3_models["flow_matcher"].evaluate_step_transitions(
                batch_action_3d, t_sample, batch_s_t, batch_s_target, embodiment_id
            )
            cfm_loss = F.mse_loss(v_theta, target_velocity)

            # Extract the raw IK trajectory payloads from our history indices
            pos_actions = torch.tensor(
                [tr["actions"] for tr in payload.pos_trajectories],
                dtype=torch.float32,
                device=device,
            ).view(-1, 7, 58)
            neg_actions = torch.tensor(
                [tr["actions"] for tr in payload.neg_trajectories],
                dtype=torch.float32,
                device=device,
            ).view(-1, 7, 58)

            # Get velocities for both trajectories
            pos_vel = pos_actions.unsqueeze(0) - x_0.unsqueeze(1)  # Shape [B, 5, 7, 58]
            neg_vel = neg_actions.unsqueeze(0) - x_0.unsqueeze(1)  # Shape [B, 8, 7, 58]

            # Expand model prediction to [B, 1, 7, 58] to get errors
            err_pos = torch.mean(
                (v_theta.unsqueeze(1) - pos_vel) ** 2, dim=(-2, -1)
            ).min(dim=1)[0]
            err_neg = torch.mean(
                (v_theta.unsqueeze(1) - neg_vel) ** 2, dim=(-2, -1)
            ).min(dim=1)[0]

            # Hinge loss formulation: push away from distractor velocities by a margin of 1.0
            margin = 1.0
            contrastive_loss = torch.mean(
                err_pos + torch.clamp(margin - err_neg, min=0.0)
            )

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
            sigreg_loss = state.stage3_models["sigreg_module"](s_next_pred.unsqueeze(0))

            # Penalty on action magnitude and jerk for overall smoothness
            reg_action_norm = torch.mean(batch_action**2)

            # 6. Smoothness regularization loss penalty on output joint space deltas
            # Multi-scale sliding window jerk regularization (1st and 2nd order deltas over H=7 horizon)
            deltas_1 = batch_action_3d[:, 1:, :] - batch_action_3d[:, :-1, :]
            deltas_2 = batch_action_3d[:, 2:, :] - batch_action_3d[:, :-2, :]
            loss_smoothness = torch.mean(deltas_1**2) + 0.5 * torch.mean(deltas_2**2)

            # Combined total optimization payload for Run 107
            loss_opsd = (
                cfm_loss
                + contrastive_loss * 0.5
                + casa_loss * 0.2
                + predictor_loss * 0.5
                + sigreg_loss * 0.03
                + reg_action_norm * 0.0045
                + loss_smoothness * 0.35
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

            opsd_pbar.set_postfix(
                {
                    "loss": f"{loss_opsd.item():.4f}",
                    "cfm": f"{cfm_loss.item():.4f}",
                    "pred": f"{predictor_loss.item():.4f}",
                    "casa": f"{casa_loss.item():.4f}",
                }
            )

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
                    phase_0_frames = raw_0.get("frames")

            eval_payloads = {}
            for fn in sorted(os.listdir(EXEMPLAR_DIR)):
                if fn.endswith(".pkl") and fn != "phase_0.pkl":
                    name = fn[:-4]
                    fp = os.path.join(EXEMPLAR_DIR, fn)
                    with open(fp, "rb") as f:
                        raw_payload = pickle.load(f)
                        curr_frames = raw_payload.get("frames", {})
                        base_frames = (
                            phase_0_frames
                            if phase_0_frames is not None
                            else curr_frames
                        )
                        raw_payload["history_frames"] = [
                            base_frames,
                            base_frames,
                            curr_frames,
                            curr_frames,
                        ]
                        eval_payloads[name] = Stage3StepPayload(
                            **raw_payload,
                            eval_mean_physical_distance=0,
                            eval_median_physical_distance=0,
                            eval_min_physical_distance=0,
                            eval_energy_distance_correlation=0,
                        )
            if eval_payloads:
                run_exemplar_diagnostic_check(
                    batch_s_target[0].unsqueeze(0), state, eval_payloads
                )

        # Synchronize frozen reference snapshot model with newly learned weights
        # if "flow_matcher_ref" in state.stage3_models:
        #     state.stage3_models["flow_matcher_ref"].load_state_dict(
        #         state.stage3_models["flow_matcher"].state_dict()
        #     )

        # Cleared the buffer
        state.stage3_trajectory_history.clear()

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
            # "txt_adapter": state.stage3_models["txt_adapter"].state_dict(),
            # "pt_adapter": state.stage3_models["pt_adapter"].state_dict(),
            "vggt_adapter": state.stage3_models["vggt_adapter"].state_dict(),
            # "tactile_adapter": state.stage3_models["tactile_adapter"].state_dict(),
            "msat": state.stage3_models["msat"].state_dict(),
            "action_adapter": state.stage3_models["action_adapter"].state_dict(),
            "state_adapter": state.stage3_models["state_adapter"].state_dict(),
            "action_down_proj": state.stage3_models["action_down_proj"].state_dict(),
            "predictor": state.stage3_models["predictor"].state_dict(),
            # "gnn_nodes": state.stage3_models["gnn_library"].nodes.state_dict(),
            # "gnn_specialists": state.stage3_models[
            #     "gnn_library"
            # ].specialists.state_dict(),
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

        distill_jobs[job_id] = {
            "status": "completed",
            "opsd_loss": float(avg_loss),
            "checkpoint": final_path,
            "error": None,
        }
    except Exception as e:
        traceback.print_exc()
        distill_jobs[job_id] = {
            "status": "failed",
            "opsd_loss": None,
            "checkpoint": None,
            "error": str(e),
        }


async def handle_stage3_distill(payload: Stage3DistillPayload):
    job_id = f"distill_{uuid.uuid4().hex[:10]}"
    distill_jobs[job_id] = {"status": "processing", "opsd_loss": None, "error": None}
    threading.Thread(
        target=run_distill_worker, args=(job_id, payload), daemon=True
    ).start()
    return {"job_id": job_id, "status": "processing"}


async def get_distill_job_status(job_id: str):
    if job_id not in distill_jobs:
        raise HTTPException(status_code=404, detail="Distill job ID not found")
    return distill_jobs[job_id]
