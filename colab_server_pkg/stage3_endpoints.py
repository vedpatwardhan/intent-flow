import os
import yaml
import numpy as np
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException
from PIL import Image
import traceback
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from colab_server_pkg.config import device
from colab_server_pkg.models_state import (
    stage3_models,
    stage3_trajectory_history,
    models,
)
from colab_server_pkg.feature_extractor import extract_stage3_obs_features
from colab_server_pkg.image_utils import decode_base64_image

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
    frames: dict  # Multi-view frames: {camera_name: base64_image}
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


def construct_goal_states(obs_dict, ui_annotations):
    """
    Construct goal state representations by rearranging crops/segments according to arrows.
    For 2 patches with 1 arrow, generates 4 positional variants (left, right, top, bottom).
    """
    view_features = obs_dict.get("view_features", {})
    if not view_features:
        return []

    primary_view = "world_center"
    if primary_view not in view_features:
        primary_view = list(view_features.keys())[0]

    features = view_features[primary_view]
    task_isolated = features.get("task_isolated_features", {})
    combined_mask_224 = task_isolated.get("combined_mask_224", np.zeros((224, 224)))

    # Get the primary frame
    frames_dict = obs_dict.get("view_features", {})
    if not frames_dict:
        return []

    # Extract crops from annotations
    crops = ui_annotations.get("crops", [])
    vectors = ui_annotations.get("vectors", [])

    if len(crops) < 2:
        # If no crops defined, use the combined mask region
        mask_indices = np.where(combined_mask_224 > 0)
        if len(mask_indices[0]) == 0:
            return []

        y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
        x_min, x_max = mask_indices[1].min(), mask_indices[1].max()

        # Create two crops from the mask region
        crops = [
            {
                "x": x_min,
                "y": y_min,
                "width": (x_max - x_min) // 2,
                "height": y_max - y_min,
            },
            {
                "x": x_min + (x_max - x_min) // 2,
                "y": y_min,
                "width": (x_max - x_min) // 2,
                "height": y_max - y_min,
            },
        ]

    # Get the primary frame from the first view
    primary_frame_str = None
    for view_name, frame_str in frames_dict.items():
        if primary_frame_str is None:
            primary_frame_str = frame_str
            break

    if primary_frame_str is None:
        return []

    frame = decode_base64_image(primary_frame_str)
    pil_frame = Image.fromarray(frame)

    # Extract the two crop regions
    crop_regions = []
    for crop in crops[:2]:  # Limit to 2 crops for single skill
        x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
        crop_region = pil_frame.crop((x, y, x + w, y + h))
        crop_regions.append(crop_region)

    if len(crop_regions) < 2:
        return []

    # Generate 4 positional arrangements
    goal_states = []
    arrangements = [
        ("left_right", 0, 1),  # crop0 left of crop1
        ("right_left", 1, 0),  # crop1 left of crop0
        ("top_bottom", 0, 1),  # crop0 above crop1
        ("bottom_top", 1, 0),  # crop1 above crop0
    ]

    canvas_size = 224
    gap = 20

    for arrangement_name, first_idx, second_idx in arrangements:
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

        crop1 = crop_regions[first_idx]
        crop2 = crop_regions[second_idx]

        w1, h1 = crop1.size
        w2, h2 = crop2.size

        if arrangement_name in ["left_right", "right_left"]:
            # Horizontal arrangement
            total_width = w1 + w2 + gap
            scale = min((canvas_size - gap) / total_width, 1.0)

            w1_scaled = int(w1 * scale)
            h1_scaled = int(h1 * scale)
            w2_scaled = int(w2 * scale)
            h2_scaled = int(h2 * scale)

            crop1_resized = crop1.resize((w1_scaled, h1_scaled))
            crop2_resized = crop2.resize((w2_scaled, h2_scaled))

            x1 = (canvas_size - w1_scaled - w2_scaled - gap) // 2
            x2 = x1 + w1_scaled + gap
            y1 = (canvas_size - h1_scaled) // 2
            y2 = (canvas_size - h2_scaled) // 2

            canvas.paste(crop1_resized, (x1, y1))
            canvas.paste(crop2_resized, (x2, y2))

        else:  # top_bottom, bottom_top
            # Vertical arrangement
            total_height = h1 + h2 + gap
            scale = min((canvas_size - gap) / total_height, 1.0)

            w1_scaled = int(w1 * scale)
            h1_scaled = int(h1 * scale)
            w2_scaled = int(w2 * scale)
            h2_scaled = int(h2 * scale)

            crop1_resized = crop1.resize((w1_scaled, h1_scaled))
            crop2_resized = crop2.resize((w2_scaled, h2_scaled))

            x1 = (canvas_size - w1_scaled) // 2
            x2 = (canvas_size - w2_scaled) // 2
            y1 = (canvas_size - h1_scaled - h2_scaled - gap) // 2
            y2 = y1 + h1_scaled + gap

            canvas.paste(crop1_resized, (x1, y1))
            canvas.paste(crop2_resized, (x2, y2))

        goal_states.append(canvas)

    return goal_states


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

    # Add goal attention head for converting predicted latent to goal latent
    state.stage3_models["goal_attention"] = torch.nn.MultiheadAttention(
        embed_dim=latent_dim, num_heads=8, batch_first=True
    ).to(device)

    # Add multi-view fusion layer
    state.stage3_models["view_fusion"] = torch.nn.Sequential(
        torch.nn.Linear(384 * 5, 384),  # Fuse 5 views of vision features
        torch.nn.ReLU(),
        torch.nn.Linear(384, 384),
    ).to(device)

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
        + list(state.stage3_models["action_down_proj"].parameters())
        + list(state.stage3_models["goal_attention"].parameters())
        + list(state.stage3_models["view_fusion"].parameters()),
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

        # Construct goal states from crops/segments/arrows
        goal_images = construct_goal_states(obs_dict, payload.ui_annotations)

        # Extract encoder representations for goal states
        goal_latents = []
        if goal_images:
            for goal_img in goal_images:
                # Convert PIL image to tensor
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                goal_tensor = transform(goal_img).unsqueeze(0).to(device)

                # Get DINO features for goal state
                with torch.no_grad():
                    goal_features = models["dino"].forward_features(goal_tensor)
                    goal_cls = goal_features[0, 0]
                    goal_patches = goal_features[0, -196:]
                    goal_cls = goal_cls / (goal_cls.norm(dim=-1, keepdim=True) + 1e-8)
                    goal_attn = torch.matmul(goal_patches, goal_cls.T).view(14, 14)
                    goal_feat = torch.tensor(
                        goal_attn.flatten()[:384], dtype=torch.float32, device=device
                    )
                    if len(goal_feat) < 384:
                        goal_feat = torch.cat(
                            [
                                goal_feat,
                                torch.zeros(384 - len(goal_feat), device=device),
                            ]
                        )

                # Pass through adapter
                goal_latent = state.stage3_models["vis_adapter"](goal_feat.unsqueeze(0))
                goal_latents.append(goal_latent)

        # Fuse goal latents (average for now)
        if goal_latents:
            s_target = torch.stack(goal_latents, dim=0).mean(dim=0)
        else:
            # Fallback: use current state as target
            with torch.no_grad():
                vis_tok = state.stage3_models["vis_adapter"](obs_dict["vision"])
                txt_tok = state.stage3_models["txt_adapter"](
                    obs_dict["text"].squeeze(1)
                )
                pt_tok = state.stage3_models["pt_adapter"](obs_dict["pointnext"])
                vggt_tok = state.stage3_models["vggt_adapter"](obs_dict["vggt"])
                tactile_emb = state.stage3_models["tactile_adapter"](
                    obs_dict["tactile"]
                )

                modality_dict = {
                    "vision": vis_tok,
                    "text": txt_tok,
                    "pointnext": pt_tok,
                    "vggt": vggt_tok,
                    "tactile": tactile_emb,
                    "proprioception": obs_dict["proprioception"],
                }
                s_target = state.stage3_models["msat"](modality_dict)

        with torch.no_grad():
            # Multi-view fusion: extract features from all views
            view_features_dict = obs_dict.get("view_features", {})
            multi_view_feats = []

            for view_name, view_feat in view_features_dict.items():
                dino_attn = view_feat.get("dino_attn")
                if dino_attn is not None:
                    view_feat_tensor = torch.tensor(
                        dino_attn.flatten()[:384], dtype=torch.float32, device=device
                    )
                    if len(view_feat_tensor) < 384:
                        view_feat_tensor = torch.cat(
                            [
                                view_feat_tensor,
                                torch.zeros(384 - len(view_feat_tensor), device=device),
                            ]
                        )
                    multi_view_feats.append(view_feat_tensor)

            # Fuse multi-view features if we have multiple views
            if len(multi_view_feats) > 1:
                multi_view_concat = torch.cat(
                    multi_view_feats, dim=0
                )  # [num_views * 384]
                if multi_view_concat.size(0) == 384 * 5:
                    vis_tok_fused = state.stage3_models["view_fusion"](
                        multi_view_concat.unsqueeze(0)
                    )
                else:
                    # Pad or truncate to match expected size
                    if multi_view_concat.size(0) < 384 * 5:
                        multi_view_concat = torch.cat(
                            [
                                multi_view_concat,
                                torch.zeros(
                                    384 * 5 - multi_view_concat.size(0), device=device
                                ),
                            ]
                        )
                    vis_tok_fused = state.stage3_models["view_fusion"](
                        multi_view_concat.unsqueeze(0)
                    )
                vis_tok = vis_tok_fused
            else:
                # Single view fallback
                vis_tok = state.stage3_models["vis_adapter"](obs_dict["vision"])

            txt_tok = state.stage3_models["txt_adapter"](obs_dict["text"].squeeze(1))
            pt_tok = state.stage3_models["pt_adapter"](obs_dict["pointnext"])
            vggt_tok = state.stage3_models["vggt_adapter"](obs_dict["vggt"])
            tactile_emb = state.stage3_models["tactile_adapter"](obs_dict["tactile"])

            # merge all modalities to get a single latent representation
            modality_dict = {
                "vision": vis_tok,
                "text": txt_tok,
                "pointnext": pt_tok,
                "vggt": vggt_tok,
                "tactile": tactile_emb,
                "proprioception": obs_dict["proprioception"],
            }
            s_t = state.stage3_models["msat"](modality_dict)

        # DAWN Loop: Generate action candidate and iteratively refine
        a_candidate = state.stage3_models["flow_matcher"].sample_with_steering(
            s_t, s_target, num_steps=10
        )
        a_candidate = a_candidate.clone().detach().requires_grad_(True)
        eta = 0.01

        for k in range(5):
            # Get the action representation
            z_action = state.stage3_models["action_adapter"](a_candidate)
            z_action_16 = state.stage3_models["action_down_proj"](z_action)

            # Predict next latent state
            s_next_pred = state.stage3_models["predictor"](s_t, z_action_16)

            # Use goal attention head to get task-space representation from prediction
            s_next_pred_expanded = s_next_pred.unsqueeze(1)  # [B, 1, D]
            s_target_expanded = s_target.unsqueeze(1)  # [B, 1, D]

            s_goal_pred, _ = state.stage3_models["goal_attention"](
                s_next_pred_expanded, s_target_expanded, s_target_expanded
            )
            s_goal_pred = s_goal_pred.squeeze(1)  # [B, D]

            # Compute energy (distance to goal)
            energy = torch.mean((s_goal_pred - s_target) ** 2)

            # Compute gradient and steer action
            grad_a = torch.autograd.grad(energy, a_candidate, retain_graph=True)[0]
            with torch.no_grad():
                a_candidate = a_candidate - eta * grad_a
            a_candidate = a_candidate.clone().detach().requires_grad_(True)

        final_action = a_candidate.detach()

        # BadWorld attack for easy tasks
        if payload.is_easy_task:
            with torch.no_grad():
                final_action = state.stage3_models[
                    "attacker"
                ].generate_perturbed_context(
                    state.stage3_models["flow_matcher"], s_t, s_target, final_action
                )

        return {
            "action": final_action.squeeze(0).cpu().numpy().tolist(),
            "active_node_key": state.stage3_models["flow_matcher"],
        }
    except Exception as e:
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
            # Multi-view fusion for current state
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

            # Multi-view fusion for next state
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

        # Dynamics loss (predictor update without goal attention)
        loss_dynamics = F.mse_loss(s_next_pred, batch_s_next)

        # Anti-collapse regularization (SIGReg-style)
        # Project latents onto random directions and enforce isotropic Gaussian
        if batch_s_t.size(0) > 1:
            random_dirs = torch.randn(batch_s_t.size(-1), 10, device=device)
            random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)
            projected = torch.matmul(batch_s_t, random_dirs)
            mean_proj = projected.mean(dim=0, keepdim=True)
            std_proj = projected.std(dim=0, keepdim=True)
            loss_sigreg = F.mse_loss(std_proj, torch.ones_like(std_proj)) + F.mse_loss(
                mean_proj, torch.zeros_like(mean_proj)
            )
            loss_total = loss_dynamics + 0.01 * loss_sigreg
        else:
            loss_total = loss_dynamics

        loss_total.backward()
        state.stage3_optimizer.step()

        tactile_spike = float(payload.next_obs.tactile[0][0] > 0.5)
        state.stage3_memory.update(s_t[0], tactile_spike)

        return {"status": "calibrated", "loss": loss_dynamics.item()}
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

        # OPSD: Offline Policy State Distillation with multiple loss components
        num_opsd_steps = 5
        batch_size = min(len(state.stage3_trajectory_history), 16)

        total_loss = 0.0

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

            # 1. Flow Matcher Loss (CFM) with reward weighting
            x_0 = torch.randn_like(batch_action)
            t_rand = torch.rand(batch_action.size(0), 1, device=device)
            t_vector = t_rand.expand(-1, state.stage3_models["flow_matcher"].action_dim)

            x_t = t_vector * batch_action + (1.0 - t_vector) * x_0
            target_vel = batch_action - x_0

            if len(state.stage3_memory.anchors) > 0:
                s_target_batch = (
                    state.stage3_memory.anchors[-1]
                    .to(device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            else:
                s_target_batch = batch_s_t.clone()

            pred_vel = state.stage3_models["flow_matcher"].velocity_field(
                x_t, t_vector, batch_s_t, s_target_batch
            )
            cfm_loss = F.mse_loss(pred_vel, target_vel)

            # 2. Predictor Loss (JEPA dynamics)
            z_action = state.stage3_models["action_adapter"](batch_action)
            z_action_16 = state.stage3_models["action_down_proj"](z_action)
            s_next_pred = state.stage3_models["predictor"](batch_s_t, z_action_16)
            predictor_loss = F.mse_loss(s_next_pred, batch_s_next)

            # 3. Goal Attention Loss (train attention head)
            s_next_pred_expanded = s_next_pred.unsqueeze(1)
            s_target_expanded = s_target_batch.unsqueeze(1)
            s_goal_pred, _ = state.stage3_models["goal_attention"](
                s_next_pred_expanded, s_target_expanded, s_target_expanded
            )
            s_goal_pred = s_goal_pred.squeeze(1)
            goal_attention_loss = F.mse_loss(s_goal_pred, s_target_batch)

            # 4. Anti-collapse regularization for all components
            random_dirs = torch.randn(batch_s_t.size(-1), 10, device=device)
            random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)
            projected = torch.matmul(batch_s_t, random_dirs)
            mean_proj = projected.mean(dim=0, keepdim=True)
            std_proj = projected.std(dim=0, keepdim=True)
            sigreg_loss = F.mse_loss(std_proj, torch.ones_like(std_proj)) + F.mse_loss(
                mean_proj, torch.zeros_like(mean_proj)
            )

            # 5. Combined loss with reward weighting
            combined_reward = max(0.05, 1.0 + payload.reward)
            loss_opsd = (
                cfm_loss * combined_reward
                + predictor_loss * 0.5
                + goal_attention_loss * 0.3
                + sigreg_loss * 0.01
            )

            loss_opsd.backward()
            state.stage3_optimizer.step()
            total_loss += loss_opsd.item()

        avg_loss = total_loss / num_opsd_steps

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
                "goal_attention": state.stage3_models["goal_attention"].state_dict(),
                "view_fusion": state.stage3_models["view_fusion"].state_dict(),
                "gnn_nodes": state.stage3_models["gnn_library"].nodes.state_dict(),
                "gnn_specialists": state.stage3_models[
                    "gnn_library"
                ].specialists.state_dict(),
            },
            final_path,
        )

        return {
            "status": "distilled",
            "opsd_loss": avg_loss,
            "checkpoint": final_path,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Stage 3 distillation error: {str(e)}"
        )
