import io
import base64
import json
import os
import yaml
import numpy as np
from pydantic import BaseModel
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

from colab_server_pkg.config import device
from colab_server_pkg.models_state import (
    stage3_models,
    stage3_trajectory_history,
    models,
)
from colab_server_pkg.feature_extractor import (
    extract_stage3_obs_features,
    extract_single_view_stage3_obs_features,
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


class Stage3CalibratePayload(BaseModel):
    transitions: List[Stage3CalibrateTransition]


class Stage3DistillPayload(BaseModel):
    reward: float


def encode_obs_to_latent(obs_dict, state):
    """
    Passes observation features through the respective adapter blocks and the
    Multi-Stream Action Transformer (MSAT) to yield the multi-modal latent state.
    """
    with torch.amp.autocast("cuda"):
        # Print input bounds for debugging NaNs
        print("   [Input Debug] bounds", end=" | ")
        for k, v in obs_dict.items():
            if torch.is_tensor(v):
                print(f"{k}: [{v.min().item():.3f}, {v.max().item():.3f}]", end=" | ")
        print()

        # Adapters
        vis_tok = state.stage3_models["vis_adapter"](obs_dict["vision"])
        txt_tok = state.stage3_models["txt_adapter"](obs_dict["text"])
        pt_tok = state.stage3_models["pt_adapter"](obs_dict["pointnext"])
        vggt_tok = state.stage3_models["vggt_adapter"](obs_dict["vggt"])
        tactile_tok = state.stage3_models["tactile_adapter"](obs_dict["tactile"])
        proprio_tok = state.stage3_models["state_adapter"](obs_dict["proprioception"])

        # MSAT
        modality_dict = {
            "vision": vis_tok,
            "text": txt_tok,
            "pointnext": pt_tok,
            "vggt": vggt_tok,
            "tactile": tactile_tok,
            "proprioception": proprio_tok,
        }
        out = state.stage3_models["msat"](modality_dict)
        return out


def construct_goal_states(obs_dict, ui_annotations):
    """
    Construct goal state representations for each annotated view by rearranging crops/segments according to arrows.
    Uses OpenCV inpainting to erase the moving patch's original position, and applies a light
    Gaussian blur to the background. Works with both rectangular crops and circular segment masks.
    """
    if not obs_dict or not ui_annotations:
        return {}

    goal_states_by_view = {}
    for view_name, view_annos in ui_annotations.items():
        features = obs_dict[view_name]["features"]
        pil_frame = features.get("pil_frame")
        if pil_frame is None:
            continue

        img_w, img_h = pil_frame.size

        crops = view_annos.get("crops", [])
        segments = view_annos.get("segments", [])
        vectors = view_annos.get("vectors", [])

        active_annos = crops + segments
        if len(active_annos) >= 2:
            p0_anno, p1_anno = active_annos[0], active_annos[1]
        else:
            # Default mock patches (crops fallback)
            p0_anno = {
                "x": int(224 * 0.3),
                "y": int(224 * 0.4),
                "width": int(224 * 0.15),
                "height": int(224 * 0.15),
            }
            p1_anno = {
                "x": int(224 * 0.55),
                "y": int(224 * 0.4),
                "width": int(224 * 0.15),
                "height": int(224 * 0.15),
            }

        task_isolated = features.get("task_isolated_features", {})
        sam_mask_224 = task_isolated.get("sam_mask_224", None)

        # Helper to extract patch and its binary alpha mask
        def extract_info(anno):
            scale_x = img_w / 224.0
            scale_y = img_h / 224.0
            is_crop_type = "width" in anno
            if is_crop_type:
                x = int(anno["x"] * scale_x)
                y = int(anno["y"] * scale_y)
                w = int(anno["width"] * scale_x)
                h = int(anno["height"] * scale_y)
                patch = pil_frame.crop((x, y, x + w, y + h))
                mask = Image.new("L", patch.size, 255)
            else:
                # Try to query the high-quality SAM segment mask
                if sam_mask_224 is not None and np.sum(sam_mask_224) > 0:
                    try:
                        mask_np_224 = (
                            np.array(sam_mask_224)
                            if isinstance(sam_mask_224, torch.Tensor)
                            else sam_mask_224
                        )
                        mask_uint8 = (mask_np_224 > 0).astype(np.uint8) * 255
                        num_labels, labels = cv2.connectedComponents(mask_uint8)

                        cx_scaled = min(223, max(0, int(anno["x"])))
                        cy_scaled = min(223, max(0, int(anno["y"])))
                        lbl = labels[cy_scaled, cx_scaled]

                        if lbl == 0:
                            # Scan a small local window if exact click landed on a zero edge
                            window = labels[
                                max(0, cy_scaled - 5) : min(224, cy_scaled + 6),
                                max(0, cx_scaled - 5) : min(224, cx_scaled + 6),
                            ]
                            non_zero = window[window > 0]
                            if len(non_zero) > 0:
                                lbl = non_zero[0]

                        if lbl > 0:
                            segment_mask_224 = (labels == lbl).astype(np.float32)
                            mask_pil = Image.fromarray(
                                (segment_mask_224 * 255).astype(np.uint8)
                            )
                            mask_resized = mask_pil.resize(
                                (img_w, img_h), Image.NEAREST
                            )
                            mask_np = np.array(mask_resized)

                            indices = np.argwhere(mask_np > 0)
                            y_min, x_min = indices.min(axis=0)
                            y_max, x_max = indices.max(axis=0)

                            x = int(x_min)
                            y = int(y_min)
                            w = int(x_max - x_min + 1)
                            h = int(y_max - y_min + 1)

                            patch = pil_frame.crop((x, y, x + w, y + h))
                            mask = mask_resized.crop((x, y, x + w, y + h))
                        else:
                            raise ValueError("No matching component label found")
                    except Exception as e:
                        print(f"Fallback to circle due to components error: {e}")
                        cx = int(anno["x"] * scale_x)
                        cy = int(anno["y"] * scale_y)
                        r = int(25 * scale_x)
                        x = max(0, cx - r)
                        y = max(0, cy - r)
                        w = min(img_w - x, 2 * r)
                        h = min(img_h - y, 2 * r)
                        patch = pil_frame.crop((x, y, x + w, y + h))
                        mask = Image.new("L", (w, h), 0)
                        draw = ImageDraw.Draw(mask)
                        draw.ellipse(
                            (cx - r - x, cy - r - y, cx + r - x, cy + r - y), fill=255
                        )
                else:
                    # Fallback circle around click point
                    cx = int(anno["x"] * scale_x)
                    cy = int(anno["y"] * scale_y)
                    r = int(25 * scale_x)
                    x = max(0, cx - r)
                    y = max(0, cy - r)
                    w = min(img_w - x, 2 * r)
                    h = min(img_h - y, 2 * r)
                    patch = pil_frame.crop((x, y, x + w, y + h))
                    mask = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse(
                        (cx - r - x, cy - r - y, cx + r - x, cy + r - y), fill=255
                    )
            return patch, mask, x, y, w, h

        try:
            patch1, mask1, x1, y1, w1, h1 = extract_info(p0_anno)
            patch2, mask2, x2, y2, w2, h2 = extract_info(p1_anno)
        except Exception as e:
            print(f"Extraction failed for view {view_name}: {e}")
            continue

        # Decide direction based on arrow vector
        scale_x = img_w / 224.0
        scale_y = img_h / 224.0
        if vectors and len(vectors) > 0:
            vec = vectors[0]
            start_x = vec["start"][0] * scale_x
            start_y = vec["start"][1] * scale_y

            ctr0_x = x1 + w1 / 2.0
            ctr0_y = y1 + h1 / 2.0
            ctr1_x = x2 + w2 / 2.0
            ctr1_y = y2 + h2 / 2.0

            d0_start = (ctr0_x - start_x) ** 2 + (ctr0_y - start_y) ** 2
            d1_start = (ctr1_x - start_x) ** 2 + (ctr1_y - start_y) ** 2

            if d1_start < d0_start:
                patch1, mask1, x1, y1, w1, h1, patch2, mask2, x2, y2, w2, h2 = (
                    patch2,
                    mask2,
                    x2,
                    y2,
                    w2,
                    h2,
                    patch1,
                    mask1,
                    x1,
                    y1,
                    w1,
                    h1,
                )

        # Inpaint original moving patch location
        try:
            cv_img = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            inpaint_mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
            cv2.rectangle(inpaint_mask, (x1, y1), (x1 + w1, y1 + h1), 255, -1)
            inpainted_cv = cv2.inpaint(
                cv_img, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
            )
            inpainted_rgb = cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB)
            clean_bg = Image.fromarray(inpainted_rgb)
        except Exception as e:
            print(f"Inpainting failed in construct_goal_states: {e}")
            clean_bg = pil_frame

        blurred_bg = clean_bg.filter(ImageFilter.GaussianBlur(radius=10))

        arrangements = [
            ("left", x2 - w1, y2 + (h2 - h1) // 2),
            ("right", x2 + w2, y2 + (h2 - h1) // 2),
            ("top", x2 + (w2 - w1) // 2, y2 - h1),
            ("bottom", x2 + (w2 - w1) // 2, y2 + h2),
        ]

        # Construct the initial frame: both segments at their original positions
        init_canvas = blurred_bg.copy()
        init_canvas.paste(patch2, (x2, y2), mask2)
        init_canvas.paste(patch1, (x1, y1), mask1)

        goal_arrangements = []
        for name, x1_new, y1_new in arrangements:
            canvas = blurred_bg.copy()
            canvas.paste(patch2, (x2, y2), mask2)
            x1_clip = max(0, min(x1_new, img_w - w1))
            y1_clip = max(0, min(y1_new, img_h - h1))
            canvas.paste(patch1, (x1_clip, y1_clip), mask1)
            goal_arrangements.append(canvas)

        goal_states_by_view[view_name] = {
            "init_frame": init_canvas,
            "goal_frames": goal_arrangements,
        }

    return goal_states_by_view


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
    state.stage3_models["txt_adapter"] = TextAdapter(d_in=512).to(device)
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
            "txt_adapter",
            "pt_adapter",
            "vggt_adapter",
            "tactile_adapter",
            "action_adapter",
            "state_adapter",
            "action_down_proj",
            "msat",
            "predictor",
            "flow_matcher",
        ]:
            m_state_dict = {}
            prefix = f"{module_name}."
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    m_state_dict[k[len(prefix) :]] = v
            if len(m_state_dict) > 0 and module_name in state.stage3_models:
                state.stage3_models[module_name].load_state_dict(
                    m_state_dict,
                    strict=(
                        module_name != "flow_matcher" and module_name != "vggt_adapter"
                    ),
                )
    else:
        print(
            "[Colab] WARNING: Neither Stage 3 nor Stage 2 checkpoints were found. Models initialized with default random weights!"
        )

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
        lr=config["stage3"]["lr"],
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

                # Construct goal states from crops/segments/arrows
                # {view_name: [goal_img1, goal_img2, goal_img3, goal_img4]}
                goal_images = construct_goal_states(obs_dict, payload.ui_annotations)

                # Save debug visualization plots via dedicated helper function
                save_stage3_debug_plots(payload, obs_dict, goal_images)

                # Extract encoder representations for goal states
                goal_latents = []
                goal_feature_maps = {}  # Store maps for debug plotting
                for view_name, goal_data in goal_images.items():
                    init_img = goal_data["init_frame"]

                    # Encode the common initial image once per view
                    buffered_init = io.BytesIO()
                    init_img.save(buffered_init, format="JPEG")
                    init_img_str = base64.b64encode(buffered_init.getvalue()).decode(
                        "utf-8"
                    )

                    goal_feature_maps[view_name] = []

                    for goal_img in goal_data["goal_frames"]:
                        # Encode PIL goal_img to base64 string
                        buffered_goal = io.BytesIO()
                        goal_img.save(buffered_goal, format="JPEG")
                        goal_img_str = base64.b64encode(
                            buffered_goal.getvalue()
                        ).decode("utf-8")

                        # We pass [init_img_str, goal_img_str] as the history to VGGT
                        goal_history = [init_img_str, goal_img_str]

                        goal_obs_dict = extract_single_view_stage3_obs_features(
                            goal_img_str,
                            goal_history,
                            payload.text_prompt,
                            payload.ui_annotations,
                            payload.tactile,
                            payload.proprioception,
                            view_name=view_name,
                        )

                        # Capture raw feature maps before adapter projection
                        goal_feature_maps[view_name].append(
                            {
                                "goal_img": goal_img,
                                "dino": goal_obs_dict["features"]["dino_attn"],
                                "clip": goal_obs_dict["features"]["clip_sim"],
                                "motion": goal_obs_dict["features"]["motion_field"],
                            }
                        )

                        # Pass features through respective adapters and MSAT
                        goal_latent = encode_obs_to_latent(goal_obs_dict, state)
                        goal_latents.append(goal_latent)

                # Save the new 4x3 debug plots showing DINO, CLIP, and VGGT motion fields
                save_stage3_goal_features_plots(goal_feature_maps)

        with torch.no_grad():
            # Get state latents [1, latent_dim]
            s_t = encode_obs_to_latent(combined_obs, state)

            # Query the goal latents using the scaled current state s_t via MultiheadAttention
            # [1, num_goals, latent_dim]
            stacked_goals = torch.stack(goal_latents, dim=1)
            tau = 10.0  # Temperature scale to break softmax saturation
            s_t_scaled = s_t.unsqueeze(0) / tau
            s_target, _ = state.stage3_models["goal_attention"](
                s_t_scaled, stacked_goals, stacked_goals
            )
            # [1, latent_dim]
            s_target = state.stage3_models["latent_adapter"](s_target.squeeze(0))

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

        # --- COMBINATORIAL OBSERVATION MINIMAX SEARCH BLOCK ---
        # 4 distinct image configurations evaluated inside the wrapper
        N_obs_variants = 4

        # Invoke the attacker pass to construct the entire grid natively
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                perturbed_payloads, s_t_ensemble, a_candidates = state.stage3_models[
                    "attacker"
                ].generate_stochastic_ensemble_pass(
                    flow_matcher=state.stage3_models["flow_matcher"],
                    raw_data=payload,
                    load_image_fn=decode_base64_image,
                    extract_obs_features_fn=extract_stage3_obs_features,
                    encode_obs_fn=encode_obs_to_latent,
                    state=state,
                    s_target=s_target,
                    steering_timelines=steering_timelines,
                    embodiment_id=embodiment_id,
                    ensemble_size=N_obs_variants,
                    horizon=horizon,
                )

        # Capture the true 16-batch size returned directly out of your attacker pass
        ensemble_size = a_candidates.shape[0]  # 16

        # Learning rate for action adjustment and timeline rollback scale
        eta = 0.2
        timeline_advance_rate = 0.05
        error_threshold = 0.02

        # Expand target constraints and master timelines to match the full 16 execution slots
        s_target_expanded = s_target.expand(ensemble_size, -1)
        steering_timelines_expanded = (
            steering_timelines.expand(ensemble_size, -1)
            .view(ensemble_size, horizon, joint_dim)
            .clone()
        )

        steering_history = {
            "episode_idx": payload.episode_idx,
            "step_idx": payload.step_idx,
            "iterations": [],
        }

        for k in range(5):
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

            # [16, 1, 512]
            s_next_pred_expanded = s_next_pred.unsqueeze(1)
            s_next_pred_scaled = s_next_pred_expanded / tau

            # Expand the 4 RAW foveated goal orientations [16, 4, 512]
            stacked_goals_expanded = stacked_goals.expand(ensemble_size, -1, -1)

            # Query the 4 goal perspectives using the scaled predicted next state vector
            s_goal_pred_attn, _ = state.stage3_models["goal_attention"](
                s_next_pred_scaled, stacked_goals_expanded, stacked_goals_expanded
            )
            s_goal_pred_attn = s_goal_pred_attn.squeeze(1)  # Shape: [16, 512]

            # Project the foveated pool back into aligned task space [16, 512]
            s_goal_pred = state.stage3_models["latent_adapter"](s_goal_pred_attn)

            # Compute energy and gradient (distance to goal)
            energy = torch.mean((s_goal_pred - s_target_expanded) ** 2)
            grad_a = torch.autograd.grad(energy, a_candidates)[0]

            # Normalize gradients per ensemble instance to stabilize steering step size
            # [16, 7, 58]
            grad_norm = grad_a.norm(dim=(1, 2), keepdim=True) + 1e-8
            grad_a_normalized = grad_a / grad_norm

            # Masked Energy Guidance matching 3D coordinates
            # Dynamically calculate the guidance mask (shape: [4, 7, 58])
            t_j = 1.0 - steering_timelines_expanded

            # Sculpt action parameters along the tracking vector field
            diff = eta * grad_a_normalized * t_j
            print(f"diff bounds: {diff.min()} {diff.max()}")
            a_candidates = a_candidates - diff

            # COMBOSTOC LOCAL REPAIR: Evaluate absolute error profile per joint channel
            # Mean over ensemble (dim 0) and horizon (dim 1) -> Shape: [58]
            joint_errors = grad_a_normalized.abs().mean(dim=(0, 1))

            # Joints that are stable (error below threshold) advance forward
            # toward clean actions (1.0)
            stable_joints_mask = joint_errors <= error_threshold
            steering_timelines_expanded[
                :, :, stable_joints_mask
            ] += timeline_advance_rate

            # Joints that are drifting (error above threshold) get pinned or
            # reset back to 0.0 (noise space)
            drifting_joints_mask = ~stable_joints_mask
            steering_timelines_expanded[:, :, drifting_joints_mask] = 0.0

            # Keep boundaries locked within standard flow matching bounds
            # [0.0, 1.0]
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

        # Extract only the immediate multi-step prediction slice if needed,
        # or output the unrolled trajectory block back to your motor script loader!
        with torch.no_grad():
            final_energies = torch.mean((s_goal_pred - s_target_expanded) ** 2, dim=-1)

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
            "perturbed_payloads": [p.dict() for p in perturbed_payloads],
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
        print(f"\n[Calibrate] Starting calibration on {num_transitions} transitions...")

        for i, trans in enumerate(payload.transitions):
            print(
                f"[Calibrate] Processing transition {i + 1}/{num_transitions} ({(i + 1) / num_transitions * 100:.1f}%)..."
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
                    print(
                        f"[CALIBRATE] s_t bounds: [{s_t.min().item():.6f}, "
                        f"{s_t.max().item():.6f}] "
                        f"s_next bounds: [{s_next.min().item():.6f}, "
                        f"{s_next.max().item():.6f}]"
                    )

            if s_t_first is None:
                s_t_first = s_t

            # action: Shape [1, 406] (406 = 7 steps * 58 action_dim)
            action = torch.tensor([trans.action_taken], dtype=torch.float32).to(device)
            print(
                f"action bounds: [{action.min().item():.6f}, {action.max().item():.6f}]"
            )

            # Track the current state, action block, true next state, energy,
            # tactile success, and the step target
            state.stage3_trajectory_history.append(
                (s_t, action, s_next, trans.energy, trans.tactile, s_t)
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

            print(
                "Shapes: \n"
                f"\tbatch_s_t: {batch_s_t.shape}\n"
                f"\tbatch_action: {batch_action.shape}\n"
                f"\tbatch_s_next: {batch_s_next.shape}\n"
                f"\tbatch_energy: {batch_energy.shape}\n"
                f"\tbatch_tactile: {batch_tactile.shape}\n"
                f"\ts_target_batch: {s_target_batch.shape}\n"
                f"\tbatch_action_3d: {batch_action_3d.shape}"
            )

            # Request unreduced batch loss elements using our new flag
            batch_size = batch_action_3d.size(0)
            embodiment_id = torch.tensor([2], dtype=torch.long, device=device).expand(
                batch_size
            )
            cfm_loss_elementwise = state.stage3_models["flow_matcher"].get_cfm_loss(
                x_1=batch_action_3d,  # [ensemble_size, horizon, action_dim]
                s_t=batch_s_t,  # [ensemble_size, latent_dim]
                s_target=s_target_batch,  # [ensemble_size, latent_dim]
                embodiment_id=embodiment_id,
                reduction="none",
            )

            # If energy is 0, exp(-0) = 1.0 (Maximum reward)
            # If energy explodes to infinity, exp(-inf) = 0.0
            energy_reward = torch.exp(-batch_energy)

            # Combine the bounded objectives: Max value is 1.0 + 2.0 = 3.0
            combined_rewards = energy_reward + 2.0 * batch_tactile

            # Apply the reward weight directly to the matching vectors
            cfm_loss = (
                cfm_loss_elementwise * combined_rewards.unsqueeze(-1).unsqueeze(-1)
            ).mean()

            # 2. Predictor Loss (JEPA dynamics)
            z_action = state.stage3_models["action_adapter"](batch_action)
            z_action_16 = state.stage3_models["action_down_proj"](z_action)
            s_next_pred = state.stage3_models["predictor"](batch_s_t, z_action_16)
            predictor_loss = F.mse_loss(s_next_pred, batch_s_next)

            # 3. Goal Attention Loss
            tau = 10.0  # Temperature scale to match inference
            s_next_pred_expanded = s_next_pred.unsqueeze(1)
            s_next_pred_scaled = s_next_pred_expanded / tau
            s_target_expanded = s_target_batch.unsqueeze(1)
            s_goal_pred, _ = state.stage3_models["goal_attention"](
                s_next_pred_scaled, s_target_expanded, s_target_expanded
            )
            s_goal_pred = s_goal_pred.squeeze(1)
            goal_attention_loss = F.mse_loss(s_goal_pred, s_target_batch)

            # 4. CASA (Contrastive Action-State Alignment) Loss integration from Stage 2 SFT
            z_s = batch_s_t / (batch_s_t.norm(dim=-1, keepdim=True) + 1e-8)
            z_a = z_action / (z_action.norm(dim=-1, keepdim=True) + 1e-8)
            sim_matrix = torch.matmul(z_s, z_a.T) / 0.07
            labels = torch.arange(sim_matrix.size(0), device=device)
            casa_loss = F.cross_entropy(sim_matrix, labels)

            # 5. Anti-collapse regularization (SIGReg)
            random_dirs = torch.randn(s_next_pred.size(-1), 10, device=device)
            random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)
            projected = torch.matmul(s_next_pred, random_dirs)
            mean_proj = projected.mean(dim=0, keepdim=True)
            std_proj = projected.std(dim=0, keepdim=True)
            sigreg_loss = F.mse_loss(std_proj, torch.ones_like(std_proj)) + F.mse_loss(
                mean_proj, torch.zeros_like(mean_proj)
            )

            # Combined total optimization payload
            loss_opsd = (
                cfm_loss
                + 0.2 * casa_loss
                + predictor_loss * 0.5
                + goal_attention_loss * 0.3
                + sigreg_loss * 0.01
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

            diagnostics = {
                "epoch_step": opsd_step,
                "loss/cfm_loss": cfm_loss.item(),
                "loss/casa_loss": casa_loss.item(),
                "loss/sigreg_loss": sigreg_loss.item(),
                "drift/state_magnitude": state_magnitude,
                "drift/state_variance": state_variance,
                "policy/action_magnitude": action_magnitude,
                "policy/action_smoothness": action_smoothness,
                "metrics/noop_loss_ratio": noop_ratio,
                "metrics/action_drift": action_drift,
            }
            print(f"📊 Stage 3 OPSD Diagnostics: {json.dumps(diagnostics, indent=2)}")

            loss_opsd.backward()
            state.stage3_optimizer.step()
            total_loss += loss_opsd.item()

        # Release fragmented allocation pools
        torch.cuda.empty_cache()

        avg_loss = total_loss / num_opsd_steps

        # ... Rest of checkpoint saving code remains identical ...
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
                "state_adapter": state.stage3_models["state_adapter"].state_dict(),
                "action_down_proj": state.stage3_models[
                    "action_down_proj"
                ].state_dict(),
                "predictor": state.stage3_models["predictor"].state_dict(),
                "goal_attention": state.stage3_models["goal_attention"].state_dict(),
                "latent_adapter": state.stage3_models["latent_adapter"].state_dict(),
                "gnn_nodes": state.stage3_models["gnn_library"].nodes.state_dict(),
                "gnn_specialists": state.stage3_models[
                    "gnn_library"
                ].specialists.state_dict(),
            },
            final_path,
        )

        return {"status": "distilled", "opsd_loss": avg_loss, "checkpoint": final_path}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Stage 3 distillation error: {str(e)}"
        )
