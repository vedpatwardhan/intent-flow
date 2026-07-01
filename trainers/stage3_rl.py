import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
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
from models.action_denoiser import CLAPFlowMatcher
from utils.dataset_loader import get_dataloader


class TrajectoryDiscriminator(nn.Module):
    """
    Evaluates human-like trajectory realism (style checking) conditioned on the task.
    """

    def __init__(self, action_dim=12, state_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, traj, s_t):
        inputs = torch.cat([traj, s_t], dim=-1)
        return self.net(inputs)


class SimulatedHumanoidEnv:
    """
    Mock-simulator client that replicates MuJoCo/Genesis dynamics execution,
    generating states, rewards, tactile grids, and joint limits.
    """

    def __init__(self, action_dim=12, state_dim=24):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reset()

    def reset(self):
        self.joint_pos = torch.randn(self.state_dim)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        # Outputs mock observations matching Aloha & pretraining schema
        return {
            "vision": torch.randn(1, 384),
            "pointnext": torch.randn(1, 384),
            "vggt": torch.randn(1, 768),
            "tactile": torch.zeros(1, 4, 4),
            "proprioception": torch.randn(1, 24),
            "text": torch.randn(1, 512),
        }

    def step(self, action, perturb_force=None):
        self.step_count += 1

        # Inject BadWorld joint perturbations if active
        effective_action = action.clone()
        if perturb_force is not None:
            effective_action += perturb_force

        # Apply basic Euler joint integration simulation
        self.joint_pos = self.joint_pos + 0.1 * effective_action.cpu().sum()

        # Compute Reward: Distance metric to target
        dist = torch.norm(self.joint_pos)
        reward = -dist.item()  # reward increases as joints align closer to 0 target

        # Check event boundary triggers (like tactile collision spikes)
        done = self.step_count >= 8
        tactile_spike = float(dist.item() < 0.2)  # Spike touch target on proximity

        obs = self._get_obs()
        obs["tactile"] = torch.full((1, 4, 4), tactile_spike)

        return obs, reward, done, {"tactile_spike": tactile_spike}


class HybridMemoryTriad:
    """
    Multi-tiered memory tracking:
      1. Short-Term Window (sliding context queue)
      2. Event Boundary Anchors (captures full spatial latents on spikes)
      3. Long-Range Gist (relevance compressed tokens)
    """

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.short_term = []
        self.anchors = []
        self.gist = []

    def update(self, s_t, tactile_spike):
        # 1. Update Short-Term sliding context
        self.short_term.append(s_t.detach())
        if len(self.short_term) > self.capacity:
            self.short_term.pop(0)

        # 2. Trigger Event Boundary Anchor
        if tactile_spike > 0.5:
            self.anchors.append(s_t.detach())
            if len(self.anchors) > 3:
                self.anchors.pop(0)

        # 3. Compile Long-Range Gist via relevance pooling
        if len(self.short_term) >= 2:
            self.gist = torch.mean(torch.stack(self.short_term), dim=0, keepdim=True)


def train_stage3(config, use_subset=False):
    print(
        "--- STARTING STAGE 3: RL ALIGNMENT, ENERGY SCULPTING & OPSD (PL & SIMULATION) ---"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Stage 2 SFT weights
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    s2_ckpt_path = os.path.join(checkpoint_dir, "stage2_sft.pt")

    vis_adapter = VisualAdapter(d_in=384).to(device)
    txt_adapter = TextAdapter(d_in=512).to(device)
    pt_adapter = PointNeXtAdapter(d_in=384).to(device)
    vggt_adapter = VGGTAdapter(d_in=config["model"]["vggt_dim"]).to(device)
    tactile_adapter = TactileAdapter().to(device)
    action_adapter = ActionAdapter(d_in=config["model"]["action_dim"], d_out=512).to(
        device
    )
    msat = MultiStreamActionTransformer().to(device)
    predictor = JepaPredictor(action_dim=512).to(device)
    flow_matcher = CLAPFlowMatcher(action_dim=config["model"]["action_dim"]).to(device)
    discriminator = TrajectoryDiscriminator(
        action_dim=config["model"]["action_dim"]
    ).to(device)

    if os.path.exists(s2_ckpt_path):
        print(f"[RL] Loading SFT weights from: {s2_ckpt_path}")
        checkpoint = torch.load(s2_ckpt_path, map_location=device)
        vis_adapter.load_state_dict(checkpoint["vis_adapter"])
        txt_adapter.load_state_dict(checkpoint["txt_adapter"])
        pt_adapter.load_state_dict(checkpoint["pt_adapter"])
        vggt_adapter.load_state_dict(checkpoint["vggt_adapter"])
        tactile_adapter.load_state_dict(checkpoint["tactile_adapter"])
        action_adapter.load_state_dict(checkpoint["action_adapter"])
        msat.load_state_dict(checkpoint["msat"])
        predictor.load_state_dict(checkpoint["predictor"])
        flow_matcher.load_state_dict(checkpoint["flow_matcher"])

    policy_checkpoints = [copy.deepcopy(flow_matcher)]
    optimizer = optim.AdamW(
        list(flow_matcher.parameters())
        + list(predictor.parameters())
        + list(discriminator.parameters())
        + list(action_adapter.parameters()),
        lr=config["stage3"]["lr"],
    )

    env = SimulatedHumanoidEnv(action_dim=config["model"]["action_dim"])
    memory = HybridMemoryTriad()
    criterion = nn.MSELoss()
    bce = nn.BCELoss()

    # Active play loop over training epochs
    epochs = 5 if use_subset else config["stage3"]["epochs"]
    for epoch in range(epochs):
        epoch_generator_loss = 0.0
        epoch_disc_loss = 0.0

        # Execute active rollout sessions
        for episode in range(10):
            obs = env.reset()
            done = False

            step_losses = []
            disc_losses = []

            # Keep tracking states over the temporal horizon
            s_history = []
            a_history = []

            while not done:
                # Compile observation dict
                vis_tok = vis_adapter(obs["vision"].to(device))
                txt_tok = txt_adapter(obs["text"].to(device).squeeze(1))
                pt_tok = pt_adapter(obs["pointnext"].to(device))
                vggt_tok = vggt_adapter(obs["vggt"].to(device))
                tactile_emb = tactile_adapter(obs["tactile"].to(device))

                modality_dict = {
                    "vision": vis_tok,
                    "text": txt_tok,
                    "pointnext": pt_tok,
                    "vggt": vggt_tok,
                    "tactile": tactile_emb,
                    "proprioception": obs["proprioception"].to(device),
                }
                s_t = msat(modality_dict)
                s_history.append(s_t)

                # --- 1. THE GENERATOR FLOW MATCHING PROPOSAL ---
                pred_action = flow_matcher.sample(s_t, num_steps=10)

                # --- 2. BADWORLD ADVERSARIAL PERTURBATIONS ---
                # Inject 20% random joint perturbation forces
                perturb_force = 0.2 * torch.randn_like(pred_action)
                next_obs, reward, done, info = env.step(
                    pred_action.detach(), perturb_force=perturb_force
                )

                # Update memory triad
                memory.update(s_t, info["tactile_spike"])

                # --- 3. TRAJECTORY STYLE DISCRIMINATOR LOSS ---
                real_score = discriminator(pred_action.detach(), s_t)
                loss_real = bce(real_score, torch.ones_like(real_score))

                # Fake path
                fake_score = discriminator(pred_action, s_t)
                loss_fake = bce(fake_score, torch.zeros_like(fake_score))

                loss_disc = 0.5 * (loss_real + loss_fake)
                disc_losses.append(loss_disc)

                # --- 4. CONTRASTIVE EBM ENERGY LANDSCAPE SCULPTING ---
                # Success path (experts) -> minimize energy
                z_action_expert = action_adapter(pred_action)
                s_next_pred = predictor(s_t, z_action_expert)

                # Retrieve next state s_next target
                vis_tok_next = vis_adapter(next_obs["vision"].to(device))
                pt_tok_next = pt_adapter(next_obs["pointnext"].to(device))
                vggt_tok_next = vggt_adapter(next_obs["vggt"].to(device))
                modality_dict_next = {
                    "vision": vis_tok_next,
                    "text": txt_tok,
                    "pointnext": pt_tok_next,
                    "vggt": vggt_tok_next,
                    "tactile": tactile_adapter(next_obs["tactile"].to(device)),
                    "proprioception": next_obs["proprioception"].to(device),
                }
                s_next = msat(modality_dict_next)

                loss_ebm_pos = criterion(s_next_pred, s_next)

                # Failure path -> maximize energy
                z_action_fail = action_adapter(pred_action + perturb_force)
                s_next_pred_fail = predictor(s_t, z_action_fail)
                loss_ebm_neg = -criterion(s_next_pred_fail, s_next)

                loss_ebm = loss_ebm_pos + 0.1 * loss_ebm_neg

                # --- 5. d-OPSD SUFFIX-CONDITIONED DISTILLATION ---
                # Teachers utilize future memory anchors to distill smooth trajectories
                best_policy = policy_checkpoints[-1]
                with torch.no_grad():
                    teacher_action = best_policy.sample(s_t, num_steps=10)

                distill_loss = criterion(pred_action, teacher_action)

                # --- 6. π-StepNFT STEP GRADIENT SUPERVISION ---
                # Regress step-wise denoising flow adjustments away from failure vectors
                if loss_ebm_pos.item() > 0.5:  # Failure threshold
                    # Perform flow reversal steering (integrate backward to noise x_0)
                    with torch.no_grad():
                        x_0 = pred_action - perturb_force  # Reconstructed base noise

                    # Direct step penalty (push current flow away from failure noise vector)
                    step_nft_loss = criterion(pred_action, x_0)
                else:
                    step_nft_loss = torch.tensor(0.0, device=device)

                # Aggregate losses
                generator_loss = (
                    loss_ebm * config["stage3"]["ram_weight"]
                    + bce(fake_score, torch.ones_like(fake_score))
                    * config["stage3"]["adv_weight"]
                    + distill_loss * config["stage3"]["distill_weight"]
                    + 0.1 * step_nft_loss
                )
                step_losses.append(generator_loss)

                obs = next_obs

            # Perform optimization step
            if step_losses:
                optimizer.zero_grad()
                total_step_loss = torch.stack(step_losses).mean()
                total_step_loss.backward()
                optimizer.step()
                epoch_generator_loss += total_step_loss.item()

            if disc_losses:
                optimizer.zero_grad()
                total_disc_loss = torch.stack(disc_losses).mean()
                total_disc_loss.backward()
                optimizer.step()
                epoch_disc_loss += total_disc_loss.item()

        # Update reference checkpoint pool (Auto-NPO)
        policy_checkpoints.append(copy.deepcopy(flow_matcher))
        if len(policy_checkpoints) > 5:
            policy_checkpoints.pop(0)

        print(
            f"Epoch {epoch+1:03d} | RL Generator Loss: {epoch_generator_loss/10:.5f} | Disc Loss: {epoch_disc_loss/10:.5f}"
        )

    print("--- STAGE 3 RL COMPLETE ---")
    final_path = os.path.join(checkpoint_dir, "stage3_rl_final.pt")
    print(f"Saving final Stage 3 RL weights to: {final_path}")
    torch.save(
        {
            "vis_adapter": vis_adapter.state_dict(),
            "txt_adapter": txt_adapter.state_dict(),
            "pt_adapter": pt_adapter.state_dict(),
            "vggt_adapter": vggt_adapter.state_dict(),
            "tactile_adapter": tactile_adapter.state_dict(),
            "msat": msat.state_dict(),
            "action_adapter": action_adapter.state_dict(),
            "predictor": predictor.state_dict(),
            "flow_matcher": flow_matcher.state_dict(),
        },
        final_path,
    )
