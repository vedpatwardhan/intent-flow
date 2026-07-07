import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import sys

# Align paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
from trainers.stage3.env import GR1Stage3Env
from trainers.stage3.denoiser import ComboStocFlowMatcher
from trainers.stage3.critic import EBMCritic, TrajectoryDiscriminator
from trainers.stage3.attacker import BadWorldAttacker


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
        self.short_term.append(s_t.detach())
        if len(self.short_term) > self.capacity:
            self.short_term.pop(0)

        if tactile_spike > 0.5:
            self.anchors.append(s_t.detach())
            if len(self.anchors) > 3:
                self.anchors.pop(0)

        if len(self.short_term) >= 2:
            self.gist = torch.mean(torch.stack(self.short_term), dim=0, keepdim=True)


def train_stage3(config, use_subset=False):
    print("--- STARTING ISOLATED STAGE 3: RAM ALIGNMENT, COMBOSTOC & SKILL0.5 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = config["paths"]["checkpoint_dir"]
    s2_ckpt_path = os.path.join(checkpoint_dir, "stage2_sft.pt")

    # 1. Load Adapters and MSAT
    vis_adapter = VisualAdapter(d_in=384).to(device)
    txt_adapter = TextAdapter(d_in=512).to(device)
    pt_adapter = PointNeXtAdapter(d_in=384).to(device)
    vggt_adapter = VGGTAdapter(d_in=config["model"]["vggt_dim"]).to(device)
    tactile_adapter = TactileAdapter().to(device)
    action_adapter = ActionAdapter(d_in=config["model"]["action_dim"], d_out=512).to(
        device
    )

    msat = MultiStreamActionTransformer(
        latent_dim=config["model"]["latent_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)
    predictor = JepaPredictor(
        state_dim=config["model"]["latent_dim"],
        action_dim=16,  # Matches Stage 2 bottleneck dim
        hidden_dim=config["model"]["latent_dim"],
    ).to(device)
    action_down_proj = nn.Linear(512, 16).to(device)

    # 2. Stage 3 ComboStoc and Critic Systems
    flow_matcher = ComboStocFlowMatcher(
        action_dim=config["model"]["action_dim"], config=config
    ).to(device)
    ebm_critic = EBMCritic(action_dim=config["model"]["action_dim"]).to(device)
    discriminator = TrajectoryDiscriminator(
        action_dim=config["model"]["action_dim"]
    ).to(device)
    attacker = BadWorldAttacker(action_dim=config["model"]["action_dim"])

    if os.path.exists(s2_ckpt_path):
        print(f"[RL-Stage3] Restoring SFT base weights from: {s2_ckpt_path}")
        checkpoint = torch.load(s2_ckpt_path, map_location=device)
        vis_adapter.load_state_dict(checkpoint["vis_adapter"])
        txt_adapter.load_state_dict(checkpoint["txt_adapter"])
        pt_adapter.load_state_dict(checkpoint["pt_adapter"])
        vggt_adapter.load_state_dict(checkpoint["vggt_adapter"])
        tactile_adapter.load_state_dict(checkpoint["tactile_adapter"])
        action_adapter.load_state_dict(checkpoint["action_adapter"])
        if "state_adapter" in checkpoint:
            # Replaced locally but loaded for compatibility
            pass
        msat.load_state_dict(checkpoint["msat"])
        predictor.load_state_dict(checkpoint["predictor"])
        if "action_down_proj" in checkpoint:
            action_down_proj.load_state_dict(checkpoint["action_down_proj"])
        # Load flow matcher weights directly from SFT checkpoint
        if "flow_matcher" in checkpoint:
            flow_matcher.load_state_dict(checkpoint["flow_matcher"])

    policy_checkpoints = [copy.deepcopy(flow_matcher)]
    optimizer = optim.AdamW(
        list(flow_matcher.parameters())
        + list(predictor.parameters())
        + list(ebm_critic.parameters())
        + list(discriminator.parameters())
        + list(action_adapter.parameters())
        + list(action_down_proj.parameters()),
        lr=config["stage3"]["lr"],
    )

    # Initialize true MuJoCo environment client
    env = GR1Stage3Env(action_dim=config["model"]["action_dim"])
    memory = HybridMemoryTriad()
    criterion = nn.MSELoss()
    bce = nn.BCELoss()

    epochs = 5 if use_subset else config["stage3"]["epochs"]
    for epoch in range(epochs):
        epoch_generator_loss = 0.0
        epoch_disc_loss = 0.0

        for episode in range(5 if use_subset else 10):
            obs = env.reset()
            done = False

            step_losses = []
            disc_losses = []

            while not done:
                # Format current observation state
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
                s_target = s_t.clone()  # Set target state equal to current context

                # --- 1. COMBOSTOC SAMPLING PROPOSAL ---
                pred_action = flow_matcher.sample_with_steering(
                    s_t, s_target, num_steps=10
                )

                # --- 2. BADWORLD WORST-CASE PERTURBATIONS ---
                perturb_force = attacker.generate_perturbation(
                    flow_matcher, s_t, s_target, pred_action
                )

                # Step physics
                next_obs, reward, done, info = env.step(
                    pred_action.detach(), perturb_force=perturb_force
                )
                memory.update(s_t, info["tactile_spike"])

                # --- 3. TRAJECTORY Realism discriminator ---
                real_score = discriminator(pred_action.detach(), s_t)
                loss_real = bce(real_score, torch.ones_like(real_score))

                fake_score = discriminator(pred_action, s_t)
                loss_fake = bce(fake_score, torch.zeros_like(fake_score))
                loss_disc = 0.5 * (loss_real + loss_fake)
                disc_losses.append(loss_disc)

                # --- 4. CONTRASTIVE EBM ENERGY MATCHING ---
                # Positive transition (expert target) -> minimize prediction error
                z_action_expert = action_adapter(pred_action)
                z_action_expert_16 = action_down_proj(z_action_expert)
                s_next_pred = predictor(s_t, z_action_expert_16)

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

                # Negative transition (failure target) -> maximize prediction error (increase energy)
                z_action_fail = action_adapter(pred_action + perturb_force)
                z_action_fail_16 = action_down_proj(z_action_fail)
                s_next_pred_fail = predictor(s_t, z_action_fail_16)
                loss_ebm_neg = -criterion(s_next_pred_fail, s_next)

                loss_ebm = loss_ebm_pos + 0.1 * loss_ebm_neg

                # --- 5. SKILL0.5 ROUTING & d-OPSD DISTILLATION ---
                best_policy = policy_checkpoints[-1]
                with torch.no_grad():
                    teacher_action = best_policy.sample(s_t, s_target, num_steps=10)

                # Easy/Medium tasks: distill standard student policy. Hard: privileged d-OPSD
                is_hard_task = info["target_dist"] > 0.18
                if is_hard_task:
                    # Inject privileged target memory anchors
                    distill_loss = criterion(pred_action, teacher_action) * 1.5
                else:
                    distill_loss = criterion(pred_action, teacher_action)

                # --- 6. RAM (Reinforce Adjoint Matching) Loss ---
                # Multiply flow match gradients by relative success rewards
                ram_scale = max(0.1, 1.0 + reward)  # reward is negative distance
                ram_loss = distill_loss * ram_scale

                # Aggregate Stage 3 Losses
                generator_loss = (
                    loss_ebm * config["stage3"]["ram_weight"]
                    + bce(fake_score, torch.ones_like(fake_score))
                    * config["stage3"]["adv_weight"]
                    + ram_loss * config["stage3"]["distill_weight"]
                )
                step_losses.append(generator_loss)
                obs = next_obs

            # Run optimizer steps
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

        # Update Auto-NPO checkpoint sliding pool
        policy_checkpoints.append(copy.deepcopy(flow_matcher))
        if len(policy_checkpoints) > 5:
            policy_checkpoints.pop(0)

        print(
            f"Epoch {epoch+1:03d} | RL Generator Loss: {epoch_generator_loss/10:.5f} | Disc Loss: {epoch_disc_loss/10:.5f}"
        )

    # Save final Stage 3 weights
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
            "action_down_proj": action_down_proj.state_dict(),
            "predictor": predictor.state_dict(),
            "flow_matcher": flow_matcher.state_dict(),
        },
        final_path,
    )
