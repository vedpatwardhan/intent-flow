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
from trainers.stage3.discriminator import TrajectoryDiscriminator
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
        self.gist = torch.zeros(1, 512)

    def update(self, s_t, tactile_spike):
        self.short_term.append(s_t.detach().cpu())
        if len(self.short_term) > self.capacity:
            self.short_term.pop(0)

        # Anchor memory snapshot on tactile spikes
        if tactile_spike > 0.5:
            self.anchors.append(s_t.detach().cpu())
            if len(self.anchors) > 3:
                self.anchors.pop(0)

        # Gist token is a moving average of recent state representations
        if len(self.short_term) >= 2:
            self.gist = torch.mean(torch.stack(self.short_term), dim=0)


def train_stage3(config, use_subset=False):
    print("--- STARTING ALIGNED STAGE 3: RAM ALIGNMENT, COMBOSTOC & SKILL0.5 ---")
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
        action_dim=16,  # Matches Stage 2 SFT bottleneck
        hidden_dim=config["model"]["latent_dim"],
    ).to(device)
    action_down_proj = nn.Linear(512, 16).to(device)

    # 2. Stage 3 ComboStoc and Discriminator Systems
    flow_matcher = ComboStocFlowMatcher(
        action_dim=config["model"]["action_dim"], config=config
    ).to(device)
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
            pass
        msat.load_state_dict(checkpoint["msat"])
        predictor.load_state_dict(checkpoint["predictor"])
        if "action_down_proj" in checkpoint:
            action_down_proj.load_state_dict(checkpoint["action_down_proj"])
        if "flow_matcher" in checkpoint:
            flow_matcher.load_state_dict(checkpoint["flow_matcher"])

    policy_checkpoints = [copy.deepcopy(flow_matcher)]
    optimizer = optim.AdamW(
        list(flow_matcher.parameters())
        + list(predictor.parameters())
        + list(discriminator.parameters())
        + list(action_adapter.parameters())
        + list(action_down_proj.parameters()),
        lr=config["stage3"]["lr"],
    )

    env = GR1Stage3Env(action_dim=config["model"]["action_dim"])
    criterion = nn.MSELoss()
    bce = nn.BCELoss()

    # Success rate tracker for Skill0.5 routing gate
    recent_successes = []

    epochs = 5 if use_subset else config["stage3"]["epochs"]
    for epoch in range(epochs):
        epoch_generator_loss = 0.0
        epoch_disc_loss = 0.0

        num_episodes = 5 if use_subset else 10
        for episode in range(num_episodes):
            obs = env.reset()
            memory = HybridMemoryTriad()
            done = False

            # Collect episode trajectory history
            trajectory_history = []
            final_reward = -2.0

            # Calculate current success rate
            success_rate = (
                sum(recent_successes) / len(recent_successes)
                if recent_successes
                else 0.5
            )
            is_easy_task = success_rate > 0.8
            is_hard_task = success_rate < 0.3

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

                # Fused state representation (MSAT)
                s_t = msat(modality_dict)

                # Memory Grounding: Add gist context residually
                if len(memory.short_term) >= 2:
                    s_t = s_t + memory.gist.to(device)

                # Determine goal configuration state from final target coordinate
                # During online rollouts, target represents task context.
                s_target = s_t.clone()

                # --- 1. POLICY ACTION PROPOSAL (COMBOSTOC) ---
                pred_action = flow_matcher.sample_with_steering(
                    s_t, s_target, num_steps=10
                )

                # --- 2. SKILL0.5 ADVERSARIAL ATTACKER (EASY TASKS ONLY) ---
                perturbed_s_t = s_t.clone()
                if is_easy_task:
                    # Run aggressive BadWorld perceptual gaslighting to destroy shortcuts
                    perturbed_s_t = attacker.generate_perturbed_context(
                        flow_matcher, s_t, s_target, pred_action
                    )

                # Step simulation
                next_obs, reward, done, info = env.step(pred_action.detach())
                memory.update(s_t, info["tactile_spike"])

                # Store transition history for RAM regression training
                trajectory_history.append(
                    {
                        "s_t": perturbed_s_t.detach(),
                        "s_target": s_target.detach(),
                        "action": pred_action.detach(),
                        "next_obs": next_obs,
                        "reward": reward,
                    }
                )

                obs = next_obs
                final_reward = reward

            # Track task success (target distance threshold)
            task_success = float(info["target_dist"] < 0.05)
            recent_successes.append(task_success)
            if len(recent_successes) > 10:
                recent_successes.pop(0)

            # --- RAM (REINFORCE ADJOINT MATCHING) EPISODIC UPDATE ---
            if len(trajectory_history) > 0:
                optimizer.zero_grad()

                # 1. Sample a random intermediate timestep from history
                sample_idx = torch.randint(0, len(trajectory_history), (1,)).item()
                transition = trajectory_history[sample_idx]

                s_t_sample = transition["s_t"]
                s_target_sample = transition["s_target"]
                a_sample = transition["action"]

                # 2. Reconstruct intermediate flow matching step (RAM step)
                x_0 = torch.randn_like(a_sample)
                t_rand = torch.rand(a_sample.size(0), 1, device=device)

                # Expand t_rand to vector for ComboStoc flow compatibility
                t_vector = t_rand.expand(-1, flow_matcher.action_dim)

                x_t = t_vector * a_sample + (1.0 - t_vector) * x_0
                target_vel = a_sample - x_0

                # Predict velocity
                pred_vel = flow_matcher.velocity_field(
                    x_t, t_vector, s_t_sample, s_target_sample
                )
                cfm_loss = criterion(pred_vel, target_vel)

                # 3. Scale flow gradients directly by final endpoint reward (RAM)
                ram_scale = max(
                    0.1, 1.0 + final_reward
                )  # final_reward is negative target distance
                ram_loss = cfm_loss * ram_scale

                # 4. Trajectory Realism Discriminator (DRL) feedback
                real_score = discriminator(a_sample, s_t_sample)
                loss_real = bce(real_score, torch.ones_like(real_score))

                fake_score = discriminator(a_sample.detach(), s_t_sample)
                loss_fake = bce(fake_score, torch.zeros_like(fake_score))
                loss_disc = 0.5 * (loss_real + loss_fake)

                # 5. Contrastive EBM dynamics predictor training
                z_action_expert = action_adapter(a_sample)
                z_action_expert_16 = action_down_proj(z_action_expert)
                s_next_pred = predictor(s_t_sample, z_action_expert_16)

                # Target state at t+1
                next_obs = transition["next_obs"]
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
                s_next = msat(modality_dict_next).detach()
                loss_ebm = criterion(s_next_pred, s_next)

                # 6. Privileged d-OPSD teacher-student distillation (HARD TASKS ONLY)
                distill_loss = torch.tensor(0.0, device=device)
                if is_hard_task and len(policy_checkpoints) > 0:
                    best_policy = policy_checkpoints[-1]
                    with torch.no_grad():
                        teacher_action = best_policy.sample(
                            s_t_sample, s_target_sample, num_steps=10
                        )
                    distill_loss = criterion(a_sample, teacher_action) * 1.5

                # Aggregate total generator loss
                total_gen_loss = (
                    loss_ebm * config["stage3"]["ram_weight"]
                    + bce(real_score, torch.zeros_like(real_score))
                    * config["stage3"]["adv_weight"]
                    + ram_loss * config["stage3"]["distill_weight"]
                    + distill_loss * 0.5
                )

                total_gen_loss.backward()
                optimizer.step()
                epoch_generator_loss += total_gen_loss.item()

                # Train discriminator separately
                optimizer.zero_grad()
                loss_disc.backward()
                optimizer.step()
                epoch_disc_loss += loss_disc.item()

        # Update Auto-NPO checkpoint sliding pool
        policy_checkpoints.append(copy.deepcopy(flow_matcher))
        if len(policy_checkpoints) > 5:
            policy_checkpoints.pop(0)

        print(
            f"Epoch {epoch+1:03d} | RL Gen Loss: {epoch_generator_loss/num_episodes:.5f} | Disc Loss: {epoch_disc_loss/num_episodes:.5f} | Success Rate: {success_rate:.2f}"
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
