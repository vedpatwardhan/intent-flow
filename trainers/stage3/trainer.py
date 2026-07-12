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
from trainers.stage3.denoiser import ComboStocFlowMatcher
from trainers.stage3.discriminator import TrajectoryDiscriminator
from trainers.stage3.attacker import BadWorldAttacker


class CLAREFeatureDiscriminator(nn.Module):
    """
    CLARE Feature Discriminator.
    Monitors internal layer statistics during training using
    autoencoder-based reconstruction checks to flag out-of-distribution shifts.
    """

    def __init__(self, state_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.GELU(), nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128), nn.GELU(), nn.Linear(128, state_dim)
        )

    def get_novelty_score(self, s_t):
        with torch.no_grad():
            reconstructed = self.decoder(self.encoder(s_t))
            error = torch.mean((s_t - reconstructed) ** 2, dim=-1)
        return error.mean().item()


class GNNSkillLibrary(nn.Module):
    """
    Programmatic Graph Neural Network Skill Library (The Automaton).
    Manages structural skill nodes (z_i) and dynamically spawns new Mixture-of-Flows
    specialists when CLARE autoencoders flag sustained OOD novelty.
    """

    def __init__(self, base_flow_matcher, state_dim=512, novelty_threshold=0.08):
        super().__init__()
        self.state_dim = state_dim
        self.novelty_threshold = novelty_threshold

        # Core structural nodes mapped as explicit skill macro vectors
        self.nodes = nn.ParameterDict(
            {"skill_0": nn.Parameter(torch.randn(1, state_dim))}
        )

        # Sparse Mixture-of-Flows (MoF) specialist dictionary
        self.specialists = nn.ModuleDict({"skill_0": copy.deepcopy(base_flow_matcher)})

        # CLARE discriminators for lifelong OOD boundaries
        self.discriminators = nn.ModuleDict(
            {"skill_0": CLAREFeatureDiscriminator(state_dim=state_dim)}
        )

    def route_or_spawn(self, s_t, base_flow_matcher):
        """
        Evaluates the current state footprint. If all active skill nodes flag
        novelty beyond the threshold, it triggers an autonomous node-spawning event.
        """
        lowest_novelty = float("inf")
        best_skill_key = "skill_0"

        for key, discriminator in self.discriminators.items():
            score = discriminator.get_novelty_score(s_t)
            if score < lowest_novelty:
                lowest_novelty = score
                best_skill_key = key

        # Node-Spawning Logic (Skill0.5 / Auto-Expansion)
        if lowest_novelty > self.novelty_threshold:
            new_idx = len(self.nodes)
            new_key = f"skill_{new_idx}"
            print(
                f"[GNN-AUTOMATON] Novelty {lowest_novelty:.4f} > Threshold {self.novelty_threshold}. Spawning node: {new_key}"
            )

            # 1. Register continuous skill embedding node
            self.nodes[new_key] = nn.Parameter(
                torch.randn(1, self.state_dim, device=s_t.device)
            )
            # 2. Allocate a fresh, un-corrupted MoF specialist clone
            self.specialists[new_key] = copy.deepcopy(base_flow_matcher)
            # 3. Instantiate dedicated CLARE monitor
            self.discriminators[new_key] = CLAREFeatureDiscriminator(
                state_dim=self.state_dim
            ).to(s_t.device)

            return self.specialists[new_key], new_key

        return self.specialists[best_skill_key], best_skill_key


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
    print(
        "--- STARTING STRUCTURALLY REFACTORED STAGE 3: GNN, ANCHORED TARGETS & RAM MULTIPLIERS ---"
    )
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

    # Initialize GNN Skill Library container before epoch cycle
    gnn_library = GNNSkillLibrary(
        flow_matcher, state_dim=config["model"]["latent_dim"]
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

    # Create a frozen copy of the initial SFT base model to act as the uncorrupted global teacher for d-OPSD distillation
    base_teacher = copy.deepcopy(flow_matcher).to(device)
    for p in base_teacher.parameters():
        p.requires_grad = False
    base_teacher.eval()

    # Node parameters are registered inside the optimizer
    optimizer = optim.AdamW(
        list(flow_matcher.parameters())
        + list(gnn_library.parameters())
        + list(predictor.parameters())
        + list(discriminator.parameters())
        + list(action_adapter.parameters())
        + list(action_down_proj.parameters()),
        lr=config["stage3"]["lr"],
    )

    from trainers.stage3.env import GR1Stage3Env
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

                # MemoryWAM Anchor Target Grounding (Instead of flat self-cloning)
                if len(memory.anchors) > 0:
                    s_target = memory.anchors[-1].to(device)
                else:
                    s_target = s_t.clone()

                # Route state tokens through the GNN Continuum to select active MoF specialist head
                active_policy, active_node_key = gnn_library.route_or_spawn(
                    s_t, flow_matcher
                )

                # --- 1. POLICY ACTION PROPOSAL (COMBOSTOC) ---
                pred_action = active_policy.sample_with_steering(
                    s_t, s_target, num_steps=10
                )

                # --- 2. SKILL0.5 ADVERSARIAL ATTACKER (EASY TASKS ONLY) ---
                perturbed_s_t = s_t.clone()
                if is_easy_task:
                    # Run aggressive BadWorld perceptual gaslighting to destroy shortcuts
                    perturbed_s_t = attacker.generate_perturbed_context(
                        active_policy, s_t, s_target, pred_action
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
                        "active_policy": active_policy,
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
                policy_target = transition["active_policy"]

                # 2. Reconstruct intermediate flow matching step (RAM step)
                x_0 = torch.randn_like(a_sample)
                t_rand = torch.rand(a_sample.size(0), 1, device=device)

                # Expand t_rand to vector for ComboStoc flow compatibility
                t_vector = t_rand.expand(-1, policy_target.action_dim)

                x_t = t_vector * a_sample + (1.0 - t_vector) * x_0
                target_vel = a_sample - x_0

                # Predict velocity
                pred_vel = policy_target.velocity_field(
                    x_t, t_vector, s_t_sample, s_target_sample
                )
                cfm_loss = criterion(pred_vel, target_vel)

                # 3. Trajectory Realism Discriminator (DRL) feedback
                real_score = discriminator(a_sample, s_t_sample)
                loss_real = bce(real_score, torch.ones_like(real_score))

                fake_score = discriminator(a_sample.detach(), s_t_sample)
                loss_fake = bce(fake_score, torch.zeros_like(fake_score))
                loss_disc = 0.5 * (loss_real + loss_fake)

                # 4. Unified RAM Scalar Equation (Multiplying CFM loss directly by the combined reward)
                combined_reward = torch.clamp(
                    1.0 + final_reward + real_score.detach(), min=0.05
                )
                ram_loss = cfm_loss * combined_reward.item()

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
                if is_hard_task:
                    # Distill from the uncorrupted global teacher copy to maintain correct trajectory manifold
                    with torch.no_grad():
                        teacher_action = base_teacher.sample(
                            s_t_sample, s_target_sample, num_steps=10
                        )
                    distill_loss = criterion(a_sample, teacher_action) * 1.5

                # Aggregate total generator loss
                total_gen_loss = (
                    loss_ebm * config["stage3"]["ram_weight"]
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

        print(
            f"Epoch {epoch+1:03d} | RL Gen Loss: {epoch_generator_loss/num_episodes:.5f} | Disc Loss: {epoch_disc_loss/num_episodes:.5f} | Success Rate: {success_rate:.2f} | Spawned Skills: {len(gnn_library.nodes)}"
        )

    # Save final Stage 3 weights
    final_path = os.path.join(checkpoint_dir, "stage3_rl_final.pt")
    print(f"Saving final Stage 3 RL weights to: {final_path}")

    # Save active GNN skill nodes and specialist dictionaries
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
            "gnn_nodes": gnn_library.nodes.state_dict(),
            "gnn_specialists": gnn_library.specialists.state_dict(),
        },
        final_path,
    )
