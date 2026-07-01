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
        """
        traj: [Batch, ActionDim]
        s_t: [Batch, StateDim]
        """
        inputs = torch.cat([traj, s_t], dim=-1)
        return self.net(inputs)


def train_stage3(config):
    print("--- STARTING STAGE 3: RL ALIGNMENT, ENERGY SCULPTING & OPSD ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Stage 2 SFT weights
    checkpoint = torch.load(
        os.path.join(config["paths"]["checkpoint_dir"], "stage2_sft.pt"),
        map_location=device,
    )

    # Initialize networks
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

    # Load weights
    vis_adapter.load_state_dict(checkpoint["vis_adapter"])
    txt_adapter.load_state_dict(checkpoint["txt_adapter"])
    pt_adapter.load_state_dict(checkpoint["pt_adapter"])
    vggt_adapter.load_state_dict(checkpoint["vggt_adapter"])
    tactile_adapter.load_state_dict(checkpoint["tactile_adapter"])
    action_adapter.load_state_dict(checkpoint["action_adapter"])
    msat.load_state_dict(checkpoint["msat"])
    predictor.load_state_dict(checkpoint["predictor"])
    flow_matcher.load_state_dict(checkpoint["flow_matcher"])

    # Checkpoint pool for Auto-NPO
    policy_checkpoints = [copy.deepcopy(flow_matcher)]

    optimizer = optim.AdamW(
        list(flow_matcher.parameters())
        + list(predictor.parameters())
        + list(discriminator.parameters())
        + list(vis_adapter.parameters())
        + list(txt_adapter.parameters())
        + list(pt_adapter.parameters())
        + list(vggt_adapter.parameters())
        + list(tactile_adapter.parameters())
        + list(action_adapter.parameters())
        + list(msat.parameters()),
        lr=config["stage3"]["lr"],
    )

    dataloader = get_dataloader(
        data_dir=config["paths"]["dataset_dir"],
        seq_len=config["model"]["horizon"],
        batch_size=config["stage3"]["batch_size"],
    )

    criterion = nn.MSELoss()
    bce = nn.BCELoss()

    for epoch in range(config["stage3"]["epochs"]):
        epoch_rl_loss = 0.0
        epoch_disc_loss = 0.0
        batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            vision = batch["vision"].to(device)
            text = batch["text"].to(device)
            pointnext = batch["pointnext"].to(device)
            vggt = batch["vggt"].to(device)
            tactile = batch["tactile"].to(device)
            proprioception = batch["proprioception"].to(device)
            actions = batch["actions"].to(device)

            batch_size = vision.size(0)
            horizon = vision.size(1)

            # --- AUTO-NPO CHECKPOINT BOOTSTRAPPING ---
            # If t is multiple of evaluation steps, evaluate candidate future selves
            if batches > 0 and batches % config["stage3"]["auto_npo_interval"] == 0:
                print("Evaluating Auto-NPO Checkpoint Pool...")
                # In mock setup, we clone current policy to represent future checkpoints
                policy_checkpoints.append(copy.deepcopy(flow_matcher))
                if len(policy_checkpoints) > 5:
                    policy_checkpoints.pop(0)

            step_losses = []
            disc_losses = []

            for t in range(horizon - 1):
                # Project modalities
                with torch.no_grad():
                    vis_tok = vis_adapter(vision[:, t, :])
                    txt_tok = txt_adapter(text.squeeze(1))
                    pt_tok = pt_adapter(pointnext[:, t, :])
                    vggt_tok = vggt_adapter(vggt[:, t, :])
                    tactile_emb = tactile_adapter(tactile[:, t, :, :])

                modality_dict = {
                    "vision": vis_tok,
                    "text": txt_tok,
                    "pointnext": pt_tok,
                    "vggt": vggt_tok,
                    "tactile": tactile_emb,
                    "proprioception": proprioception[:, t, :],
                }
                s_t = msat(modality_dict)

                # Target future representation
                with torch.no_grad():
                    vis_tok_next = vis_adapter(vision[:, t + 1, :])
                    pt_tok_next = pt_adapter(pointnext[:, t + 1, :])
                    vggt_tok_next = vggt_adapter(vggt[:, t + 1, :])
                    modality_dict_next = {
                        "vision": vis_tok_next,
                        "text": txt_tok,
                        "pointnext": pt_tok_next,
                        "vggt": vggt_tok_next,
                        "tactile": tactile_adapter(tactile[:, t + 1, :, :]),
                        "proprioception": proprioception[:, t + 1, :],
                    }
                    s_next = msat(modality_dict_next)

                # --- 1. ACTIVE POLICY FLOW MATCHING PROPOSAL ---
                pred_action = flow_matcher.sample(s_t, num_steps=10)

                # --- 2. TRAJECTORY STYLE DISCRIMINATOR LOSS ---
                # Real: Teleoperated human action
                real_score = discriminator(actions[:, t, :], s_t)
                loss_real = bce(real_score, torch.ones_like(real_score))

                # Fake: Proposed RL action
                fake_score = discriminator(pred_action.detach(), s_t)
                loss_fake = bce(fake_score, torch.zeros_like(fake_score))

                loss_disc = 0.5 * (loss_real + loss_fake)
                disc_losses.append(loss_disc)

                # --- 3. EBM CONTRASTIVE LANDSCAPE SCULPTING ---
                # A. Positive path (teleoperated/expert action) -> Minimize energy
                z_action_expert = action_adapter(actions[:, t, :])
                s_next_pred_expert = predictor(s_t, z_action_expert)
                ebm_loss_positive = criterion(s_next_pred_expert, s_next)

                # B. Negative path (bad/failed action proposal, e.g. random perturbation)
                # Inject force perturbation to simulate BadWorld failure
                perturbed_action = pred_action.detach() + 0.5 * torch.randn_like(
                    pred_action
                )
                z_action_fail = action_adapter(perturbed_action)
                s_next_pred_fail = predictor(s_t, z_action_fail)
                # Maximize energy (prediction error) for failure path
                ebm_loss_negative = -criterion(s_next_pred_fail, s_next)

                loss_ebm = ebm_loss_positive + 0.1 * ebm_loss_negative

                # --- 4. d-OPSD SUFFIX-CONDITIONED DISTILLATION ---
                # Calculate OPSD regression target using the best policy checkpoint in pool
                best_policy = policy_checkpoints[-1]
                with torch.no_grad():
                    teacher_action = best_policy.sample(s_t, num_steps=10)

                distill_loss = criterion(pred_action, teacher_action)

                # Combine losses
                generator_loss = (
                    loss_ebm * config["stage3"]["ram_weight"]
                    + bce(discriminator(pred_action, s_t), torch.ones_like(fake_score))
                    * config["stage3"]["adv_weight"]
                    + distill_loss * config["stage3"]["distill_weight"]
                )
                step_losses.append(generator_loss)

            # Perform optimization step
            total_step_loss = torch.stack(step_losses).mean()
            total_step_loss.backward()
            optimizer.step()

            # Discriminator update
            optimizer.zero_grad()
            total_disc_loss = torch.stack(disc_losses).mean()
            total_disc_loss.backward()
            optimizer.step()

            epoch_rl_loss += total_step_loss.item()
            epoch_disc_loss += total_disc_loss.item()
            batches += 1

        mean_rl = epoch_rl_loss / batches
        mean_disc = epoch_disc_loss / batches
        print(
            f"Epoch {epoch+1:03d} | RL Loss: {mean_rl:.5f} | Disc Loss: {mean_disc:.5f}"
        )

    print("--- STAGE 3 RL COMPLETE ---")
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
        os.path.join(config["paths"]["checkpoint_dir"], "stage3_rl_final.pt"),
    )
