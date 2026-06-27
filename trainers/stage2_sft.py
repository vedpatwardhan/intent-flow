import torch
import torch.optim as optim
import torch.nn as nn
from models.adapters import (
    VisualAdapter,
    TextAdapter,
    PointNeXtAdapter,
    TactileAdapter,
    ActionAdapter,
)
from models.msat import MultiStreamActionTransformer
from models.jepa_predictor import JepaPredictor
from models.action_denoiser import CLAPFlowMatcher
from utils.dataset_loader import get_dataloader
from utils.safety_filter import SafetyFilter


def train_stage2(config):
    print("--- STARTING STAGE 2: SUPERVISED FINE-TUNING & ACTION GROUNDING ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Stage 1 Pre-trained weights
    checkpoint = torch.load(
        os.path.join(config["paths"]["checkpoint_dir"], "stage1_pretrained.pt"),
        map_location=device,
    )

    vis_adapter = VisualAdapter().to(device)
    txt_adapter = TextAdapter().to(device)
    pt_adapter = PointNeXtAdapter().to(device)
    msat = MultiStreamActionTransformer().to(device)
    predictor = JepaPredictor(action_dim=512).to(
        device
    )  # Now receives projected 512 action embeddings

    vis_adapter.load_state_dict(checkpoint["vis_adapter"])
    txt_adapter.load_state_dict(checkpoint["txt_adapter"])
    pt_adapter.load_state_dict(checkpoint["pt_adapter"])
    msat.load_state_dict(checkpoint["msat"])

    # 2. Initialize Stage 2 Specific Networks
    tactile_adapter = TactileAdapter().to(device)
    action_adapter = ActionAdapter(d_in=config["model"]["action_dim"], d_out=512).to(
        device
    )
    flow_matcher = CLAPFlowMatcher(action_dim=config["model"]["action_dim"]).to(device)
    safety_filter = SafetyFilter(urdf_path=config["paths"]["urdf_path"])

    optimizer = optim.AdamW(
        list(vis_adapter.parameters())
        + list(txt_adapter.parameters())
        + list(pt_adapter.parameters())
        + list(tactile_adapter.parameters())
        + list(msat.parameters())
        + list(action_adapter.parameters())
        + list(flow_matcher.parameters())
        + list(predictor.parameters()),
        lr=config["stage2"]["lr"],
    )

    dataloader = get_dataloader(
        data_dir=config["paths"]["dataset_dir"],
        seq_len=config["model"]["horizon"],
        batch_size=config["stage2"]["batch_size"],
    )

    for epoch in range(config["stage2"]["epochs"]):
        epoch_loss = 0.0
        batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # Map inputs to device
            vision = batch["vision"].to(device)
            text = batch["text"].to(device)
            pointnext = batch["pointnext"].to(device)
            tactile = batch["tactile"].to(device)
            proprioception = batch["proprioception"].to(device)
            actions = batch["actions"].to(device)

            batch_size = vision.size(0)
            horizon = vision.size(1)

            step_losses = []
            for t in range(horizon - 1):
                # ComboStoc: Asynchronous time-stepping regularizes cross-modal attention
                # We inject random masking/noise independently across different streams
                noise_mask = (
                    torch.rand(batch_size, 1, device=device)
                    > config["stage2"]["combostoc_noise_ratio"]
                )

                # Project current modalities
                vis_tok = vis_adapter(vision[:, t, :])
                txt_tok = txt_adapter(text.squeeze(1))
                pt_tok = pt_adapter(pointnext[:, t, :])

                # Tactile is now fully active
                tactile_emb = tactile_adapter(tactile[:, t, :, :])
                if not noise_mask.all():
                    # ComboStoc: occasionally mask tactile input to train robustness to contact noise
                    tactile_emb = tactile_emb * noise_mask.float()

                modality_dict = {
                    "vision": vis_tok,
                    "text": txt_tok,
                    "pointnext": pt_tok,
                    "tactile": tactile_emb,
                    "proprioception": proprioception[:, t, :],
                }
                s_t = msat(modality_dict)

                # Ground truth joint torque target at step t
                a_target = actions[:, t, :]

                # Apply CFM Flow Matching Loss
                cfm_loss = flow_matcher.get_cfm_loss(a_target, s_t)

                # Apply pycapacity limit constraints to proposed actions during training evaluation
                with torch.no_grad():
                    pred_action = flow_matcher.sample(s_t, num_steps=10)
                    filtered_action = safety_filter.filter_actions(pred_action)
                    constraint_loss = torch.mean((pred_action - filtered_action) ** 2)

                total_loss = cfm_loss + 0.1 * constraint_loss
                step_losses.append(total_loss)

            total_step_loss = torch.stack(step_losses).mean()
            total_step_loss.backward()
            optimizer.step()

            epoch_loss += total_step_loss.item()
            batches += 1

        mean_loss = epoch_loss / batches
        print(f"Epoch {epoch+1:03d} | SFT CFM Loss: {mean_loss:.5f}")

    print("--- STAGE 2 SFT COMPLETE ---")
    # Save SFT checkpoints
    torch.save(
        {
            "vis_adapter": vis_adapter.state_dict(),
            "txt_adapter": txt_adapter.state_dict(),
            "pt_adapter": pt_adapter.state_dict(),
            "tactile_adapter": tactile_adapter.state_dict(),
            "msat": msat.state_dict(),
            "action_adapter": action_adapter.state_dict(),
            "predictor": predictor.state_dict(),
            "flow_matcher": flow_matcher.state_dict(),
        },
        os.path.join(config["paths"]["checkpoint_dir"], "stage2_sft.pt"),
    )
