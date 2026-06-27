import torch
import torch.optim as optim
import torch.nn as nn
from models.adapters import VisualAdapter, TextAdapter, PointNeXtAdapter
from models.msat import MultiStreamActionTransformer
from models.jepa_predictor import LatentActionEncoder, JepaPredictor
from utils.dataset_loader import get_dataloader


def train_stage1(config):
    print("--- STARTING STAGE 1: LATENT DYNAMICS PRE-TRAINING ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    vis_adapter = VisualAdapter().to(device)
    txt_adapter = TextAdapter().to(device)
    pt_adapter = PointNeXtAdapter().to(device)
    msat = MultiStreamActionTransformer().to(device)

    latent_action_encoder = LatentActionEncoder(
        bottleneck_dim=config["model"]["bottleneck_dim"]
    ).to(device)
    predictor = JepaPredictor(action_dim=config["model"]["bottleneck_dim"]).to(device)

    optimizer = optim.AdamW(
        list(vis_adapter.parameters())
        + list(txt_adapter.parameters())
        + list(pt_adapter.parameters())
        + list(msat.parameters())
        + list(latent_action_encoder.parameters())
        + list(predictor.parameters()),
        lr=config["stage1"]["lr"],
        weight_decay=config["stage1"]["weight_decay"],
    )

    dataloader = get_dataloader(
        data_dir=config["paths"]["dataset_dir"],
        seq_len=config["model"]["horizon"],
        batch_size=config["stage1"]["batch_size"],
    )

    criterion = nn.MSELoss()

    for epoch in range(config["stage1"]["epochs"]):
        epoch_loss = 0.0
        no_op_losses = 0.0
        drift_sums = 0.0
        batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # Map inputs to device
            vision = batch["vision"].to(device)
            text = batch["text"].to(device)
            pointnext = batch["pointnext"].to(device)

            batch_size = vision.size(0)
            horizon = vision.size(1)

            # Loop over transitions in horizon
            step_losses = []
            for t in range(horizon - 1):
                # 1. Project current modal states
                vis_tok = vis_adapter(vision[:, t, :])
                txt_tok = txt_adapter(text.squeeze(1))
                pt_tok = pt_adapter(pointnext[:, t, :])

                # Fused context s_t (Tactile is masked to zeroes during pretraining)
                tactile_mask = torch.zeros(batch_size, 512, device=device)
                modality_dict = {
                    "vision": vis_tok,
                    "text": txt_tok,
                    "pointnext": pt_tok,
                    "tactile": tactile_mask,
                }
                s_t = msat(modality_dict)

                # 2. Project target future state s_next
                with torch.no_grad():
                    vis_tok_next = vis_adapter(vision[:, t + 1, :])
                    pt_tok_next = pt_adapter(pointnext[:, t + 1, :])
                    modality_dict_next = {
                        "vision": vis_tok_next,
                        "text": txt_tok,
                        "pointnext": pt_tok_next,
                        "tactile": tactile_mask,
                    }
                    s_next = msat(modality_dict_next)

                # 3. Extract latent action using training-only helper h_ψ
                z_latent = latent_action_encoder(s_t, s_next)

                # 4. Predict future state using dynamics model g_θ
                s_next_pred = predictor(s_t, z_latent)

                # 5. Compute predictive transition loss
                loss = criterion(s_next_pred, s_next)
                step_losses.append(loss)

                # --- DIAGNOSTIC CALCULATION ---
                # A. No-Op Loss metric
                with torch.no_grad():
                    no_op_loss = criterion(s_t, s_next).item()
                    no_op_losses += loss.item() / max(no_op_loss, 1e-6)

                    # B. Action Perturbation Drift metric (using random noise action)
                    z_random = torch.randn_like(z_latent)
                    s_next_pred_rand = predictor(s_t, z_random)
                    drift = criterion(s_next_pred, s_next_pred_rand).item()
                    drift_sums += drift

            total_step_loss = torch.stack(step_losses).mean()
            total_step_loss.backward()
            optimizer.step()

            epoch_loss += total_step_loss.item()
            batches += 1

        mean_loss = epoch_loss / batches
        mean_noop = no_op_losses / (batches * (horizon - 1))
        mean_drift = drift_sums / (batches * (horizon - 1))

        print(
            f"Epoch {epoch+1:03d} | Loss: {mean_loss:.5f} | No-Op Ratio: {mean_noop:.3f} | Action Drift: {mean_drift:.4f}"
        )

    print("--- STAGE 1 PRE-TRAINING COMPLETE ---")
    # Save pre-trained checkpoints
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    torch.save(
        {
            "vis_adapter": vis_adapter.state_dict(),
            "txt_adapter": txt_adapter.state_dict(),
            "pt_adapter": pt_adapter.state_dict(),
            "msat": msat.state_dict(),
            "predictor": predictor.state_dict(),
        },
        os.path.join(config["paths"]["checkpoint_dir"], "stage1_pretrained.pt"),
    )
