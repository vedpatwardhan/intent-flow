import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.adapters import VisualAdapter, TextAdapter, PointNeXtAdapter, VGGTAdapter
from models.msat import MultiStreamActionTransformer
from models.jepa_predictor import LatentActionEncoder, JepaPredictor
from utils.dataset_loader import get_dataloader


class JEPAStage1Module(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the Stage 1 JEPA pre-training loop.
    Optimizes latent dynamics prediction: s_{t+1} = predictor(s_t, z_t)
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Load hyperparams from config
        self.latent_dim = config["model"].get("latent_dim", 512)
        self.bottleneck_dim = config["model"].get("bottleneck_dim", 16)
        self.vggt_dim = config["model"].get("vggt_dim", 768)
        self.lr = config["stage1"].get("lr", 1e-4)
        self.weight_decay = config["stage1"].get("weight_decay", 1e-4)

        # Initialize network modules
        # DINOv3 output dimension is 384
        self.vis_adapter = VisualAdapter(d_in=384, d_out=self.latent_dim)
        self.txt_adapter = TextAdapter(d_in=512, d_out=self.latent_dim)
        self.pt_adapter = PointNeXtAdapter(d_in=384, d_out=self.latent_dim)
        self.vggt_adapter = VGGTAdapter(d_in=self.vggt_dim, d_out=self.latent_dim)

        self.msat = MultiStreamActionTransformer(
            latent_dim=self.latent_dim,
            num_heads=config["model"].get("num_heads", 8),
            num_layers=config["model"].get("num_layers", 4),
            dropout=config["model"].get("dropout", 0.1),
        )

        self.latent_action_encoder = LatentActionEncoder(
            state_dim=self.latent_dim, bottleneck_dim=self.bottleneck_dim
        )

        self.predictor = JepaPredictor(
            state_dim=self.latent_dim,
            action_dim=self.bottleneck_dim,
            hidden_dim=self.latent_dim,
        )

        self.criterion = nn.MSELoss()

    def forward(self, batch):
        # Forward pass is implemented as sequence transition dynamics
        vision = batch["vision"]  # [B, T, 384]
        text = batch["text"]  # [B, 1, 768]
        pointnext = batch["pointnext"]  # [B, T, 384]
        vggt = batch["vggt"]  # [B, T, 768]

        batch_size = vision.size(0)
        horizon = vision.size(1)

        step_losses = []
        no_op_ratios = []
        drifts = []

        for t in range(horizon - 1):
            # 1. Project current modal states
            vis_tok = self.vis_adapter(vision[:, t, :])
            txt_tok = self.txt_adapter(text.squeeze(1))
            pt_tok = self.pt_adapter(pointnext[:, t, :])
            vggt_tok = self.vggt_adapter(vggt[:, t, :])

            # Fused context s_t (Tactile is masked to zeroes during pretraining)
            tactile_mask = torch.zeros(batch_size, self.latent_dim, device=self.device)
            modality_dict = {
                "vision": vis_tok,
                "text": txt_tok,
                "pointnext": pt_tok,
                "vggt": vggt_tok,
                "tactile": tactile_mask,
            }
            s_t = self.msat(modality_dict)

            # 2. Project target future state s_next
            # We keep gradients active for msat/adapters during target projection
            vis_tok_next = self.vis_adapter(vision[:, t + 1, :])
            pt_tok_next = self.pt_adapter(pointnext[:, t + 1, :])
            vggt_tok_next = self.vggt_adapter(vggt[:, t + 1, :])

            modality_dict_next = {
                "vision": vis_tok_next,
                "text": txt_tok,
                "pointnext": pt_tok_next,
                "vggt": vggt_tok_next,
                "tactile": tactile_mask,
            }
            s_next = self.msat(modality_dict_next)

            # 3. Extract latent action using training helper h_ψ
            z_latent = self.latent_action_encoder(s_t, s_next)

            # 4. Predict future state using dynamics predictor g_θ
            s_next_pred = self.predictor(s_t, z_latent)

            # 5. Compute predictive transition loss
            loss = self.criterion(s_next_pred, s_next)
            step_losses.append(loss)

            # --- DIAGNOSTIC CALCULATION ---
            with torch.no_grad():
                # A. No-Op Loss Ratio
                no_op_loss = self.criterion(s_t, s_next).item()
                no_op_ratios.append(loss.item() / max(no_op_loss, 1e-6))

                # B. Action Perturbation Drift metric (using random noise action)
                z_random = torch.randn_like(z_latent)
                s_next_pred_rand = self.predictor(s_t, z_random)
                drift = self.criterion(s_next_pred, s_next_pred_rand).item()
                drifts.append(drift)

        # Average losses over the horizon
        mean_step_loss = torch.stack(step_losses).mean()
        mean_noop = sum(no_op_ratios) / len(no_op_ratios)
        mean_drift = sum(drifts) / len(drifts)

        return mean_step_loss, mean_noop, mean_drift

    def training_step(self, batch, batch_idx):
        loss, noop, drift = self(batch)

        # Log training metrics: only step loss goes to the progress bar to avoid wrapping
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_noop_ratio", noop, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train_action_drift", drift, on_step=False, on_epoch=True, prog_bar=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, noop, drift = self(batch)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_noop_ratio", noop, on_epoch=True, prog_bar=False)
        self.log("val_action_drift", drift, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        # Group parameters across all sub-modules
        params = (
            list(self.vis_adapter.parameters())
            + list(self.txt_adapter.parameters())
            + list(self.pt_adapter.parameters())
            + list(self.vggt_adapter.parameters())
            + list(self.msat.parameters())
            + list(self.latent_action_encoder.parameters())
            + list(self.predictor.parameters())
        )
        return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


def train_stage1(config, use_subset=False):
    print("--- STARTING STAGE 1: LATENT DYNAMICS PRE-TRAINING (PL & W&B) ---")

    # Resolve local dataset path dynamically
    data_dir = config["paths"]["dataset_dir"]
    if os.path.exists("latent-flow/data/processed"):
        data_dir = "latent-flow/data/processed"
        print(
            f"[Trainer] Overriding dataset directory with local processed path: {data_dir}"
        )

    # 1. Initialize PyTorch Dataloaders with centralized helper
    num_workers = config.get("num_workers", 2)
    train_loader, val_loader = get_dataloader(
        data_dir=data_dir,
        seq_len=config["model"]["horizon"],
        batch_size=config["stage1"]["batch_size"],
        use_subset=use_subset,
        validation_split=0.1,
        num_workers=num_workers,
    )

    # 2. Setup W&B Logger
    wandb_config = config.get("wandb", {})
    log_model_val = wandb_config.get("log_model", False)
    if isinstance(log_model_val, str):
        if log_model_val.lower() == "false":
            log_model_val = False
        elif log_model_val.lower() == "true":
            log_model_val = True

    wandb_logger = WandbLogger(
        project=wandb_config.get("project", "latentflow-stage1"),
        entity=wandb_config.get("entity", None),
        log_model=log_model_val,
    )

    # 3. Setup Checkpoint Callbacks
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    subdir = config["paths"].get("subdir", "")
    if subdir:
        checkpoint_dir = os.path.join(checkpoint_dir, subdir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=(
            "stage1-{epoch:02d}-{val_loss:.5f}" if val_loader else "stage1-{epoch:02d}"
        ),
        save_top_k=3,
        monitor="val_loss" if val_loader else "train_loss",
        mode="min",
    )

    # 4. Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config["stage1"]["epochs"],
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )

    # 5. Build Model Module
    model = JEPAStage1Module(config)

    # 6. Start Training
    if val_loader:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)
    print("--- STAGE 1 PRE-TRAINING COMPLETE ---")
