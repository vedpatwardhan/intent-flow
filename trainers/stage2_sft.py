import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
from utils.safety_filter import SafetyFilter


class Stage2SFTSimplified(pl.LightningModule):
    """
    SFT training module incorporating:
      - pl.LightningModule loop structure
      - WandB logging
      - InfoNCE CASA (Contrastive Action-State Alignment) Loss
      - Multi-stream ComboStoc asynchronous masking
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # 1. Initialize Adapters and MSAT
        self.vis_adapter = VisualAdapter(d_in=384)
        self.txt_adapter = TextAdapter(d_in=512)
        self.pt_adapter = PointNeXtAdapter(d_in=384)
        self.vggt_adapter = VGGTAdapter(d_in=config["model"]["vggt_dim"])
        self.tactile_adapter = TactileAdapter()
        self.action_adapter = ActionAdapter(d_in=config["model"]["action_dim"])
        self.state_adapter = ActionAdapter(d_in=config["model"]["state_dim"])

        self.msat = MultiStreamActionTransformer()
        self.predictor = JepaPredictor(action_dim=512)
        self.flow_matcher = CLAPFlowMatcher(
            action_dim=config["model"]["action_dim"], config=config
        )
        self.safety_filter = SafetyFilter(urdf_path=config["paths"]["urdf_path"])

    def forward(self, batch):
        vision = batch["vision"]
        text = batch["text"]
        pointnext = batch["pointnext"]
        vggt = batch["vggt"]
        tactile = batch["tactile"]
        proprioception = batch["proprioception"]
        actions = batch["actions"]

        batch_size = vision.size(0)
        horizon = vision.size(1)

        step_losses = []
        casa_losses = []

        # Compute s_target (goal configuration state) at the end of the window (horizon - 1)
        t_target = horizon - 1
        vis_tok_tgt = self.vis_adapter(vision[:, t_target, :])
        txt_tok_tgt = self.txt_adapter(text.squeeze(1))
        pt_tok_tgt = self.pt_adapter(pointnext[:, t_target, :])
        vggt_tok_tgt = self.vggt_adapter(vggt[:, t_target, :])
        tactile_emb_tgt = self.tactile_adapter(tactile[:, t_target, :, :])
        proprio_tok_tgt = self.state_adapter(proprioception[:, t_target, :])

        modality_dict_tgt = {
            "vision": vis_tok_tgt,
            "text": txt_tok_tgt,
            "pointnext": pt_tok_tgt,
            "vggt": vggt_tok_tgt,
            "tactile": tactile_emb_tgt,
            "proprioception": proprio_tok_tgt,
        }
        s_target = self.msat(modality_dict_tgt)

        # Iterate step-by-step to compute CFM + CASA alignment
        for t in range(horizon - 1):
            # ComboStoc: Asynchronous multi-stream masking
            noise_ratio = self.config["stage2"]["combostoc_noise_ratio"]
            mask_vis = (
                torch.rand(batch_size, 1, 1, device=self.device) > noise_ratio
            ).float()
            mask_pt = (
                torch.rand(batch_size, 1, 1, device=self.device) > noise_ratio
            ).float()
            mask_tac = (
                torch.rand(batch_size, 1, 1, device=self.device) > noise_ratio
            ).float()

            vis_tok = self.vis_adapter(vision[:, t, :]) * mask_vis
            txt_tok = self.txt_adapter(text.squeeze(1))
            pt_tok = self.pt_adapter(pointnext[:, t, :]) * mask_pt
            vggt_tok = self.vggt_adapter(vggt[:, t, :])
            tactile_emb = self.tactile_adapter(tactile[:, t, :, :]) * mask_tac
            proprio_tok = self.state_adapter(proprioception[:, t, :])

            modality_dict = {
                "vision": vis_tok,
                "text": txt_tok,
                "pointnext": pt_tok,
                "vggt": vggt_tok,
                "tactile": tactile_emb,
                "proprioception": proprio_tok,
            }
            s_t = self.msat(modality_dict)

            # Ground truth joint target at step t
            a_target = actions[:, t, :]

            # CFM Loss with dual-state conditioning
            cfm_loss = self.flow_matcher.get_cfm_loss(a_target, s_t, s_target)

            # CASA Contrastive Alignment (InfoNCE)
            # Projects target action and state into unified latent alignment space
            z_s = s_t / (s_t.norm(dim=-1, keepdim=True) + 1e-8)
            z_a = self.action_adapter(a_target)
            z_a = z_a / (z_a.norm(dim=-1, keepdim=True) + 1e-8)

            # InfoNCE Similarity Matrix
            sim_matrix = torch.matmul(z_s, z_a.T) / 0.07  # temperature=0.07
            labels = torch.arange(batch_size, device=self.device)
            casa_loss = F.cross_entropy(sim_matrix, labels)
            casa_losses.append(casa_loss)

            # Safety filter constraint loss evaluation
            with torch.no_grad():
                pred_action = self.flow_matcher.sample(s_t, s_target, num_steps=10)
                filtered_action = self.safety_filter.filter_actions(pred_action)
                constraint_loss = torch.mean((pred_action - filtered_action) ** 2)

            total_loss = cfm_loss + 0.1 * constraint_loss
            step_losses.append(total_loss)

        mean_step_loss = torch.stack(step_losses).mean()
        mean_casa_loss = torch.stack(casa_losses).mean()
        return mean_step_loss, mean_casa_loss

    def training_step(self, batch, batch_idx):
        loss_cfm, loss_casa = self(batch)
        total_loss = loss_cfm + 0.2 * loss_casa

        self.log(
            "train_cfm_loss", loss_cfm, on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train_casa_loss", loss_casa, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_cfm, loss_casa = self(batch)
        total_loss = loss_cfm + 0.2 * loss_casa
        self.log("val_cfm_loss", loss_cfm, on_epoch=True, prog_bar=False)
        self.log("val_casa_loss", loss_casa, on_epoch=True, prog_bar=False)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=False)
        return total_loss

    def configure_optimizers(self):
        params = (
            list(self.vis_adapter.parameters())
            + list(self.txt_adapter.parameters())
            + list(self.pt_adapter.parameters())
            + list(self.vggt_adapter.parameters())
            + list(self.tactile_adapter.parameters())
            + list(self.state_adapter.parameters())
            + list(self.action_adapter.parameters())
            + list(self.msat.parameters())
            + list(self.flow_matcher.parameters())
            + list(self.predictor.parameters())
        )
        return optim.AdamW(params, lr=self.config["stage2"]["lr"])


class EpochMetricsTableCallback(pl.Callback):
    """Prints a beautiful table with training and validation metrics at the end of each epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        train_cfm = metrics.get("train_cfm_loss") or metrics.get("train_cfm_loss_step")
        train_casa = metrics.get("train_casa_loss")
        train_total = metrics.get("train_total_loss")

        val_cfm = metrics.get("val_cfm_loss")
        val_casa = metrics.get("val_casa_loss")
        val_total = metrics.get("val_total_loss")

        def fmt(val):
            return f"{val.item():.5f}" if val is not None else "N/A"

        print(
            f"\n================ EPOCH {epoch} METRICS SUMMARY ================"
            f"\n  Metric              | Training    | Validation"
            "\n  --------------------+-------------+-------------"
            f"\n  CFM Loss            | {fmt(train_cfm):<11} | {fmt(val_cfm):<11}"
            f"\n  CASA Loss           | {fmt(train_casa):<11} | {fmt(val_casa):<11}"
            f"\n  Total Loss          | {fmt(train_total):<11} | {fmt(val_total):<11}"
            f"\n===============================================================\n"
        )


def train_stage2(config, use_subset=False):
    print(
        "--- STARTING STAGE 2: SUPERVISED FINE-TUNING & ACTION GROUNDING (PL & W&B) ---"
    )

    # 1. Initialize SFT Module
    model = Stage2SFTSimplified(config)

    # 2. Load Stage 1 Pre-trained weights if available
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    stage1_ckpt_path = os.path.join(checkpoint_dir, "stage1_pretrained.pt")
    if os.path.exists(stage1_ckpt_path):
        print(f"[SFT] Loading Stage 1 Pretrained weights from: {stage1_ckpt_path}")
        checkpoint = torch.load(stage1_ckpt_path, map_location="cpu")

        # Load matched dictionary parameters
        model.vis_adapter.load_state_dict(checkpoint["vis_adapter"])
        model.txt_adapter.load_state_dict(checkpoint["txt_adapter"])
        model.pt_adapter.load_state_dict(checkpoint["pt_adapter"])
        model.vggt_adapter.load_state_dict(checkpoint["vggt_adapter"])
        model.msat.load_state_dict(checkpoint["msat"])

    # 3. Setup dataloader (pulls from Aloha SFT split)
    s2_data_dir = os.path.join(config["paths"]["dataset_dir"], "sft")
    train_loader, val_loader = get_dataloader(
        data_dir=s2_data_dir,
        seq_len=config["model"]["horizon"],
        batch_size=config["stage2"]["batch_size"],
        use_subset=use_subset,
        validation_split=0.1,
    )

    # 4. Setup W&B Logger
    wandb_config = config.get("wandb", {})
    wandb_logger = WandbLogger(
        project=wandb_config.get("project", "latentflow-stage2"),
        entity=wandb_config.get("entity", None),
        log_model=False,
    )

    # 5. Checkpointing callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="stage2-{epoch:02d}-{val_total_loss:.5f}",
        save_top_k=3,
        monitor="val_total_loss",
        mode="min",
    )

    # 6. Trainer Execution
    trainer = pl.Trainer(
        max_epochs=config["stage2"]["epochs"],
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, EpochMetricsTableCallback()],
    )
    trainer.fit(model, train_loader, val_loader)

    # Save final stage2 dict weights
    final_path = os.path.join(checkpoint_dir, "stage2_sft.pt")
    print(f"Saving final Stage 2 SFT weights to: {final_path}")
    torch.save(
        {
            "vis_adapter": model.vis_adapter.state_dict(),
            "txt_adapter": model.txt_adapter.state_dict(),
            "pt_adapter": model.pt_adapter.state_dict(),
            "vggt_adapter": model.vggt_adapter.state_dict(),
            "tactile_adapter": model.tactile_adapter.state_dict(),
            "state_adapter": model.state_adapter.state_dict(),
            "msat": model.msat.state_dict(),
            "action_adapter": model.action_adapter.state_dict(),
            "predictor": model.predictor.state_dict(),
            "flow_matcher": model.flow_matcher.state_dict(),
        },
        final_path,
    )
