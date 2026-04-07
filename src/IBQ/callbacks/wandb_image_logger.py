"""Callback that logs GT vs reconstructed images from the first validation batch to WandB."""

import torch
import lightning.pytorch as pl


class WandbReconstructionLogger(pl.Callback):
    """Log ground-truth vs reconstructed images for the first validation batch each epoch."""

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0:
            return
        logger = trainer.logger
        if logger is None:
            return
        # Only log from rank 0
        if trainer.global_rank != 0:
            return

        try:
            import wandb
        except ImportError:
            return

        with torch.no_grad():
            log_dict = pl_module.log_images(batch)

        if not log_dict:
            return

        inputs = log_dict["inputs"]          # [B, 3, H, W] in [-1, 1]
        recons = log_dict["reconstructions"]  # [B, 3, H, W] in [-1, 1]

        # Clamp and convert to [0, 1]
        inputs = (inputs.clamp(-1, 1) + 1) / 2
        recons = (recons.clamp(-1, 1) + 1) / 2

        images = []
        n = min(inputs.size(0), 8)  # log up to 8 pairs
        for i in range(n):
            gt = wandb.Image(inputs[i].cpu().float(), caption=f"GT_{i}")
            rc = wandb.Image(recons[i].cpu().float(), caption=f"Recon_{i}")
            images.extend([gt, rc])

        step = trainer.global_step
        logger.experiment.log({"val/reconstructions": images, "trainer/global_step": step})
