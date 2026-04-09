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


class WandbStepReconstructionLogger(pl.Callback):
    """
    Every `log_every_n_steps` training batches, run the model on a cached val batch
    and log [GT | Reconstruction] side-by-side to WandB.

    Args:
        log_every_n_steps: how often to log (in batch_idx, not global_step)
        num_images: number of image pairs per log (default 2)
    """

    def __init__(self, log_every_n_steps=1000, num_images=2):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_images = num_images
        self._val_batch = None
        self._batch_count = 0

    def _get_wandb_logger(self, trainer):
        from lightning.pytorch.loggers import WandbLogger
        loggers = trainer.loggers if hasattr(trainer, "loggers") else []
        if not loggers and trainer.logger is not None:
            loggers = [trainer.logger]
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def _cache_val_batch(self, trainer, pl_module):
        if self._val_batch is not None:
            return
        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        x = batch["image"]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous().float()
        self._val_batch = x[: self.num_images]

    @torch.no_grad()
    def _log_reconstructions(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            print("[WandbStepReconstructionLogger] WARNING: no WandbLogger found")
            return

        import wandb

        self._cache_val_batch(trainer, pl_module)
        x = self._val_batch.to(pl_module.device)

        if hasattr(pl_module, "use_ema") and pl_module.use_ema:
            with pl_module.ema_scope():
                xrec, _ = pl_module(x)
        else:
            xrec, _ = pl_module(x)
        xrec = xrec.clamp(-1, 1)

        images = []
        for i in range(min(self.num_images, x.shape[0])):
            gt = ((x[i].cpu().float() + 1) * 127.5).clamp(0, 255).byte()
            rec = ((xrec[i].cpu().float() + 1) * 127.5).clamp(0, 255).byte()
            # Side by side: [C, H, 2W]
            side_by_side = torch.cat([gt, rec], dim=2)
            img_np = side_by_side.permute(1, 2, 0).numpy()
            images.append(wandb.Image(img_np, caption=f"GT (left) | Recon (right) #{i}"))

        wandb_logger.experiment.log(
            {"val/reconstruction_sidebyside": images, "trainer/global_step": trainer.global_step},
        )
        print(f"[WandbStepReconstructionLogger] Logged {len(images)} images at batch {self._batch_count}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_count += 1
        if self._batch_count > 0 and self._batch_count % self.log_every_n_steps == 0:
            torch.cuda.empty_cache()
            pl_module.eval()
            try:
                self._log_reconstructions(trainer, pl_module)
            except Exception as e:
                print(f"[WandbStepReconstructionLogger] Failed to log: {e}")
            pl_module.train()
