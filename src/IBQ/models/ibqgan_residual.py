"""
Residual IBQ (IB-RQ): Two-level Index Backpropagation Residual Quantization.

Level 1: Frozen pretrained IBQ encoder + codebook (e.g. TencentARC/IBQ-Tokenizer-1024).
Level 2: Trainable residual IBQ codebook that quantizes the residual r = z - z_q1.
Final reconstruction: z_final = z_q1 + r_q2 -> decoder -> image.

Only the level-2 codebook, post_quant_conv, decoder, and discriminator receive gradients.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import os
import torch
import torch.nn.functional as F
import lightning as L
from collections import OrderedDict

from main import instantiate_from_config
from src.IBQ.modules.diffusionmodules.model import Encoder, Decoder
from src.IBQ.modules.vqvae.quantize import IndexPropagationQuantize
from src.IBQ.modules.scheduler.lr_scheduler import (
    Scheduler_LinearWarmup,
    Scheduler_LinearWarmup_CosineDecay,
)
from src.IBQ.modules.ema import LitEma


def _load_pretrained_ibq(pretrained_path, config_file="config.yaml"):
    """Load pretrained IBQ config and state dict."""
    from omegaconf import OmegaConf

    cfg_path = os.path.join(pretrained_path, config_file)
    cfg = OmegaConf.load(cfg_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Try .ckpt files in directory
    ckpt_path = None
    for f in sorted(os.listdir(pretrained_path)):
        if f.endswith(".ckpt"):
            ckpt_path = os.path.join(pretrained_path, f)
            break

    if ckpt_path is None:
        raise FileNotFoundError(f"No .ckpt file found in {pretrained_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return cfg_dict, sd


class ResidualIBQ(L.LightningModule):
    """
    Two-level IBQ with frozen level-1 and trainable level-2 residual quantizer.
    """

    def __init__(
        self,
        pretrained_path,
        lossconfig,
        # Level-2 quantizer config
        l2_n_embed=1024,
        l2_embed_dim=256,
        l2_beta=0.25,
        l2_use_entropy_loss=True,
        l2_entropy_temperature=0.01,
        l2_sample_minimization_weight=1.0,
        l2_batch_maximization_weight=1.0,
        # Residual scaling
        residual_scale=1.0,
        # Training config
        config_file="config.yaml",
        image_key="image",
        learning_rate=1e-4,
        warmup_epochs=0,
        scheduler_type="None",
        min_learning_rate=0,
        gradient_clip_val=0,
        lr_drop_epoch=None,
        lr_drop_rate=0.1,
        resume_lr=None,
        use_ema=True,
        accumulate_grad_batches=1,
        ckpt_path=None,
        ignore_keys=None,
    ):
        super().__init__()
        ignore_keys = ignore_keys or []
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.gradient_clip_val = gradient_clip_val
        self.resume_lr = resume_lr
        self.lr_drop_epoch = lr_drop_epoch
        self.lr_drop_rate = lr_drop_rate
        self.accumulate_grad_batches = accumulate_grad_batches
        self.residual_scale = residual_scale

        # ---- Load pretrained level-1 config ----
        cfg_dict, state_dict = _load_pretrained_ibq(pretrained_path, config_file)
        ddconfig = cfg_dict["ddconfig"]
        l1_n_embed = cfg_dict["n_embed"]
        l1_embed_dim = cfg_dict["embed_dim"]
        l1_beta = cfg_dict.get("beta", 0.25)
        l1_use_entropy = cfg_dict.get("use_entropy_loss", True)
        l1_entropy_temp = cfg_dict.get("entropy_temperature", 0.01)

        # ---- Build level-1 components (frozen) ----
        self.encoder = Encoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], l1_embed_dim, 1)
        self.quantize_l1 = IndexPropagationQuantize(
            l1_n_embed,
            l1_embed_dim,
            l1_beta,
            l1_use_entropy,
            entropy_temperature=l1_entropy_temp,
        )

        # Load pretrained weights for level-1
        l1_prefixes = ("encoder.", "quant_conv.", "quantize.")
        l1_sd = OrderedDict()
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in l1_prefixes):
                # Map quantize.* -> quantize_l1.*
                new_k = k.replace("quantize.", "quantize_l1.") if k.startswith("quantize.") else k
                l1_sd[new_k] = v
        missing, unexpected = self.load_state_dict(l1_sd, strict=False)
        print(f"[ResidualIBQ] Loaded {len(l1_sd)} level-1 keys. Missing (expected): {len(missing)}")

        # Freeze level-1
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.quant_conv.parameters():
            p.requires_grad = False
        for p in self.quantize_l1.parameters():
            p.requires_grad = False

        # ---- Build level-2 components (trainable) ----
        self.residual_norm = torch.nn.LayerNorm(l2_embed_dim)
        self.quantize_l2 = IndexPropagationQuantize(
            l2_n_embed,
            l2_embed_dim,
            l2_beta,
            l2_use_entropy_loss,
            entropy_temperature=l2_entropy_temperature,
            sample_minimization_weight=l2_sample_minimization_weight,
            batch_maximization_weight=l2_batch_maximization_weight,
        )

        # ---- Build decoder (trainable, initialized from pretrained) ----
        self.post_quant_conv = torch.nn.Conv2d(l1_embed_dim, ddconfig["z_channels"], 1)
        self.decoder = Decoder(**ddconfig)

        # Load pretrained decoder weights as initialization
        dec_sd = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("decoder.") or k.startswith("post_quant_conv."):
                dec_sd[k] = v
        self.load_state_dict(dec_sd, strict=False)
        print(f"[ResidualIBQ] Initialized decoder from pretrained ({len(dec_sd)} keys)")

        # ---- Loss / Discriminator ----
        self.loss = instantiate_from_config(lossconfig)

        # ---- EMA ----
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)

        # ---- Resume from checkpoint ----
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.load_state_dict(sd, strict=False)
            print(f"[ResidualIBQ] Resumed from {ckpt_path}")

        self.strict_loading = False

        # Skip tracking (same as parent IBQ)
        self._skip_reasons = [
            "image_retrieval_failed", "non_finite_input", "non_finite_input_std",
            "low_std_input", "non_finite_forward", "non_finite_disc_loss",
            "non_finite_ae_loss",
        ]
        self._train_skip_total = {k: 0 for k in self._skip_reasons}
        self._train_skip_epoch = {k: 0 for k in self._skip_reasons}
        self._val_skip_total = {k: 0 for k in self._skip_reasons}
        self._val_skip_epoch = {k: 0 for k in self._skip_reasons}
        self.min_std_threshold = 1e-2

    def load_state_dict(self, *args, strict=False):
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return {
            k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items()
            if "inception_model" not in k and "lpips_vgg" not in k
               and "lpips_alex" not in k and "ocr_lpips" not in k
        }

    from contextlib import contextmanager

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    # ---- Forward path ----

    @torch.no_grad()
    def encode_l1(self, x):
        """Frozen level-1 encoding: image -> z, z_q1."""
        h = self.encoder(x)
        z = self.quant_conv(h)
        z_q1, _, info1 = self.quantize_l1(z)
        return z, z_q1, info1

    def encode_l2(self, z, z_q1):
        """Trainable level-2 residual quantization."""
        residual = (z - z_q1.detach()) * self.residual_scale

        # Apply LayerNorm in channel dimension: (B, C, H, W) -> (B, H, W, C) -> norm -> (B, C, H, W)
        B, C, H, W = residual.shape
        r_norm = residual.permute(0, 2, 3, 1).reshape(-1, C)
        r_norm = self.residual_norm(r_norm)
        r_norm = r_norm.reshape(B, H, W, C).permute(0, 3, 1, 2)

        r_q, l2_loss, info2 = self.quantize_l2(r_norm)
        return r_q, l2_loss, info2

    def encode(self, x):
        """Full encode: returns combined quantized output and losses."""
        z, z_q1, info1 = self.encode_l1(x)
        r_q, l2_loss, info2 = self.encode_l2(z, z_q1)
        z_final = z_q1 + r_q / self.residual_scale
        return z_final, l2_loss, (info1, info2)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    # ---- Helpers ----

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def _all_finite(self, value):
        if torch.is_tensor(value):
            return torch.isfinite(value).all().item()
        if isinstance(value, (tuple, list)):
            return all(self._all_finite(v) for v in value)
        if isinstance(value, dict):
            return all(self._all_finite(v) for v in value.values())
        return True

    def _record_skip(self, split, reason):
        if reason not in self._skip_reasons:
            return
        if split == "train":
            self._train_skip_total[reason] += 1
            self._train_skip_epoch[reason] += 1
            self.log("train/skip_total", float(sum(self._train_skip_total.values())),
                     prog_bar=False, logger=True, on_step=True, on_epoch=False)
        elif split == "val":
            self._val_skip_total[reason] += 1
            self._val_skip_epoch[reason] += 1

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # ---- Training ----

    def on_train_start(self):
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for pg1, pg2 in zip(opt_gen.param_groups, opt_disc.param_groups):
                pg1["lr"] = self.resume_lr
                pg2["lr"] = self.resume_lr

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def on_train_epoch_end(self):
        self._train_skip_epoch = {k: 0 for k in self._skip_reasons}
        self.lr_annealing()

    def lr_annealing(self):
        if self.lr_drop_epoch is not None:
            current_epoch = self.trainer.current_epoch
            if (current_epoch + 1) in self.lr_drop_epoch:
                opt_gen, opt_disc = self.optimizers()
                for pg1, pg2 in zip(opt_gen.param_groups, opt_disc.param_groups):
                    pg1["lr"] = pg1["lr"] * self.lr_drop_rate
                    pg2["lr"] = pg2["lr"] * self.lr_drop_rate

    def training_step(self, batch, batch_idx):
        load_failed = batch.get("image_load_failed")
        if load_failed is not None:
            if torch.is_tensor(load_failed) and load_failed.any():
                self._record_skip("train", "image_retrieval_failed")
                return None
        x = self.get_input(batch, self.image_key)
        if not self._all_finite(x):
            self._record_skip("train", "non_finite_input")
            return None

        min_channel_std = x.view(x.size(0), x.size(1), -1).std(dim=-1).min(dim=1)[0]
        global_std = x.view(x.size(0), -1).std(dim=-1)
        if not self._all_finite(min_channel_std) or not self._all_finite(global_std):
            self._record_skip("train", "non_finite_input_std")
            return None
        if (min_channel_std < self.min_std_threshold).any() or (global_std < self.min_std_threshold).any():
            self._record_skip("train", "low_std_input")
            return None

        xrec, qloss = self(x)
        if not self._all_finite(xrec) or not self._all_finite(qloss):
            self._record_skip("train", "non_finite_forward")
            return None

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        acc = self.accumulate_grad_batches
        step_count = getattr(self, "_accum_step_count", 0)
        self._accum_step_count = step_count + 1
        if (self._accum_step_count - 1) % acc == 0:
            opt_disc.zero_grad()
            opt_gen.zero_grad()

        # Discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        if not self._all_finite(discloss):
            self._record_skip("train", "non_finite_disc_loss")
            return None
        self.manual_backward(discloss)
        if self._accum_step_count % acc == 0:
            if self.gradient_clip_val > 0:
                self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            opt_disc.step()
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # Generator (level-2 quantizer + decoder)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        if not self._all_finite(aeloss):
            self._record_skip("train", "non_finite_ae_loss")
            return None
        self.manual_backward(aeloss)
        if self.gradient_clip_val > 0:
            self.clip_gradients(opt_gen, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
        if self._accum_step_count % acc == 0:
            opt_gen.step()
            if self.scheduler_type != "None":
                scheduler_disc.step()
                scheduler_gen.step()
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    # ---- Validation ----

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                return self._validation_step(batch, batch_idx, suffix="_ema")
        return self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        load_failed = batch.get("image_load_failed")
        if load_failed is not None:
            if torch.is_tensor(load_failed) and load_failed.any():
                self._record_skip("val", "image_retrieval_failed")
                return {}
        x = self.get_input(batch, self.image_key)
        if not self._all_finite(x):
            self._record_skip("val", "non_finite_input")
            return {}

        quant, qloss, _ = self.encode(x)
        if not self._all_finite(quant) or not self._all_finite(qloss):
            self._record_skip("val", "non_finite_forward")
            return {}
        x_rec = self.decode(quant).clamp(-1, 1)
        if not self._all_finite(x_rec):
            self._record_skip("val", "non_finite_forward")
            return {}
        aeloss, log_dict_ae = self.loss(qloss, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return {}

    # ---- Optimizer ----

    def configure_optimizers(self):
        lr = self.learning_rate
        # Only optimize trainable params: level-2 quantizer, residual_norm, decoder, post_quant_conv
        gen_params = (
            list(self.quantize_l2.parameters())
            + list(self.residual_norm.parameters())
            + list(self.decoder.parameters())
            + list(self.post_quant_conv.parameters())
        )
        opt_gen = torch.optim.Adam(gen_params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        if self.trainer.is_global_zero:
            trainable = sum(p.numel() for p in gen_params if p.requires_grad)
            frozen = sum(p.numel() for p in self.parameters()) - trainable
            print(f"[ResidualIBQ] Trainable params: {trainable:,}  Frozen params: {frozen:,}")

        if self.scheduler_type == "None":
            return (
                {"optimizer": opt_gen},
                {"optimizer": opt_disc, "do_not_count_global_step": True},
            )

        batches_per_epoch_raw = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        limit = self.trainer.limit_train_batches
        if isinstance(limit, int):
            batches_per_epoch = min(batches_per_epoch_raw, limit)
        elif isinstance(limit, float) and 0 < limit <= 1:
            batches_per_epoch = int(batches_per_epoch_raw * limit)
        else:
            batches_per_epoch = batches_per_epoch_raw
        optimizer_steps_per_epoch = max(1, batches_per_epoch // self.accumulate_grad_batches)
        warmup_steps = optimizer_steps_per_epoch * self.warmup_epochs
        training_steps = optimizer_steps_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))
        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError(f"scheduler_type={self.scheduler_type}")

        return (
            {"optimizer": opt_gen, "lr_scheduler": scheduler_ae},
            {"optimizer": opt_disc, "lr_scheduler": scheduler_disc},
        )
