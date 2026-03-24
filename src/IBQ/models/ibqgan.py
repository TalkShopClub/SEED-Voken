import sys
from pathlib import Path

# Add src/ so that taming package (src/taming/) is importable
_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import lightning as L

from main import instantiate_from_config
from collections import OrderedDict
from contextlib import contextmanager

from taming.modules.losses.lpips import OCR_CRAFT_LPIPS
from src.IBQ.modules.diffusionmodules.model import Encoder, Decoder
from src.IBQ.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from src.IBQ.modules.vqvae.quantize import IndexPropagationQuantize
from src.IBQ.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from src.IBQ.modules.ema import LitEma

class VQModel(L.LightningModule):
    def __init__(self,
                ddconfig,
                lossconfig,
                #Quantize Related
                n_embed,
                embed_dim,
                ckpt_path = None,
                ignore_keys = [],
                image_key = "image",
                colorize_nlabels = None,
                monitor = None,
                remap = None,
                sane_index_shape = False,  # tell vector quantizer to return indices as bhw
                learning_rate = None,
                l2_normalize = False,
                ### scheduler config
                warmup_epochs = 0,  # warmup epochs
                scheduler_type = "None",
                min_learning_rate = 0,
                gradient_clip_val = 0,
                resume_lr = None,
                lr_drop_epoch = None,
                lr_drop_rate = 0.1,
                use_ema = False,
                stage = None,
                ocr_loss = False,
                accumulate_grad_batches = 1,
                min_std_threshold = 1e-2,
                # Mask channel parameters
                mask_loss_type = "bce",  # "bce" or "mse"
                text_mask_loss_weight = 0.0,
                icon_mask_loss_weight = 0.0,
                # Region-weighted RGB reconstruction loss
                text_region_weight = 0.0,
                icon_region_weight = 0.0,
                # Text-only L1: zero out L1 loss outside text regions
                text_only_l1 = False,
                # Composite mode: blend decoder output with original using text mask
                composite_mode = False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape, l2_normalize=l2_normalize)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.stage = stage
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if self.use_ema and stage is None: #no need to construct EMA when training Transformer
            self.model_ema = LitEma(self)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)

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
        self.ocr_loss = ocr_loss
        self.ocr_lpips = None
        self.min_std_threshold = min_std_threshold
        if self.ocr_loss:
            self.ocr_lpips = OCR_CRAFT_LPIPS().eval()
            for p in self.ocr_lpips.parameters():
                p.requires_grad = False

        # Mask channel loss config
        self.mask_loss_type = mask_loss_type
        self.text_mask_loss_weight = text_mask_loss_weight
        self.icon_mask_loss_weight = icon_mask_loss_weight
        self._use_mask_loss = (text_mask_loss_weight > 0 or icon_mask_loss_weight > 0)
        # Whether the model has mask channels (5ch input/output)
        self._has_mask_channels = ddconfig.get("in_channels", 3) > 3
        # Region-weighted RGB reconstruction loss
        self.text_region_weight = text_region_weight
        self.icon_region_weight = icon_region_weight
        self._use_region_weight = (text_region_weight > 0 or icon_region_weight > 0)
        # Text-only L1 mode: L1 loss is zero outside text regions
        self.text_only_l1 = text_only_l1
        # Composite mode: blend decoder RGB with original using text mask before loss
        self.composite_mode = composite_mode

        self.strict_loading = False
        self._skip_reasons = [
            "image_retrieval_failed",
            "non_finite_input",
            "non_finite_input_std",
            "low_std_input",
            "non_finite_forward",
            "non_finite_disc_loss",
            "non_finite_ae_loss",
            "non_finite_ocr_loss",
            "non_finite_ae_plus_ocr",
        ]
        self._train_skip_total = {k: 0 for k in self._skip_reasons}
        self._train_skip_epoch = {k: 0 for k in self._skip_reasons}
        self._val_skip_total = {k: 0 for k in self._skip_reasons}
        self._val_skip_epoch = {k: 0 for k in self._skip_reasons}

    def _build_region_weight_map(self, x_masks):
        """Build a (B, 1, H, W) spatial weight map from ground truth masks.
        Pixels in text regions get extra weight text_region_weight,
        pixels in icon regions get extra weight icon_region_weight,
        all pixels start with base weight 1.0.
        The map is normalized so its spatial mean is 1.0 — this preserves
        relative emphasis on text/icon regions without inflating the overall
        loss magnitude (which would destabilize the L1 vs perceptual vs GAN balance).
        """
        B, C, H, W = x_masks.shape
        weight_map = torch.ones(B, 1, H, W, device=x_masks.device, dtype=x_masks.dtype)
        if self.text_region_weight > 0:
            weight_map = weight_map + self.text_region_weight * x_masks[:, 0:1]
        if self.icon_region_weight > 0 and C > 1:
            weight_map = weight_map + self.icon_region_weight * x_masks[:, 1:2]
        # Normalize so spatial mean = 1.0 per sample
        mean = weight_map.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        weight_map = weight_map / mean
        return weight_map

    def _split_rgb_masks(self, x):
        """Split 5-channel tensor into RGB (3ch) and mask targets (2ch).
        Returns (x_rgb, x_masks) where x_masks is None if input is 3ch."""
        if self._has_mask_channels and x.shape[1] > 3:
            return x[:, :3], x[:, 3:]
        return x, None

    def _compute_mask_loss(self, xrec_masks, x_masks, split="train"):
        """Compute mask reconstruction loss for text (and optionally icon) channels.
        xrec_masks: (B, N, H, W) raw decoder output for mask channels (N=1 or 2)
        x_masks: (B, N, H, W) ground truth masks in [0, 1]
        """
        log_dict = {}
        total_mask_loss = torch.tensor(0.0, device=xrec_masks.device)

        if self.text_mask_loss_weight > 0:
            if self.mask_loss_type == "bce":
                text_loss = F.binary_cross_entropy_with_logits(
                    xrec_masks[:, 0:1], x_masks[:, 0:1])
            else:  # mse
                text_loss = F.mse_loss(torch.sigmoid(xrec_masks[:, 0:1]), x_masks[:, 0:1])
            total_mask_loss = total_mask_loss + self.text_mask_loss_weight * text_loss
            log_dict[f"{split}/text_mask_loss"] = text_loss.detach()

        if self.icon_mask_loss_weight > 0 and x_masks.shape[1] > 1:
            if self.mask_loss_type == "bce":
                icon_loss = F.binary_cross_entropy_with_logits(
                    xrec_masks[:, 1:2], x_masks[:, 1:2])
            else:  # mse
                icon_loss = F.mse_loss(torch.sigmoid(xrec_masks[:, 1:2]), x_masks[:, 1:2])
            total_mask_loss = total_mask_loss + self.icon_mask_loss_weight * icon_loss
            log_dict[f"{split}/icon_mask_loss"] = icon_loss.detach()

        log_dict[f"{split}/total_mask_loss"] = total_mask_loss.detach()
        return total_mask_loss, log_dict

    def _ensure_ocr_lpips_frozen_eval(self):
        if self.ocr_lpips is None:
            return
        # Keep OCR perceptual module deterministic/non-trainable even if parent train()/eval() toggles.
        self.ocr_lpips.eval()
        for p in self.ocr_lpips.parameters():
            p.requires_grad = False

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

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k and "ocr_lpips" not in k)}

    def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue
                    if "embedding" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "quant" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
                    elif "embedding" in k:
                        new_params[k] = v
                    elif "quant" in k:
                        new_params[k] = v              
        missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
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

    def _record_skip(self, split: str, reason: str):
        if reason not in self._skip_reasons:
            return

        if split == "train":
            self._train_skip_total[reason] += 1
            self._train_skip_epoch[reason] += 1
            total = float(sum(self._train_skip_total.values()))
            reason_total = float(self._train_skip_total[reason])
            self.log("train/skip_total", total, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log(f"train/skip_{reason}_total", reason_total, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        elif split == "val":
            self._val_skip_total[reason] += 1
            self._val_skip_epoch[reason] += 1
            total = float(sum(self._val_skip_total.values()))
            reason_total = float(self._val_skip_total[reason])
            self.log("val/skip_total", total, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log(f"val/skip_{reason}_total", reason_total, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def on_train_start(self):
        """
        change lr after resuming
        """
        self._ensure_ocr_lpips_frozen_eval()
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def on_train_epoch_end(self):
        ### update lr
        for reason in self._skip_reasons:
            self.log(f"train/skip_{reason}_epoch", float(self._train_skip_epoch[reason]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("train/skip_epoch_total", float(sum(self._train_skip_epoch.values())), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self._train_skip_epoch = {k: 0 for k in self._skip_reasons}
        self.lr_annealing()

    def on_validation_epoch_end(self):
        for reason in self._skip_reasons:
            self.log(f"val/skip_{reason}_epoch", float(self._val_skip_epoch[reason]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/skip_epoch_total", float(sum(self._val_skip_epoch.values())), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self._val_skip_epoch = {k: 0 for k in self._skip_reasons}

    def lr_annealing(self):
        """
        Perform Lr decay
        """
        if self.lr_drop_epoch is not None:
            current_epoch = self.trainer.current_epoch
            if (current_epoch + 1) in self.lr_drop_epoch:
                opt_gen, opt_disc = self.optimizers()
                for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                    opt_gen_param_group["lr"] = opt_gen_param_group["lr"] * self.lr_drop_rate
                    opt_disc_param_group["lr"] = opt_disc_param_group["lr"] * self.lr_drop_rate
    
    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        self._ensure_ocr_lpips_frozen_eval()
        # Skip batch if any sample in the batch failed to load (e.g. truncated image)
        load_failed = batch.get("image_load_failed")
        if load_failed is not None:
            if torch.is_tensor(load_failed) and load_failed.any():
                self._record_skip("train", "image_retrieval_failed")
                return None
            if isinstance(load_failed, (list, tuple)) and any(load_failed):
                self._record_skip("train", "image_retrieval_failed")
                return None
        x = self.get_input(batch, self.image_key)

        if not self._all_finite(x): # skip batch if any value is nan/inf
            self._record_skip("train", "non_finite_input")
            return None
        # Sanity check: skip batch if any image has low channel or low global std (avoids BatchNorm/div issues)
        # Only check std on RGB channels to avoid mask channels (which can be all-zero) triggering skip
        x_rgb_for_std = x[:, :3] if self._has_mask_channels else x
        # Per-image min std across channels: (B, C, H, W) -> std over (H,W) per (B,C) -> (B,C) -> min over C -> (B,)
        min_channel_std = x_rgb_for_std.view(x_rgb_for_std.size(0), x_rgb_for_std.size(1), -1).std(dim=-1).min(dim=1)[0]
        # Per-image global std: (B, C, H, W) -> (B,)
        global_std = x_rgb_for_std.view(x_rgb_for_std.size(0), -1).std(dim=-1)
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

        # Split into RGB and mask channels for loss computation
        x_rgb, x_masks = self._split_rgb_masks(x)
        xrec_rgb, xrec_masks = self._split_rgb_masks(xrec)

        # Composite mode: blend decoder RGB with original using text mask
        if self.composite_mode and x_masks is not None:
            text_mask = x_masks[:, 0:1]  # (B, 1, H, W)
            xrec_rgb = xrec_rgb * text_mask + x_rgb * (1.0 - text_mask)

        # Build region weight map or text-only mask for RGB reconstruction loss
        region_weight_map = None
        text_only_mask = None
        if self.text_only_l1 and x_masks is not None:
            text_only_mask = x_masks[:, 0:1]  # (B, 1, H, W) binary
        elif self._use_region_weight and x_masks is not None:
            region_weight_map = self._build_region_weight_map(x_masks)

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        # Manual gradient accumulation: use a counter that increments every training_step,
        # since global_step only increments on optimizer.step() (and we use do_not_count for disc).
        acc = self.accumulate_grad_batches
        step_count = getattr(self, "_accum_step_count", 0)
        self._accum_step_count = step_count + 1
        # Start of accumulation window: clear gradients
        if (self._accum_step_count - 1) % acc == 0:
            opt_disc.zero_grad()
            opt_gen.zero_grad()
        # original VQGAN first optimizes G, then D. We first optimize D then G, following traditional GAN
        # optimize discriminator (RGB only)
        discloss, log_dict_disc = self.loss(qloss, x_rgb, xrec_rgb, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            region_weight_map=region_weight_map,
                                            text_only_mask=text_only_mask)
        if not self._all_finite(discloss):
            self._record_skip("train", "non_finite_disc_loss")
            return None

        self.manual_backward(discloss)
        if self._accum_step_count % acc == 0:
            if self.gradient_clip_val > 0:
                self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")
            opt_disc.step()
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)


        # optimize generator (RGB losses)
        aeloss, log_dict_ae = self.loss(qloss, x_rgb, xrec_rgb, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train",
                                        region_weight_map=region_weight_map,
                                        text_only_mask=text_only_mask)
        if not self._all_finite(aeloss):
            self._record_skip("train", "non_finite_ae_loss")
            return None
        if self.ocr_loss and self.ocr_lpips is not None:
            ocr_loss_val = self.ocr_lpips(x_rgb, xrec_rgb).mean()
            if not self._all_finite(ocr_loss_val):
                self._record_skip("train", "non_finite_ocr_loss")
                return None
            aeloss = aeloss + ocr_loss_val
            if not self._all_finite(aeloss):
                self._record_skip("train", "non_finite_ae_plus_ocr")
                return None
            log_dict_ae["train/ocr_loss"] = ocr_loss_val.detach()

        # Mask reconstruction loss
        if self._use_mask_loss and x_masks is not None and xrec_masks is not None:
            mask_loss, mask_log_dict = self._compute_mask_loss(xrec_masks, x_masks, split="train")
            if self._all_finite(mask_loss):
                aeloss = aeloss + mask_loss
                log_dict_ae.update(mask_log_dict)

        self.manual_backward(aeloss)


        if self.gradient_clip_val > 0: # for cosine similarity
            self.clip_gradients(opt_gen, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")

        if self._accum_step_count % acc == 0:
            opt_gen.step()
            if self.scheduler_type != "None":
                scheduler_disc.step()
                scheduler_gen.step()
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
            return log_dict_ema
        else:
            log_dict = self._validation_step(batch, batch_idx)
            return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        self._ensure_ocr_lpips_frozen_eval()
        load_failed = batch.get("image_load_failed")
        if load_failed is not None:
            if torch.is_tensor(load_failed) and load_failed.any():
                self._record_skip("val", "image_retrieval_failed")
                return {}
            if isinstance(load_failed, (list, tuple)) and any(load_failed):
                self._record_skip("val", "image_retrieval_failed")
                return {}
        x = self.get_input(batch, self.image_key)

        if not self._all_finite(x):
            self._record_skip("val", "non_finite_input")
            return {}
        # Same guards as training: skip batch if any image has low channel or low global std (RGB only)
        x_rgb_for_std = x[:, :3] if self._has_mask_channels else x
        min_channel_std = x_rgb_for_std.view(x_rgb_for_std.size(0), x_rgb_for_std.size(1), -1).std(dim=-1).min(dim=1)[0]
        global_std = x_rgb_for_std.view(x_rgb_for_std.size(0), -1).std(dim=-1)
        if not self._all_finite(min_channel_std) or not self._all_finite(global_std):
            self._record_skip("val", "non_finite_input_std")
            return {}
        if (min_channel_std < self.min_std_threshold).any() or (global_std < self.min_std_threshold).any():
            self._record_skip("val", "low_std_input")
            return {}

        quant, qloss, (_, _, min_encoding_indices) = self.encode(x)
        if not self._all_finite(quant) or not self._all_finite(qloss):
            self._record_skip("val", "non_finite_forward")
            return {}
        x_rec = self.decode(quant).clamp(-1, 1)
        if not self._all_finite(x_rec):
            self._record_skip("val", "non_finite_forward")
            return {}

        # Split into RGB and mask channels
        x_rgb, x_masks = self._split_rgb_masks(x)
        xrec_rgb, xrec_masks = self._split_rgb_masks(x_rec)

        # Composite mode
        if self.composite_mode and x_masks is not None:
            text_mask = x_masks[:, 0:1]
            xrec_rgb = xrec_rgb * text_mask + x_rgb * (1.0 - text_mask)

        region_weight_map = None
        text_only_mask = None
        if self.text_only_l1 and x_masks is not None:
            text_only_mask = x_masks[:, 0:1]
        elif self._use_region_weight and x_masks is not None:
            region_weight_map = self._build_region_weight_map(x_masks)

        aeloss, log_dict_ae = self.loss(qloss, x_rgb, xrec_rgb, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",
                                        region_weight_map=region_weight_map,
                                        text_only_mask=text_only_mask)
        if not self._all_finite(aeloss):
            self._record_skip("val", "non_finite_ae_loss")
            return {}
        if self.ocr_loss and self.ocr_lpips is not None:
            ocr_loss_val = self.ocr_lpips(x_rgb, xrec_rgb).mean()
            if not self._all_finite(ocr_loss_val):
                self._record_skip("val", "non_finite_ocr_loss")
                return {}
            log_dict_ae["val/ocr_loss"] = ocr_loss_val.detach()

        # Mask reconstruction loss
        if self._use_mask_loss and x_masks is not None and xrec_masks is not None:
            mask_loss, mask_log_dict = self._compute_mask_loss(xrec_masks, x_masks, split="val")
            if self._all_finite(mask_loss):
                log_dict_ae.update(mask_log_dict)

        discloss, log_dict_disc = self.loss(qloss, x_rgb, xrec_rgb, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",
                                            region_weight_map=region_weight_map,
                                            text_only_mask=text_only_mask)
        if not self._all_finite(discloss):
            self._record_skip("val", "non_finite_disc_loss")
            return {}
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return {}

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.trainer.is_global_zero:
            print(
                "[IBQGAN] configure_optimizers: learning_rate={}, scheduler_type={}, warmup_epochs={}, min_learning_rate={}, lr_drop_epoch={}".format(
                    lr, self.scheduler_type, self.warmup_epochs, self.min_learning_rate, self.lr_drop_epoch
                )
            )
        opt_gen = torch.optim.Adam(list(self.encoder.parameters()) +
                                   list(self.decoder.parameters()) +
                                   list(self.quantize.parameters()) +
                                   list(self.quant_conv.parameters()) +
                                   list(self.post_quant_conv.parameters()),
                                   lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        if self.scheduler_type == "None":
            return (
                {"optimizer": opt_gen},
                {"optimizer": opt_disc, "do_not_count_global_step": True},
            )

        # Batches per epoch: full dataloader length (per process), capped by limit_train_batches
        batches_per_epoch_raw = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        limit = self.trainer.limit_train_batches
        if isinstance(limit, int):
            batches_per_epoch = min(batches_per_epoch_raw, limit)
        elif isinstance(limit, float) and 0 < limit <= 1:
            batches_per_epoch = int(batches_per_epoch_raw * limit)
        else:
            batches_per_epoch = batches_per_epoch_raw
        # Scheduler is stepped once per optimizer step (every accumulate_grad_batches batches)
        optimizer_steps_per_epoch = max(1, batches_per_epoch // self.accumulate_grad_batches)
        warmup_steps = optimizer_steps_per_epoch * self.warmup_epochs
        training_steps = optimizer_steps_per_epoch * self.trainer.max_epochs

        if self.trainer.is_global_zero:
            print(
                "scheduler: batches_per_epoch={}, optimizer_steps_per_epoch={}, warmup_steps={}, training_steps={}".format(
                    batches_per_epoch, optimizer_steps_per_epoch, warmup_steps, training_steps
                )
            )

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))

        elif self.scheduler_type == "cosine_annealing":
            # Step-based cosine annealing: LR from learning_rate to min_learning_rate over training_steps
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_gen, T_max=training_steps, eta_min=self.min_learning_rate
            )
            scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_disc, T_max=training_steps, eta_min=self.min_learning_rate
            )
        else:
            raise NotImplementedError("scheduler_type must be one of: None, linear-warmup, linear-warmup_cosine-decay, cosine_annealing")
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        # Log only RGB channels for visualization
        x_rgb = x[:, :3] if self._has_mask_channels else x
        xrec_rgb = xrec[:, :3] if self._has_mask_channels else xrec
        if self.composite_mode and self._has_mask_channels and x.shape[1] > 3:
            text_mask = x[:, 3:4]
            xrec_rgb = xrec_rgb * text_mask + x_rgb * (1.0 - text_mask)
        log["inputs"] = x_rgb
        log["reconstructions"] = xrec_rgb
        # Log mask channels if present
        if self._has_mask_channels and x.shape[1] > 3:
            log["text_mask_gt"] = x[:, 3:4].repeat(1, 3, 1, 1)  # expand to 3ch for viz
            log["text_mask_pred"] = torch.sigmoid(xrec[:, 3:4]).repeat(1, 3, 1, 1)
            if x.shape[1] > 4:
                log["icon_mask_gt"] = x[:, 4:5].repeat(1, 3, 1, 1)
                log["icon_mask_pred"] = torch.sigmoid(xrec[:, 4:5]).repeat(1, 3, 1, 1)
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class IBQ(VQModel):
    def __init__(self,
                ddconfig,
                lossconfig,
                n_embed,
                embed_dim,
                ckpt_path = None,
                ignore_keys = [],
                image_key = "image",
                colorize_nlabels = None,
                monitor = None,
                remap = None,
                sane_index_shape = False,  # tell vector quantizer to return indices as bhw
                learning_rate = None,
                l2_normalize = False,
                ### scheduler config
                warmup_epochs = 0,  # warmup epochs
                scheduler_type = "None",
                min_learning_rate = 0,
                cosine_similarity = False,
                gradient_clip_val = 0,
                use_entropy_loss = False,
                sample_minimization_weight = 1.0,
                batch_maximization_weight = 1.0,
                entropy_temperature = 0.01,
                beta = 0.25,
                lr_drop_epoch = None,
                lr_drop_rate = 0.1,
                resume_lr = None,
                use_ema = False,
                stage = None,
                ocr_loss = False,
                accumulate_grad_batches = 1,
                # Mask channel parameters
                mask_loss_type = "bce",
                text_mask_loss_weight = 0.0,
                icon_mask_loss_weight = 0.0,
                # Region-weighted RGB reconstruction loss
                text_region_weight = 0.0,
                icon_region_weight = 0.0,
                # Text-only L1
                text_only_l1 = False,
                # Composite mode
                composite_mode = False,
                 ):
        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                        lossconfig,
                        n_embed,
                        embed_dim,
                        ckpt_path=None,
                        ignore_keys=ignore_keys,
                        image_key=image_key,
                        colorize_nlabels = colorize_nlabels,
                        monitor = monitor,
                        remap = remap,
                        sane_index_shape = sane_index_shape,
                        learning_rate = learning_rate,
                        l2_normalize = l2_normalize,
                        warmup_epochs = warmup_epochs,
                        scheduler_type = scheduler_type,
                        min_learning_rate = min_learning_rate,
                        gradient_clip_val = gradient_clip_val,
                        resume_lr = resume_lr,
                        use_ema = use_ema,
                        stage = stage,
                        lr_drop_epoch = lr_drop_epoch,
                        lr_drop_rate = lr_drop_rate,
                        ocr_loss = ocr_loss,
                        accumulate_grad_batches = accumulate_grad_batches,
                        mask_loss_type = mask_loss_type,
                        text_mask_loss_weight = text_mask_loss_weight,
                        icon_mask_loss_weight = icon_mask_loss_weight,
                        text_region_weight = text_region_weight,
                        icon_region_weight = icon_region_weight,
                        text_only_l1 = text_only_l1,
                        composite_mode = composite_mode,
                        )
        self.quantize = IndexPropagationQuantize(n_embed, embed_dim, beta, use_entropy_loss,
                                          remap=remap, cosine_similarity=cosine_similarity,
                                          entropy_temperature=entropy_temperature,
                                          sample_minimization_weight=sample_minimization_weight, batch_maximization_weight=batch_maximization_weight)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)


def _load_pretrained_state_dict(pretrained_path, ckpt_file, config_file="config.yaml"):
    """Load state dict from pretrained dir (vision_tokenizer format). Used by IBQFromPretrained."""
    import os.path as osp
    from omegaconf import OmegaConf

    cfg_path = osp.join(pretrained_path, config_file)
    if not osp.exists(cfg_path):
        raise FileNotFoundError(f"Pretrained config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    ckpt_path = osp.join(pretrained_path, ckpt_file)
    if not osp.exists(ckpt_path):
        # try safetensors
        alt = ckpt_file.replace(".ckpt", ".safetensors")
        if alt != ckpt_file:
            alt_path = osp.join(pretrained_path, alt)
            if osp.exists(alt_path):
                try:
                    from safetensors.torch import load_file
                    sd = load_file(alt_path)
                    return cfg_dict, sd
                except Exception as e:
                    raise FileNotFoundError(f"Could not load {alt_path}: {e}") from e
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return cfg_dict, sd


class IBQFromPretrained(IBQ):
    """
    IBQ model that loads generator weights from a pretrained vision tokenizer
    (e.g. pretrained/ with config.yaml + model.ckpt as in src/vision_tokenizer).
    Same training behavior as IBQ; only initialization differs.
    """

    def __init__(
        self,
        pretrained_path,
        lossconfig,
        config_file="config.yaml",
        ckpt_file="model.ckpt",
        ignore_keys=None,
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        learning_rate=1e-4,
        l2_normalize=False,
        warmup_epochs=0,
        scheduler_type="None",
        min_learning_rate=0,
        cosine_similarity=False,
        gradient_clip_val=0,
        use_entropy_loss=True,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        entropy_temperature=0.01,
        beta=0.25,
        lr_drop_epoch=None,
        lr_drop_rate=0.1,
        resume_lr=None,
        use_ema=True,
        stage=None,
        ocr_loss: bool = False,
        accumulate_grad_batches = 1,
        # Mask channel parameters
        mask_loss_type = "bce",
        text_mask_loss_weight = 0.0,
        icon_mask_loss_weight = 0.0,
        # Region-weighted RGB reconstruction loss
        text_region_weight = 0.0,
        icon_region_weight = 0.0,
        # Text-only L1
        text_only_l1 = False,
        # Composite mode
        composite_mode = False,
        # Override ddconfig channels for 5ch mode
        override_in_channels = None,
        override_out_ch = None,
        **kwargs,
    ):
        ignore_keys = ignore_keys or []
        pretrained_cfg = _load_pretrained_state_dict(pretrained_path, ckpt_file, config_file)
        cfg_dict, state_dict = pretrained_cfg

        ddconfig = cfg_dict.get("ddconfig")
        n_embed = cfg_dict.get("n_embed")
        embed_dim = cfg_dict.get("embed_dim")
        if ddconfig is None or n_embed is None or embed_dim is None:
            raise KeyError(
                "Pretrained config must contain ddconfig, n_embed, embed_dim. "
                f"Got keys: {list(cfg_dict.keys())}"
            )

        # Remember original channel counts from pretrained model before overriding
        pretrained_in_channels = ddconfig.get("in_channels", 3)
        pretrained_out_ch = ddconfig.get("out_ch", 3)

        # Override ddconfig channel counts for 5-channel mode
        if override_in_channels is not None:
            ddconfig["in_channels"] = override_in_channels
        if override_out_ch is not None:
            ddconfig["out_ch"] = override_out_ch

        # Optional overrides from pretrained config
        beta = cfg_dict.get("beta", beta)
        use_entropy_loss = cfg_dict.get("use_entropy_loss", use_entropy_loss)
        cosine_similarity = cfg_dict.get("cosine_similarity", cosine_similarity)
        entropy_temperature = cfg_dict.get("entropy_temperature", entropy_temperature)
        sample_minimization_weight = cfg_dict.get("sample_minimization_weight", sample_minimization_weight)
        batch_maximization_weight = cfg_dict.get("batch_maximization_weight", batch_maximization_weight)

        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=None,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
            remap=remap,
            sane_index_shape=sane_index_shape,
            learning_rate=learning_rate,
            l2_normalize=l2_normalize,
            warmup_epochs=warmup_epochs,
            scheduler_type=scheduler_type,
            min_learning_rate=min_learning_rate,
            gradient_clip_val=gradient_clip_val,
            use_entropy_loss=use_entropy_loss,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
            entropy_temperature=entropy_temperature,
            beta=beta,
            lr_drop_epoch=lr_drop_epoch,
            lr_drop_rate=lr_drop_rate,
            resume_lr=resume_lr,
            use_ema=use_ema,
            stage=stage,
            cosine_similarity=cosine_similarity,
            ocr_loss=ocr_loss,
            accumulate_grad_batches=accumulate_grad_batches,
            mask_loss_type=mask_loss_type,
            text_mask_loss_weight=text_mask_loss_weight,
            icon_mask_loss_weight=icon_mask_loss_weight,
            text_region_weight=text_region_weight,
            icon_region_weight=icon_region_weight,
            text_only_l1=text_only_l1,
            composite_mode=composite_mode,
        )

        # Load pretrained generator weights, expanding conv_in/conv_out if channel count changed
        new_in_channels = ddconfig.get("in_channels", 3)
        new_out_ch = ddconfig.get("out_ch", 3)

        generator_prefixes = ("encoder.", "decoder.", "quantize.", "quant_conv.", "post_quant_conv.")
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in generator_prefixes):
                if not any(ig in k for ig in ignore_keys):
                    new_sd[k] = v

        # Expand encoder.conv_in weights if in_channels increased (e.g. 3 -> 5)
        if new_in_channels > pretrained_in_channels and "encoder.conv_in.weight" in new_sd:
            old_w = new_sd["encoder.conv_in.weight"]  # (out_ch, old_in, kH, kW)
            new_w = torch.zeros(old_w.shape[0], new_in_channels, old_w.shape[2], old_w.shape[3],
                                dtype=old_w.dtype)
            new_w[:, :pretrained_in_channels] = old_w
            new_sd["encoder.conv_in.weight"] = new_w
            print(f"Expanded encoder.conv_in.weight: ({old_w.shape[1]}) -> ({new_in_channels}) channels, "
                  f"new channels zero-initialized")
            # bias stays the same shape (out_ch,) — no change needed

        # Expand decoder.conv_out weights if out_ch increased (e.g. 3 -> 5)
        if new_out_ch > pretrained_out_ch and "decoder.conv_out.weight" in new_sd:
            old_w = new_sd["decoder.conv_out.weight"]  # (old_out, in_ch, kH, kW)
            new_w = torch.zeros(new_out_ch, old_w.shape[1], old_w.shape[2], old_w.shape[3],
                                dtype=old_w.dtype)
            new_w[:pretrained_out_ch] = old_w
            new_sd["decoder.conv_out.weight"] = new_w
            print(f"Expanded decoder.conv_out.weight: ({old_w.shape[0]}) -> ({new_out_ch}) channels, "
                  f"new channels zero-initialized")
            # Expand bias too
            if "decoder.conv_out.bias" in new_sd:
                old_b = new_sd["decoder.conv_out.bias"]  # (old_out,)
                new_b = torch.zeros(new_out_ch, dtype=old_b.dtype)
                new_b[:pretrained_out_ch] = old_b
                new_sd["decoder.conv_out.bias"] = new_b

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"IBQFromPretrained: missing keys (expected for loss/disc): {len(missing)}")
        if unexpected:
            print(f"IBQFromPretrained: unexpected keys from checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        print(f"Restored generator from {pretrained_path} ({len(new_sd)} keys)")


class IBQResidualFromPretrained(IBQFromPretrained):
    """
    IBQ with residual codebook (RVQ-style).

    Frozen pretrained encoder + codebook_1 produce z_q1. A second randomly-
    initialized codebook quantizes the residual (h - z_q1). A zero-initialized
    1x1 conv gates the residual contribution so the model starts at pretrained
    quality and the new codebook's influence grows organically during training.

    Trainable: quantize_residual, residual_proj, post_quant_conv, decoder.
    Frozen:    encoder, quant_conv, quantize (codebook_1).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Second codebook: same size as pretrained, randomly initialized
        n_e = self.quantize.n_e
        e_dim = self.quantize.e_dim
        self.quantize_residual = IndexPropagationQuantize(
            n_e, e_dim,
            beta=self.quantize.beta,
            use_entropy_loss=True,
            entropy_temperature=self.quantize.entropy_temperature,
            sample_minimization_weight=self.quantize.sample_minimization_weight,
            batch_maximization_weight=self.quantize.batch_maximization_weight,
        )

        # Zero-initialized projection: at init, residual contribution is zero
        # so model outputs are identical to pretrained baseline
        self.residual_proj = torch.nn.Conv2d(e_dim, e_dim, 1, bias=True)
        torch.nn.init.zeros_(self.residual_proj.weight)
        torch.nn.init.zeros_(self.residual_proj.bias)

        # Freeze pretrained encoder path
        for module in [self.encoder, self.quant_conv, self.quantize]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

        print(f"[IBQResidual] Residual codebook: {n_e} entries x {e_dim} dim (randomly init)")
        print(f"[IBQResidual] Frozen: encoder, quant_conv, quantize")
        print(f"[IBQResidual] Trainable: quantize_residual, residual_proj, post_quant_conv, decoder")

    # ------------------------------------------------------------------
    # Keep frozen modules in eval even when Lightning calls model.train()
    # ------------------------------------------------------------------
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.encoder.eval()
            self.quant_conv.eval()
            self.quantize.eval()
        return self

    # ------------------------------------------------------------------
    # Dual-codebook encode
    # ------------------------------------------------------------------
    def encode(self, x):
        # Frozen encoder + quant_conv + codebook_1
        with torch.no_grad():
            h = self.encoder(x)
            h = self.quant_conv(h)
            z_q1, _loss1, _info1 = self.quantize(h)

        # Residual: what the pretrained codebook discarded
        residual = h - z_q1  # both detached via no_grad

        # Quantize residual with new codebook (trained)
        z_q2, emb_loss2, info2 = self.quantize_residual(residual)

        # Zero-init gate — starts as no-op, grows during training
        z_q2_proj = self.residual_proj(z_q2)

        z_combined = z_q1 + z_q2_proj

        # Diagnostics (logged in on_train_batch_end)
        self._diag_residual_norm = residual.detach().norm(dim=1).mean()
        self._diag_proj_norm = z_q2_proj.detach().norm(dim=1).mean()
        self._diag_proj_scale = (
            z_q2_proj.detach().norm() / z_q1.detach().norm().clamp(min=1e-8)
        )

        return z_combined, emb_loss2, info2

    # ------------------------------------------------------------------
    # Log residual diagnostics
    # ------------------------------------------------------------------
    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        for attr, name in [
            ("_diag_residual_norm", "train/residual_norm"),
            ("_diag_proj_norm", "train/residual_proj_norm"),
            ("_diag_proj_scale", "train/residual_proj_scale"),
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                self.log(name, val, prog_bar=False, logger=True,
                         on_step=True, on_epoch=False)

    # ------------------------------------------------------------------
    # Optimizer: only trainable params in generator optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.learning_rate
        if self.trainer.is_global_zero:
            print(
                "[IBQResidual] configure_optimizers: lr={}, scheduler={}".format(
                    lr, self.scheduler_type
                )
            )

        gen_params = (
            list(self.decoder.parameters()) +
            list(self.post_quant_conv.parameters()) +
            list(self.quantize_residual.parameters()) +
            list(self.residual_proj.parameters())
        )
        opt_gen = torch.optim.Adam(gen_params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        if self.scheduler_type == "None":
            return (
                {"optimizer": opt_gen},
                {"optimizer": opt_disc, "do_not_count_global_step": True},
            )

        # Replicate parent scheduler logic for non-None cases
        batches_per_epoch_raw = (
            len(self.trainer.datamodule._train_dataloader())
            // self.trainer.world_size
        )
        limit = self.trainer.limit_train_batches
        if isinstance(limit, int):
            batches_per_epoch = min(batches_per_epoch_raw, limit)
        elif isinstance(limit, float) and 0 < limit <= 1:
            batches_per_epoch = int(batches_per_epoch_raw * limit)
        else:
            batches_per_epoch = batches_per_epoch_raw
        optimizer_steps_per_epoch = max(
            1, batches_per_epoch // self.accumulate_grad_batches
        )
        warmup_steps = optimizer_steps_per_epoch * self.warmup_epochs
        training_steps = optimizer_steps_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
                opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
                opt_disc, Scheduler_LinearWarmup(warmup_steps))
        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
                opt_gen,
                Scheduler_LinearWarmup_CosineDecay(
                    warmup_steps=warmup_steps, max_steps=training_steps,
                    multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
                opt_disc,
                Scheduler_LinearWarmup_CosineDecay(
                    warmup_steps=warmup_steps, max_steps=training_steps,
                    multipler_min=multipler_min))
        elif self.scheduler_type == "cosine_annealing":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_gen, T_max=training_steps, eta_min=self.min_learning_rate)
            scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_disc, T_max=training_steps, eta_min=self.min_learning_rate)
        else:
            raise NotImplementedError(
                f"scheduler_type {self.scheduler_type} not supported"
            )

        return (
            {"optimizer": opt_gen, "lr_scheduler": scheduler_ae},
            {"optimizer": opt_disc, "lr_scheduler": scheduler_disc},
        )


class IBQDualDecoderFromPretrained(IBQFromPretrained):
    """
    IBQ with dual decoder: frozen pretrained pipeline for global reconstruction,
    plus a lightweight text-specialist decoder with its own codebook.

    Path 1 (frozen):  encoder → codebook_1 → decoder_1 → xrec_global
    Path 2 (trained): text_proj(encoder_features) → codebook_2 → text_decoder → xrec_text
    Output: pixel-space blend using the text mask.

    Gradients only flow through the text decoder in text regions, so the
    pretrained quality is fully preserved for non-text areas.
    """

    def __init__(self, *args, text_decoder_config=None, mask_blur_sigma=2.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        n_e = self.quantize.n_e
        e_dim = self.quantize.e_dim
        z_ch = self.post_quant_conv.out_channels  # 256

        # ---- text specialist path ----
        # text_proj: identity-initialized 1x1 conv so text path starts
        # seeing the same features as the pretrained codebook
        self.text_proj = torch.nn.Conv2d(e_dim, e_dim, 1, bias=True)
        torch.nn.init.eye_(self.text_proj.weight.view(e_dim, e_dim))
        self.text_proj.weight.data = self.text_proj.weight.data.view(
            e_dim, e_dim, 1, 1)
        torch.nn.init.zeros_(self.text_proj.bias)

        # text codebook: copy pretrained codebook weights
        self.quantize_text = IndexPropagationQuantize(
            n_e, e_dim,
            beta=self.quantize.beta,
            use_entropy_loss=True,
            entropy_temperature=self.quantize.entropy_temperature,
            sample_minimization_weight=self.quantize.sample_minimization_weight,
            batch_maximization_weight=self.quantize.batch_maximization_weight,
        )
        self.quantize_text.embedding.weight.data.copy_(
            self.quantize.embedding.weight.data)

        # post_quant_conv_text: copy from pretrained
        self.post_quant_conv_text = torch.nn.Conv2d(e_dim, z_ch, 1)
        self.post_quant_conv_text.weight.data.copy_(
            self.post_quant_conv.weight.data)
        self.post_quant_conv_text.bias.data.copy_(
            self.post_quant_conv.bias.data)

        # text decoder: same architecture as pretrained, initialized from
        # pretrained weights. Uses out_ch=3 (RGB only, no mask channel).
        pretrained_ddconfig = self.encoder.ch  # just need to read arch
        td_cfg = dict(
            ch=self.decoder.ch,
            out_ch=3,
            ch_mult=[1, 1, 2, 2, 4],
            num_res_blocks=self.decoder.num_res_blocks,
            attn_resolutions=[16],
            dropout=0.0,
            in_channels=3,
            resolution=256,
            z_channels=z_ch,
        )
        if text_decoder_config is not None:
            td_cfg.update(text_decoder_config)
        self.text_decoder = Decoder(**td_cfg)

        # Copy pretrained decoder weights into text decoder
        pretrained_sd = self.decoder.state_dict()
        text_sd = self.text_decoder.state_dict()
        new_text_sd = OrderedDict()
        for k, v in pretrained_sd.items():
            if k not in text_sd:
                continue
            if text_sd[k].shape == v.shape:
                new_text_sd[k] = v.clone()
            elif k == "conv_out.weight":
                # pretrained: (4, 128, 3, 3) → text: (3, 128, 3, 3)
                new_text_sd[k] = v[:3].clone()
            elif k == "conv_out.bias":
                # pretrained: (4,) → text: (3,)
                new_text_sd[k] = v[:3].clone()
        missing, unexpected = self.text_decoder.load_state_dict(
            new_text_sd, strict=False)
        n_copied = len(new_text_sd)
        n_total = len(text_sd)
        print(f"[IBQDualDecoder] Text decoder: copied {n_copied}/{n_total} "
              f"tensors from pretrained decoder"
              f"{f' (missing: {missing})' if missing else ''}")

        self.mask_blur_sigma = mask_blur_sigma

        # ---- freeze entire pretrained pipeline ----
        for module in [self.encoder, self.quant_conv, self.quantize,
                       self.post_quant_conv, self.decoder]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

        text_dec_params = sum(p.numel() for p in self.text_decoder.parameters())
        print(f"[IBQDualDecoder] Text codebook: {n_e} x {e_dim} "
              f"(init from pretrained)")
        print(f"[IBQDualDecoder] Text decoder: {text_dec_params:,} params")
        print(f"[IBQDualDecoder] Frozen: encoder, quant_conv, quantize, "
              f"post_quant_conv, decoder")
        print(f"[IBQDualDecoder] Trainable: text_proj, quantize_text, "
              f"post_quant_conv_text, text_decoder")

    # ------------------------------------------------------------------
    def train(self, mode=True):
        super().train(mode)
        if mode:
            for m in [self.encoder, self.quant_conv, self.quantize,
                      self.post_quant_conv, self.decoder]:
                m.eval()
        return self

    # ------------------------------------------------------------------
    def _blur_mask(self, mask):
        """Apply Gaussian blur to text mask for soft blending edges."""
        if self.mask_blur_sigma <= 0:
            return mask
        import torchvision.transforms.functional as TF
        ks = int(6 * self.mask_blur_sigma + 1)
        if ks % 2 == 0:
            ks += 1
        return TF.gaussian_blur(mask, kernel_size=[ks, ks],
                                sigma=[self.mask_blur_sigma] * 2)

    # ------------------------------------------------------------------
    def encode(self, x):
        # Frozen pretrained encoder + codebook
        with torch.no_grad():
            h = self.encoder(x)
            h = self.quant_conv(h)
            z_q1, _loss1, _info1 = self.quantize(h)

        # Text specialist: project encoder features → text codebook
        h_text = self.text_proj(h)  # h detached via no_grad
        z_q2, emb_loss2, info2 = self.quantize_text(h_text)

        # Cache for decode()
        self._cached_z_q1 = z_q1
        self._cached_input = x

        return z_q2, emb_loss2, info2

    # ------------------------------------------------------------------
    def decode(self, quant_text):
        # Frozen global reconstruction
        with torch.no_grad():
            xrec_global = self.decoder(
                self.post_quant_conv(self._cached_z_q1))

        # Text specialist reconstruction (RGB only)
        xrec_text = self.text_decoder(
            self.post_quant_conv_text(quant_text))

        # Blend using text mask from input
        inp = self._cached_input
        if inp.shape[1] > 3:
            text_mask = inp[:, 3:4]
            mask_soft = self._blur_mask(text_mask)
            xrec_rgb = (xrec_global[:, :3] * (1 - mask_soft)
                        + xrec_text * mask_soft)
            # Keep mask channel from frozen decoder for mask-loss logging
            xrec = torch.cat([xrec_rgb, xrec_global[:, 3:]], dim=1)
        else:
            xrec = xrec_global

        return xrec

    # ------------------------------------------------------------------
    def get_last_layer(self):
        # Adaptive disc weight must reference the trainable decoder
        return self.text_decoder.conv_out.weight

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.learning_rate
        if self.trainer.is_global_zero:
            print(f"[IBQDualDecoder] configure_optimizers: lr={lr}, "
                  f"scheduler={self.scheduler_type}")

        gen_params = (
            list(self.text_proj.parameters()) +
            list(self.quantize_text.parameters()) +
            list(self.post_quant_conv_text.parameters()) +
            list(self.text_decoder.parameters())
        )
        opt_gen = torch.optim.Adam(gen_params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        if self.scheduler_type == "None":
            return (
                {"optimizer": opt_gen},
                {"optimizer": opt_disc, "do_not_count_global_step": True},
            )

        batches_per_epoch_raw = (
            len(self.trainer.datamodule._train_dataloader())
            // self.trainer.world_size
        )
        limit = self.trainer.limit_train_batches
        if isinstance(limit, int):
            batches_per_epoch = min(batches_per_epoch_raw, limit)
        elif isinstance(limit, float) and 0 < limit <= 1:
            batches_per_epoch = int(batches_per_epoch_raw * limit)
        else:
            batches_per_epoch = batches_per_epoch_raw
        optimizer_steps_per_epoch = max(
            1, batches_per_epoch // self.accumulate_grad_batches
        )
        warmup_steps = optimizer_steps_per_epoch * self.warmup_epochs
        training_steps = optimizer_steps_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
                opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
                opt_disc, Scheduler_LinearWarmup(warmup_steps))
        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(
                opt_gen,
                Scheduler_LinearWarmup_CosineDecay(
                    warmup_steps=warmup_steps, max_steps=training_steps,
                    multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(
                opt_disc,
                Scheduler_LinearWarmup_CosineDecay(
                    warmup_steps=warmup_steps, max_steps=training_steps,
                    multipler_min=multipler_min))
        elif self.scheduler_type == "cosine_annealing":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_gen, T_max=training_steps, eta_min=self.min_learning_rate)
            scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_disc, T_max=training_steps, eta_min=self.min_learning_rate)
        else:
            raise NotImplementedError(
                f"scheduler_type {self.scheduler_type} not supported"
            )

        return (
            {"optimizer": opt_gen, "lr_scheduler": scheduler_ae},
            {"optimizer": opt_disc, "lr_scheduler": scheduler_disc},
        )
