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

        self.strict_loading = False
        self._skip_reasons = [
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
        x = self.get_input(batch, self.image_key)

        if not self._all_finite(x): # skip batch if any value is nan/inf
            self._record_skip("train", "non_finite_input")
            return None
        # Sanity check: skip batch if any image has low channel or low global std (avoids BatchNorm/div issues)
        
        # Per-image min std across channels: (B, C, H, W) -> std over (H,W) per (B,C) -> (B,C) -> min over C -> (B,)
        min_channel_std = x.view(x.size(0), x.size(1), -1).std(dim=-1).min(dim=1)[0]
        # Per-image global std: (B, C, H, W) -> (B,) 
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
        # optimize discriminator
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


        # optimize generator
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        if not self._all_finite(aeloss):
            self._record_skip("train", "non_finite_ae_loss")
            return None
        if self.ocr_loss and self.ocr_lpips is not None:
            ocr_loss_val = self.ocr_lpips(x, xrec).mean()
            if not self._all_finite(ocr_loss_val):
                self._record_skip("train", "non_finite_ocr_loss")
                return None
            aeloss = aeloss + ocr_loss_val
            if not self._all_finite(aeloss):
                self._record_skip("train", "non_finite_ae_plus_ocr")
                return None
            log_dict_ae["train/ocr_loss"] = ocr_loss_val.detach()
            
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
        x = self.get_input(batch, self.image_key)

        if not self._all_finite(x):
            self._record_skip("val", "non_finite_input")
            return {}
        # Same guards as training: skip batch if any image has low channel or low global std
        min_channel_std = x.view(x.size(0), x.size(1), -1).std(dim=-1).min(dim=1)[0]
        global_std = x.view(x.size(0), -1).std(dim=-1)
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
        aeloss, log_dict_ae = self.loss(qloss, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        if not self._all_finite(aeloss):
            self._record_skip("val", "non_finite_ae_loss")
            return {}
        if self.ocr_loss and self.ocr_lpips is not None:
            ocr_loss_val = self.ocr_lpips(x, x_rec).mean()
            if not self._all_finite(ocr_loss_val):
                self._record_skip("val", "non_finite_ocr_loss")
                return {}
            log_dict_ae["val/ocr_loss"] = ocr_loss_val.detach()

        discloss, log_dict_disc = self.loss(qloss, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        if not self._all_finite(discloss):
            self._record_skip("val", "non_finite_disc_loss")
            return {}
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

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
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
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
        )

        # Load pretrained generator weights (encoder, decoder, quantize, quant_conv, post_quant_conv)
        generator_prefixes = ("encoder.", "decoder.", "quantize.", "quant_conv.", "post_quant_conv.")
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in generator_prefixes):
                if not any(ig in k for ig in ignore_keys):
                    new_sd[k] = v
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"IBQFromPretrained: missing keys (expected for loss/disc): {len(missing)}")
        if unexpected:
            print(f"IBQFromPretrained: unexpected keys from checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        print(f"Restored generator from {pretrained_path} ({len(new_sd)} keys)")
