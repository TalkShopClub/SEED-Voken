"""
OCR-Enhanced IBQ: Fuses PaddleOCR-VL-1.5 extracted text tokens with IBQ discrete image
tokens, passes them through a bidirectional fusion transformer, then decodes with the
(trainable) IBQ decoder.

Architecture:
  Image
    ├─[frozen IBQ encoder]──► discrete indices (B, 256)
    │                          ├─ embed + 2D position → (B, 256, d_model)
    │                          └─ modality type embedding (image)
    └─[frozen PaddleOCR-VL-1.5]── OCR text
         └─ byte-level tokenize ── text_ids (B, T)
              ├─ embed + 1D position → (B, T, d_model)
              └─ modality type embedding (text)

  [text | SEP | image_tokens]  →  bidirectional Fusion Transformer
                                   └─ extract image positions (B, 256, d_model)
                                        └─ linear proj → (B, 256, embed_dim)
                                             └─ reshape → (B, embed_dim, 16, 16)
                                                  └─ [trainable post_quant_conv + decoder]
                                                       └─ (B, 3, H, W)

Frozen:  IBQ encoder, quant_conv, quantize, PaddleOCR-VL-1.5
Trained: img_tok_emb, img_pos_emb, txt_tok_emb, txt_pos_emb, modality_type_emb,
         sep_emb, fusion_transformer, proj_to_codebook, post_quant_conv, decoder
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from PIL import Image
from collections import OrderedDict

from main import instantiate_from_config
from src.IBQ.modules.diffusionmodules.model import Encoder, Decoder
from src.IBQ.modules.vqvae.quantize import IndexPropagationQuantize
from src.IBQ.modules.losses.lpips import LPIPS


def disabled_train(self, mode=True):
    """Override model.train so frozen modules stay in eval mode."""
    return self


# ---------------------------------------------------------------------------
# Text tokenization
# ---------------------------------------------------------------------------

class ByteTextTokenizer:
    """
    Byte-level tokenizer for OCR output text.

    Vocabulary:
        0  : PAD  – padding
        1  : EMPTY – image has no detectable text
        2  : SEP  – separator token between text sequence and image sequence
        3+ : raw UTF-8 byte value + BYTE_OFFSET  (byte b → token b+3)

    Total vocab size: 259  (256 bytes + 3 special tokens)
    """
    PAD_ID   = 0
    EMPTY_ID = 1
    SEP_ID   = 2
    BYTE_OFFSET = 3
    VOCAB_SIZE  = 259   # 256 byte tokens + 3 specials

    def encode(self, text: str, max_len: int) -> list:
        """Encode text string to a list of token ids (no padding, no SEP)."""
        if not text or not text.strip():
            return [self.EMPTY_ID]
        byte_ids = [b + self.BYTE_OFFSET for b in text.encode("utf-8")[:max_len]]
        return byte_ids

    def pad(self, ids: list, length: int) -> list:
        """Pad or truncate ids to exactly `length` entries."""
        ids = ids[:length]
        return ids + [self.PAD_ID] * (length - len(ids))


# ---------------------------------------------------------------------------
# OCR extractor (lazy-loaded, frozen)
# ---------------------------------------------------------------------------

class OCRExtractor:
    """
    Wraps PaddleOCR-VL-1.5 for text extraction.

    Lazy-loaded on first use so the training process can start without
    loading the VLM immediately. Results are cached to `cache_path` (JSON)
    keyed by file path so subsequent epochs skip the VLM entirely.
    """

    def __init__(
        self,
        model_name: str = "PaddlePaddle/PaddleOCR-VL-1.5",
        max_new_tokens: int = 256,
        cache_path: str = "ocr_cache.json",
    ):
        self.model_name    = model_name
        self.max_new_tokens = max_new_tokens
        self.cache_path    = cache_path
        self._model        = None
        self._processor    = None
        self._cache: dict  = {}
        self._cache_dirty  = False

        # Load existing cache from disk
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    # ------------------------------------------------------------------
    def _ensure_loaded(self, device):
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise ImportError(
                "PaddleOCR-VL-1.5 requires `transformers>=5.0.0`. "
                "Install with: pip install 'transformers>=5.0.0'\n"
                "Or pre-extract OCR text with: python scripts/preextract_ocr.py"
            ) from e

        print(f"[OCRExtractor] Loading {self.model_name} …")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        ).to(device).eval()
        # Permanently keep in eval; freeze
        self._model.train = disabled_train.__get__(self._model)
        for p in self._model.parameters():
            p.requires_grad_(False)
        print("[OCRExtractor] Loaded and frozen.")

    @property
    def is_available(self):
        """True if the OCR model can be loaded (transformers installed)."""
        try:
            import importlib
            return importlib.util.find_spec("transformers") is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    def flush_cache(self):
        if self._cache_dirty:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            self._cache_dirty = False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(
        self,
        pil_images: list,
        file_paths: list = None,
        device: str = "cuda",
        fallback_on_missing: bool = True,
    ) -> list:
        """
        Extract text from `pil_images`.

        If `file_paths` is provided, cached results are used when available
        and new results are stored in the cache.

        If `fallback_on_missing=True` and the OCR model cannot be loaded
        (e.g. transformers not installed), returns empty strings for
        un-cached images instead of raising an error. This allows training
        to proceed using only cached OCR text.
        """
        results = []
        uncached_idx = []
        uncached_imgs = []

        for i, img in enumerate(pil_images):
            fp = file_paths[i] if file_paths else None
            if fp is not None and fp in self._cache:
                results.append(self._cache[fp])
            else:
                results.append(None)
                uncached_idx.append(i)
                uncached_imgs.append(img)

        if uncached_imgs:
            try:
                self._ensure_loaded(device)
            except ImportError as e:
                if fallback_on_missing:
                    print(f"[OCRExtractor] WARNING: {e}  → using empty text for un-cached images.")
                    for i in uncached_idx:
                        results[i] = ""
                    return results
                raise

            for local_i, img in enumerate(uncached_imgs):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text",  "text": "OCR:"},
                    ],
                }]
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                text = self._processor.decode(out[0][inputs["input_ids"].shape[-1]:-1])

                global_i = uncached_idx[local_i]
                results[global_i] = text

                fp = file_paths[global_i] if file_paths else None
                if fp is not None:
                    self._cache[fp] = text
                    self._cache_dirty = True

            self.flush_cache()

        return results


# ---------------------------------------------------------------------------
# Bidirectional fusion transformer block (pre-norm, no causal mask)
# ---------------------------------------------------------------------------

class FusionBlock(nn.Module):
    """Pre-norm bidirectional transformer block."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # Self-attention (bidirectional – no attn_mask)
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = residual + self.drop1(attn_out)
        # Feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OCREnhancedIBQ(L.LightningModule):
    """
    Trains the IBQ decoder to reconstruct images from a fusion of:
      • discrete image tokens (from frozen IBQ encoder)
      • OCR text tokens (from frozen PaddleOCR-VL-1.5)

    Only the fusion transformer, embeddings, projection, post_quant_conv,
    and decoder are updated during training.
    """

    def __init__(
        self,
        # ── IBQ architecture ──────────────────────────────────────────
        ddconfig: dict,            # same ddconfig used to train the IBQ model
        n_embed:   int  = 1024,    # codebook size
        embed_dim: int  = 256,     # codebook vector dimension
        ibq_ckpt_path: str = None, # path to pretrained IBQ .ckpt

        # ── Resolution handling ───────────────────────────────────────
        # `ibq_input_size`: what resolution images are resized to before
        # being fed to the IBQ encoder.  The encoder's ddconfig["resolution"]
        # describes its training resolution and determines the grid size.
        # `data_resolution`: what resolution the dataset provides (and what
        # resolution is used for OCR extraction).  Must be >= ibq_input_size.
        # Example: data_resolution=1024, ibq_input_size=256 → OCR at 1024px,
        #   IBQ encoding at 256px → 16×16=256 discrete tokens.
        ibq_input_size: int = None,   # None = use ddconfig["resolution"]

        # ── Fusion transformer ────────────────────────────────────────
        d_model:          int   = 512,
        n_heads:          int   = 8,
        n_layers:         int   = 6,
        ffn_dim_multiplier: int = 4,
        dropout:          float = 0.1,

        # ── Text / OCR ────────────────────────────────────────────────
        text_max_len:    int = 128,  # max byte-tokens per image
        ocr_model_name:  str = "PaddlePaddle/PaddleOCR-VL-1.5",
        ocr_max_new_tokens: int = 256,
        ocr_cache_path:  str = "ocr_cache.json",

        # ── Loss weights ──────────────────────────────────────────────
        rec_loss_weight:        float = 1.0,
        perceptual_loss_weight: float = 1.0,

        # ── Optimiser ─────────────────────────────────────────────────
        learning_rate: float = 1e-4,
        weight_decay:  float = 1e-2,
        warmup_steps:  int   = 500,

        # ── Resolution ────────────────────────────────────────────────
        # output_size: resolution at which reconstruction loss is computed.
        # Set equal to ibq_input_size (default) to keep behaviour unchanged,
        # or set to the data resolution (e.g. 1024) to compute loss at full res.
        output_size: int = None,  # None → same as ibq_input_size

        # ── Misc ──────────────────────────────────────────────────────
        image_key:  str = "image",
        fp_key:     str = "file_path_",  # batch key carrying image file paths
    ):
        super().__init__()
        self.image_key  = image_key
        self.fp_key     = fp_key
        self.n_embed    = n_embed
        self.embed_dim  = embed_dim
        self.d_model    = d_model
        self.text_max_len = text_max_len
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.warmup_steps  = warmup_steps
        self.rec_loss_weight        = rec_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.strict_loading = False

        z_channels = ddconfig["z_channels"]

        # Resolution for the IBQ encoder (resize images to this before encoding)
        self.ibq_input_size = ibq_input_size or ddconfig["resolution"]
        # Resolution at which reconstruction loss is computed
        _effective_ibq = self.ibq_input_size
        self.output_size = output_size if output_size is not None else _effective_ibq

        # Compute the spatial grid size from ddconfig:
        #   resolution / 2^(num_downsampling_levels)  where num_down = len(ch_mult)-1
        _res       = self.ibq_input_size
        _num_down  = len(ddconfig["ch_mult"]) - 1
        self._grid = _res // (2 ** _num_down)          # e.g. 256/16=16
        self.n_img_tokens = self._grid * self._grid     # e.g. 256

        # ── 1. Frozen IBQ encoder side ─────────────────────────────────
        self.encoder    = Encoder(**ddconfig)
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, 1)
        self.quantize   = IndexPropagationQuantize(
            n_embed, embed_dim, beta=0.25, use_entropy_loss=False
        )

        # ── 2. Trainable IBQ decoder side ─────────────────────────────
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder         = Decoder(**ddconfig)

        # ── 3. Load pretrained IBQ weights ────────────────────────────
        if ibq_ckpt_path is not None:
            self._load_ibq_weights(ibq_ckpt_path)

        # ── 4. Freeze encoder components ──────────────────────────────
        self._freeze_module(self.encoder)
        self._freeze_module(self.quant_conv)
        self._freeze_module(self.quantize)

        # ── 5. OCR extractor (not an nn.Module – kept as plain Python) ─
        self.ocr_extractor = OCRExtractor(
            model_name=ocr_model_name,
            max_new_tokens=ocr_max_new_tokens,
            cache_path=ocr_cache_path,
        )
        self.text_tokenizer = ByteTextTokenizer()

        # ── 6. Modality type embeddings ───────────────────────────────
        # 0 = text token,  1 = SEP token,  2 = image token
        self.modality_emb = nn.Embedding(3, d_model)

        # ── 7. Image token embeddings + 2D position ───────────────────
        self.img_tok_emb = nn.Embedding(n_embed, d_model)
        # Flat 2D positions: one per spatial location (n_img_tokens total)
        self.img_pos_emb = nn.Embedding(self.n_img_tokens, d_model)

        # ── 8. Text token embeddings + 1D position ────────────────────
        self.txt_tok_emb = nn.Embedding(ByteTextTokenizer.VOCAB_SIZE, d_model)
        # +1 for the SEP token position
        self.txt_pos_emb = nn.Embedding(text_max_len + 1, d_model)

        # ── 9. Bidirectional fusion transformer ───────────────────────
        ffn_dim = d_model * ffn_dim_multiplier
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.fusion_norm = nn.LayerNorm(d_model)

        # ── 10. Project fused image features → codebook dimension ─────
        self.proj_to_codebook = nn.Linear(d_model, embed_dim, bias=True)

        # ── 10b. Trainable 4× upsampler (256 → 1024) ─────────────────
        # Used when output_size > ibq_input_size to compute loss at full res.
        # Architecture: bilinear 4× + lightweight residual conv refinement.
        if self.output_size != self.ibq_input_size:
            scale = self.output_size // self.ibq_input_size
            assert scale >= 2 and (scale & (scale - 1)) == 0, \
                "output_size must be a power-of-2 multiple of ibq_input_size"
            layers = []
            in_ch = 3
            for _ in range(int(math.log2(scale))):
                layers += [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(64, 3, 3, padding=1, bias=False),
                ]
                in_ch = 3
            self.upsampler = nn.Sequential(*layers)
        else:
            self.upsampler = None

        # ── 11. Perceptual loss (frozen VGG-LPIPS) ────────────────────
        self.perceptual_loss = LPIPS().eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad_(False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_ibq_weights(self, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        def _extract(prefix):
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        self.encoder.load_state_dict(_extract("encoder."), strict=False)
        self.decoder.load_state_dict(_extract("decoder."), strict=False)
        self.quantize.load_state_dict(_extract("quantize."), strict=False)
        self.quant_conv.load_state_dict(_extract("quant_conv."), strict=False)
        self.post_quant_conv.load_state_dict(_extract("post_quant_conv."), strict=False)
        print(f"[OCREnhancedIBQ] Loaded IBQ weights from {ckpt_path}")

    @staticmethod
    def _freeze_module(module: nn.Module):
        module.eval()
        module.train = disabled_train.__get__(module)
        for p in module.parameters():
            p.requires_grad_(False)

    def _init_weights(self):
        """Xavier-uniform init for new (non-IBQ) layers."""
        for module in [
            self.img_tok_emb, self.img_pos_emb,
            self.txt_tok_emb, self.txt_pos_emb,
            self.modality_emb,
        ]:
            nn.init.normal_(module.weight, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.proj_to_codebook:
                if module.weight.requires_grad:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.proj_to_codebook.weight)
        nn.init.zeros_(self.proj_to_codebook.bias)

    # ------------------------------------------------------------------
    # IBQ encode (frozen) → discrete indices
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor):
        """
        x : (B, 3, H, W) normalised image tensor (at any resolution)
        Resizes to ibq_input_size before encoding.
        Returns indices (B, grid*grid) and quantised latents (B, embed_dim, grid, grid)
        """
        s = self.ibq_input_size
        if x.shape[-1] != s or x.shape[-2] != s:
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        h = self.encoder(x)
        h = self.quant_conv(h)
        z_q, _, info = self.quantize(h)
        indices = info[2].view(x.shape[0], -1)   # (B, grid*grid)
        return indices, z_q

    # ------------------------------------------------------------------
    # OCR text extraction + tokenisation
    # ------------------------------------------------------------------

    def get_pil_images(self, x_norm: torch.Tensor) -> list:
        """Convert normalised tensor (B, 3, H, W) → list of PIL images."""
        x_uint8 = ((x_norm.detach().cpu().float() + 1.0) * 127.5).clamp(0, 255).byte()
        return [
            Image.fromarray(x_uint8[i].permute(1, 2, 0).numpy())
            for i in range(x_uint8.shape[0])
        ]

    def tokenize_texts(self, texts: list, device) -> torch.Tensor:
        """
        texts : list[str] of length B
        Returns (B, text_max_len) LongTensor  (byte-level, padded)
        """
        all_ids = [
            self.text_tokenizer.pad(
                self.text_tokenizer.encode(t, self.text_max_len),
                self.text_max_len
            )
            for t in texts
        ]
        return torch.tensor(all_ids, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, file_paths: list = None):
        """
        x          : (B, 3, H, W) normalised image tensor (data resolution, e.g. 1024px)
        file_paths : optional list[str] for OCR caching
        Returns    : (xrec, x_ibq) where xrec and x_ibq are both at ibq_input_size
        """
        B, _, H, W = x.shape
        device = x.device
        n_img  = self.n_img_tokens                           # e.g. 256 (16×16)

        # ── Step 1: Frozen IBQ encoder → discrete indices ──────────────
        # encode_to_indices internally resizes x to ibq_input_size
        img_indices, _ = self.encode_to_indices(x)          # (B, n_img)

        # ── Step 2: OCR text extraction on full-res image (frozen) ─────
        # Use the full data-resolution image for OCR – much better quality
        # for text-rich screenshots than downscaled versions.
        pil_images = self.get_pil_images(x)
        texts = self.ocr_extractor.extract(
            pil_images, file_paths=file_paths, device=str(device),
            fallback_on_missing=True,
        )

        # Resize x to ibq_input_size for the reconstruction target
        s = self.ibq_input_size
        if H != s or W != s:
            x_ibq = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        else:
            x_ibq = x

        # ── Step 3: Tokenise OCR text → (B, T) ────────────────────────
        text_ids = self.tokenize_texts(texts, device)        # (B, T)
        T = text_ids.shape[1]                                # == text_max_len

        # ── Step 4: Build token embeddings ────────────────────────────
        # 4a. Text tokens: embed + 1D position + modality-type 0
        txt_pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        txt_emb = (self.txt_tok_emb(text_ids)
                   + self.txt_pos_emb(txt_pos)
                   + self.modality_emb(
                       torch.zeros(B, T, dtype=torch.long, device=device)))
        # (B, T, d_model)

        # 4b. SEP token: position T, modality 1
        sep_id  = torch.full((B, 1), ByteTextTokenizer.SEP_ID, dtype=torch.long, device=device)
        sep_pos = torch.full((B, 1), T, dtype=torch.long, device=device)
        sep_emb = (self.txt_tok_emb(sep_id)
                   + self.txt_pos_emb(sep_pos)
                   + self.modality_emb(
                       torch.ones(B, 1, dtype=torch.long, device=device)))
        # (B, 1, d_model)

        # 4c. Image tokens: embed + flat 2D position + modality-type 2
        img_pos = torch.arange(n_img, device=device).unsqueeze(0).expand(B, -1)
        img_emb = (self.img_tok_emb(img_indices)
                   + self.img_pos_emb(img_pos)
                   + self.modality_emb(
                       torch.full((B, n_img), 2, dtype=torch.long, device=device)))
        # (B, n_img, d_model)

        # ── Step 5: Concatenate [text | SEP | image_tokens] ───────────
        seq = torch.cat([txt_emb, sep_emb, img_emb], dim=1)  # (B, T+1+n_img, d_model)

        # ── Step 6: Build key_padding_mask (True = ignore position) ───
        text_pad_mask    = (text_ids == ByteTextTokenizer.PAD_ID)          # (B, T)
        sep_img_mask     = torch.zeros(B, 1 + n_img, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([text_pad_mask, sep_img_mask], dim=1) # (B, T+1+n_img)

        # ── Step 7: Bidirectional fusion transformer ───────────────────
        for block in self.fusion_blocks:
            seq = block(seq, key_padding_mask)
        seq = self.fusion_norm(seq)                          # (B, T+1+n_img, d_model)

        # ── Step 8: Extract image token outputs (last n_img positions) ─
        img_out = seq[:, T + 1:, :]                         # (B, n_img, d_model)

        # ── Step 9: Project → codebook dimension, reshape to 2D grid ──
        img_out = self.proj_to_codebook(img_out)            # (B, n_img, embed_dim)
        img_out = (img_out
                   .view(B, self._grid, self._grid, self.embed_dim)
                   .permute(0, 3, 1, 2)
                   .contiguous())                           # (B, embed_dim, grid, grid)

        # ── Step 10: Trainable IBQ decoder ────────────────────────────
        xrec = self.post_quant_conv(img_out)                # (B, z_channels, grid, grid)
        xrec = self.decoder(xrec)                           # (B, 3, ibq_input_size, ibq_input_size)

        # ── Step 11: Optional upsampler → output_size ─────────────────
        if self.upsampler is not None:
            # residual: bilinear base + learned refinement
            xrec_base = F.interpolate(xrec, size=(self.output_size, self.output_size),
                                      mode="bilinear", align_corners=False)
            xrec = xrec_base + self.upsampler(xrec)

        # Target: resize original image to output_size for loss
        if H != self.output_size or W != self.output_size:
            x_target = F.interpolate(x, size=(self.output_size, self.output_size),
                                     mode="bilinear", align_corners=False)
        else:
            x_target = x

        return xrec, x_target

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, x: torch.Tensor, xrec: torch.Tensor):
        rec_loss  = F.l1_loss(xrec, x)
        perc_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous()).mean()
        total     = self.rec_loss_weight * rec_loss + self.perceptual_loss_weight * perc_loss
        return total, {
            "loss/rec":         rec_loss,
            "loss/perceptual":  perc_loss,
            "loss/total":       total,
        }

    # ------------------------------------------------------------------
    # Dataset input helpers
    # ------------------------------------------------------------------

    def get_input(self, batch):
        x = batch[self.image_key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def get_file_paths(self, batch):
        if self.fp_key in batch:
            fp = batch[self.fp_key]
            # Lightning may deliver these as a tuple/list from collate
            if isinstance(fp, (list, tuple)):
                return list(fp)
            return None
        return None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x  = self.get_input(batch).to(self.device)
        fp = self.get_file_paths(batch)
        xrec, x_ibq = self(x, file_paths=fp)
        loss, log_dict = self.compute_loss(x_ibq, xrec)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x  = self.get_input(batch).to(self.device)
        fp = self.get_file_paths(batch)
        xrec, x_ibq = self(x, file_paths=fp)
        loss, log_dict = self.compute_loss(x_ibq, xrec)
        val_log = {f"val/{k.split('/')[-1]}": v for k, v in log_dict.items()}
        self.log_dict(val_log, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, step / max(1, self.warmup_steps)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer

    # ------------------------------------------------------------------
    # Checkpoint helpers – exclude frozen VGG and OCR model weights
    # ------------------------------------------------------------------

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        skip = ("perceptual_loss.net", "perceptual_loss.scaling_layer")
        return {k: v for k, v in sd.items() if not any(k.startswith(prefix + s) for s in skip)}

    def load_state_dict(self, *args, strict=False):
        return super().load_state_dict(*args, strict=strict)

    def log_images(self, batch, **kwargs):
        x  = self.get_input(batch).to(self.device)
        fp = self.get_file_paths(batch)
        xrec, x_ibq = self(x, file_paths=fp)
        return {"inputs": x_ibq, "reconstructions": xrec.clamp(-1, 1)}
