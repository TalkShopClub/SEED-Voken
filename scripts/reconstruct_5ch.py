#!/usr/bin/env python3
"""
Reconstruct images from a trained 5-channel IBQ model and save side-by-side comparisons.

Usage:
    python scripts/reconstruct_5ch.py \
        --checkpoint /workspace/users/mike/experiments/ibq_5ch_test/checkpoints/last.ckpt \
        --config configs/IBQ/gpu/finetune_1024_5ch_test.yaml \
        --output_dir /workspace/users/mike/experiments/ibq_5ch_test/reconstructions \
        --num_images 10
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_ema", action="store_true", help="Use training weights instead of EMA")
    return parser.parse_args()


def tensor_to_pil(tensor):
    """Convert (C, H, W) tensor in [-1, 1] to PIL Image."""
    img = tensor.clamp(-1, 1).cpu().float()
    img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(img.permute(1, 2, 0).numpy())


def mask_to_pil(tensor):
    """Convert (1, H, W) mask tensor to grayscale PIL Image."""
    mask = torch.sigmoid(tensor).clamp(0, 1).cpu().float()
    mask = (mask * 255).to(torch.uint8)
    return Image.fromarray(mask.squeeze(0).numpy(), mode='L')


def gt_mask_to_pil(tensor):
    """Convert (1, H, W) ground truth mask in [0, 1] to grayscale PIL Image."""
    mask = tensor.clamp(0, 1).cpu().float()
    mask = (mask * 255).to(torch.uint8)
    return Image.fromarray(mask.squeeze(0).numpy(), mode='L')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config to get data settings
    from omegaconf import OmegaConf
    from jsonargparse import ArgumentParser as JAP
    cfg = OmegaConf.load(args.config) if args.config.endswith('.yaml') else None

    # Build dataset manually
    from src.IBQ.data.base import IterableImagePaths

    # Extract data config from yaml
    data_cfg = cfg.data.init_args
    train_cfg = data_cfg.train.params.config
    manifest_path = train_cfg.manifest_path
    annotations_path = train_cfg.get('annotations_path', None)
    size = train_cfg.get('size', 1024)

    with open(manifest_path) as f:
        manifest = json.load(f)
    paths = manifest['paths']

    dataset = IterableImagePaths(
        paths=paths[:args.num_images],
        size=size,
        random_crop=False,  # Use center crop for reproducible evaluation
        original_reso=False,
        annotations_path=annotations_path,
    )

    # Load model from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    import importlib
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Reconstruct model from config (dynamically resolve class from class_path)
    model_cfg = cfg.model.init_args
    class_path = cfg.model.class_path
    module_path, class_name = class_path.rsplit('.', 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)
    model = ModelClass(**OmegaConf.to_container(model_cfg, resolve=True))

    # Load trained weights
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model = model.to(args.device).eval()

    # Switch to EMA weights if available and requested
    use_ema = not args.no_ema
    if use_ema and model.use_ema:
        model.model_ema.copy_to(model)
        print("Model loaded (using EMA weights)")
    else:
        print("Model loaded (using training weights)")

    # Check if composite mode is enabled
    composite_mode = getattr(model, 'composite_mode', False)
    if composite_mode:
        print("Composite mode: blending decoder output with original using text mask")

    # Reconstruct images
    with torch.no_grad():
        for i in range(min(args.num_images, len(dataset))):
            sample = dataset[i]
            image = sample["image"]  # (H, W, 4+) float32
            x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(args.device)

            xrec, _ = model(x)

            # Composite mode: blend decoder RGB with original using text mask
            if composite_mode and x.shape[1] > 3:
                text_mask = x[:, 3:4]
                xrec_rgb = xrec[:, :3] * text_mask + x[:, :3] * (1.0 - text_mask)
            else:
                xrec_rgb = xrec[:, :3]

            # Save original RGB
            orig_rgb = tensor_to_pil(x[0, :3])
            orig_rgb.save(os.path.join(args.output_dir, f"{i:03d}_original.png"))

            # Save reconstructed RGB
            rec_rgb = tensor_to_pil(xrec_rgb[0])
            rec_rgb.save(os.path.join(args.output_dir, f"{i:03d}_reconstructed.png"))

            # Save side-by-side
            w, h = orig_rgb.size
            sidebyside = Image.new("RGB", (w * 2, h))
            sidebyside.paste(orig_rgb, (0, 0))
            sidebyside.paste(rec_rgb, (w, 0))
            sidebyside.save(os.path.join(args.output_dir, f"{i:03d}_comparison.png"))

            # In composite mode, also save the raw decoder output (before compositing)
            if composite_mode and x.shape[1] > 3:
                raw_rgb = tensor_to_pil(xrec[0, :3])
                raw_rgb.save(os.path.join(args.output_dir, f"{i:03d}_raw_decoder.png"))

            # Save mask comparisons if >3ch
            if x.shape[1] > 3:
                gt_text = gt_mask_to_pil(x[0, 3:4])
                pred_text = mask_to_pil(xrec[0, 3:4])
                gt_text.save(os.path.join(args.output_dir, f"{i:03d}_text_mask_gt.png"))
                pred_text.save(os.path.join(args.output_dir, f"{i:03d}_text_mask_pred.png"))
                if x.shape[1] > 4:
                    gt_icon = gt_mask_to_pil(x[0, 4:5])
                    pred_icon = mask_to_pil(xrec[0, 4:5])
                    gt_icon.save(os.path.join(args.output_dir, f"{i:03d}_icon_mask_gt.png"))
                    pred_icon.save(os.path.join(args.output_dir, f"{i:03d}_icon_mask_pred.png"))

            fname = os.path.basename(sample.get("file_path_", f"image_{i}"))
            print(f"  [{i+1}/{args.num_images}] {fname}")

    print(f"\nReconstructions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
