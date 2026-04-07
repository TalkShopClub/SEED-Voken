#!/usr/bin/env python3
"""Reconstruct images using pretrained IBQ tokenizer and save GT vs Recon side by side."""
import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.vision_tokenizer.ibq import IBQ


DDCONFIG = dict(
    double_z=False,
    z_channels=256,
    resolution=256,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=4,
    attn_resolutions=[16],
    dropout=0.0,
)

DS_FACTOR = 16  # total downsampling factor from ch_mult


def load_model(ckpt_path, n_embed, device):
    model = IBQ(
        ddconfig=DDCONFIG,
        n_embed=n_embed,
        embed_dim=256,
        beta=0.25,
        use_entropy_loss=True,
        entropy_temperature=0.01,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Filter out EMA and non-model keys
    sd = {k: v for k, v in sd.items() if not k.startswith("model_ema.")}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


def pad_to_multiple(t, factor=DS_FACTOR):
    """Pad tensor [1,3,H,W] so H and W are multiples of factor."""
    _, _, h, w = t.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
    return t, h, w


def load_and_preprocess(path, size=None):
    """Load image. If size is set, resize+crop to square. If None, keep original resolution."""
    img = Image.open(path).convert("RGB")
    if size is not None and size > 0:
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    return transform(img).unsqueeze(0)


def tensor_to_pil(t):
    """Convert [-1,1] tensor [1,3,H,W] to PIL Image."""
    t = t.squeeze(0).clamp(-1, 1)
    t = (t + 1) / 2  # [0, 1]
    t = (t * 255).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="pretrained/IBQ-Tokenizer-262144-Pretrain/IBQ_pretrain_262144.ckpt")
    parser.add_argument("--n_embed", type=int, default=262144)
    parser.add_argument("--image_dir", default="/workspace/AgentNet/ubuntu_images_raw")
    parser.add_argument("--output_dir", default="results/reconstruct_pretrained_262144")
    parser.add_argument("--n_images", type=int, default=1000)
    parser.add_argument("--size", type=int, default=None, help="Resize to square. None=original resolution")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.ckpt} (n_embed={args.n_embed})...")
    model = load_model(args.ckpt, args.n_embed, device)

    # Collect image paths
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_paths = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])[:args.n_images]
    print(f"Reconstructing {len(all_paths)} images (size={'original' if args.size is None else args.size})...")

    # Process one-by-one for original resolution (varying sizes can't batch)
    if args.size is None:
        for p in tqdm(all_paths):
            try:
                x = load_and_preprocess(p, size=None).to(device)
            except Exception as e:
                print(f"Skip {p}: {e}")
                continue
            x_padded, orig_h, orig_w = pad_to_multiple(x)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x_rec, _ = model(x_padded)
                x_rec = x_rec[:, :, :orig_h, :orig_w].float().clamp(-1, 1)
            x = x[:, :, :orig_h, :orig_w]
            gt = tensor_to_pil(x.float())
            rec = tensor_to_pil(x_rec)
            combined = Image.new("RGB", (orig_w * 2, orig_h))
            combined.paste(gt, (0, 0))
            combined.paste(rec, (orig_w, 0))
            fname = os.path.splitext(os.path.basename(p))[0] + ".png"
            combined.save(os.path.join(args.output_dir, fname))
    else:
        # Batched processing for fixed-size
        for i in tqdm(range(0, len(all_paths), args.batch_size)):
            batch_paths = all_paths[i : i + args.batch_size]
            batch_tensors = []
            valid_paths = []
            for p in batch_paths:
                try:
                    batch_tensors.append(load_and_preprocess(p, args.size))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Skip {p}: {e}")
                    continue
            if not batch_tensors:
                continue
            x = torch.cat(batch_tensors, dim=0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x_rec, _ = model(x)
                x_rec = x_rec.float().clamp(-1, 1)
            for j, p in enumerate(valid_paths):
                gt = tensor_to_pil(x[j : j + 1].float())
                rec = tensor_to_pil(x_rec[j : j + 1])
                combined = Image.new("RGB", (args.size * 2, args.size))
                combined.paste(gt, (0, 0))
                combined.paste(rec, (args.size, 0))
                fname = os.path.splitext(os.path.basename(p))[0] + ".png"
                combined.save(os.path.join(args.output_dir, fname))

    print(f"Done! Saved {len(all_paths)} reconstructions to {args.output_dir}")


if __name__ == "__main__":
    main()
