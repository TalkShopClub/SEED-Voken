"""
Shard inference loop for DDP eval: each rank runs the same inference as evaluation_image.py
on its share of the validation set, then we aggregate metrics in metrics.py.
"""

import os
import sys
import contextlib
import warnings

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Repo root must be on sys.path before run_worker is called (main.py does this).
def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_repo_path():
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    _src = os.path.join(root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)


def run_worker(rank: int, world_size: int, args):
    """Run inference on this rank's shard of the validation set; then participate in metric reduction."""
    _ensure_repo_path()

    from evaluation_image import (
        load_config,
        load_vqgan_new,
        get_encoder_spatial_align,
        pad_to_encoder_multiple,
        unpad_to_region,
        instantiate_from_config,
    )
    from finetune import custom_collate
    from metrics.inception import InceptionV3
    import lpips
    from skimage.metrics import peak_signal_noise_ratio as psnr_loss
    from skimage.metrics import structural_similarity as ssim_loss

    from eval.ddp.metrics import (
        gather_inception_preds,
        reduce_scalar_metrics,
        reduce_usage,
        calculate_frechet_distance,
    )

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Load config and resolve (same as evaluation_image)
    config_data = __import__("omegaconf").OmegaConf.load(args.config_file)
    __import__("omegaconf").OmegaConf.resolve(config_data)
    if hasattr(config_data.data.init_args.validation.params, "config"):
        config_data.data.init_args.validation.params.config.original_reso = True
        config_data.data.init_args.validation.params.config.size = 0
    config_data.data.init_args.batch_size = args.batch_size

    config_model = load_config(args.config_file, display=(rank == 0 and getattr(args, "display_config", False)))
    model = load_vqgan_new(config_model, model_type=args.model, ckpt_path=args.ckpt_path)
    model = model.to(device).to(torch.bfloat16)
    spatial_align = get_encoder_spatial_align(model)
    _q = model.quantize
    codebook_size = (
        config_model.model.init_args.get("n_embed")
        or getattr(_q, "n_embed", None)
        or getattr(_q, "n_e", None)
    )
    if codebook_size is None:
        emb = getattr(_q, "embedding", None) or getattr(_q, "embed", None)
        codebook_size = emb.weight.shape[0] if emb is not None else 0

    usage = torch.zeros(codebook_size, dtype=torch.long, device=device)

    # Inception on CPU for FID (same as evaluation_image)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx])
    inception_model.eval()

    # Dataset and dataloader with DistributedSampler for this rank's shard
    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    val_ds = dataset.datasets["validation"]
    sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
        collate_fn=custom_collate,
        pin_memory=True,
    )

    pred_xs = []
    pred_recs = []
    loss_fn_alex = lpips.LPIPS(net="alex").eval()
    loss_fn_vgg = lpips.LPIPS(net="vgg").eval()
    if device.type != "cuda":
        loss_fn_alex, loss_fn_vgg = loss_fn_alex.to(device), loss_fn_vgg.to(device)

    lpips_alex_sum = 0.0
    lpips_vgg_sum = 0.0
    ssim_sum = 0.0
    psnr_sum = 0.0
    num_images = 0
    num_iter = 0
    skipped_batches = 0
    skipped_oom = 0

    oom_exceptions = (RuntimeError,)
    if hasattr(torch, "OutOfMemoryError"):
        oom_exceptions = (*oom_exceptions, torch.OutOfMemoryError)
    if hasattr(torch.cuda, "OutOfMemoryError") and torch.cuda.OutOfMemoryError not in oom_exceptions:
        oom_exceptions = (*oom_exceptions, torch.cuda.OutOfMemoryError)

    save_dir = getattr(args, "save_comparison_dir", None)
    save_native = getattr(args, "save_native_resolution", False)
    if save_dir and rank == 0:
        save_dir = os.path.expanduser(save_dir)
        for d in ("input", "output", "comparison"):
            os.makedirs(os.path.join(save_dir, d), exist_ok=True)
        if save_native:
            for d in ("input_native", "output_native", "comparison_native"):
                os.makedirs(os.path.join(save_dir, d), exist_ok=True)
    saved_count = 0  # for rank 0 save filenames

    torch.set_grad_enabled(False)
    num_batches = len(dataloader)
    pbar = tqdm(total=num_batches, unit="batch", position=rank, leave=(rank == 0), disable=(world_size > 1 and rank != 0))

    with torch.inference_mode():
        for batch in dataloader:
            try:
                images = batch["image"].permute(0, 3, 1, 2).detach().to(device, non_blocking=True)
            except OSError as e:
                skipped_batches += 1
                path_hint = f" (paths: {batch.get('file_path_', batch.get('path'))})" if batch.get("file_path_") is not None else ""
                warnings.warn(f"[rank {rank}] Skip batch OSError: {e}{path_hint}")
                pbar.update(1)
                continue
            except oom_exceptions as e:
                if getattr(args, "skip_oom", False) and "out of memory" in str(e).lower():
                    skipped_oom += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    pbar.update(1)
                    continue
                raise

            try:
                _, _, h, w = images.shape
                scale = getattr(args, "image_size", 1.0)
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                images = torch.nn.functional.interpolate(
                    images, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                num_images += images.shape[0]
                images = images.to(torch.bfloat16)
                images_padded, crop_region = pad_to_encoder_multiple(images, spatial_align)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    if model.use_ema:
                        with model.ema_scope():
                            if args.model == "Open-MAGVIT2":
                                quant, diff, indices, _ = model.encode(images_padded)
                            elif args.model == "IBQ":
                                quant, qloss, (_, _, indices) = model.encode(images_padded)
                            reconstructed_padded = model.decode(quant)
                    else:
                        if args.model == "Open-MAGVIT2":
                            quant, diff, indices, _ = model.encode(images_padded)
                        elif args.model == "IBQ":
                            quant, qloss, (_, _, indices) = model.encode(images_padded)
                        reconstructed_padded = model.decode(quant)

                reconstructed_padded = reconstructed_padded.clamp(-1, 1)
                images = unpad_to_region(images_padded, crop_region)
                reconstructed_images = unpad_to_region(reconstructed_padded, crop_region)
                del images_padded, reconstructed_padded, quant
                if args.model == "IBQ":
                    del qloss
                else:
                    del diff
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                for index in indices:
                    usage[index.item()] += 1

                images = images.float()
                reconstructed_images = reconstructed_images.float()

                if device.type == "cuda":
                    loss_fn_alex.to(device)
                lpips_alex_sum += loss_fn_alex(images, reconstructed_images).sum().item()
                if device.type == "cuda":
                    loss_fn_alex.cpu()
                    torch.cuda.empty_cache()
                if device.type == "cuda":
                    loss_fn_vgg.to(device)
                lpips_vgg_sum += loss_fn_vgg(images, reconstructed_images).sum().item()
                if device.type == "cuda":
                    loss_fn_vgg.cpu()
                    torch.cuda.empty_cache()

                images = (images + 1) / 2
                reconstructed_images = (reconstructed_images + 1) / 2

                if save_dir and rank == 0:
                    B = images.shape[0]
                    paths = batch.get("file_path_")
                    for i in range(B):
                        inp = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        out = (reconstructed_images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        idx = saved_count + i
                        Image.fromarray(inp).save(os.path.join(save_dir, "input", f"{idx:05d}.png"))
                        Image.fromarray(out).save(os.path.join(save_dir, "output", f"{idx:05d}.png"))
                        side = Image.new("RGB", (inp.shape[1] * 2, inp.shape[0]))
                        side.paste(Image.fromarray(inp), (0, 0))
                        side.paste(Image.fromarray(out), (inp.shape[1], 0))
                        side.save(os.path.join(save_dir, "comparison", f"{idx:05d}.png"))
                        if save_native and paths is not None:
                            try:
                                p = paths[i]
                                path = str(p.item()) if hasattr(p, "item") else str(p)
                                raw = Image.open(path).convert("RGB")
                                w, h = raw.size
                                out_pil = Image.fromarray(out)
                                out_upscaled = out_pil.resize((w, h), Image.Resampling.LANCZOS)
                                raw.save(os.path.join(save_dir, "input_native", f"{idx:05d}.png"))
                                out_upscaled.save(os.path.join(save_dir, "output_native", f"{idx:05d}.png"))
                                comp = Image.new("RGB", (w * 2, h))
                                comp.paste(raw, (0, 0))
                                comp.paste(out_upscaled, (w, 0))
                                comp.save(os.path.join(save_dir, "comparison_native", f"{idx:05d}.png"))
                            except Exception as e:
                                warnings.warn(f"Native save failed idx {idx}: {e}")
                    saved_count += B

                images_cpu = images.cpu()
                reconstructed_cpu = reconstructed_images.cpu()
                pred_x = inception_model(images_cpu)[0].squeeze(3).squeeze(2).numpy()
                pred_rec = inception_model(reconstructed_cpu)[0].squeeze(3).squeeze(2).numpy()
                pred_xs.append(pred_x)
                pred_recs.append(pred_rec)

                rgb_restored = (
                    (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                )
                rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                rgb_restored = rgb_restored.astype(np.float32) / 255.0
                rgb_gt = rgb_gt.astype(np.float32) / 255.0
                B = rgb_restored.shape[0]
                for i in range(B):
                    ssim_sum += ssim_loss(rgb_restored[i], rgb_gt[i], data_range=1.0, channel_axis=-1)
                    psnr_sum += psnr_loss(rgb_gt[i], rgb_restored[i])
                num_iter += 1

            except oom_exceptions as e:
                if getattr(args, "skip_oom", False) and "out of memory" in str(e).lower():
                    skipped_oom += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    pbar.update(1)
                    continue
                raise
            pbar.update(1)

    pbar.close()

    pred_xs = np.concatenate(pred_xs, axis=0) if pred_xs else np.zeros((0, dims), dtype=np.float32)
    pred_recs = np.concatenate(pred_recs, axis=0) if pred_recs else np.zeros((0, dims), dtype=np.float32)

    # Distributed reduction
    pred_xs_all, pred_recs_all = gather_inception_preds(pred_xs, pred_recs)
    (
        lpips_alex_sum,
        lpips_vgg_sum,
        ssim_sum,
        psnr_sum,
        num_images,
        num_iter,
        skipped_batches,
        skipped_oom,
    ) = reduce_scalar_metrics(
        lpips_alex_sum, lpips_vgg_sum, ssim_sum, psnr_sum,
        num_images, num_iter, skipped_batches, skipped_oom,
    )
    usage = reduce_usage(usage)
    if device.type == "cuda":
        usage = usage.cpu()

    # Rank 0: compute FID and print/save
    if rank != 0:
        return

    if pred_xs_all is not None and pred_recs_all is not None and len(pred_xs_all) > 0 and len(pred_recs_all) > 0:
        mu_x = np.mean(pred_xs_all, axis=0)
        sigma_x = np.cov(pred_xs_all, rowvar=False)
        mu_rec = np.mean(pred_recs_all, axis=0)
        sigma_rec = np.cov(pred_recs_all, rowvar=False)
        fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    else:
        fid_value = float("nan")

    lpips_alex_value = lpips_alex_sum / num_images if num_images else 0.0
    lpips_vgg_value = lpips_vgg_sum / num_images if num_images else 0.0
    ssim_value = ssim_sum / num_iter if num_iter else 0.0
    psnr_value = psnr_sum / num_iter if num_iter else 0.0
    usage_np = usage.numpy()
    num_used = (usage_np > 0).sum()
    utilization = num_used / codebook_size if codebook_size else 0.0

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batch(es) due to OSError when loading images.")
    if skipped_oom > 0:
        print(f"Skipped {skipped_oom} batch(es) due to out of memory.")
    print("FID: ", fid_value)
    print("LPIPS_ALEX: ", lpips_alex_value)
    print("LPIPS_VGG: ", lpips_vgg_value)
    print("SSIM: ", ssim_value)
    print("PSNR: ", psnr_value)
    print("utilization: ", utilization)

    if save_dir:
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"FID: {fid_value}\n")
            f.write(f"LPIPS_ALEX: {lpips_alex_value}\n")
            f.write(f"LPIPS_VGG: {lpips_vgg_value}\n")
            f.write(f"SSIM: {ssim_value}\n")
            f.write(f"PSNR: {psnr_value}\n")
            f.write(f"utilization: {utilization}\n")
            f.write(f"num_images: {num_images}\n")
            f.write(f"skipped_batches (OSError): {skipped_batches}\n")
            f.write(f"skipped_batches (OOM): {skipped_oom}\n")
        print(f"Summary written to {summary_path}")
