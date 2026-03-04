"""
We provide Tokenizer Evaluation code here.
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
"""

import os
import sys
import contextlib
_cwd = os.getcwd()
sys.path.append(_cwd)
# So that "taming" resolves to src/taming (used by IBQ OCR loss)
_src = os.path.join(_cwd, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
import torch
try:
    import torch_npu
except: 
    pass

from omegaconf import OmegaConf
import importlib
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg
#### Note When using original imagenet setup
from src.Open_MAGVIT2.models.lfqgan import VQModel
### When using pretrain setup
# from src.Open_MAGVIT2.models.lfqgan_pretrain import VQModel
from src.IBQ.models.ibqgan import IBQ
### When using pretrain setup (use alias so Open-MAGVIT2 keeps lfqgan.VQModel)
# from src.IBQ.models.ibqgan_pretrain import VQModel
from metrics.inception import InceptionV3
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse
import warnings

if hasattr(torch, "npu"):
    DEVICE = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## for different model configuration
MODEL_TYPE = {
    "Open-MAGVIT2": VQModel,
    "IBQ": IBQ
}

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
    if hasattr(config.model, "class_path") and config.model.class_path:
        init_args = OmegaConf.to_container(config.model.get("init_args", {}), resolve=True)
        model = get_obj_from_str(str(config.model.class_path))(**init_args)
    else:
        model = MODEL_TYPE[model_type](**config.model.init_args)
    if ckpt_path is not None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint file is corrupted or incomplete: {ckpt_path}\n"
                "Try another checkpoint (e.g. an earlier epoch) or re-save the checkpoint."
            ) from e
        sd = ckpt["state_dict"]
        # Load in bf16 to reduce memory (bf16 mixed inference)
        sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Replace inf/NaN in covariances and regularize if product would be singular
    offset = np.eye(sigma1.shape[0]) * eps
    if not np.isfinite(sigma1).all() or not np.isfinite(sigma2).all():
        sigma1 = np.where(np.isfinite(sigma1), sigma1, 0.0) + offset
        sigma2 = np.where(np.isfinite(sigma2), sigma2, 0.0) + offset
    cov_product = sigma1.dot(sigma2)
    if not np.isfinite(cov_product).all():
        msg = (
            "fid calculation produces singular product or non-finite cov; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        sigma1 = sigma1 + offset
        sigma2 = sigma2 + offset
        cov_product = sigma1.dot(sigma2)
    covmean, _ = linalg.sqrtm(cov_product, disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def pad_to_encoder_multiple(batch, spatial_align):
    """Pad a batch of images (B, C, H, W) so H and W are divisible by spatial_align.
    Returns padded batch and crop_region (y1, x1, y2, x2) to restore original size."""
    _, _, height, width = batch.shape
    height_to_pad = (spatial_align - height % spatial_align) if height % spatial_align != 0 else 0
    width_to_pad = (spatial_align - width % spatial_align) if width % spatial_align != 0 else 0
    crop_region = (
        height_to_pad >> 1,
        width_to_pad >> 1,
        height + (height_to_pad >> 1),
        width + (width_to_pad >> 1),
    )
    padded = torch.nn.functional.pad(
        batch,
        (
            width_to_pad >> 1,
            width_to_pad - (width_to_pad >> 1),
            height_to_pad >> 1,
            height_to_pad - (height_to_pad >> 1),
        ),
        mode="constant",
        value=0,
    )
    return padded, crop_region


def unpad_to_region(batch, crop_region):
    """Crop batch (B, C, H, W) to region (y1, x1, y2, x2)."""
    y1, x1, y2, x2 = crop_region
    return batch[:, :, y1:y2, x1:x2]


def get_encoder_spatial_align(model):
    """Return spatial alignment (downsample factor) required by the encoder."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "num_resolutions"):
        return 2 ** (model.encoder.num_resolutions - 1)
    return 16


def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--image_size", default=1.0, type=float, help="Scale factor for H and W (default 1 = native resolution).")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for validation (use 1 for native resolution).")
    parser.add_argument("--model", choices=["Open-MAGVIT2", "IBQ"])
    parser.add_argument(
        "--save_comparison_dir",
        default=None,
        type=str,
        help="If set, save input and reconstructed images here for comparison (input/, output/, comparison/ side-by-side).",
    )
    parser.add_argument(
        "--save_native_resolution",
        action="store_true",
        help="When saving comparisons, also save raw input and output at native (original) resolution. Requires file paths in the batch (e.g. LocalImages). Output is upscaled to match input size.",
    )
    parser.add_argument(
        "--skip_oom",
        action="store_true",
        help="Skip any batch that causes a CUDA/device out of memory error instead of failing.",
    )

    return parser.parse_args()

def main(args):
    config_data = OmegaConf.load(args.config_file)
    OmegaConf.resolve(config_data)  # resolve ${oc.env:MAX_PATHS} etc. before dataloader uses manifest_path
    # Native resolution: no resizing before model inference; tokenize/detokenize at original size
    if hasattr(config_data.data.init_args.validation.params, "config"):
        config_data.data.init_args.validation.params.config.original_reso = True
        config_data.data.init_args.validation.params.config.size = 0
    config_data.data.init_args.batch_size = args.batch_size

    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, model_type=args.model, ckpt_path=args.ckpt_path).to(DEVICE).to(torch.bfloat16)  # bf16 mixed inference
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
    
    #usage
    usage = {}
    for i in range(codebook_size):
        usage[i] = 0


    # FID score related: keep Inception on CPU to save GPU VRAM (run on CPU each batch)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx])
    inception_model.eval()

    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []

    # LPIPS score related: on CUDA keep on CPU and move to GPU one at a time per batch to minimize VRAM
    loss_fn_alex = lpips.LPIPS(net='alex').eval()  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').eval()   # closer to "traditional" perceptual loss, when used for optimization
    if DEVICE.type != "cuda":
        loss_fn_alex, loss_fn_vgg = loss_fn_alex.to(DEVICE), loss_fn_vgg.to(DEVICE)
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0

    num_images = 0
    num_iter = 0
    save_dir = getattr(args, "save_comparison_dir", None)
    save_native = getattr(args, "save_native_resolution", False)
    if save_dir:
        save_dir = os.path.expanduser(save_dir)
        input_dir = os.path.join(save_dir, "input")
        output_dir = os.path.join(save_dir, "output")
        comparison_dir = os.path.join(save_dir, "comparison")
        for d in (input_dir, output_dir, comparison_dir):
            os.makedirs(d, exist_ok=True)
        if save_native:
            input_native_dir = os.path.join(save_dir, "input_native")
            output_native_dir = os.path.join(save_dir, "output_native")
            comparison_native_dir = os.path.join(save_dir, "comparison_native")
            for d in (input_native_dir, output_native_dir, comparison_native_dir):
                os.makedirs(d, exist_ok=True)
            print(f"Saving input/output comparison images to {save_dir} (including native resolution)")
        else:
            print(f"Saving input/output comparison images to {save_dir}")
        global_idx = 0

    # Inference-only: no gradients, no computation graph, minimal memory
    torch.set_grad_enabled(False)
    skipped_batches = 0
    skipped_oom = 0
    oom_exceptions = [RuntimeError]
    if hasattr(torch, "OutOfMemoryError"):
        oom_exceptions.append(torch.OutOfMemoryError)
    if hasattr(torch.cuda, "OutOfMemoryError") and torch.cuda.OutOfMemoryError not in oom_exceptions:
        oom_exceptions.append(torch.cuda.OutOfMemoryError)
    oom_exceptions = tuple(oom_exceptions)
    dataloader = dataset._val_dataloader()
    try:
        num_batches = len(dataloader)
    except TypeError:
        num_batches = None
    loader_iter = iter(dataloader)
    pbar = tqdm(total=num_batches, unit="batch")
    with torch.inference_mode():
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            except OSError as e:
                skipped_batches += 1
                warnings.warn(f"Skipping batch due to OSError when loading images: {e}")
                pbar.update(1)
                continue

            try:
                # Detach input so we never build a graph from the dataloader
                images = batch["image"].permute(0, 3, 1, 2).detach().to(DEVICE, non_blocking=True)  # (B, C, H, W)
            except OSError as e:
                skipped_batches += 1
                paths = batch.get("file_path_", batch.get("path", None))
                path_hint = f" (paths: {paths})" if paths is not None else ""
                warnings.warn(f"Skipping batch due to OSError when loading images: {e}{path_hint}")
                pbar.update(1)
                continue
            except oom_exceptions as e:
                if args.skip_oom and "out of memory" in str(e).lower():
                    skipped_oom += 1
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
                    warnings.warn(f"Skipping batch due to out of memory: {e}")
                    pbar.update(1)
                    continue
                raise

            try:
                # Resize by scalar factor image_size (1 = native resolution)
                _, _, h, w = images.shape
                scale = args.image_size
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                images = torch.nn.functional.interpolate(
                    images, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                num_images += images.shape[0]
                # Match model dtype (bf16) for encode/decode
                images = images.to(torch.bfloat16)

                # Pad to encoder multiple for tokenize/detokenize at native resolution
                images_padded, crop_region = pad_to_encoder_multiple(images, spatial_align)
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else contextlib.nullcontext()
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
                # Crop back to native resolution for metrics and saving
                images = unpad_to_region(images_padded, crop_region)
                reconstructed_images = unpad_to_region(reconstructed_padded, crop_region)
                # Free large intermediates before metrics to reduce peak memory
                del images_padded, reconstructed_padded, quant
                if args.model == "IBQ":
                    del qloss
                else:
                    del diff  # Open-MAGVIT2
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                ### usage
                for index in indices:
                    usage[index.item()] += 1

                # Metrics expect float32
                images = images.float()
                reconstructed_images = reconstructed_images.float()

                # calculate lpips (one net on GPU at a time to save VRAM)
                if DEVICE.type == "cuda":
                    loss_fn_alex.to(DEVICE)
                lpips_alex += loss_fn_alex(images, reconstructed_images).sum()
                if DEVICE.type == "cuda":
                    loss_fn_alex.cpu()
                    torch.cuda.empty_cache()
                if DEVICE.type == "cuda":
                    loss_fn_vgg.to(DEVICE)
                lpips_vgg += loss_fn_vgg(images, reconstructed_images).sum()
                if DEVICE.type == "cuda":
                    loss_fn_vgg.cpu()
                    torch.cuda.empty_cache()

                images = (images + 1) / 2
                reconstructed_images = (reconstructed_images + 1) / 2

                # save input and output for comparison (optional)
                if save_dir:
                    B = images.shape[0]
                    paths = batch.get("file_path_")  # available for LocalImages / ImagePaths
                    for i in range(B):
                        inp = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        out = (reconstructed_images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        idx = global_idx + i
                        Image.fromarray(inp).save(os.path.join(input_dir, f"{idx:05d}.png"))
                        Image.fromarray(out).save(os.path.join(output_dir, f"{idx:05d}.png"))
                        # side-by-side: input | output
                        side_by_side = Image.new("RGB", (inp.shape[1] * 2, inp.shape[0]))
                        side_by_side.paste(Image.fromarray(inp), (0, 0))
                        side_by_side.paste(Image.fromarray(out), (inp.shape[1], 0))
                        side_by_side.save(os.path.join(comparison_dir, f"{idx:05d}.png"))
                        # native resolution: raw input + output upscaled to match
                        if save_native and paths is not None:
                            try:
                                p = paths[i]
                                path = str(p.item()) if hasattr(p, "item") else str(p)
                                raw = Image.open(path).convert("RGB")
                                raw_arr = np.array(raw)
                                w, h = raw.size
                                out_pil = Image.fromarray(out)
                                out_upscaled = out_pil.resize((w, h), Image.Resampling.LANCZOS)
                                raw.save(os.path.join(input_native_dir, f"{idx:05d}.png"))
                                out_upscaled.save(os.path.join(output_native_dir, f"{idx:05d}.png"))
                                comp_native = Image.new("RGB", (w * 2, h))
                                comp_native.paste(raw, (0, 0))
                                comp_native.paste(out_upscaled, (w, 0))
                                comp_native.save(os.path.join(comparison_native_dir, f"{idx:05d}.png"))
                            except Exception as e:
                                warnings.warn(f"Native-resolution save failed for index {idx}: {e}")
                    global_idx += B

                # calculate fid (Inception on CPU; move batch to CPU for forward, then numpy)
                images_cpu = images.cpu()
                reconstructed_cpu = reconstructed_images.cpu()
                pred_x = inception_model(images_cpu)[0].squeeze(3).squeeze(2).numpy()
                pred_rec = inception_model(reconstructed_cpu)[0].squeeze(3).squeeze(2).numpy()
                pred_xs.append(pred_x)
                pred_recs.append(pred_rec)

                #calculate PSNR and SSIM
                rgb_restored = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                rgb_restored = rgb_restored.astype(np.float32) / 255.
                rgb_gt = rgb_gt.astype(np.float32) / 255.
                ssim_temp = 0
                psnr_temp = 0
                B, _, _, _ = rgb_restored.shape
                for i in range(B):
                    rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                    ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                    psnr_temp += psnr_loss(rgb_gt_s, rgb_restored_s)
                ssim_value += ssim_temp / B
                psnr_value += psnr_temp / B
                num_iter += 1
                pbar.update(1)

            except oom_exceptions as e:
                if args.skip_oom and "out of memory" in str(e).lower():
                    skipped_oom += 1
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
                    warnings.warn(f"Skipping batch due to out of memory: {e}")
                    pbar.update(1)
                    continue
                raise

    pbar.close()
    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)


    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    lpips_alex_value = lpips_alex / num_images
    lpips_vgg_value = lpips_vgg / num_images
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter

    num_count = sum([1 for key, value in usage.items() if value > 0])
    utilization = num_count / codebook_size

    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batch(es) due to OSError when loading images.")
    if skipped_oom > 0:
        print(f"Skipped {skipped_oom} batch(es) due to out of memory.")
    print("FID: ", fid_value)
    print("LPIPS_ALEX: ", lpips_alex_value.item())
    print("LPIPS_VGG: ", lpips_vgg_value.item())
    print("SSIM: ", ssim_value)
    print("PSNR: ", psnr_value)
    print("utilization", utilization)

    save_dir = getattr(args, "save_comparison_dir", None)
    if save_dir:
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"FID: {fid_value}\n")
            f.write(f"LPIPS_ALEX: {lpips_alex_value.item()}\n")
            f.write(f"LPIPS_VGG: {lpips_vgg_value.item()}\n")
            f.write(f"SSIM: {ssim_value}\n")
            f.write(f"PSNR: {psnr_value}\n")
            f.write(f"utilization: {utilization}\n")
            f.write(f"num_images: {num_images}\n")
            f.write(f"skipped_batches (OSError): {skipped_batches}\n")
            f.write(f"skipped_batches (OOM): {skipped_oom}\n")
        print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    args = get_args()
    main(args)