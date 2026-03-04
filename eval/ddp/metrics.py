"""
Distributed metric aggregation for DDP eval: gather Inception activations for FID,
all-reduce scalar sums/counts for LPIPS/SSIM/PSNR, and all-reduce codebook usage.
"""

import numpy as np
from scipy import linalg
import torch
import torch.distributed as dist


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance (FID).
    Stable version by Dougal J. Sutherland.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    if not np.isfinite(sigma1).all() or not np.isfinite(sigma2).all():
        sigma1 = np.where(np.isfinite(sigma1), sigma1, 0.0) + offset
        sigma2 = np.where(np.isfinite(sigma2), sigma2, 0.0) + offset
    cov_product = sigma1.dot(sigma2)
    if not np.isfinite(cov_product).all():
        sigma1 = sigma1 + offset
        sigma2 = sigma2 + offset
        cov_product = sigma1.dot(sigma2)
    covmean, _ = linalg.sqrtm(cov_product, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in covmean")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def gather_inception_preds(pred_xs: np.ndarray, pred_recs: np.ndarray):
    """
    Gather per-rank Inception activations to rank 0 for global FID.
    pred_xs, pred_recs: (N_local, dim) numpy arrays.
    Returns (pred_xs_all, pred_recs_all) on rank 0, (None, None) on other ranks.
    """
    if not dist.is_initialized():
        return pred_xs, pred_recs
    obj = (pred_xs, pred_recs)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj)
    if dist.get_rank() != 0:
        return None, None
    pred_xs_all = np.concatenate([g[0] for g in gathered], axis=0)
    pred_recs_all = np.concatenate([g[1] for g in gathered], axis=0)
    return pred_xs_all, pred_recs_all


def reduce_scalar_metrics(
    lpips_alex_sum: float,
    lpips_vgg_sum: float,
    ssim_sum: float,
    psnr_sum: float,
    num_images: int,
    num_iter: int,
    skipped_batches: int,
    skipped_oom: int,
):
    """
    All-reduce scalar sums and counts so rank 0 has global totals.
    Returns (lpips_alex_sum, lpips_vgg_sum, ssim_sum, psnr_sum, num_images, num_iter,
             skipped_batches, skipped_oom) as tensors on current device; same on all ranks.
    """
    if not dist.is_initialized():
        return (
            lpips_alex_sum, lpips_vgg_sum, ssim_sum, psnr_sum,
            num_images, num_iter, skipped_batches, skipped_oom,
        )
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    buf = torch.tensor(
        [
            lpips_alex_sum, lpips_vgg_sum, ssim_sum, psnr_sum,
            float(num_images), float(num_iter), float(skipped_batches), float(skipped_oom),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return (
        buf[0].item(), buf[1].item(), buf[2].item(), buf[3].item(),
        int(buf[4].item()), int(buf[5].item()), int(buf[6].item()), int(buf[7].item()),
    )


def reduce_usage(usage_tensor: torch.Tensor):
    """All-reduce codebook usage counts (long tensor of shape [codebook_size])."""
    if not dist.is_initialized():
        return usage_tensor
    dist.all_reduce(usage_tensor, op=dist.ReduceOp.SUM)
    return usage_tensor
