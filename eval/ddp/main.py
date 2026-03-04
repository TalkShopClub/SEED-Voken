"""
Entrypoint for DDP shard inference. Run with torchrun so each process gets RANK/LOCAL_RANK/WORLD_SIZE.

  torchrun --nproc_per_node=N -m eval.ddp.main --config_file CONFIG --ckpt_path CKPT --model IBQ [options]

Single-GPU (no torchrun): same CLI, runs one process (rank 0, world_size 1) and skips process group init.
"""

import os
import sys
import argparse

# Ensure repo root is on path so evaluation_image, finetune, metrics, src resolve
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.distributed as dist


def get_args():
    parser = argparse.ArgumentParser(
        description="DDP shard inference for image eval (FID, LPIPS, SSIM, PSNR)."
    )
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument(
        "--model",
        required=True,
        choices=["Open-MAGVIT2", "IBQ"],
    )
    parser.add_argument(
        "--image_size",
        default=1.0,
        type=float,
        help="Scale factor for H and W (1 = native resolution).",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="DataLoader num_workers per process.",
    )
    parser.add_argument(
        "--save_comparison_dir",
        default=None,
        type=str,
        help="If set, rank 0 saves input/output comparison images here.",
    )
    parser.add_argument(
        "--save_native_resolution",
        action="store_true",
        help="When saving, also save at native resolution.",
    )
    parser.add_argument(
        "--skip_oom",
        action="store_true",
        help="Skip batches that cause OOM instead of failing.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT; env:// uses them
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
        if rank == 0:
            print(f"DDP eval: world_size={world_size}")
    else:
        if rank == 0:
            print("Single-process eval (world_size=1).")

    from eval.ddp.inference import run_worker

    run_worker(rank, world_size, args)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
