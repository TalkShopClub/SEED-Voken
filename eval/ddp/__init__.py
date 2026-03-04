"""
Distributed data-parallel (DDP) shard inference for image evaluation.

Run multi-GPU evaluation with sharded validation set and reduced metrics:
  torchrun --nproc_per_node=N -m eval.ddp.main --config_file ... --ckpt_path ... --model IBQ [other args]
"""

from eval.ddp.main import main

__all__ = ["main"]
