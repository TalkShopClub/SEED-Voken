"""
Fine-tune script: same entrypoint as main.py but intended for loading a pretrained
vision tokenizer (e.g. from pretrained/ with model definitions in src/vision_tokenizer/)
and fine-tuning it on the given data in the same fashion as IBQ tokenizers.

Usage:
  python finetune.py fit -c configs/IBQ/gpu/finetune_256.yaml

Use a config with model.class_path: src.IBQ.models.ibqgan.IBQFromPretrained
and model.init_args.pretrained_path: "pretrained" (or path to your pretrained dir).
"""
import argparse
import os
import sys

# Allow imports of top-level "taming" (src/taming) and "src.*" when run from repo root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
import datetime
import glob
import importlib
from torch.utils.data import random_split, DataLoader, Dataset, IterableDataset

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything

from torch.utils.data.dataloader import default_collate as custom_collate

import torch

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        wrap=False,
        num_workers=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict()
        for k in self.dataset_configs:
            if "pretrain" not in self.dataset_configs[k]["target"]:
                self.datasets[k] = instantiate_from_config(self.dataset_configs[k])
            else:
                self.datasets[k] = instantiate_from_config(
                    self.dataset_configs[k]
                ).create_dataset()
        if self.wrap:
            for k in self.datasets:
                if not isinstance(self.datasets[k], IterableDataset):
                    self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        train_ds = self.datasets["train"]
        if "pretrain" in self.dataset_configs["train"]["target"]:
            return DataLoader(
                train_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        shuffle = not isinstance(train_ds, IterableDataset)
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=custom_collate,
            pin_memory=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
            shuffle=False,
            pin_memory=True,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
            shuffle=False,
            pin_memory=True,
        )


class WandbLoggerCallback(Callback):
    """Adds WandbLogger to the trainer at fit/test start so training and evaluation are logged to wandb."""

    def __init__(self, project: str = "seed-voken-finetune", **kwargs):
        super().__init__()
        self.project = project
        self.wandb_kwargs = kwargs

    def _ensure_wandb_logger(self, trainer):
        loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
        if loggers is None:
            loggers = []
        if not isinstance(loggers, list):
            loggers = [loggers]
        if any(isinstance(lg, WandbLogger) for lg in loggers):
            return
        wandb_logger = WandbLogger(project=self.project, **self.wandb_kwargs)
        trainer._loggers = list(loggers) + [wandb_logger]

    def on_fit_start(self, trainer, pl_module):
        self._ensure_wandb_logger(trainer)

    def on_test_start(self, trainer, pl_module):
        self._ensure_wandb_logger(trainer)


class TotalLossProgressBar(TQDMProgressBar):
    """Progress bar that includes total training loss (ae + disc) in the display."""

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        # Add total training loss from logged metrics (ae total + disc loss)
        cm = trainer.callback_metrics
        total = None
        if "train/total_loss" in cm and "train/disc_loss" in cm:
            v_ae, v_disc = cm["train/total_loss"], cm["train/disc_loss"]
            total = (v_ae.item() if hasattr(v_ae, "item") else v_ae) + (
                v_disc.item() if hasattr(v_disc, "item") else v_disc
            )
        elif "train/total_loss" in cm:
            v = cm["train/total_loss"]
            total = v.item() if hasattr(v, "item") else v
        elif "train/disc_loss" in cm:
            v = cm["train/disc_loss"]
            total = v.item() if hasattr(v, "item") else v
        if total is not None:
            items["loss"] = f"{total:.4f}"
        return items


class SetIterableDatasetEpoch(Callback):
    """Set train IterableDataset.data.epoch to current_epoch at epoch start.
    Ensures each epoch gets a different shuffle order (IterableImagePaths uses epoch in RNG seed).
    Does not fix mid-epoch resume: if you resume from a checkpoint saved mid-epoch, the dataloader
    restarts from the beginning of that epoch so some batches are seen twice in that epoch."""
    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if not hasattr(dm, "datasets") or "train" not in dm.datasets:
            return
        ds = dm.datasets["train"]
        if hasattr(ds, "data") and hasattr(ds.data, "epoch"):
            ds.data.epoch = trainer.current_epoch


class FinetuneCLI(LightningCLI):
    """CLI that adds WandbLogger for training and evaluation logging."""

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="seed-voken-finetune",
            help="Wandb project name for logging.",
        )

    def before_instantiate_classes(self) -> None:
        # Inject callbacks: progress bar with total loss, WandbLogger, and epoch for IterableDataset
        sub = getattr(self.config, str(self.subcommand), None) if self.subcommand else None
        wandb_project = getattr(sub, "wandb_project", None) or getattr(
            self.config, "wandb_project", "seed-voken-finetune"
        )
        defaults = self.trainer_defaults or {}
        extra_callbacks = list(defaults.get("callbacks", []))
        extra_callbacks.append(TotalLossProgressBar())
        extra_callbacks.append(WandbLoggerCallback(project=wandb_project))
        extra_callbacks.append(SetIterableDatasetEpoch())
        self.trainer_defaults = {**defaults, "callbacks": extra_callbacks}


def main():
    cli = FinetuneCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
