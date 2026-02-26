import argparse, os, sys, datetime, glob, importlib
from torch.utils.data import random_split, DataLoader, Dataset, IterableDataset

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything

from torch.utils.data.dataloader import default_collate as custom_collate

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
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
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
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
            if "pretrain" not in self.dataset_configs[k]["target"]: ##laion should use webdataset
                self.datasets[k] = instantiate_from_config(self.dataset_configs[k])
            else:
                self.datasets[k] = instantiate_from_config(self.dataset_configs[k]).create_dataset()
        if self.wrap:
            for k in self.datasets:
                # Don't wrap IterableDataset; DataLoader must use iter() for sequential reads
                if not isinstance(self.datasets[k], IterableDataset):
                    self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        """
        laion serves as the train loader
        """
        train_ds = self.datasets["train"]
        if "pretrain" in self.dataset_configs["train"]["target"]:  # webdataset no need for shuffle=True
            return DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        # IterableDataset shuffles inside __iter__; DataLoader must not shuffle
        shuffle = not isinstance(train_ds, IterableDataset)
        return DataLoader(train_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle, collate_fn=custom_collate, pin_memory=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate, shuffle=False, pin_memory=True)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate, shuffle=False, pin_memory=True)


class WandbLoggerCallback(Callback):
    """Adds WandbLogger to the trainer at fit/test start so training and evaluation are logged to wandb."""

    def __init__(self, project: str = "seed-voken", **kwargs):
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


class MainCLI(LightningCLI):
    """CLI that adds WandbLogger for training and evaluation logging."""

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="seed-voken",
            help="Wandb project name for logging.",
        )

    def before_instantiate_classes(self) -> None:
        sub = getattr(self.config, str(self.subcommand), None) if self.subcommand else None
        wandb_project = getattr(sub, "wandb_project", None) or getattr(
            self.config, "wandb_project", "seed-voken"
        )
        defaults = self.trainer_defaults or {}
        extra_callbacks = list(defaults.get("callbacks", []))
        extra_callbacks.append(TotalLossProgressBar())
        extra_callbacks.append(WandbLoggerCallback(project=wandb_project))
        self.trainer_defaults = {**defaults, "callbacks": extra_callbacks}


def main():
    cli = MainCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
