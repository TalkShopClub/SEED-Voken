"""
Multi-source dataset: combines local image directories and HuggingFace ImageNet parquets.
Supports random crop to a target size (e.g. 384x384).
Balances domains via oversampling: AgentNet images are repeated to match ImageNet count.
"""

import io
import os
import glob
import random
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate as custom_collate
import lightning as L
import albumentations
import pyarrow.parquet as pq

from main import instantiate_from_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _list_images(root):
    """Recursively list all image files under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


class _LazyParquetImages:
    """
    Lazily reads images from parquet files with row-group level indexing.
    Only reads metadata at init; image bytes are fetched on demand per row group.
    Caches the last-read row group so consecutive accesses within the same group
    (e.g. from RowGroupBatchSampler) share a single IO.
    """

    def __init__(self, parquet_dir, split="train"):
        pattern = os.path.join(parquet_dir, f"{split}-*.parquet")
        self._files = sorted(glob.glob(pattern))
        if not self._files:
            raise FileNotFoundError(f"No parquet files matching {pattern}")

        # Build row-group level index: (file_idx, rg_idx, rg_num_rows)
        self._rg_index = []
        self._cum_rows = []  # cumulative row count before each row group
        total = 0
        for fi, pf in enumerate(self._files):
            meta = pq.read_metadata(pf)
            for rg in range(meta.num_row_groups):
                n = meta.row_group(rg).num_rows
                self._rg_index.append((fi, rg, n))
                self._cum_rows.append(total)
                total += n
        self._total = total
        # Row group cache (1 entry) — avoids re-reading the same group for consecutive indices
        self._cached_rg_pos = -1
        self._cached_rg_data = None
        print(f"[LazyParquet] Indexed {len(self._files)} files, "
              f"{len(self._rg_index)} row groups, {self._total} images")

    @property
    def num_row_groups(self):
        return len(self._rg_index)

    def row_group_indices(self, rg_pos):
        """Return the global index range [start, end) for row group rg_pos."""
        start = self._cum_rows[rg_pos]
        _, _, n = self._rg_index[rg_pos]
        return start, start + n

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        import bisect
        rg_pos = bisect.bisect_right(self._cum_rows, idx) - 1
        row_in_rg = idx - self._cum_rows[rg_pos]

        # Cache hit?
        if rg_pos != self._cached_rg_pos:
            file_idx, rg_idx, _ = self._rg_index[rg_pos]
            pf = pq.ParquetFile(self._files[file_idx])
            table = pf.read_row_group(rg_idx, columns=["image"])
            self._cached_rg_data = table.to_pydict()["image"]
            self._cached_rg_pos = rg_pos

        img_bytes = self._cached_rg_data[row_in_rg]["bytes"]
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


class RowGroupBatchSampler:
    """
    Batch sampler that groups ImageNet indices by row group so that each batch
    reads from the same parquet row group, amortising the IO cost.
    Local indices are shuffled and interleaved.

    Usage: pass as batch_sampler to DataLoader (replaces both sampler and batch_size).
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        """
        Args:
            dataset: MultiSourceImageDataset instance
            batch_size: samples per batch
            shuffle: whether to shuffle row-group order and within-group indices
            drop_last: drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        n_local = dataset._n_local
        n_imagenet = dataset._n_imagenet
        imagenet_obj = dataset._imagenet

        # Build batches: mix of local and imagenet indices
        # Strategy: create imagenet batches grouped by row group,
        # then interleave with local batches.
        self._batches = []

        # ImageNet batches grouped by row group
        imagenet_batches = []
        if imagenet_obj is not None and n_imagenet > 0:
            offset = n_local  # imagenet indices start after local
            for rg_pos in range(imagenet_obj.num_row_groups):
                start, end = imagenet_obj.row_group_indices(rg_pos)
                rg_indices = list(range(offset + start, offset + end))
                if shuffle:
                    random.shuffle(rg_indices)
                for i in range(0, len(rg_indices), batch_size):
                    batch = rg_indices[i : i + batch_size]
                    if drop_last and len(batch) < batch_size:
                        continue
                    imagenet_batches.append(batch)

        # Local batches
        local_batches = []
        if n_local > 0:
            local_indices = list(range(n_local))
            if shuffle:
                random.shuffle(local_indices)
            for i in range(0, len(local_indices), batch_size):
                batch = local_indices[i : i + batch_size]
                if drop_last and len(batch) < batch_size:
                    continue
                local_batches.append(batch)

        # Interleave: alternate local and imagenet batches
        self._batches = []
        li, ii = 0, 0
        while li < len(local_batches) or ii < len(imagenet_batches):
            if li < len(local_batches):
                self._batches.append(local_batches[li])
                li += 1
            if ii < len(imagenet_batches):
                self._batches.append(imagenet_batches[ii])
                ii += 1

        if shuffle:
            random.shuffle(self._batches)

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class MultiSourceImageDataset(Dataset):
    """
    Combines multiple local image directories + HuggingFace ImageNet parquet files.
    Balances the two domains: AgentNet local images are oversampled so that each epoch
    sees roughly equal counts from both domains.

    Args:
        local_roots: list of directories containing images (AgentNet)
        imagenet_parquet_dir: path to imagenet-1k/ root (with data/ subdir), or None
        imagenet_split: 'train' or 'test'
        size: target crop size (e.g. 384)
        random_crop: whether to random crop (True) or center crop (False)
        oversample_local: if True, repeat local images to balance with imagenet count
    """

    def __init__(
        self,
        local_roots=None,
        imagenet_parquet_dir=None,
        imagenet_split="train",
        size=384,
        random_crop=True,
        oversample_local=True,
    ):
        self.size = size
        self.random_crop = random_crop

        transforms = [
            albumentations.SmallestMaxSize(max_size=size),
        ]
        if random_crop:
            transforms.append(albumentations.RandomCrop(height=size, width=size))
            transforms.append(albumentations.HorizontalFlip())
        else:
            transforms.append(albumentations.CenterCrop(height=size, width=size))
        self.preprocessor = albumentations.Compose(transforms)

        # Collect local image paths
        self._local_paths = []
        if local_roots:
            for root in local_roots:
                root = os.path.expanduser(root)
                if os.path.isdir(root):
                    paths = _list_images(root)
                    print(f"[MultiSource] Found {len(paths)} images in {root}")
                    self._local_paths.extend(paths)
                else:
                    print(f"[MultiSource] WARNING: {root} not found, skipping")

        # Load ImageNet lazily
        self._imagenet = None
        self._n_imagenet = 0
        if imagenet_parquet_dir and os.path.isdir(imagenet_parquet_dir):
            data_dir = os.path.join(imagenet_parquet_dir, "data") if os.path.isdir(
                os.path.join(imagenet_parquet_dir, "data")
            ) else imagenet_parquet_dir
            self._imagenet = _LazyParquetImages(data_dir, split=imagenet_split)
            self._n_imagenet = len(self._imagenet)

        self._n_local_raw = len(self._local_paths)

        # Oversample local images to balance with ImageNet
        if oversample_local and self._n_local_raw > 0 and self._n_imagenet > 0:
            repeat = max(1, round(self._n_imagenet / self._n_local_raw))
            self._n_local = self._n_local_raw * repeat
            print(f"[MultiSource] Oversampling local {repeat}x: {self._n_local_raw} -> {self._n_local}")
        else:
            self._n_local = self._n_local_raw

        self._total = self._n_local + self._n_imagenet
        if self._total == 0:
            raise ValueError("No images found in any source!")
        print(f"[MultiSource] Total per epoch: {self._total} ({self._n_local} local + {self._n_imagenet} imagenet)")

    def __len__(self):
        return self._total

    def _load_image(self, idx):
        """Load image as numpy uint8 HWC array."""
        if idx < self._n_local:
            # Wrap around for oversampled local images
            real_idx = idx % self._n_local_raw
            img = Image.open(self._local_paths[real_idx]).convert("RGB")
        else:
            img = self._imagenet[idx - self._n_local]
        return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx):
        example = {}
        try:
            image = self._load_image(idx)
            image = self.preprocessor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
            example["image"] = image
            example["image_load_failed"] = False
        except Exception:
            example["image"] = np.full((self.size, self.size, 3), -1.0, dtype=np.float32)
            example["image_load_failed"] = True
        return example


class MultiSourceImageDatasetVal(Dataset):
    """Validation variant: center crop, no flip, limited samples."""

    def __init__(self, local_roots=None, imagenet_parquet_dir=None,
                 imagenet_split="test", size=384, max_samples=2000):
        self.size = size

        transforms = [
            albumentations.SmallestMaxSize(max_size=size),
            albumentations.CenterCrop(height=size, width=size),
        ]
        self.preprocessor = albumentations.Compose(transforms)

        # Collect local paths (no oversampling for val)
        self._local_paths = []
        if local_roots:
            for root in local_roots:
                root = os.path.expanduser(root)
                if os.path.isdir(root):
                    paths = _list_images(root)
                    self._local_paths.extend(paths)

        # ImageNet val
        self._imagenet = None
        self._n_imagenet = 0
        if imagenet_parquet_dir and os.path.isdir(imagenet_parquet_dir):
            data_dir = os.path.join(imagenet_parquet_dir, "data") if os.path.isdir(
                os.path.join(imagenet_parquet_dir, "data")
            ) else imagenet_parquet_dir
            self._imagenet = _LazyParquetImages(data_dir, split=imagenet_split)
            self._n_imagenet = len(self._imagenet)

        self._n_local = len(self._local_paths)
        self._total = self._n_local + self._n_imagenet

        # Limit
        if max_samples and self._total > max_samples:
            # Take proportional from each source
            ratio = self._n_local / self._total if self._total > 0 else 0.5
            n_local_keep = min(self._n_local, int(max_samples * ratio))
            n_imagenet_keep = min(self._n_imagenet, max_samples - n_local_keep)
            self._local_paths = self._local_paths[:n_local_keep]
            self._n_local = len(self._local_paths)
            self._n_imagenet = n_imagenet_keep
            self._total = self._n_local + self._n_imagenet

        print(f"[MultiSourceVal] Total: {self._total} ({self._n_local} local + {self._n_imagenet} imagenet)")

    def __len__(self):
        return self._total

    def _load_image(self, idx):
        if idx < self._n_local:
            return np.array(Image.open(self._local_paths[idx]).convert("RGB"), dtype=np.uint8)
        else:
            img = self._imagenet[idx - self._n_local]
            return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx):
        example = {}
        try:
            image = self._load_image(idx)
            image = self.preprocessor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
            example["image"] = image
            example["image_load_failed"] = False
        except Exception:
            example["image"] = np.full((self.size, self.size, 3), -1.0, dtype=np.float32)
            example["image_load_failed"] = True
        return example


class MultiSourceDataModule(L.LightningDataModule):
    """
    DataModule that uses RowGroupBatchSampler for training (IO-efficient parquet reads)
    and standard DataLoader for validation.
    """

    def __init__(self, batch_size, train=None, validation=None, num_workers=16):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_configs = {}
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation

    def setup(self, stage=None):
        self.datasets = {}
        for k in self.dataset_configs:
            self.datasets[k] = instantiate_from_config(self.dataset_configs[k])

    def _train_dataloader(self):
        train_ds = self.datasets["train"]
        return DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=custom_collate,
            pin_memory=True, persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._train_dataloader()

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"], batch_size=self.batch_size,
            num_workers=self.num_workers, collate_fn=custom_collate,
            shuffle=False, pin_memory=True, persistent_workers=self.num_workers > 0,
        )
