import bisect
import random
import numpy as np
import torch
import albumentations
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, get_worker_info
from torchvision.io import read_image
from PIL import Image

def load_image(path):
    """Load image as (C, H, W) uint8 tensor. Falls back to PIL when torchvision raises
    (e.g. 'At most 8-bit PNG images are supported' for 16-bit PNG or mislabeled files)."""
    try:
        return read_image(path)
    except RuntimeError:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, original_reso=False, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop
        self.original_reso = original_reso

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            # Ensure crop size never exceeds image size: resize so smallest side >= size first
            self.preprocessor = albumentations.Compose([
                albumentations.SmallestMaxSize(max_size=self.size),
                self.cropper,
            ])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)  # (C, H, W) uint8, or (T, C, H, W) for multi-frame
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        if len(image.shape) == 4:  # multi-frame: torchvision returns (T, C, H, W), take first frame
            image = image[0]
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        if not self.original_reso:
            image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class IterableImagePaths(IterableDataset):
    """IterableDataset version of ImagePaths for sequential reads (mitigates random I/O)."""
    def __init__(self, paths, original_reso=False, size=None, random_crop=False, labels=None, shuffle=True, epoch=0):
        self.size = size
        self.random_crop = random_crop
        self.original_reso = original_reso
        self.shuffle = shuffle
        self.epoch = epoch
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        if self.size is not None and self.size > 0:
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            # Ensure crop size never exceeds image size: resize so smallest side >= size first
            self.preprocessor = albumentations.Compose([
                albumentations.SmallestMaxSize(max_size=self.size),
                self.cropper,
            ])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        if len(image.shape) == 4:  # multi-frame: torchvision returns (T, C, H, W), take first frame
            image = image[0]
        try:
            image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")
        if not self.original_reso:
            image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def _get_sample(self, i):
        example = dict()

        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start, iter_end = 0, self._length
            seed = self.epoch
        else:
            per_worker = self._length // worker_info.num_workers
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker if worker_id < worker_info.num_workers - 1 else self._length
            seed = worker_info.seed + self.epoch
        indices = list(range(iter_start, iter_end))
        if self.shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)
        for i in indices:
            yield self._get_sample(i)

    def __getitem__(self, i):
        return self._get_sample(i)


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        if not self.original_reso:
            image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
