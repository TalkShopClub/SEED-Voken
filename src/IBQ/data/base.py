import bisect
import io
import json
import os
import random
import numpy as np
import torch
import albumentations
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, get_worker_info
from torchvision.io import read_image
from PIL import Image

_PIL_FIRST_EXTENSIONS = {".png"}


def render_mask_from_bboxes(bboxes, height, width):
    """Render a binary mask (uint8, 0/255) from a list of [x1, y1, x2, y2] ratio-coordinate bboxes."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        px1 = int(x1 * width)
        py1 = int(y1 * height)
        px2 = int(x2 * width)
        py2 = int(y2 * height)
        # Clamp
        px1 = max(0, min(width, px1))
        py1 = max(0, min(height, py1))
        px2 = max(0, min(width, px2))
        py2 = max(0, min(height, py2))
        if py2 > py1 and px2 > px1:
            mask[py1:py2, px1:px2] = 255
    return mask


def load_annotations(annotations_path):
    """Load pre-computed OmniParser annotations from JSON file."""
    if annotations_path is None:
        return None
    annotations_path = os.path.expanduser(annotations_path)
    if not os.path.isfile(annotations_path):
        return None
    with open(annotations_path, "r") as f:
        return json.load(f)


def _load_image_with_pil_path(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)


def _load_image_with_pil_bytes(image_data):
    with Image.open(io.BytesIO(image_data)) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)


def load_image(path):
    """Load image as (C, H, W) uint8 tensor.

    PNGs are decoded with PIL first to avoid known torchvision/libpng failures on
    malformed iCCP metadata. Other formats use torchvision first and PIL fallback.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in _PIL_FIRST_EXTENSIONS:
        try:
            return _load_image_with_pil_path(path)
        except Exception:
            return read_image(path)
    try:
        return read_image(path)
    except Exception:
        return _load_image_with_pil_path(path)


def load_image_bytes(image_data, source_hint=""):
    """Load image bytes as (C, H, W) uint8 tensor with torchvision->PIL fallback."""
    hint = source_hint.lower() if source_hint else ""
    if hint.endswith(".png"):
        try:
            return _load_image_with_pil_bytes(image_data)
        except Exception:
            pass
    try:
        return read_image(torch.frombuffer(memoryview(image_data), dtype=torch.uint8))
    except Exception:
        return _load_image_with_pil_bytes(image_data)


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
        try:
            example["image"] = self.preprocess_image(self.labels["file_path_"][i])
            example["image_load_failed"] = False
        except (OSError, IOError, RuntimeError, ValueError) as e:
            # Placeholder so collate and training_step can run; training_step will skip the batch
            size = self.size if self.size else 256
            example["image"] = np.full((size, size, 3), -1.0, dtype=np.float32)
            example["image_load_failed"] = True
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class IterableImagePaths(IterableDataset):
    """IterableDataset version of ImagePaths for sequential reads (mitigates random I/O)."""
    def __init__(self, paths, original_reso=False, size=None, random_crop=False, labels=None,
                 shuffle=True, epoch=0, annotations_path=None):
        self.size = size
        self.random_crop = random_crop
        self.original_reso = original_reso
        self.shuffle = shuffle
        self.epoch = epoch
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        # Load OmniParser annotations for mask channels
        self.annotations = load_annotations(annotations_path)
        self._use_masks = self.annotations is not None

        additional_targets = None
        if self._use_masks:
            additional_targets = {"text_mask": "mask"}

        if self.size is not None and self.size > 0:
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            # Ensure crop size never exceeds image size: resize so smallest side >= size first
            self.preprocessor = albumentations.Compose(
                [
                    albumentations.SmallestMaxSize(max_size=self.size),
                    self.cropper,
                ],
                additional_targets=additional_targets,
            )
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

        if self._use_masks:
            h, w = image.shape[:2]
            fname = os.path.basename(image_path)
            ann = self.annotations.get(fname, {})
            text_mask = render_mask_from_bboxes(ann.get("text_bboxes", []), h, w)

            if not self.original_reso:
                result = self.preprocessor(image=image, text_mask=text_mask)
                image = result["image"]
                text_mask = result["text_mask"]

            image = (image / 127.5 - 1.0).astype(np.float32)
            # Normalize mask to [0, 1] float32
            text_mask = (text_mask / 255.0).astype(np.float32)
            # Stack: (H, W, 4) = RGB + text_mask
            image = np.concatenate([image, text_mask[..., None]], axis=-1)
        else:
            if not self.original_reso:
                image = self.preprocessor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def _get_sample(self, i):
        example = dict()
        try:
            example["image"] = self.preprocess_image(self.labels["file_path_"][i])
            example["image_load_failed"] = False
        except (OSError, IOError, RuntimeError, ValueError) as e:
            size = self.size if self.size else 256
            n_channels = 4 if self._use_masks else 3
            example["image"] = np.full((size, size, n_channels), -1.0, dtype=np.float32)
            example["image_load_failed"] = True
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
