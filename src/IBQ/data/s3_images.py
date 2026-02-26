import os
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import albumentations
import boto3
import random
from botocore.exceptions import ClientError

from src.IBQ.util import retrieve, KeyNotFoundError
from src.IBQ.data.base import IterableImagePaths, load_image
from src.IBQ.data.image_resize import SmartResize
from src.manifest_utils import (
    ensure_manifest,
    get_failed_samples_path_from_manifest,
    load_failed_paths,
)
from torchvision.io import decode_image, write_png

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class S3ImagePaths(IterableDataset):
    """
    IterableDataset that loads images sequentially from S3 (mitigates random reads).
    Uses albumentations for preprocessing (same as IBQ ImagePaths).
    """
    def __init__(self, s3_keys, s3_client, bucket, size=None, random_crop=False,
                 original_reso=False, labels=None, cache_dir=None, shuffle=True, epoch=0):
        self.s3_client = s3_client
        self.bucket = bucket
        self.size = size
        self.random_crop = random_crop
        self.original_reso = original_reso
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.epoch = epoch
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = s3_keys
        self._length = len(s3_keys)
        if self.size is not None and self.size > 0:
            self.rescaler = SmartResize(area=self.size * self.size, ds_factor=16)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            # Ensure crop size never exceeds image size (SmartResize may leave small images unchanged)
            self.preprocessor = albumentations.Compose([
                self.rescaler,
                albumentations.SmallestMaxSize(max_size=self.size),
                self.cropper,
            ])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def _load_from_s3(self, s3_key):
        if self.cache_dir is not None:
            safe_key = s3_key.replace("\\", "/")
            cache_path = os.path.join(self.cache_dir, safe_key)
            if os.path.exists(cache_path):
                try:
                    return load_image(cache_path)
                except Exception as e:
                    print(f"Warning: Cached file {cache_path} is corrupted, re-downloading. Error: {e}")
                    try:
                        os.remove(cache_path)
                    except Exception:
                        pass
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            image_data = response['Body'].read()
            image = decode_image(torch.frombuffer(memoryview(image_data), dtype=torch.uint8), mode="RGB")
            if self.cache_dir is not None:
                safe_key = s3_key.replace("\\", "/")
                cache_path = os.path.join(self.cache_dir, safe_key)
                cache_dir_path = os.path.dirname(cache_path)
                if cache_dir_path:
                    os.makedirs(cache_dir_path, exist_ok=True)
                write_png(image, cache_path)
            return image
        except ClientError as e:
            raise FileNotFoundError(f"Failed to load image from S3: s3://{self.bucket}/{s3_key}. Error: {e}")

    def preprocess_image(self, s3_key):
        image = self._load_from_s3(s3_key)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        if not self.original_reso:
            image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
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


class S3ImagesBase(IterableDataset):
    """
    Base class for loading images from S3, similar to ImageNetBase.
    Optional skip_files (default True): exclude keys listed in failed_samples_path JSON if set.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)

        skip_files = retrieve(self.config, "skip_files", default=True)
        failed_samples_path = retrieve(self.config, "failed_samples_path", default=None)
        self._keys_to_skip = set()
        if skip_files and failed_samples_path:
            self._keys_to_skip = load_failed_paths(failed_samples_path, normalize=False)
            if self._keys_to_skip:
                log.info("Skipping %d keys from failed_samples: %s", len(self._keys_to_skip), failed_samples_path)

        # Get S3 configuration
        self.bucket = retrieve(self.config, "bucket", default=None)
        self.object_prefix = retrieve(self.config, "object", default="")
        try:
            self.aws_access_key_id = retrieve(self.config, "aws_access_key_id", default=None)
            self.aws_secret_access_key = retrieve(self.config, "aws_secret_access_key", default=None)
            self.aws_session_token = retrieve(self.config, "aws_session_token", default=None)
        except KeyNotFoundError:
            log.info("No AWS credentials found in config, fetching from environment variables.")
            self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", None)
            self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", None)
            self.aws_session_token = os.getenv("AWS_SESSION_TOKEN", None)
        self.region_name = retrieve(self.config, "region_name", default=None)
        self.endpoint_url = retrieve(self.config, "endpoint_url", default=None) or os.getenv("S3_ENDPOINT_URL", None)
        try:
            self.cache_dir = retrieve(self.config, "cache_dir", default=None)
            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
        except KeyNotFoundError:
            log.info("No cache directory found in config")
            self.cache_dir = None

        if self.bucket is None:
            raise ValueError("'bucket' must be specified in config")

        # Initialize S3 client
        s3_kwargs = {}
        if self.aws_access_key_id:
            s3_kwargs['aws_access_key_id'] = self.aws_access_key_id
        if self.aws_secret_access_key:
            s3_kwargs['aws_secret_access_key'] = self.aws_secret_access_key
        if self.aws_session_token:
            s3_kwargs['aws_session_token'] = self.aws_session_token
        if self.region_name:
            s3_kwargs['region_name'] = self.region_name
        if self.endpoint_url:
            s3_kwargs['endpoint_url'] = self.endpoint_url

        self.s3_client = boto3.client('s3', **s3_kwargs)

        self.random_crop = retrieve(self.config, "random_crop", default=False)

        self._load()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        yield from self.data

    def __getitem__(self, i):
        return self.data[i]

    def _list_s3_images(self, bucket, prefix=""):
        """
        Recursively list all image files from S3 bucket with given prefix.
        Returns a list of S3 keys (relative paths within the bucket).
        """
        image_keys = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        try:
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        continue

                    _, ext = os.path.splitext(key.lower())
                    if ext in self.IMAGE_EXTENSIONS:
                        if prefix and key.startswith(prefix):
                            rel_key = key[len(prefix):].lstrip('/')
                        else:
                            rel_key = key
                        image_keys.append(key)

        except ClientError as e:
            raise RuntimeError(f"Failed to list objects from S3 bucket '{bucket}' with prefix '{prefix}'. Error: {e}")

        image_keys = sorted(image_keys)
        print(f"Found {len(image_keys)} images in s3://{bucket}/{prefix}")
        return image_keys

    def _filter_keys(self, keys):
        """Filter S3 keys if needed (e.g. exclude failed samples when skip_files is True)."""
        if not self._keys_to_skip:
            return keys
        return [k for k in keys if k not in self._keys_to_skip]

    def _load(self):
        """Load all image keys from S3 and create dataset."""
        s3_keys = self._list_s3_images(self.bucket, self.object_prefix)

        if len(s3_keys) == 0:
            raise ValueError(f"No images found in s3://{self.bucket}/{self.object_prefix}")

        l1 = len(s3_keys)
        s3_keys = self._filter_keys(s3_keys)
        if l1 != len(s3_keys):
            print(f"Removed {l1 - len(s3_keys)} files during filtering.")

        if self.object_prefix:
            relpaths = [key[len(self.object_prefix):].lstrip('/') if key.startswith(self.object_prefix) else key
                       for key in s3_keys]
        else:
            relpaths = s3_keys

        labels = {
            "relpath": np.array(relpaths),
            "s3_key": np.array(s3_keys),
        }

        self.data = S3ImagePaths(
            s3_keys=s3_keys,
            s3_client=self.s3_client,
            bucket=self.bucket,
            size=retrieve(self.config, "size", default=0),
            random_crop=self.random_crop,
            original_reso=retrieve(self.config, "original_reso", default=False),
            labels=labels,
            cache_dir=self.cache_dir
        )


class S3Images(S3ImagesBase):
    """
    Main dataset class for loading images from S3.
    Usage:
        config = {
            "bucket": "my-bucket",
            "object": "path/to/images/",
            "size": 256,
            "random_crop": True,
            "aws_access_key_id": "optional",
            "aws_secret_access_key": "optional",
            "region_name": "us-east-1",
            "endpoint_url": "https://...",  # optional, for S3-compatible APIs (e.g. RunPod)
            "cache_dir": "/tmp/s3_cache"
        }
        dataset = S3Images(config=config)
    """
    pass


# ---------------------------------------------------------------------------
# Local directory dataloader (same interface as S3, but reads from disk)
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_ROOT = "/workspace/models/EWM-DataCollection/images"


def _list_local_images(root):
    """Recursively list all image file paths under root. Returns sorted list of absolute paths."""
    image_paths = []
    for ext in S3ImagesBase.IMAGE_EXTENSIONS:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if os.path.splitext(f)[1].lower() == ext:
                    image_paths.append(os.path.join(dirpath, f))
    return sorted(image_paths)


class LocalImagesBase(IterableDataset):
    """
    Base class for loading images from a local directory (IterableDataset for sequential reads).
    Same config options as S3 (size, random_crop, original_reso) plus root path.
    If manifest_path is set in config: use that JSON cache (build via build_local_ibqgan_image_paths.py if missing).
    Optional skip_files (default True): exclude paths listed in the config's failed_samples JSON if present.
    """
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)

        skip_files = retrieve(self.config, "skip_files", default=True)
        failed_samples_path = retrieve(self.config, "failed_samples_path", default=None)
        manifest_path = retrieve(self.config, "manifest_path", default=None)
        # When failed_samples_path is None, default to derived path from manifest so skip_files can apply
        if failed_samples_path is None and manifest_path:
            derived = get_failed_samples_path_from_manifest(manifest_path)
            if derived:
                derived = os.path.abspath(os.path.expanduser(derived))
                if os.path.isfile(derived):
                    failed_samples_path = derived
        self._paths_to_skip = set()
        if skip_files and failed_samples_path:
            self._paths_to_skip = load_failed_paths(failed_samples_path, normalize=True)
            if self._paths_to_skip:
                log.info("Skipping %d paths from failed_samples: %s", len(self._paths_to_skip), failed_samples_path)

        if manifest_path:
            root_for_build = retrieve(self.config, "root", default=None)
            if root_for_build:
                root_for_build = os.path.abspath(os.path.expanduser(root_for_build))
            self.root, abspaths = ensure_manifest(manifest_path, root=root_for_build)
            self.random_crop = retrieve(self.config, "random_crop", default=False)
            self._load_from_paths(abspaths)
            return

        self.root = retrieve(
            self.config, "root", default=DEFAULT_LOCAL_ROOT
        )
        self.root = os.path.abspath(os.path.expanduser(self.root))
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Image root directory not found: {self.root}")

        self.random_crop = retrieve(self.config, "random_crop", default=False)
        self._load()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        yield from self.data

    def __getitem__(self, i):
        return self.data[i]

    def _filter_paths(self, paths):
        """Filter paths if needed (e.g. exclude failed samples when skip_files is True)."""
        if not self._paths_to_skip:
            return paths
        normalized = {os.path.normpath(os.path.abspath(p)) for p in paths}
        kept = [p for p in paths if os.path.normpath(os.path.abspath(p)) not in self._paths_to_skip]
        return kept

    def _load_from_paths(self, abspaths):
        """Create dataset from a precomputed list of absolute paths (e.g. from manifest)."""
        if len(abspaths) == 0:
            raise ValueError("No image paths in manifest.")
        n_before = len(abspaths)
        abspaths = self._filter_paths(abspaths)
        if n_before != len(abspaths):
            log.info("Removed %d files during filtering.", n_before - len(abspaths))
        relpaths = [os.path.relpath(p, self.root) for p in abspaths]
        labels = {"relpath": np.array(relpaths)}
        self.data = IterableImagePaths(
            abspaths,
            size=retrieve(self.config, "size", default=0),
            random_crop=self.random_crop,
            original_reso=retrieve(self.config, "original_reso", default=False),
            labels=labels,
        )
        log.info("Loaded %d images from manifest (root: %s)", len(self.data), self.root)

    def _load(self):
        """Discover all images under root and create dataset using ImagePaths (same preprocessing as IBQ)."""
        abspaths = _list_local_images(self.root)
        if len(abspaths) == 0:
            raise ValueError(f"No images found under {self.root}")

        n_before = len(abspaths)
        abspaths = self._filter_paths(abspaths)
        if n_before != len(abspaths):
            log.info("Removed %d files during filtering.", n_before - len(abspaths))

        relpaths = [os.path.relpath(p, self.root) for p in abspaths]
        labels = {
            "relpath": np.array(relpaths),
        }

        self.data = IterableImagePaths(
            abspaths,
            size=retrieve(self.config, "size", default=0),
            random_crop=self.random_crop,
            original_reso=retrieve(self.config, "original_reso", default=False),
            labels=labels,
        )
        log.info("Found %d images in %s", len(self.data), self.root)


class LocalImages(LocalImagesBase):
    """
    Dataset that loads all images from a local directory (recursive).
    Same preprocessing as S3Images (albumentations, size, random_crop, original_reso).
    Config:
        root: path to image directory (default: /workspace/models/EWM-DataCollection/images)
        size: resize/crop size (default 0 = no resize)
        random_crop: use random crop for training (default False)
        original_reso: if True, do not resize/crop (default False)
    """
    pass
