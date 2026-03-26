"""
Simple dataset for raw image folders (e.g. ubuntu desktop screenshots).
Returns {"image": HxWx3 float32 array in [-1,1], "file_path_": str}.
Compatible with the DataModuleFromConfig loader in main.py.
"""

import os
import glob
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class OCRImageDataset(Dataset):
    """
    Loads all images from a directory tree.

    Args:
        data_root  : root directory to scan for images
        size       : resize+crop to this square resolution
        extensions : image file extensions to include
        random_crop: use random crop (True for train, False for val)
    """

    EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

    def __init__(
        self,
        data_root: str,
        size: int = 256,
        random_crop: bool = True,
        extensions: tuple = None,
    ):
        self.size        = size
        self.random_crop = random_crop
        exts = extensions or self.EXTENSIONS

        self.paths = sorted([
            p for p in glob.glob(os.path.join(data_root, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in exts
        ])
        assert len(self.paths) > 0, f"No images found in {data_root}"

        # Albumentations preprocessing pipeline
        rescaler = albumentations.SmallestMaxSize(max_size=size)
        cropper  = (
            albumentations.RandomCrop(height=size, width=size)
            if random_crop
            else albumentations.CenterCrop(height=size, width=size)
        )
        self.preprocessor = albumentations.Compose([rescaler, cropper])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img).astype(np.uint8)
        img = self.preprocessor(image=img)["image"]
        img = (img / 127.5 - 1.0).astype(np.float32)   # → [-1, 1]
        return {"image": img, "file_path_": path}


class OCRImageTrain(OCRImageDataset):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(
            data_root=config.get("data_root", "data"),
            size=config.get("size", 1024),
            random_crop=config.get("random_crop", True),
        )


class OCRImageValidation(OCRImageDataset):
    def __init__(self, config=None):
        config = config or {}
        super().__init__(
            data_root=config.get("data_root", "data"),
            size=config.get("size", 1024),
            random_crop=False,
        )
